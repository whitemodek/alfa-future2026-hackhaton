"""Offline paired research of strictly past entity statistics; never a runtime module.

All configurations are fixed before the two outer folds are evaluated. Those
folds were previously used during development, so this is exploratory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path
import time

import numpy as np
import pandas as pd


STAT_COLUMNS = ('__history_target_lag', '__history_available', '__history_mean_last2',
                '__history_trend_per_period', '__history_age', '__history_count')


def causal_stats(train, target, query, columns, time_column, deadline=math.inf):
    """Build statistics using strictly earlier periods, never query labels.

    Rows at a collided entity-period can use the preceding unambiguous history;
    from the end of that period onward the key is disabled permanently. Missing
    identity has NaN statistics, availability zero and count zero. Count is the
    number of distinct usable past observations, not a count of query rows.
    """
    if (not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
            or time_column in columns
            or any(c not in train or c not in query for c in [*columns, time_column])):
        raise ValueError('Distinct entity columns and a separate time column are required')
    if not len(train) or not len(query):
        raise ValueError('Nonempty training and query frames are required')
    for frame in (train, query):
        periods = frame[time_column]
        if (not pd.api.types.is_numeric_dtype(periods) or periods.isna().any()
                or not np.isfinite(periods.to_numpy(dtype=float)).all()):
            raise ValueError('Periods must be numeric, finite and nonmissing')
    periods = train[time_column].to_numpy(dtype=float)
    query_periods = query[time_column].to_numpy(dtype=float)
    if query_periods.min() <= periods.max():
        raise ValueError('Every query period must be later than every training period')
    targets = np.asarray(target, dtype=float).reshape(-1)
    if len(targets) != len(train) or not np.isfinite(targets).all():
        raise ValueError('Training targets must be finite and aligned')

    def identity(row):
        for value in row:
            try:
                if pd.isna(value) or (isinstance(value, numbers.Number) and not math.isfinite(value)):
                    return None
            except (TypeError, ValueError, OverflowError):
                return None
        try:
            hash(row)
        except TypeError:
            return None
        return row

    keys = [identity(row) for row in train[columns].itertuples(index=False, name=None)]
    query_keys = [identity(row) for row in query[columns].itertuples(index=False, name=None)]
    training = np.full((len(train), 6), np.nan)
    prediction = np.full((len(query), 6), np.nan)
    training[:, [1, 5]] = 0
    prediction[:, [1, 5]] = 0
    # key -> (last_period, last_target, previous_period, previous_target, count)
    history, ambiguous = {}, set()

    def features(key, period):
        item = history.get(key)
        if item is None or key in ambiguous:
            return [np.nan, 0, np.nan, np.nan, np.nan, 0]
        last_time, last_value, previous_time, previous_value, count = item
        mean = (last_value + previous_value) / 2 if count >= 2 else last_value
        trend = ((last_value - previous_value) / (last_time - previous_time)
                 if count >= 2 else np.nan)
        return [last_value, 1, mean, trend, period - last_time, count]

    ordered = np.argsort(periods, kind='stable')
    position = 0
    while position < len(ordered):
        if time.monotonic() >= deadline:
            raise TimeoutError('Causal statistic deadline reached')
        stop = position + 1
        period = periods[ordered[position]]
        while stop < len(ordered) and periods[ordered[stop]] == period:
            stop += 1
        counts = {}
        current = ordered[position:stop]
        for i in current:
            key = keys[i]
            if key is not None:
                training[i] = features(key, period)
                counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            if count > 1:
                ambiguous.add(key)
                history.pop(key, None)
        for i in current:
            key = keys[i]
            if key is not None and key not in ambiguous:
                previous = history.get(key)
                history[key] = (period, targets[i], previous[0] if previous else np.nan,
                                previous[1] if previous else np.nan,
                                previous[4] + 1 if previous else 1)
        position = stop
    for i, (key, period) in enumerate(zip(query_keys, query_periods)):
        if key is not None:
            prediction[i] = features(key, period)
    return (pd.DataFrame(training, index=train.index, columns=STAT_COLUMNS),
            pd.DataFrame(prediction, index=query.index, columns=STAT_COLUMNS),
            dict(training_coverage=float(training[:, 1].mean()),
                 query_coverage=float(prediction[:, 1].mean()), ambiguous_keys=len(ambiguous),
                 missing_training_rows=sum(k is None for k in keys)))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(root, output, seconds=360):
    from ml_backend import load_problem, _features
    from ml_history import entity_columns
    from ml_lag import XGB_REQUEST, causal_lags
    from ml_models import fit

    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, model='xgboost', model_request=dict(XGB_REQUEST),
        variants={'control': list(STAT_COLUMNS[:2]), 'stats6': list(STAT_COLUMNS)},
        outer_periods=[5, 4], inner='last available training period', threads=2,
        total_seconds=seconds, seed=42, source_sha256=_sha(__file__),
        continuation_gate=dict(minimum_rmse_reduction_each_outer=.1,
            meaning='Engineering threshold, not a significance claim. '
                    'Both folds must pass before separate full-pipeline confirmation.'),
        inputs={name: _sha(root / name) for name in ('train.csv', 'test.csv', 'task.md')},
        runtime_sources={name: _sha(Path(__file__).resolve().parents[1] / name)
                         for name in ('ml_models.py', 'ml_backend.py', 'ml_lag.py',
                                      'ml_history.py', 'ml_validation.py')},
        replay_note='If repeated, the model request, folds, feature definitions and '
                    'weights remain unchanged. Saving models/predictions adds provenance only.',
        note='Exploratory: these outer periods were previously used in development. '
             'Fixed hypotheses, inner-only early stopping and exact outer refit. '
             'No query target input; no platform score or improvement claim.')
    protocol_path = output.with_name(output.stem + '-protocol.json')
    _save(protocol_path, protocol)  # Freeze before any new outer scoring.
    report = dict(protocol_sha256=_sha(protocol_path), records=[], complete=False,
                  production_changed=False, external_api_requests=0)
    started = time.monotonic()
    deadline = started + seconds
    problem = load_problem(dict(task_dir=str(root), task_path=str(root / 'task.md'), request={}))
    columns, reason = entity_columns(problem, {})
    if not columns:
        raise ValueError(reason)
    time_column = 'Year_Factor'
    report.update(entity_columns=columns, time_column=time_column)
    raw = problem['train']
    y = problem['y']
    x, _, categorical = _features(problem)
    if any(c in x for c in STAT_COLUMNS):
        raise ValueError('Research feature name collision')
    predictions = {}
    artifacts = output.with_name(output.stem + '-artifacts')
    artifacts.mkdir(exist_ok=True)

    for outer in protocol['outer_periods']:
        outer_ti = np.flatnonzero(raw[time_column].to_numpy() < outer)
        outer_vi = np.flatnonzero(raw[time_column].to_numpy() == outer)
        inner_period = raw.iloc[outer_ti][time_column].max()
        inner_ti = np.flatnonzero(raw[time_column].to_numpy() < inner_period)
        inner_vi = np.flatnonzero(raw[time_column].to_numpy() == inner_period)
        raw_columns = [*columns, time_column]
        it, iv, inner_diag = causal_stats(raw.iloc[inner_ti][raw_columns], y.iloc[inner_ti],
            raw.iloc[inner_vi][raw_columns], columns, time_column, deadline)
        ot, ov, outer_diag = causal_stats(raw.iloc[outer_ti][raw_columns], y.iloc[outer_ti],
            raw.iloc[outer_vi][raw_columns], columns, time_column, deadline)
        # Independently prove that the first two features reproduce real runtime.
        for ti, vi, a, b in ((inner_ti, inner_vi, it, iv), (outer_ti, outer_vi, ot, ov)):
            control_t, control_v, _ = causal_lags(raw.iloc[ti][raw_columns], y.iloc[ti],
                raw.iloc[vi][raw_columns], columns, time_column, deadline)
            np.testing.assert_allclose(a.iloc[:, 0], control_t, equal_nan=True)
            np.testing.assert_allclose(b.iloc[:, 0], control_v, equal_nan=True)
            np.testing.assert_array_equal(a.iloc[:, 1], np.isfinite(control_t).astype(int))
            np.testing.assert_array_equal(b.iloc[:, 1], np.isfinite(control_v).astype(int))
        for variant, names in protocol['variants'].items():
            record = dict(outer_period=outer, inner_period=int(inner_period), variant=variant,
                inner_training_rows=len(inner_ti), inner_validation_rows=len(inner_vi),
                outer_training_rows=len(outer_ti), outer_validation_rows=len(outer_vi),
                inner_diagnostics=inner_diag, outer_diagnostics=outer_diag, status='started')
            report['records'].append(record)
            _save(output, report)
            trial_started = time.monotonic()
            try:
                if deadline - time.monotonic() < 15:
                    raise TimeoutError('Insufficient shared research budget')
                xt = pd.concat([x.iloc[inner_ti], it[names]], axis=1)
                xv = pd.concat([x.iloc[inner_vi], iv[names]], axis=1)
                model = fit('xgboost', xt, y.iloc[inner_ti], xv, y.iloc[inner_vi],
                    categorical, dict(XGB_REQUEST), min(deadline - 10, time.monotonic() + 55),
                    2, 'regression', 'rmse')
                record['selected_iterations'] = model.tree_count_
                record['inner_rmse'] = float(np.sqrt(np.mean((model.predict(xv) - y.iloc[inner_vi]) ** 2)))
                full_x = pd.concat([x.iloc[outer_ti], ot[names]], axis=1)
                outer_x = pd.concat([x.iloc[outer_vi], ov[names]], axis=1)
                final = model.refit(full_x, y.iloc[outer_ti],
                                    min(deadline - 2, time.monotonic() + 45), 2)
                prediction = final.predict(outer_x)
                if not np.isfinite(prediction).all():
                    raise ValueError('Nonfinite outer prediction')
                predictions[(outer, variant)] = prediction
                stem = f'period{outer}-{variant}'
                inner_model_path = artifacts / (stem + '-inner.joblib')
                outer_model_path = artifacts / (stem + '-outer.joblib')
                arrays_path = artifacts / (stem + '.npz')
                model.save_model(inner_model_path)
                final.save_model(outer_model_path)
                np.savez_compressed(arrays_path, prediction=prediction,
                    outer_truth=y.iloc[outer_vi].to_numpy(),
                    inner_training_indices=inner_ti, inner_validation_indices=inner_vi,
                    outer_training_indices=outer_ti, outer_validation_indices=outer_vi,
                    inner_predictions=model.predict(xv), inner_truth=y.iloc[inner_vi].to_numpy())
                record['artifacts'] = {key: dict(path=str(path), sha256=_sha(path))
                    for key, path in (('inner_model', inner_model_path),
                                      ('outer_model', outer_model_path), ('arrays', arrays_path))}
                record['feature_names'] = list(full_x.columns)
                record.update(status='fitted', exact_refit_iterations=final.tree_count_)
            except Exception as exc:
                record.update(status='failed', error_type=type(exc).__name__, message=str(exc)[:300])
            record['seconds'] = round(time.monotonic() - trial_started, 3)
            _save(output, report)

    # Score only after every predeclared training run has finished. No choice is
    # made using these outer labels or scores, and no runtime changes are made.
    for record in report['records']:
        key = record['outer_period'], record['variant']
        if key in predictions:
            indices = np.flatnonzero(raw[time_column].to_numpy() == key[0])
            record['outer_rmse'] = float(np.sqrt(np.mean((predictions[key] - y.iloc[indices]) ** 2)))
            record['status'] = 'completed'
    report['complete'] = all(r['status'] == 'completed' for r in report['records'])
    report['seconds'] = round(time.monotonic() - started, 3)
    report['inputs_unchanged'] = all(_sha(root / name) == digest
                                     for name, digest in protocol['inputs'].items())
    report['pairs'] = []
    for outer in protocol['outer_periods']:
        pair = {r['variant']: r for r in report['records']
                if r['outer_period'] == outer and r['status'] == 'completed'}
        if len(pair) == 2:
            report['pairs'].append(dict(outer_period=outer,
                control_rmse=pair['control']['outer_rmse'], stats6_rmse=pair['stats6']['outer_rmse'],
                delta=pair['stats6']['outer_rmse'] - pair['control']['outer_rmse']))
    report['improves_both'] = len(report['pairs']) == 2 and all(p['delta'] < 0 for p in report['pairs'])
    report['passes_continuation_gate'] = (len(report['pairs']) == 2
        and all(p['delta'] <= -.1 for p in report['pairs']))
    _save(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.test-temp/organizer/regression'))
    parser.add_argument('--output', type=Path, default=Path('.test-temp/causal-stats-report.json'))
    parser.add_argument('--seconds', type=int, default=360)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve(), args.seconds), indent=2))
