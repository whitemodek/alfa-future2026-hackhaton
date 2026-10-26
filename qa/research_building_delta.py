"""Offline building residual forecasting with strictly past entity labels.

One frozen hypothesis, two temporal folds, inner-only early stopping. Never
loaded by the runtime. Previously inspected outer folds make this exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_causal_stats import _sha, _save


def residual_history(train, target, query, columns, time_column, deadline=math.inf):
    """Training residuals and strictly past query lags; no query-target input."""
    from ml_lag import causal_lags
    raw_columns = [*columns, time_column]
    train_lag, query_lag, diagnostics = causal_lags(
        train[raw_columns], target, query[raw_columns], columns, time_column, deadline)
    available = np.isfinite(train_lag)
    residual = np.full(len(train), np.nan)
    residual[available] = np.asarray(target, dtype=float)[available] - train_lag[available]
    return dict(train_lag=train_lag, query_lag=query_lag,
                train_available=available, query_available=np.isfinite(query_lag),
                residual_target=residual, diagnostics=diagnostics)


def delta_pipeline(existing, history, query_lag, delta_available):
    """Change only rows with usable history; missing rows keep exact baseline."""
    existing, history, query_lag = [np.asarray(a, dtype=float).reshape(-1)
                                    for a in (existing, history, query_lag)]
    mask = np.isfinite(query_lag)
    delta_available = np.asarray(delta_available, dtype=float).reshape(-1)
    if (len(existing) != len(history) or len(existing) != len(query_lag)
            or len(delta_available) != int(mask.sum())
            or not np.isfinite(existing).all() or not np.isfinite(history).all()
            or not np.isfinite(delta_available).all()):
        raise ValueError('Finite, aligned baseline/history/delta arrays are required')
    result = existing.copy()
    result[mask] = .5 * history[mask] + .5 * (query_lag[mask] + delta_available)
    return result


def _residual_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def _load_current(root, release, history_report):
    """Load hashed history vectors and actual runtime validation predictions."""
    release_records = json.loads(release.read_text(encoding='utf-8'))
    primary = release_records[0]
    release_root = Path(primary['work_dir']) / 'data'
    journal = next(release_root.glob('.agent-*/events.jsonl'))
    events = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
    results = [e['result'].get('response', {}) for e in events if e['event'] == 'backend']
    final = [r for r in results if r.get('model') == 'lag' and r.get('fit_scope') == 'full_train'][-1]
    manifest_path = Path(final['manifest_path'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    validation_path = Path(manifest['validation_path'])
    if _sha(validation_path) != manifest['validation_sha256']:
        raise ValueError('Actual primary validation predictions changed')
    primary_prediction = np.load(validation_path, allow_pickle=False)['prediction']
    secondary_lag_path = root.parents[1] / 'late-lag-feature' / 'xgboost-lagTrue-period4.npy'
    history_records = json.loads(history_report.read_text(encoding='utf-8'))['records']
    current, provenance = {}, {}
    for period in (5, 4):
        record = next(r for r in history_records if r['outer_period'] == period)
        artifact = record['arrays']
        path = Path(artifact['path'])
        if _sha(path) != artifact['sha256']:
            raise ValueError('Saved history confirmation arrays changed')
        saved = np.load(path, allow_pickle=False)
        history = saved['history_prediction']
        prediction = (primary_prediction if period == 5 else
                      .5 * history + .5 * np.load(secondary_lag_path, allow_pickle=False))
        expected = float(primary['best']['score']) if period == 5 else 32.64325416847114
        score = float(np.sqrt(np.mean((prediction - saved['truth']) ** 2)))
        if not np.isclose(score, expected, rtol=1e-10, atol=1e-10):
            raise ValueError('Actual baseline score no longer reproduces')
        current[period] = dict(prediction=prediction, history=history,
            indices=saved['outer_indices'], truth=saved['truth'], expected_score=expected)
        provenance[str(period)] = dict(history_arrays=artifact, history_sources=record['provenance'])
    provenance['primary_validation'] = dict(path=str(validation_path), sha256=_sha(validation_path),
                                           manifest_path=str(manifest_path), manifest_sha256=_sha(manifest_path))
    provenance['secondary_lag'] = dict(path=str(secondary_lag_path), sha256=_sha(secondary_lag_path))
    return current, provenance


def run(root, output, release, history_report, seconds=300):
    import ml_backend as b
    import ml_history as h
    from ml_lag import XGB_REQUEST, _augment
    from ml_models import fit

    output.parent.mkdir(parents=True, exist_ok=True)
    current, baseline_provenance = _load_current(root, release, history_report)
    protocol = dict(version=1, hypothesis='predict last strictly past target plus learned residual',
        model='xgboost', model_request=dict(XGB_REQUEST), lag_features='existing two features only',
        outer_periods=[5, 4], inner='latest available outer-training period', threads=2,
        seconds=seconds, history_weight=.5, residual_weight=.5,
        missing_history_fallback='exact existing full runtime prediction for that row',
        gate='At least 0.1 RMSE reduction against actual current runtime on BOTH periods; '
             'engineering threshold, not statistical significance. No further tuning.',
        note='Research only until the gate and an independent integration review pass. '
             'Outer periods were previously inspected; not an unseen test.',
        source_sha256=_sha(__file__), baseline_provenance=baseline_provenance,
        inputs={name: _sha(root / name) for name in ('train.csv', 'test.csv', 'task.md')},
        runtime_sources={name: _sha(Path(__file__).resolve().parents[1] / name)
                         for name in ('ml_models.py', 'ml_backend.py', 'ml_lag.py', 'ml_history.py')},
        release_report_sha256=_sha(release), history_report_sha256=_sha(history_report))
    protocol_path = output.with_name(output.stem + '-protocol.json')
    _save(protocol_path, protocol)
    report = dict(protocol_sha256=_sha(protocol_path), records=[], complete=False,
                  production_changed=False, external_api_requests=0)
    _save(output, report)
    started, deadline = time.monotonic(), time.monotonic() + seconds
    problem = b.load_problem(dict(task_dir=str(root), task_path=str(root / 'task.md'), request={}))
    columns, _ = h.entity_columns(problem, {})
    if not columns:
        raise ValueError('No conservative entity signature available')
    time_column = 'Year_Factor'
    x, _, categorical = b._features(problem)
    raw, y = problem['train'], problem['y']
    report.update(entity_columns=columns, time_column=time_column)
    artifacts = output.with_name(output.stem + '-artifacts')
    artifacts.mkdir(exist_ok=True)
    candidates = {}
    for period in (5, 4):
        ti = np.flatnonzero(raw[time_column].to_numpy() < period)
        vi = np.flatnonzero(raw[time_column].to_numpy() == period)
        inner_period = raw.iloc[ti][time_column].max()
        iti = np.flatnonzero(raw[time_column].to_numpy() < inner_period)
        ivi = np.flatnonzero(raw[time_column].to_numpy() == inner_period)
        np.testing.assert_array_equal(vi, current[period]['indices'])
        np.testing.assert_allclose(y.iloc[vi], current[period]['truth'], atol=0, rtol=0)
        inner = residual_history(raw.iloc[iti], y.iloc[iti], raw.iloc[ivi], columns, time_column, deadline)
        outer = residual_history(raw.iloc[ti], y.iloc[ti], raw.iloc[vi], columns, time_column, deadline)
        ia, iq = inner['train_available'], inner['query_available']
        oa, oq = outer['train_available'], outer['query_available']
        inner_residual = inner['residual_target'][ia]
        inner_valid_residual = y.iloc[ivi].to_numpy()[iq] - inner['query_lag'][iq]
        outer_residual = outer['residual_target'][oa]
        record = dict(outer_period=period, inner_period=int(inner_period), status='started',
            inner_available_training_rows=int(ia.sum()), inner_available_validation_rows=int(iq.sum()),
            outer_available_training_rows=int(oa.sum()), outer_available_validation_rows=int(oq.sum()),
            inner_diagnostics=inner['diagnostics'], outer_diagnostics=outer['diagnostics'],
            inner_training_residual_sha256=_residual_hash(inner_residual),
            outer_training_residual_sha256=_residual_hash(outer_residual),
            actual_current_rmse=current[period]['expected_score'])
        report['records'].append(record)
        _save(output, report)
        began = time.monotonic()
        try:
            if min(int(ia.sum()), int(iq.sum()), int(oa.sum()), int(oq.sum())) < 2:
                raise ValueError('Insufficient usable history for residual forecasting')
            xt = _augment(x.iloc[iti], inner['train_lag']).iloc[np.flatnonzero(ia)]
            xv = _augment(x.iloc[ivi], inner['query_lag']).iloc[np.flatnonzero(iq)]
            model = fit('xgboost', xt, inner_residual, xv, inner_valid_residual, categorical,
                        dict(XGB_REQUEST), min(deadline - 15, time.monotonic() + 75),
                        2, 'regression', 'rmse')
            inner_prediction = model.predict(xv)
            record.update(selected_iterations=model.tree_count_,
                inner_residual_rmse=float(np.sqrt(np.mean((inner_prediction - inner_valid_residual) ** 2))))
            full_x = _augment(x.iloc[ti], outer['train_lag']).iloc[np.flatnonzero(oa)]
            query_x = _augment(x.iloc[vi], outer['query_lag']).iloc[np.flatnonzero(oq)]
            final = model.refit(full_x, outer_residual,
                                min(deadline - 2, time.monotonic() + 75), 2)
            delta_prediction = final.predict(query_x)
            candidate = delta_pipeline(current[period]['prediction'], current[period]['history'],
                                       outer['query_lag'], delta_prediction)
            np.testing.assert_array_equal(candidate[~oq], current[period]['prediction'][~oq])
            stem = f'period{period}'
            inner_model_path = artifacts / (stem + '-inner.joblib')
            outer_model_path = artifacts / (stem + '-outer.joblib')
            arrays_path = artifacts / (stem + '.npz')
            model.save_model(inner_model_path)
            final.save_model(outer_model_path)
            np.savez_compressed(arrays_path, inner_training_indices=iti[ia],
                inner_validation_indices=ivi[iq], outer_training_indices=ti[oa],
                outer_validation_indices=vi, outer_available_indices=vi[oq],
                inner_training_residual=inner_residual, inner_validation_residual=inner_valid_residual,
                outer_training_residual=outer_residual, inner_prediction=inner_prediction,
                query_lag=outer['query_lag'], delta_prediction=delta_prediction,
                candidate=candidate, baseline=current[period]['prediction'],
                history=current[period]['history'], truth=y.iloc[vi].to_numpy())
            record['artifacts'] = {key: dict(path=str(path), sha256=_sha(path))
                for key, path in (('inner_model', inner_model_path), ('outer_model', outer_model_path),
                                  ('arrays', arrays_path))}
            record.update(status='fitted', exact_refit_iterations=final.tree_count_,
                          feature_names=list(full_x.columns), fallback_unchanged=True)
            candidates[period] = candidate
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, message=str(exc)[:400])
        record['seconds'] = round(time.monotonic() - began, 3)
        _save(output, report)

    # No model/hyperparameter/weight changes are permitted after outer scores.
    for record in report['records']:
        period = record['outer_period']
        if period in candidates:
            score = float(np.sqrt(np.mean((candidates[period] - current[period]['truth']) ** 2)))
            record.update(status='completed', candidate_rmse=score,
                          delta_vs_actual_current=score - record['actual_current_rmse'])
    report['complete'] = len(report['records']) == 2 and all(r['status'] == 'completed' for r in report['records'])
    report['passes_gate'] = report['complete'] and all(r['delta_vs_actual_current'] <= -.1 for r in report['records'])
    report['seconds'] = round(time.monotonic() - started, 3)
    report['inputs_unchanged'] = all(_sha(root / name) == digest for name, digest in protocol['inputs'].items())
    report['recommendation'] = ('Separate integration review required' if report['passes_gate'] else
                                'Research only: frozen gate not passed')
    _save(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.test-temp/organizer/regression'))
    parser.add_argument('--release', type=Path, default=Path('.test-temp/platform-upgrade-report.json'))
    parser.add_argument('--history-report', type=Path, default=Path('.test-temp/causal-stats-pipeline-report.json'))
    parser.add_argument('--output', type=Path, default=Path('.test-temp/building-delta-report.json'))
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve(), args.release.resolve(),
                         args.history_report.resolve(), args.seconds), indent=2))
