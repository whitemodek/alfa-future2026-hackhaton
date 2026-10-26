"""Fixed causal weather-change residual model, two temporal backtests.

The task explicitly supplies observed weather for the prediction year. Delta
features compare it with the same entity's last strictly earlier TRAIN row;
no query targets or same-period features enter the historical snapshot.
"""
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import sha256


def causal_feature_deltas(train, query, columns, time_column, feature_columns, deadline=math.inf):
    """Only covariates enter the interface; ambiguous/missing identities fall back."""
    if (not columns or not feature_columns or time_column in columns
            or any(c not in train or c not in query for c in [*columns, time_column, *feature_columns])):
        raise ValueError('Complete entity/time/feature columns are required')
    a = train[time_column].to_numpy(dtype=float)
    q = query[time_column].to_numpy(dtype=float)
    if not len(a) or not len(q) or not np.isfinite(a).all() or not np.isfinite(q).all() or q.min() <= a.max():
        raise ValueError('Queries must occur strictly after finite training periods')
    def keys(frame):
        result = []
        for row in frame[columns].itertuples(index=False, name=None):
            result.append(None if any(pd.isna(v) or (isinstance(v, (int, float)) and not math.isfinite(v))
                                      for v in row) else row)
        return result
    key, query_key = keys(train), keys(query)
    values = train[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    target_values = query[feature_columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    result = np.full_like(values, np.nan)
    query_result = np.full_like(target_values, np.nan)
    latest, ambiguous = {}, set()
    order = np.argsort(a, kind='stable')
    position = 0
    while position < len(order):
        if time.monotonic() >= deadline:
            raise TimeoutError('Weather snapshot deadline exhausted')
        stop = position + 1
        while stop < len(order) and a[order[stop]] == a[order[position]]:
            stop += 1
        current = order[position:stop]
        counts = {}
        for i in current:
            k = key[i]
            if k is not None:
                if k in latest and k not in ambiguous:
                    result[i] = values[i] - latest[k]
                counts[k] = counts.get(k, 0) + 1
        # Only now make this period visible to subsequent periods.
        ambiguous.update(k for k, count in counts.items() if count > 1)
        for i in current:
            k = key[i]
            if k is not None and k not in ambiguous:
                latest[k] = values[i].copy()
        position = stop
    for i, k in enumerate(query_key):
        if k is not None and k in latest and k not in ambiguous:
            query_result[i] = target_values[i] - latest[k]
    result[~np.isfinite(result)] = np.nan
    query_result[~np.isfinite(query_result)] = np.nan
    return result, query_result


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'weather-delta-study'
    out.mkdir(exist_ok=False)
    from qa.research_building_delta import residual_history, delta_pipeline
    from ml_lag import XGB_REQUEST, _augment
    import ml_backend as b
    from ml_history import entity_columns
    from ml_models import fit
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/regression'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, cats = b._features(p)
    p['validation'] = dict(strategy='time', diagnostics={'time_column': 'Year_Factor'})
    columns, rationale = entity_columns(p, {})
    time_column = 'Year_Factor'
    weather = [c for c in p['features'] if c not in columns and c != time_column
               and re.search('temp|degree|days|precip|snow|wind|fog', c, re.I)]
    protocol = dict(periods=[5, 4], feature_columns=weather, entity_columns=columns, rationale=rationale,
        model_request=XGB_REQUEST, residual_only_weight_in_delta_pipeline=.5,
        delta_pipeline_weight_in_final=.5, grid_search=False, api_calls=0, platform_submissions=0,
        gate='Relative RMSE gain >= .001 on BOTH complete outer periods, before runtime follow-up',
        source_sha256={n: sha256(root / n) for n in ('qa/research_weather_delta.py', 'ml_models.py', 'ml_lag.py')},
        limitation='Previously inspected temporal folds; fresh model/feature hypothesis, not a blind estimate or platform score')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, records=[], completed=False, input_sha256=b._input_hashes(p))
    started = time.monotonic()
    deadline = started + 600

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def augment(frame, lag, deltas):
        result = _augment(frame, lag)
        for i, name in enumerate(weather):
            derived = '__weather_change_' + name
            if derived in result:
                raise ValueError('Derived feature collides with an input column')
            result[derived] = deltas[:, i]
        return result

    save()
    for period in protocol['periods']:
        print('FIT', period, flush=True)
        a = np.flatnonzero(p['train'][time_column].to_numpy() < period - 1)
        v = np.flatnonzero(p['train'][time_column].to_numpy() == period - 1)
        train = np.flatnonzero(p['train'][time_column].to_numpy() < period)
        valid = np.flatnonzero(p['train'][time_column].to_numpy() == period)
        inner = residual_history(p['train'].iloc[a], p['y'].iloc[a], p['train'].iloc[v], columns, time_column, deadline)
        outer = residual_history(p['train'].iloc[train], p['y'].iloc[train], p['train'].iloc[valid], columns, time_column, deadline)
        inner_delta, inner_query_delta = causal_feature_deltas(p['train'].iloc[a], p['train'].iloc[v], columns,
            time_column, weather, deadline)
        outer_delta, outer_query_delta = causal_feature_deltas(p['train'].iloc[train], p['train'].iloc[valid], columns,
            time_column, weather, deadline)
        mask, query_mask = inner['train_available'], inner['query_available']
        outer_mask, outer_query_mask = outer['train_available'], outer['query_available']
        with threadpool_limits(limits=2):
            model = fit('xgboost', augment(x.iloc[a], inner['train_lag'], inner_delta).iloc[np.flatnonzero(mask)],
                inner['residual_target'][mask], augment(x.iloc[v], inner['query_lag'], inner_query_delta).iloc[np.flatnonzero(query_mask)],
                p['y'].iloc[v].to_numpy()[query_mask] - inner['query_lag'][query_mask], cats, XGB_REQUEST,
                min(deadline, time.monotonic() + 100), 2, 'regression', 'rmse')
            row = dict(period=period, selected_iterations=model.tree_count_, feature_names=model.feature_names_)
            report['records'].append(row)
            save()
            full = model.refit(augment(x.iloc[train], outer['train_lag'], outer_delta).iloc[np.flatnonzero(outer_mask)],
                outer['residual_target'][outer_mask], min(deadline, time.monotonic() + 120), 2)
            residual = full.predict(augment(x.iloc[valid], outer['query_lag'], outer_query_delta).iloc[np.flatnonzero(outer_query_mask)])
        cache = scratch / f'building-delta-report-artifacts/period{period}.npz'
        row['control_cache_sha256'] = sha256(cache)
        with np.load(cache, allow_pickle=False) as source:
            if not np.array_equal(source['outer_validation_indices'], valid):
                raise ValueError('Full-period cache alignment changed')
            base, history = source['baseline'].copy(), source['history'].copy()
        candidate_delta = delta_pipeline(base, history, outer['query_lag'], residual)
        prediction = .5 * base + .5 * candidate_delta
        np.savez_compressed(out / f'period{period}-frozen.npz', prediction=prediction, valid_indices=valid)
        truth = p['y'].iloc[valid].to_numpy()
        old = float(np.sqrt(np.mean((base - truth) ** 2)))
        new = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        row.update(control_rmse=old, candidate_rmse=new, relative_gain=(old - new) / old)
        save()
        print('RESULT', period, old, new, row['relative_gain'], flush=True)
    report.update(completed=True, passed=all(r['relative_gain'] >= .001 for r in report['records']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
