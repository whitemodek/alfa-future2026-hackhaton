"""Actual train/validate/refit/predict pipeline on two complete future periods.

Outer answers are removed from solver directories before any fitting. Baseline
and residual variant are selected using INTERNAL validation only, then refit
on all available past rows before the parent scores future-period predictions.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import sha256, array_hash


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'forward-residual-pipeline'
    out.mkdir(exist_ok=False)
    protocol = dict(periods=[5, 4], threads=4, control_seconds=240, residual_seconds=180, refit_seconds=360,
        selection='Add residual only on a positive INTERNAL RMSE gain; outer future labels never select models/counts/weights',
        blend='Fixed .5 direct + .25 history + .25 past-target-plus-residual on known entities',
        cold_start='If fewer than10 inner causal pairs: exactly128 rounds, no outer-label selection',
        gate='Selected variant improves BOTH full future periods; pooled relative RMSE gain >=.005',
        api_calls=0, platform_submissions=0, max_seconds=2400,
        deviation='The earlier cached half-blend missed its per-period .1% gate (latest gain .0625%). '
                  'That gate remains failed. This new protocol tests the ACTUAL complete nested forward pipeline '
                  'with a pooled .5% gate and improvement on both full future periods.',
        source_sha256={n: sha256(root / n) for n in ('ml_residual.py', 'ml_models.py', 'ml_lag.py',
            'ml_backend.py', 'ml_portfolio.py', 'qa/forward_residual_pipeline.py')},
        limitation='Previously explored dataset/periods, not a statistically untouched experiment or platform result; '
                   'extra compute for the variant. Full execution excludes outer labels from fitting/selection.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    deadline = started + 2400
    report = dict(protocol=protocol, records=[], completed=False)
    import ml_backend as b
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/regression'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    report['original_input_sha256'] = b._input_hashes(p)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def run(env, request, budget):
        left = min(budget, deadline - time.monotonic())
        if left < 3:
            raise TimeoutError('Forward comparison deadline exhausted')
        return b.run(dict(env, request=request, time_budget_seconds=left))

    save()
    errors = {'control': [], 'candidate': [], 'selected': []}
    for period in protocol['periods']:
        print('START', period, flush=True)
        train = np.flatnonzero(p['train']['Year_Factor'].to_numpy() < period)
        valid = np.flatnonzero(p['train']['Year_Factor'].to_numpy() == period)
        folder = out / f'period{period}'
        folder.mkdir()
        p['train'].iloc[train].to_csv(folder / 'train.csv', index=False)
        p['train'].iloc[valid].drop(columns=p['target']).to_csv(folder / 'test.csv', index=False)
        template = p['train'].iloc[valid][p['id_columns']].copy()
        template['site_eui'] = 0.
        template.to_csv(folder / 'sample_submission.csv', index=False)
        (folder / 'task.md').write_text('Tabular regression, predict site_eui. Primary metric RMSE.\n'
            'Train: train.csv. Test: test.csv. Template: sample_submission.csv. ID: id, never a feature.\n'
            'Year_Factor is a chronological year code; test is a later complete period. '
            'Weather in that year is observed and may be used. Write submission.csv.\n', encoding='utf-8')
        row = dict(period=period, source_rows=len(train), future_rows=len(valid),
                   train_indices_sha256=array_hash(train), future_indices_sha256=array_hash(valid), completed=False)
        report['records'].append(row)
        env = dict(task_dir=str(folder), task_path=str(folder / 'task.md'), threads=4)
        with threadpool_limits(limits=4):
            control = run(env, dict(action='portfolio'), protocol['control_seconds'])
            row['control'] = control
            save()
            print('CONTROL', period, control['model'], control['score'], flush=True)
            if control['model'] == 'lag':
                candidate = run(env, dict(action='residual', manifest_path=control['manifest_path']), protocol['residual_seconds'])
            else:
                row['candidate_skipped'] = 'Residual path requires a causal-lag control; preserve other source types'
                candidate = control
            if any(candidate[k] != control[k] for k in ('validation_id', 'metric', 'direction')):
                raise ValueError('Residual candidate changed native validation identity')
            selected = candidate['score'] < control['score']
            row['candidate'] = candidate
            row['selected_residual'] = bool(selected)
            save()
            print('RESIDUAL', period, candidate['score'], 'SELECTED', selected, flush=True)
            same = candidate['manifest_path'] == control['manifest_path']
            if control['fit_scope'] == 'holdout_train':
                control = run(env, dict(action='refit', manifest_path=control['manifest_path']), protocol['refit_seconds'])
            candidate = control if same else run(env, dict(action='refit', manifest_path=candidate['manifest_path']), protocol['refit_seconds'])
        if control['fit_scope'] != 'full_train' or candidate['fit_scope'] != 'full_train':
            raise RuntimeError('Both forward arms must complete full-data refit')
        row['final_control'], row['final_candidate'] = control, candidate
        save()  # Freeze full models and the choice before reading future answers.
        vectors = {}
        for label, response in (('control', control), ('candidate', candidate)):
            frame = pd.read_csv(response['candidate_path'])
            if list(frame) != list(template) or not np.array_equal(frame['id'], template['id']):
                raise ValueError('Prediction format/order changed')
            vectors[label] = frame['site_eui'].to_numpy(dtype=float)
        vectors['selected'] = vectors['candidate' if selected else 'control']
        truth = p['y'].iloc[valid].to_numpy()
        row['outer_rmse'] = {}
        for label, prediction in vectors.items():
            squared = (prediction - truth) ** 2
            errors[label].extend(squared.tolist())
            row['outer_rmse'][label] = float(np.sqrt(np.mean(squared)))
        row['relative_gain'] = (row['outer_rmse']['control'] - row['outer_rmse']['selected']) / row['outer_rmse']['control']
        row['completed'] = True
        save()
        print('RESULT', period, row['outer_rmse'], 'GAIN', row['relative_gain'], flush=True)
    report['pooled_rmse'] = {label: float(np.sqrt(np.mean(squared))) for label, squared in errors.items()}
    pooled = report['pooled_rmse']
    report['pooled_relative_gain'] = (pooled['control'] - pooled['selected']) / pooled['control']
    report.update(completed=True, passed=bool(all(r['relative_gain'] > 0 for r in report['records'])
        and report['pooled_relative_gain'] >= .005))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '4'
    main()
