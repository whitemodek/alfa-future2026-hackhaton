"""Exact-runtime, prospective external group comparison of fixed seed bags.

The old bagging pilot missed its numerical promotion gate despite two positive
deltas. This NEW protocol retains that failure and tests a fixed implementation
on new outer seeds; it does not retroactively pass the previous protocol.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import sha256, array_hash


SEEDS = (20261104, 20261105)


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'seed-bag-prospective'
    out.mkdir(exist_ok=False)
    protocol = dict(seeds=SEEDS, threads=4, api_calls=0, platform_submissions=0,
        budgets=dict(control=300, bag=480, refit_each=360),
        selection='Frozen seeds [17,29] (or next distinct 43); same per-member tree counts. '
                  'Select bag only when INNER native Macro F1 improves by >=.001; '
                  'outer answers never enter selection/early stopping/refit.',
        gate='Selected variant strictly improves BOTH new outer folds; mean gain >= .001; '
             'all original classes present; all models refit on the complete outer training portion.',
        deviation='Prior historical pilot: mean gain .00155646, old gate .002 missed. This is a new '
                  'prospective .001 gate on different outer folds; the old gate remains FAILED.',
        max_seconds=3000, target='attack_cat', metric='macro_f1',
        source_sha256={n: sha256(root / n) for n in ('ml_models.py', 'ml_bagging.py', 'ml_backend.py',
            'ml_portfolio.py', 'ml_validation.py', 'qa/prospective_seed_bag.py')},
        limitation='Previously explored dataset; new untouched outer splits of known training labels. '
                   'Not original platform test labels, official score or proof of a large competitive gain.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, folds=[], completed=False)
    started = time.monotonic()
    deadline = started + 3000
    import ml_backend as b
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    original = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    features = original['train'][original['features']]
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    report['input_sha256'] = b._input_hashes(original)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def run(env, request, allowance):
        remaining = min(allowance, deadline - time.monotonic())
        if remaining < 3:
            raise TimeoutError('Prospective protocol deadline exhausted')
        return b.run(dict(env, request=request, time_budget_seconds=remaining))

    save()
    for seed in SEEDS:
        print('START', seed, flush=True)
        candidates = GroupShuffleSplit(20, test_size=.2, random_state=seed).split(features, original['y'], groups)
        train, valid = next((a, c) for a, c in candidates if set(original['y'].iloc[a]) == set(original['y'])
                           and set(original['y'].iloc[c]) == set(original['y']))
        if set(groups[train]) & set(groups[valid]):
            raise ValueError('Outer group leakage')
        folder = out / str(seed)
        folder.mkdir()
        source = original['train'].iloc[train].copy()
        target = original['train'].iloc[valid].drop(columns=original['target']).copy()
        source.to_csv(folder / 'train.csv', index=False)
        target.to_csv(folder / 'test.csv', index=False)
        template = target[original['id_columns']].copy()
        template[original['target']] = original['classes'][0]
        template.to_csv(folder / 'sample_submission.csv', index=False)
        (folder / 'task.md').write_text('Tabular classification. Target: attack_cat. Primary metric: macro_f1.\n'
            'Inputs: train.csv, test.csv, sample_submission.csv. ID: id, never a feature.\n'
            'Output submission.csv, original string labels and template row order.\n', encoding='utf-8')
        np.savez_compressed(out / f'{seed}-indices.npz', train=train, valid=valid)
        row = dict(seed=seed, train_rows=len(train), outer_rows=len(valid),
                   train_indices_sha256=array_hash(train), outer_indices_sha256=array_hash(valid),
                   outer_group_overlap=0, completed=False)
        report['folds'].append(row)
        save()
        env = dict(task_dir=str(folder), task_path=str(folder / 'task.md'), threads=4)
        with threadpool_limits(limits=4):
            control = run(env, dict(action='portfolio'), protocol['budgets']['control'])
            row['inner_control_score'] = control['score']
            row['control_model'] = control['model']
            row['control_response'] = control
            save()
            print('CONTROL', seed, control['model'], control['score'], flush=True)
            candidate = run(env, dict(action='seed_bag', manifest_path=control['manifest_path']), protocol['budgets']['bag'])
            if any(candidate[k] != control[k] for k in ('validation_id', 'metric', 'direction')):
                raise ValueError('Native validation identity changed')
            row['inner_candidate_score'] = candidate['score']
            selected_bag = candidate['score'] >= control['score'] + .001
            row['selected_bag'] = bool(selected_bag)
            row['candidate_response'] = candidate
            save()
            print('BAG', seed, candidate['score'], 'SELECTED', selected_bag, flush=True)
            if control['fit_scope'] == 'holdout_train':
                control = run(env, dict(action='refit', manifest_path=control['manifest_path']), protocol['budgets']['refit_each'])
            if candidate['fit_scope'] == 'holdout_train':
                candidate = run(env, dict(action='refit', manifest_path=candidate['manifest_path']), protocol['budgets']['refit_each'])
        if control['fit_scope'] != 'full_train' or candidate['fit_scope'] != 'full_train':
            raise RuntimeError('Both arms must finish full refit')
        row['final_control_response'] = control
        row['final_candidate_response'] = candidate
        # Persist final frozen models and selection BEFORE opening outer task answers.
        save()
        predictions = {}
        for label, response in (('control', control), ('candidate', candidate)):
            frame = pd.read_csv(response['candidate_path'])
            if list(frame) != list(template) or not np.array_equal(frame['id'], template['id']):
                raise ValueError('Submission schema or order changed')
            predictions[label] = frame['attack_cat'].astype(str).to_numpy()
        truth = original['train'].iloc[valid]['attack_cat'].astype(str).to_numpy()
        row['outer_scores'] = {name: float(f1_score(truth, prediction, labels=original['classes'],
            average='macro', zero_division=0)) for name, prediction in predictions.items()}
        selected = row['outer_scores']['candidate' if selected_bag else 'control']
        row['selected_outer_score'] = selected
        row['gain'] = selected - row['outer_scores']['control']
        row['completed'] = True
        save()
        print('RESULT', seed, row['gain'], flush=True)
    report.update(completed=True, passed=bool(all(r['gain'] > 0 for r in report['folds']) and
        np.mean([r['gain'] for r in report['folds']]) >= .001))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '4'
    main()
