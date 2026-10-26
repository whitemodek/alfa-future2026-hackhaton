"""Predeclared paired study: reduce the influence of repeated training features.

Two historical group holdouts; every model is trained again. Outer labels
never choose hyperparameters, tree counts or weights. No API or network.
"""
import json
import os
from pathlib import Path
import time

import numpy as np

from qa.research_attack_specialist import FOLDS, guarded_splits, sha256, array_hash, metrics


CONFIGS = (
    ('lightgbm', dict(iterations=700, num_leaves=63, depth=-1, learning_rate=.05)),
    ('xgboost', dict(iterations=700, depth=6, learning_rate=.05)),
    ('extra_trees', dict(iterations=100, min_samples_leaf=3)),
)


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'duplicate-weighting-study'
    out.mkdir(exist_ok=False)
    protocol = dict(folds=FOLDS, configs=CONFIGS, correction='1/sqrt(training feature-group count)',
        selection='Per-arm tree counts selected on INNER only; exact refit on outer training; '
                  'fixed equal three-model ensemble; no outer tuning',
        gate='Macro F1 gain >= .002 on BOTH outer folds. Otherwise no production promotion.',
        threads=2, total_seconds=1200, api_calls=0, platform_submissions=0,
        source_sha256={name: sha256(root / name) for name in ('ml_models.py', 'qa/research_duplicate_weighting.py')},
        limitation='Previously inspected historical splits; paired exploratory result, not blind platform evidence')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, trials=[], completed=False)
    started = time.monotonic()
    deadline = started + 1200
    import pandas as pd
    import ml_backend as b
    from ml_models import fit
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    problem = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, categorical = b._features(problem)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    count = len(problem['classes'])
    report['input_sha256'] = b._input_hashes(problem)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, inner_train, inner_valid = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, inner_train, inner_valid)
        row = dict(fold=fold, guards=guards, arms={})
        report['trials'].append(row)
        outer_predictions = {}
        for mode in ('none', 'inverse_sqrt'):
            arm = dict(models=[])
            row['arms'][mode] = arm
            parts = []
            for family, config in CONFIGS:
                print('FIT', fold, mode, family, flush=True)
                request = dict(config, auto_class_weights='SqrtBalanced', early_stopping_rounds=70,
                               random_seed=42, duplicate_weighting=mode)
                remaining_jobs = ((len(FOLDS) - FOLDS.index(fold)) * 6
                                  - ('none', 'inverse_sqrt').index(mode) * 3
                                  - [name for name, _ in CONFIGS].index(family))
                allowance = min(120, (deadline - time.monotonic()) / max(remaining_jobs, 1))
                if allowance < 3:
                    raise TimeoutError('Paired study deadline exhausted')
                with threadpool_limits(limits=2):
                    model = fit(family, x.iloc[inner_train], problem['y'].iloc[inner_train],
                        x.iloc[inner_valid], problem['y'].iloc[inner_valid], categorical, request,
                        time.monotonic() + allowance * .45, 2, 'classification', 'macro_f1')
                    record = dict(model=family, tree_count=model.tree_count_, parameters=model.parameters,
                                  fit_indices_sha256=array_hash(inner_train),
                                  selection_indices_sha256=array_hash(inner_valid))
                    arm['models'].append(record)
                    save()  # Freeze selected counts before any outer refit/evaluation.
                    full = model.refit(x.iloc[train], problem['y'].iloc[train],
                                       min(deadline, time.monotonic() + allowance * .55), 2)
                    p = b._aligned_proba(full, x.iloc[valid], count)
                if full.tree_count_ != model.tree_count_:
                    raise RuntimeError('Selected tree count changed during refit')
                path = out / f'{fold}-{mode}-{family}.npz'
                np.savez_compressed(path, probability=p, valid_indices=valid)
                record['probability_sha256'] = sha256(path)
                parts.append(p)
                save()
            outer_predictions[mode] = np.mean(parts, axis=0)
        # First access to outer targets, after both frozen-arm predictions exist.
        for mode, p in outer_predictions.items():
            row['arms'][mode]['outer'] = metrics(problem['y'].iloc[valid].to_numpy(), p, problem['classes'])
        row['gain'] = (row['arms']['inverse_sqrt']['outer']['macro_f1'] - row['arms']['none']['outer']['macro_f1'])
        save()
        print('RESULT', fold, row['gain'], flush=True)
    report.update(completed=True, passed=all(r['gain'] >= .002 for r in report['trials']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
