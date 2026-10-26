"""Frozen three-seed family ensemble against the cached three-family control.

Counts are inherited from historical INNER selection, never outer answers.
No choice of seed/weight on outer outcomes. A positive result still requires
an exact current-runtime, prospective comparison before release.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import FOLDS, sha256, guarded_splits, metrics


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'classification-seed-bag-study'
    out.mkdir(exist_ok=False)
    protocol = dict(new_seeds=[17, 29], total_seeds=3, family_weights='Equal LGB63/XGB6/ET100 per seed',
        selection='Fixed equal seed weights; exact historical INNER-selected counts; no outer selection',
        gate='Mean Macro F1 gain >= .002 and BOTH folds strictly positive, before exact-runtime follow-up',
        api_calls=0, platform_submissions=0, max_seconds=1200,
        source_sha256={n: sha256(root / n) for n in ('ml_models.py', 'qa/research_classification_bag.py')},
        limitation='Historical folds/control caches, same-family approximated adapters; not a final-runtime score')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, trials=[], completed=False)
    started = time.monotonic()
    deadline = started + 1200
    import ml_backend as b
    from ml_models import fit
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, cats = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    history = json.loads((scratch / 'night-classification-report.json').read_text(encoding='utf-8'))
    counts = {(t['fold'], t['name']): t['iterations'] for t in history['trials'] if 'iterations' in t}
    report['input_sha256'] = b._input_hashes(p)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, inner_train, inner_valid = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        row = dict(fold=fold, guards=guarded_splits(groups, train, valid, inner_train, inner_valid), models=[])
        report['trials'].append(row)
        parts = []
        for rel in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                    f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                    f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / rel, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not np.array_equal(a[key], valid):
                    raise ValueError('Cached control indices changed')
                parts.append(a['probability'].copy())
        base = np.mean(parts, axis=0)
        configs = (
            ('lightgbm', counts[fold, 'lightgbm63_sqrt'],
             dict(num_leaves=63, learning_rate=.05, min_child_samples=30, l2_leaf_reg=2, subsample=.9, colsample_bytree=.9)),
            ('xgboost', counts[fold, 'xgboost6_sqrt'],
             dict(depth=6, learning_rate=.05, min_child_weight=3, l2_leaf_reg=3, subsample=.9, colsample_bytree=.9)),
            ('extra_trees', 100, dict(min_samples_leaf=3, max_features=1.)),
        )
        seed_predictions = [base]
        for seed in protocol['new_seeds']:
            members = []
            for family, rounds, parameters in configs:
                print('FIT', fold, seed, family, rounds, flush=True)
                request = dict(parameters, random_seed=seed, auto_class_weights='SqrtBalanced', iterations=1)
                with threadpool_limits(limits=2):
                    # Factory sees TRAIN as validation; its one fitted iteration is discarded.
                    prototype = fit(family, x.iloc[train], p['y'].iloc[train], x.iloc[train], p['y'].iloc[train],
                        cats, request, min(deadline, time.monotonic() + 15), 2, 'classification', 'macro_f1')
                    prototype.tree_count_ = rounds
                    full = prototype.refit(x.iloc[train], p['y'].iloc[train],
                                           min(deadline, time.monotonic() + 140), 2)
                    probability = b._aligned_proba(full, x.iloc[valid], len(p['classes']))
                if full.tree_count_ != rounds:
                    raise RuntimeError('Frozen count changed during fit')
                np.savez_compressed(out / f'{fold}-{seed}-{family}.npz', probability=probability, valid_indices=valid)
                row['models'].append(dict(seed=seed, model=family, rounds=rounds, parameters=full.parameters))
                members.append(probability)
                save()
            seed_predictions.append(np.mean(members, axis=0))
        combined = np.mean(seed_predictions, axis=0)
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=combined, valid_indices=valid)
        # Only now score against the outer task labels.
        row['base'] = metrics(p['y'].iloc[valid].to_numpy(), base, p['classes'])
        row['candidate'] = metrics(p['y'].iloc[valid].to_numpy(), combined, p['classes'])
        row['gain'] = row['candidate']['macro_f1'] - row['base']['macro_f1']
        save()
        print('RESULT', fold, row['gain'], flush=True)
    report.update(completed=True, passed=bool(all(t['gain'] > 0 for t in report['trials']) and
        np.mean([t['gain'] for t in report['trials']]) >= .002))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
