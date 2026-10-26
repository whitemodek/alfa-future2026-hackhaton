"""Fixed flow ratios/categorical interactions, inner-selected blend only."""
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
    out = scratch / 'flow-feature-study'
    out.mkdir(exist_ok=False)
    config = dict(iterations=1000, num_leaves=63, depth=-1, learning_rate=.05,
                  min_child_samples=30, l2_leaf_reg=2, subsample=.9, colsample_bytree=.9,
                  auto_class_weights='SqrtBalanced', early_stopping_rounds=70, random_seed=42)
    protocol = dict(folds=FOLDS, model=config, blend_grid=[0., .25, .5],
        selection='Macro F1 on inner group holdout, least intervention on ties; exact-count outer refit',
        gate='Outer gain >=.002 on BOTH group folds before exact-runtime follow-up',
        api_calls=0, platform_submissions=0, max_seconds=600,
        source_sha256={n: sha256(root / n) for n in ('ml_flow.py', 'ml_models.py', 'qa/research_flow_features.py')},
        limitation='Previously inspected dataset/folds and cached historical control; not a blind or official score')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    deadline = started + 600
    report = dict(protocol=protocol, trials=[], completed=False)
    import ml_backend as b
    from ml_flow import add_flow_features
    from ml_models import fit
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    raw, _, cats = b._features(p)
    x, derived = add_flow_features(raw)
    cats = cats + derived
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    report['input_sha256'] = b._input_hashes(p)
    report['added_features'] = [c for c in x if c not in raw]

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, it, iv = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, it, iv)
        inner, outer = [], []
        for rel in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                    f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                    f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / rel, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not np.array_equal(a[key], valid) or not np.array_equal(a['inner_indices'], iv):
                    raise ValueError('Control probability alignment changed')
                inner.append(a['inner_probability'].copy())
                outer.append(a['probability'].copy())
        inner, outer = np.mean(inner, axis=0), np.mean(outer, axis=0)
        print('INNER', fold, flush=True)
        with threadpool_limits(limits=2):
            model = fit('lightgbm', x.iloc[it], p['y'].iloc[it], x.iloc[iv], p['y'].iloc[iv], cats,
                        config, min(deadline, time.monotonic() + 120), 2, 'classification', 'macro_f1')
            probability = b._aligned_proba(model, x.iloc[iv], len(p['classes']))
            scores = [float(f1_score(p['y'].iloc[iv], ((1 - w) * inner + w * probability).argmax(axis=1),
                labels=np.arange(len(p['classes'])), average='macro', zero_division=0)) for w in protocol['blend_grid']]
            strength = protocol['blend_grid'][int(np.argmax(scores))]
            row = dict(fold=fold, guards=guards, selected_iterations=model.tree_count_, inner_scores=scores,
                       selected_strength=strength)
            report['trials'].append(row)
            save()
            print('OUTER', fold, strength, flush=True)
            if strength:
                full = model.refit(x.iloc[train], p['y'].iloc[train], min(deadline, time.monotonic() + 150), 2)
                candidate = (1 - strength) * outer + strength * b._aligned_proba(full, x.iloc[valid], len(p['classes']))
            else:
                candidate = outer.copy()
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=candidate, valid_indices=valid)
        row['base'] = metrics(p['y'].iloc[valid].to_numpy(), outer, p['classes'])
        row['candidate'] = metrics(p['y'].iloc[valid].to_numpy(), candidate, p['classes'])
        row['gain'] = row['candidate']['macro_f1'] - row['base']['macro_f1']
        save()
        print('RESULT', fold, row['gain'], flush=True)
    report.update(completed=True, passed=all(r['gain'] >= .002 for r in report['trials']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
