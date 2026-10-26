"""Fixed conditional pair specialists, restricted to ambiguous top-two decisions."""
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import FOLDS, sha256, guarded_splits, metrics


PAIRS = (('Analysis', 'Backdoor'), ('DoS', 'Exploits'), ('Fuzzers', 'Normal'))
REQUEST = dict(iterations=700, num_leaves=15, depth=-1, learning_rate=.04,
               l2_leaf_reg=5, min_child_samples=30, auto_class_weights='SqrtBalanced',
               early_stopping_rounds=70, random_seed=42)


def refine(base, pair_probability, pair, margin=.25, strength=.5):
    from qa.research_attack_specialist import validate_probability
    base = validate_probability(base)
    expert = validate_probability(pair_probability, 2)
    pair = np.asarray(pair, dtype=int)
    if pair.shape != (2,) or len(set(pair)) != 2 or (pair < 0).any() or (pair >= base.shape[1]).any():
        raise ValueError('Two distinct global classes are required')
    if len(expert) != len(base) or not 0 <= margin <= 1 or not 0 <= strength <= 1:
        raise ValueError('Aligned expert probabilities and bounded refinement are required')
    top = np.argpartition(base, -2, axis=1)[:, -2:]
    mask = np.isin(top, pair).all(axis=1) & (np.abs(base[np.arange(len(base)), top[:, 0]] -
                                                               base[np.arange(len(base)), top[:, 1]]) <= margin)
    result = base.copy()
    positions = np.flatnonzero(mask)
    mass = base[np.ix_(positions, pair)].sum(axis=1, keepdims=True)
    result[np.ix_(positions, pair)] = ((1 - strength) * base[np.ix_(positions, pair)] + strength * mass * expert[positions])
    return result, mask


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'pair-specialist-study'
    out.mkdir(exist_ok=False)
    protocol = dict(pairs=PAIRS, request=REQUEST, binary_selection_metric='log_loss', strength=.5, margin=.25,
        selection='All three predeclared disjoint pairs; fixed top-two/margin gate; no outer selection',
        gate='Each outer gain >=.001; mean gain >=.002 before exact-runtime follow-up',
        api_calls=0, platform_submissions=0, max_seconds=600, source_sha256=sha256(__file__),
        limitation='Previously inspected dataset/folds, cached base; hypothesis informed by known confusion classes. '
                   'Not a blind or official score; no target identities or test answers are hardcoded.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    deadline = started + 600
    report = dict(protocol=protocol, trials=[], completed=False)
    import ml_backend as b
    from ml_models import fit
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, cats = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, it, iv = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, it, iv)
        parts = []
        for rel in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                    f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                    f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / rel, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not np.array_equal(a[key], valid):
                    raise ValueError('Cached row alignment changed')
                parts.append(a['probability'].copy())
        base = np.mean(parts, axis=0)
        candidate = base.copy()
        row = dict(fold=fold, guards=guards, pairs=[])
        report['trials'].append(row)
        assigned = np.zeros(len(base), dtype=bool)
        for labels in PAIRS:
            pair = np.array([p['classes'].index(label) for label in labels])
            a = it[np.isin(p['y'].iloc[it], pair)]
            v = iv[np.isin(p['y'].iloc[iv], pair)]
            outer = train[np.isin(p['y'].iloc[train], pair)]
            mapping = {int(label): i for i, label in enumerate(pair)}
            y_a, y_v, y_outer = (p['y'].iloc[indices].map(mapping).to_numpy() for indices in (a, v, outer))
            print('FIT', fold, labels, flush=True)
            with threadpool_limits(limits=2):
                model = fit('lightgbm', x.iloc[a], y_a, x.iloc[v], y_v, cats, REQUEST,
                            min(deadline, time.monotonic() + 70), 2, 'classification', 'log_loss')
                record = dict(labels=labels, selected_iterations=model.tree_count_, train_rows=len(outer))
                row['pairs'].append(record)
                save()
                full = model.refit(x.iloc[outer], y_outer, min(deadline, time.monotonic() + 100), 2)
                probability = b._aligned_proba(full, x.iloc[valid], 2)
            refined, mask = refine(base, probability, pair)
            if np.any(mask & assigned):
                raise ValueError('Predeclared disjoint pair gates overlap')
            candidate[mask] = refined[mask]
            assigned |= mask
            record['changed_rows'] = int(mask.sum())
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=candidate, valid_indices=valid)
        row['base'] = metrics(p['y'].iloc[valid].to_numpy(), base, p['classes'])
        row['candidate'] = metrics(p['y'].iloc[valid].to_numpy(), candidate, p['classes'])
        row['gain'] = row['candidate']['macro_f1'] - row['base']['macro_f1']
        save()
        print('RESULT', fold, row['gain'], flush=True)
    gains = [r['gain'] for r in report['trials']]
    report.update(completed=True, passed=bool(min(gains) >= .001 and np.mean(gains) >= .002))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
