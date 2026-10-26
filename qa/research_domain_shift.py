"""Target-like group holdouts and a paired train-only domain-weighting study.

Only unlabeled organizer test covariates define the domain; no test answers
are fetched or used. A single fixed LGB model tests the weighting mechanism
before any expensive full-portfolio follow-up.
"""
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.domain_weights import estimate_domain_weights
from qa.research_attack_specialist import sha256, array_hash, guarded_splits, metrics


def target_like_folds(groups, y, weights, seeds=(20261026, 20261027)):
    """Weighted sampling of feature groups; never rank by task errors/scores."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=float)
    if (groups.shape != y.shape or y.shape != weights.shape or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError('Aligned finite positive source weights are required')
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=weights) / counts
    grouped_rows = np.argsort(inverse, kind='stable')
    offsets = np.concatenate([[0], np.cumsum(counts)])
    used_validation = set()
    folds = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = np.argsort(-np.log(np.maximum(rng.random(len(unique)), 1e-12)) / means)
        left = pd.Series(y).value_counts().to_dict()
        selected, size = [], 0
        for g in order:
            if int(g) in used_validation:
                continue
            members = grouped_rows[offsets[g]:offsets[g + 1]]
            labels, numbers = np.unique(y[members], return_counts=True)
            labels = dict(zip(labels, numbers))
            if any(left[label] <= number for label, number in labels.items()):
                continue
            selected.append(int(g))
            size += len(members)
            for label, number in labels.items():
                left[label] -= number
            if size >= int(np.ceil(len(y) * .2)):
                break
        mask = np.isin(inverse, selected)
        train, valid = np.flatnonzero(~mask), np.flatnonzero(mask)
        if not len(valid) or set(y[train]) != set(y) or set(groups[train]) & set(groups[valid]):
            raise ValueError('Cannot construct a target-like independent group fold')
        used_validation.update(selected)
        folds.append((train, valid))
    return folds


def weighted_training(original, expected_frame, domain_weights):
    """Temporary research adapter; production refit must recompute its own weights."""
    expected = expected_frame.index.to_numpy()

    def calculate(x, y, mode, duplicate_weighting='none'):
        if not np.array_equal(x.index.to_numpy(), expected):
            raise ValueError('Domain weights belong to a different training subset')
        base = original(x, y, mode, duplicate_weighting)
        combined = np.asarray(domain_weights) * (1 if base is None else base)
        return combined / combined.mean()

    return calculate


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'domain-shift-study'
    out.mkdir(exist_ok=False)
    protocol = dict(folds='Two disjoint target-like group validation parts, seeds 20261026/20261027',
        model=dict(iterations=700, num_leaves=63, depth=-1, learning_rate=.05,
                   auto_class_weights='SqrtBalanced', early_stopping_rounds=70),
        proposal='Fixed training density ratios from GROUP OOF domain classifier, clip [.25,4] before mean normalization',
        gate='Macro F1 gain >=.003 on BOTH target-like folds before a full-runtime follow-up; no outer tuning',
        api_calls=0, platform_submissions=0, target_labels_used=False, max_seconds=900,
        source_sha256={name: sha256(root / name) for name in
                       ('qa/domain_weights.py', 'qa/research_domain_shift.py', 'ml_models.py')},
        limitation='Transductive use of test covariates; covariate shift assumption may fail. '
                   'Fresh splits but a known dataset; single-model ablation, not a full solution score.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, completed=False, folds=[])
    started = time.monotonic()
    deadline = started + 900
    import ml_backend as b
    import ml_models
    from sklearn.model_selection import GroupShuffleSplit
    from threadpoolctl import threadpool_limits
    data = scratch / 'organizer/multiclass'
    problem = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, test, categorical = b._features(problem)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    report['input_sha256'] = b._input_hashes(problem)
    with np.load(scratch / 'organizer-domain-diagnostic/weights.npz', allow_pickle=False) as a:
        fold_weights = a['weights'].copy()
    report['domain_cache_sha256'] = sha256(scratch / 'organizer-domain-diagnostic/weights.npz')
    folds = target_like_folds(groups, problem['y'].to_numpy(), fold_weights)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for ordinal, (train, valid) in enumerate(folds):
        inner_a, inner_b = next(GroupShuffleSplit(1, test_size=.2, random_state=42).split(
            x.iloc[train], problem['y'].iloc[train], groups[train]))
        inner_train, inner_valid = train[inner_a], train[inner_b]
        guards = guarded_splits(groups, train, valid, inner_train, inner_valid)
        row = dict(number=ordinal, guards=guards, indices_sha256={name: array_hash(a) for name, a in
            (('train', train), ('valid', valid), ('inner_train', inner_train), ('inner_valid', inner_valid))}, arms={})
        report['folds'].append(row)
        np.savez_compressed(out / f'fold-{ordinal}-indices.npz', train=train, valid=valid,
                            inner_train=inner_train, inner_valid=inner_valid)
        print('DOMAIN', ordinal, flush=True)
        inner_weights, inner_diagnostic = estimate_domain_weights(x.iloc[inner_train], test, categorical,
                                                                  min(deadline, time.monotonic() + 90))
        outer_weights, outer_diagnostic = estimate_domain_weights(x.iloc[train], test, categorical,
                                                                  min(deadline, time.monotonic() + 90))
        row['domain'] = dict(inner=inner_diagnostic, outer=outer_diagnostic)
        predictions = {}
        for mode in ('control', 'domain_weighted'):
            print('FIT', ordinal, mode, flush=True)
            original = ml_models._training_weights
            inner_function = weighted_training(original, x.iloc[inner_train], inner_weights) if mode != 'control' else original
            outer_function = weighted_training(original, x.iloc[train], outer_weights) if mode != 'control' else original
            with threadpool_limits(limits=2):
                with patch('ml_models._training_weights', inner_function):
                    model = ml_models.fit('lightgbm', x.iloc[inner_train], problem['y'].iloc[inner_train],
                        x.iloc[inner_valid], problem['y'].iloc[inner_valid], categorical, protocol['model'],
                        min(deadline, time.monotonic() + 90), 2, 'classification', 'macro_f1')
                row['arms'][mode] = dict(selected_iterations=model.tree_count_)
                save()
                with patch('ml_models._training_weights', outer_function):
                    full = model.refit(x.iloc[train], problem['y'].iloc[train], min(deadline, time.monotonic() + 120), 2)
                predictions[mode] = b._aligned_proba(full, x.iloc[valid], len(problem['classes']))
            np.savez_compressed(out / f'fold-{ordinal}-{mode}.npz', probability=predictions[mode], valid_indices=valid)
        for mode, probability in predictions.items():
            row['arms'][mode]['outer'] = metrics(problem['y'].iloc[valid].to_numpy(), probability, problem['classes'])
        row['gain'] = row['arms']['domain_weighted']['outer']['macro_f1'] - row['arms']['control']['outer']['macro_f1']
        save()
        print('RESULT', ordinal, row['gain'], flush=True)
    report.update(completed=True, passed=all(f['gain'] >= .003 for f in report['folds']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '2'
    main()
