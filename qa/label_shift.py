"""Exploratory bounded label-shift adjustment; no target labels in the API.

Inspired by maximum-likelihood label-shift adaptation (Alexandari et al.,
ICML 2020). Calibration and the label-shift assumption are NOT established
for this dataset; fixed shrinkage/clipping are engineering safeguards.
"""
import numpy as np

from qa.decision_weights import checked_probability


def class_prior(y, count):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=count).astype(float)
    if counts.shape != (count,) or (counts <= 0).any():
        raise ValueError('Every source training class must be present')
    return counts / counts.sum()


def normalize(p):
    return p / p.sum(axis=1, keepdims=True)


def adapt_label_shift(source_probability, target_probability, source_fit_prior, target_fit_prior):
    source = checked_probability(source_probability)
    target = checked_probability(target_probability, source.shape[1])
    priors = [np.asarray(p, dtype=float) for p in (source_fit_prior, target_fit_prior)]
    if any(p.shape != (source.shape[1],) or not np.isfinite(p).all() or (p <= 0).any()
           or not np.isclose(p.sum(), 1) for p in priors):
        raise ValueError('Aligned positive normalized source training priors are required')
    # Undo SqrtBalanced's idealized class-prior distortion. No held-out answers.
    source_posterior = normalize(np.maximum(source, 1e-12) * np.sqrt(priors[0]))
    target_posterior = normalize(np.maximum(target, 1e-12) * np.sqrt(priors[1]))
    reference = source_posterior.mean(axis=0)
    prior = reference.copy()
    shrink = .05
    for iteration in range(100):
        responsibility = normalize(target_posterior * (prior / reference))
        updated = (1 - shrink) * responsibility.mean(axis=0) + shrink * reference
        difference = np.max(np.abs(updated - prior))
        prior = updated
        if difference < 1e-7:
            break
    ratio = np.clip(prior / reference, .25, 4.)
    # For a SqrtBalanced decision rule, ideal label-prior adaptation has
    # sqrt(target/source) rather than the full posterior ratio.
    adjusted = normalize(target * np.sqrt(ratio))
    return adjusted, dict(reference_prior=reference.tolist(), estimated_target_prior=prior.tolist(),
        clipped_ratio=ratio.tolist(), iterations=iteration + 1, max_update=float(difference),
        target_labels_used=False, shrink=shrink, decision_exponent=.5)


def main():
    import json
    from pathlib import Path
    from qa.research_attack_specialist import FOLDS, sha256, metrics, guarded_splits
    import pandas as pd
    import ml_backend as b
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'label-shift-study'
    out.mkdir(exist_ok=False)
    protocol = dict(folds=FOLDS, grid_search=False, api_calls=0, platform_submissions=0,
        gate='At least +.002 on both historical group holdouts before full-runtime work',
        source_sha256=sha256(__file__), source='https://proceedings.mlr.press/v119/alexandari20a.html',
        limitation='Historical caches/holdouts; model-calibration/label-shift assumptions unverified. '
                   'Not a replication of the cited paper or an official platform score.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, _ = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    report = dict(protocol=protocol, folds=[])
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, inner_train, inner_valid = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, inner_train, inner_valid)
        inner, outer = [], []
        for relative in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                         f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                         f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / relative, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not (np.array_equal(a[key], valid) and np.array_equal(a['inner_indices'], inner_valid)):
                    raise ValueError('Cache row alignment changed')
                inner.append(a['inner_probability'].copy())
                outer.append(a['probability'].copy())
        inner, outer = np.mean(inner, axis=0), np.mean(outer, axis=0)
        adjusted, diagnostic = adapt_label_shift(inner, outer,
            class_prior(p['y'].iloc[inner_train], len(p['classes'])),
            class_prior(p['y'].iloc[train], len(p['classes'])))
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=adjusted, valid_indices=valid)
        base = metrics(p['y'].iloc[valid].to_numpy(), outer, p['classes'])
        new = metrics(p['y'].iloc[valid].to_numpy(), adjusted, p['classes'])
        row = dict(fold=fold, guards=guards, diagnostic=diagnostic, base=base, candidate=new,
                   gain=new['macro_f1'] - base['macro_f1'])
        report['folds'].append(row)
        print(json.dumps(row), flush=True)
    report['passed'] = all(f['gain'] >= .002 for f in report['folds'])
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    main()
