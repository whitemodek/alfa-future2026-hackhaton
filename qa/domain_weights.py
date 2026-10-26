"""Research-only covariate-shift diagnostic using features and domain labels.

No task targets enter this interface. Group OOF prevents a domain classifier
from memorizing duplicate feature vectors across its fitting/diagnostic folds.
"""
import time

import numpy as np
import pandas as pd


def estimate_domain_weights(source, target, categorical, deadline, threads=2):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from ml_models import fit
    from threadpoolctl import threadpool_limits
    if list(source) != list(target) or min(len(source), len(target)) < 20:
        raise ValueError('Compatible source and target features are required')
    combined = pd.concat([source, target], ignore_index=True)
    domain = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    groups = pd.util.hash_pandas_object(combined, index=False).to_numpy()
    prediction = np.full(len(combined), np.nan)
    rounds = 120
    for train, valid in GroupKFold(3).split(combined, domain, groups):
        if set(domain[train]) != {0, 1} or set(groups[train]) & set(groups[valid]):
            raise ValueError('Cannot form independent domain folds')
        remaining = deadline - time.monotonic()
        if remaining < 2:
            raise TimeoutError('Domain diagnostic deadline exhausted')
        with threadpool_limits(limits=threads):
            # One-round factory supplies fold-only preprocessing and fixed parameters.
            prototype = fit('lightgbm', combined.iloc[train], domain[train],
                combined.iloc[train], domain[train], categorical,
                dict(iterations=1, num_leaves=15, min_child_samples=100,
                     learning_rate=.05, auto_class_weights='None'),
                deadline, threads, 'classification', 'roc_auc')
            prototype.tree_count_ = rounds
            model = prototype.refit(combined.iloc[train], domain[train], deadline, threads)
            prediction[valid] = model.predict_proba(combined.iloc[valid])[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete domain predictions')
    auc = float(roc_auc_score(domain, prediction))
    odds = np.clip(prediction[:len(source)], 1e-4, 1 - 1e-4)
    ratio = (odds / (1 - odds)) * len(source) / len(target)
    weights = np.clip(ratio, .25, 4.)
    weights /= weights.mean()
    diagnostic = dict(group_oof_auc=auc, fixed_rounds=rounds, source_rows=len(source), target_rows=len(target),
        source_domain_probability_quantiles=np.quantile(prediction[:len(source)], [0, .1, .5, .9, 1]).tolist(),
        target_domain_probability_quantiles=np.quantile(prediction[len(source):], [0, .1, .5, .9, 1]).tolist(),
        normalized_weight_quantiles=np.quantile(weights, [0, .1, .5, .9, 1]).tolist(),
        effective_source_rows=float(weights.sum() ** 2 / np.sum(weights ** 2)), target_labels_used=False,
        note='AUC is source-vs-target distinguishability, not task quality. Density-ratio weights '
             'assume covariate shift and may be wrong under conditional/label shift.')
    return weights, diagnostic


def main():
    import json
    from pathlib import Path
    import ml_backend as b
    root = Path(__file__).resolve().parents[1]
    out = root / '.test-temp/organizer-domain-diagnostic'
    out.mkdir(exist_ok=False)
    data = root / '.test-temp/organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, target, categorical = b._features(p)
    # Task labels are never passed to the domain estimator.
    weights, report = estimate_domain_weights(x, target, categorical, time.monotonic() + 120)
    np.savez_compressed(out / 'weights.npz', weights=weights)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
