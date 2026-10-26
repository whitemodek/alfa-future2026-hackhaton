"""Paired replay on pre-existing nested probability caches. No network/API.

Outer folds have been viewed before; this is exploratory, not a blind result.
The predeclared gate requires +.002 Macro F1 on BOTH outer folds.
"""
import json
from pathlib import Path
import time

import numpy as np

from qa.decision_weights import fit_decision_weights, decision_probability, macro_f1
from qa.research_attack_specialist import FOLDS, guarded_splits, sha256, validate_probability


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'decision-weights-study'
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Preserve the previous study artifacts')
    import pandas as pd
    import ml_backend as b
    data = scratch / 'organizer/multiclass'
    problem = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, _ = b._features(problem)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    count = len(problem['classes'])
    protocol = dict(folds=FOLDS, gate='Delta >= .002 on each outer fold; no further grid search',
        source_sha256=sha256(root / 'qa/decision_weights.py'), api_calls=0, platform_submissions=0,
        limitation='Historical folds/caches; base inner early stopping reused inner labels. '
                   'Outer labels do not fit weights; not a blind estimate or full runtime comparison.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    trials = []
    started = time.monotonic()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            ti, vi, it, iv = (a[key].copy() for key in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, ti, vi, it, iv)
        inner, outer, hashes = [], [], {}
        for relative in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                         f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                         f'late-extra100-results/{fold}-raw100.npz'):
            path = scratch / relative
            hashes[relative] = sha256(path)
            with np.load(path, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not (np.array_equal(a[key], vi) and np.array_equal(a['inner_indices'], iv)):
                    raise ValueError('Probability cache indices changed')
                inner.append(validate_probability(a['inner_probability'].copy(), count))
                outer.append(validate_probability(a['probability'].copy(), count))
        inner, outer = np.mean(inner, axis=0), np.mean(outer, axis=0)
        # Freeze the proposal before reading any outer answers.
        weights = fit_decision_weights(inner, problem['y'].iloc[iv].to_numpy())
        np.savez_compressed(out / f'{fold}-frozen.npz', weights=weights, indices=vi,
                            raw_probability=outer, weighted_probability=decision_probability(outer, weights))
        y = problem['y'].iloc[vi].to_numpy()
        base = macro_f1(y, outer.argmax(axis=1), count)
        score = macro_f1(y, decision_probability(outer, weights).argmax(axis=1), count)
        trials.append(dict(fold=fold, guards=guards, weights=weights.tolist(), base=base,
                           challenger=score, gain=score - base, cache_sha256=hashes))
        print(json.dumps(trials[-1]), flush=True)
    report = dict(protocol=protocol, trials=trials, passed=all(t['gain'] >= .002 for t in trials),
                  seconds=round(time.monotonic() - started, 3))
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
