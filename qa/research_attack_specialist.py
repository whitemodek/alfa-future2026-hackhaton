"""Exploratory, inner-selected attack specialist; never imported by the solution.

Uses only local organizer training data and pre-existing, aligned probability
caches. These folds have been used before: this is not a blind test or a claim
about the production pipeline. No API calls, network access or runtime edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


SPECIALIST_LABELS = ('Analysis', 'Backdoor', 'DoS', 'Exploits')
LAMBDAS = (0., .25, .5)
FOLDS = ('primary', 'confirmation17')
REQUEST = dict(iterations=800, num_leaves=31, depth=-1, learning_rate=.05,
               random_seed=42, early_stopping_rounds=70,
               auto_class_weights='SqrtBalanced', encoding='native')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_probability(values, columns=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or (columns is not None and values.shape[1] != columns)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1., atol=1e-6)):
        raise ValueError('Expected finite, nonnegative normalized probabilities')
    return values


def conditional_blend(base, specialist, subset, strength):
    """Redistribute only the base probability mass inside a fixed class subset."""
    base = validate_probability(base)
    specialist = validate_probability(specialist, len(subset))
    subset = np.asarray(subset, dtype=int)
    if (len(base) != len(specialist) or len(set(subset)) != len(subset)
            or not len(subset) or (subset < 0).any() or (subset >= base.shape[1]).any()
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid specialist subset, rows or blend strength')
    result = base.copy()
    mass = base[:, subset].sum(axis=1, keepdims=True)
    result[:, subset] = (1 - strength) * base[:, subset] + strength * mass * specialist
    return validate_probability(result, base.shape[1])


def guarded_splits(groups, train, valid, inner_train, inner_valid):
    """Reject row/group overlap and non-nested inner folds before any fitting."""
    rows = [np.asarray(v) for v in (train, valid, inner_train, inner_valid)]
    if any(v.ndim != 1 or not len(v) or not np.issubdtype(v.dtype, np.integer)
           or (v < 0).any() or (v >= len(groups)).any()
           or len(np.unique(v)) != len(v) for v in rows):
        raise ValueError('Invalid split indices')
    train, valid, inner_train, inner_valid = rows
    if (set(train) & set(valid) or set(inner_train) & set(inner_valid)
            or set(inner_train) | set(inner_valid) != set(train)
            or set(groups[train]) & set(groups[valid])
            or set(groups[inner_train]) & set(groups[inner_valid])):
        raise ValueError('Validation leakage or non-nested inner folds')
    return dict(outer_row_overlap=0, inner_row_overlap=0,
                outer_group_overlap=0, inner_group_overlap=0,
                inner_folds_partition_outer_train=True)


def select_strength(base, specialist, subset, truth):
    from sklearn.metrics import f1_score
    scores = []
    for strength in LAMBDAS:
        prediction = conditional_blend(base, specialist, subset, strength).argmax(axis=1)
        scores.append(float(f1_score(truth, prediction, labels=range(base.shape[1]),
                                     average='macro', zero_division=0)))
    # np.argmax keeps the first (least intervention) candidate on exact ties.
    return float(LAMBDAS[int(np.argmax(scores))]), scores


def metrics(truth, probability, classes):
    from sklearn.metrics import accuracy_score, f1_score
    truth = np.asarray(truth)
    prediction = probability.argmax(axis=1)
    per_class = f1_score(truth, prediction, labels=range(len(classes)),
                         average=None, zero_division=0)
    return dict(macro_f1=float(per_class.mean()), accuracy=float(accuracy_score(truth, prediction)),
                per_class_f1=dict(zip(classes, map(float, per_class))),
                support={name: int(np.sum(truth == i)) for i, name in enumerate(classes)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    scratch = root / '.test-temp'
    protocol_path = scratch / 'attack-specialist-protocol.json'
    report_path = scratch / 'attack-specialist-report.json'
    if protocol_path.exists() or report_path.exists():
        raise FileExistsError('A previous protocol/report exists; do not silently overwrite an experiment')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['MPLCONFIGDIR'] = str(scratch / 'matplotlib')
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import ml_backend as backend
    from ml_models import fit

    started = time.monotonic()
    deadline = started + 300
    data = scratch / 'organizer' / 'multiclass'
    protocol = dict(specialist_labels=SPECIALIST_LABELS, blend_strengths=LAMBDAS,
                    folds=FOLDS, model='lightgbm', request=REQUEST, threads=2,
                    max_seconds=300, api_calls=0, production_changed=False,
                    source_sha256=sha256(__file__), runtime_adapter_sha256=sha256(root / 'ml_models.py'),
                    backend_sha256=sha256(root / 'ml_backend.py'),
                    historical_cache_producer_sha256={name: sha256(scratch / name)
                                                       for name in ('night-classification.py', 'late-extra100.py')},
                    base='Equal cached LGB63 sqrt + XGB6 sqrt + standalone ET100',
                    selection='Specialist tree count on inner subset only; strength on full10class inner only; '
                              'exact selected-count specialist refit on outer train subset; outer labels only for reporting.',
                    continuation_gate='Only consider exact-runtime follow-up if inner-selected strength>0 '
                                      'and paired outer MacroF1 delta>=0.002 on EACH fold. '
                                      'Engineering threshold, not statistical significance. No further group/strength search.',
                    warning='Exploratory: previously used folds and historical cached base, '
                            'not a blind test, not a measured production improvement.')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, protocol_sha256=sha256(protocol_path), trials=[], completed=False)

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    problem = backend.load_problem(dict(task_path=str(data / 'task.md'), task_dir=str(data), request={}))
    features, _, categorical = backend._features(problem)
    truth, classes = problem['y'], problem['classes']
    if len(classes) != 10 or not all(label in classes for label in SPECIALIST_LABELS):
        raise ValueError('This predeclared experiment requires the known 10-class organizer task')
    subset = np.asarray([classes.index(label) for label in SPECIALIST_LABELS])
    mapping = {int(original): local for local, original in enumerate(subset)}
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    sources = [problem[name] for name in ('train_path', 'test_path', 'sample_path') if problem[name]]
    input_hashes = {str(path): sha256(path) for path in sources}
    report.update(classes=classes, subset_indices=subset.tolist(), input_sha256=input_hashes,
                  feature_names=list(features.columns))
    save()
    for fold_no, fold in enumerate(FOLDS):
        print('START', fold, flush=True)
        fold_started = time.monotonic()
        index_path = scratch / 'night-classification-results' / f'{fold}-indices.npz'
        with np.load(index_path, allow_pickle=False) as indices:
            train, valid, inner_train, inner_valid = (indices[name].copy() for name in
                                                      ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, inner_train, inner_valid)
        inner_parts, outer_parts, cache_hashes = [], [], {str(index_path): sha256(index_path)}
        for relative in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                         f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                         f'late-extra100-results/{fold}-raw100.npz'):
            path = scratch / relative
            cache_hashes[str(path)] = sha256(path)
            with np.load(path, allow_pickle=False) as cache:
                valid_key = 'validation_indices' if 'validation_indices' in cache else 'val_indices'
                if not (np.array_equal(cache[valid_key], valid)
                        and np.array_equal(cache['inner_indices'], inner_valid)):
                    raise ValueError('Cached probability rows do not match pinned fold indices')
                inner_parts.append(validate_probability(cache['inner_probability'].copy(), 10))
                outer_parts.append(validate_probability(cache['probability'].copy(), 10))
        inner_base, outer_base = np.mean(inner_parts, axis=0), np.mean(outer_parts, axis=0)
        inner_fit = inner_train[np.isin(truth.iloc[inner_train], subset)]
        inner_stop = inner_valid[np.isin(truth.iloc[inner_valid], subset)]
        outer_fit = train[np.isin(truth.iloc[train], subset)]
        # Only training-fold labels are read here, never labels in valid.
        inner_y = truth.iloc[inner_fit].map(mapping).to_numpy()
        inner_valid_y = truth.iloc[inner_stop].map(mapping).to_numpy()
        outer_y = truth.iloc[outer_fit].map(mapping).to_numpy()
        if any(set(labels) != set(range(4)) for labels in (inner_y, inner_valid_y, outer_y)):
            raise ValueError('Specialist train/stop subsets must contain all four declared classes')
        if set(outer_fit) & set(valid) or set(inner_fit) & set(inner_valid):
            raise ValueError('Outer validation rows reached training')
        available = deadline - time.monotonic()
        allowance = available / (len(FOLDS) - fold_no)
        if allowance < 10:
            raise TimeoutError('Insufficient remaining research budget')
        with threadpool_limits(limits=2):
            model = fit('lightgbm', features.iloc[inner_fit], inner_y,
                        features.iloc[inner_stop], inner_valid_y, categorical, REQUEST,
                        time.monotonic() + allowance * .5, 2, 'classification', 'macro_f1')
            specialist_inner = validate_probability(model.predict_proba(features.iloc[inner_valid]), 4)
            strength, inner_scores = select_strength(inner_base, specialist_inner, subset,
                                                      truth.iloc[inner_valid])
            record = dict(fold=fold, guards=guards, cache_sha256=cache_hashes,
                          selected_iterations=model.tree_count_, selected_strength=strength,
                          inner_scores=[dict(strength=value, macro_f1=score)
                                        for value, score in zip(LAMBDAS, inner_scores)],
                          inner_fit_rows=len(inner_fit), inner_stop_rows=len(inner_stop),
                          outer_fit_rows=len(outer_fit), outer_validation_rows=len(valid),
                          fit_index_sha256={name: array_hash(value) for name, value in
                                            [('inner_fit', inner_fit), ('inner_stop', inner_stop),
                                             ('outer_fit', outer_fit), ('outer_valid', valid)]},
                          outer_labels_used_for_selection=False, exact_outer_refit=False,
                          training_parameters=model.parameters)
            report['trials'].append(record)
            save()  # Persist selected strength/count BEFORE outer refit or evaluation.
            outer_model = model.refit(features.iloc[outer_fit], outer_y,
                                      min(deadline, fold_started + allowance), 2)
            if outer_model.tree_count_ != model.tree_count_:
                raise RuntimeError('Outer refit did not complete the selected tree count')
            specialist_outer = validate_probability(outer_model.predict_proba(features.iloc[valid]), 4)
        blended_outer = conditional_blend(outer_base, specialist_outer, subset, strength)
        prediction_path = scratch / f'attack-specialist-{fold}-probabilities.npz'
        np.savez_compressed(prediction_path, train_indices=train, valid_indices=valid,
                            inner_train_indices=inner_train, inner_valid_indices=inner_valid,
                            inner_fit_indices=inner_fit, inner_stop_indices=inner_stop,
                            outer_fit_indices=outer_fit, inner_base=inner_base, outer_base=outer_base,
                            inner_specialist=specialist_inner, outer_specialist=specialist_outer,
                            outer_selected=blended_outer)
        # First access to outer labels: both fitting and all choices are finished.
        outer_truth = truth.iloc[valid].to_numpy()
        record.update(control=metrics(outer_truth, outer_base, classes),
                      selected=metrics(outer_truth, blended_outer, classes),
                      probability_path=str(prediction_path), probability_sha256=sha256(prediction_path),
                      exact_outer_refit=True,
                      inner_vocabulary_sha256=hashlib.sha256(json.dumps(model.vocabulary, sort_keys=True).encode()).hexdigest(),
                      outer_vocabulary_sha256=hashlib.sha256(json.dumps(outer_model.vocabulary, sort_keys=True).encode()).hexdigest(),
                      seconds=round(time.monotonic() - fold_started, 3))
        record['delta'] = record['selected']['macro_f1'] - record['control']['macro_f1']
        save()
        print(json.dumps(dict(fold=fold, strength=strength, rounds=model.tree_count_,
                              control=record['control']['macro_f1'], selected=record['selected']['macro_f1'],
                              delta=record['delta'])), flush=True)
    if input_hashes != {str(path): sha256(path) for path in sources}:
        raise RuntimeError('Original input files changed')
    report.update(completed=True, input_hashes_unchanged=True,
                  improves_both=all(trial['delta'] > 0 for trial in report['trials']),
                  qualifies_for_runtime_followup=all(trial['selected_strength'] > 0 and trial['delta'] >= .002
                                                      for trial in report['trials']))
    save()
    print('COMPLETE', report_path, flush=True)


if __name__ == '__main__':
    main()
