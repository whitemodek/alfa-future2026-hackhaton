"""Fixed learned stacking experiment on historical out-of-sample probabilities.

Research only: the production runtime never imports this module. Base models
did not fit any inner-validation row, but tree-count/leaf selection used all
inner labels, including the later meta holdout. A group-isolated split inside
those rows selects a fixed stacker or the unchanged uniform average; it is NOT
an independent validation of the complete base-plus-meta pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from qa.research_attack_specialist import (
    FOLDS, array_hash, guarded_splits, metrics, sha256, validate_probability,
)


PARAMETERS = dict(C=.1, max_iter=500, solver='lbfgs', multi_class='multinomial',
                  random_state=42, tol=1e-4)
CLIP = 1e-6


def meta_features(probabilities):
    if len(probabilities) != 3:
        raise ValueError('The predeclared stacker requires exactly three base models')
    probabilities = [validate_probability(value) for value in probabilities]
    if len(set(value.shape for value in probabilities)) != 1:
        raise ValueError('Base probability shapes differ')
    return np.concatenate([np.log(np.clip(value, CLIP, 1.)) for value in probabilities], axis=1)


def sqrt_weights(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any()):
        raise ValueError('Expected nonnegative encoded labels')
    counts = np.bincount(labels)
    if (counts == 0).any():
        raise ValueError('Stacker labels must contain every encoded class')
    weights = np.sqrt(counts.max() / counts)[labels]
    return weights / weights.mean()


def meta_group_split(raw_features, labels, classes):
    """Group by ORIGINAL features; labels only preserve training classes."""
    import pandas as pd
    from ml_validation import plan_validation
    raw_features = raw_features.reset_index(drop=True)
    labels = pd.Series(np.asarray(labels)).reset_index(drop=True)
    if len(raw_features) != len(labels):
        raise ValueError('Meta rows and labels differ')
    problem = dict(train=raw_features, test=raw_features.iloc[:0], y=labels,
                   features=list(raw_features.columns), classes=classes, target='__stack_target__', text='')
    plan = plan_validation(problem, dict(split='group', group_columns=list(raw_features.columns),
                                         validation_fraction=.2))
    train, valid = plan['train_indices'], plan['valid_indices']
    if train is None or valid is None:
        raise ValueError('Cannot form a meta validation split')
    groups = pd.util.hash_pandas_object(raw_features, index=False).to_numpy()
    if (set(train) & set(valid) or set(groups[train]) & set(groups[valid])
            or set(labels.iloc[train]) != set(range(len(classes)))):
        raise ValueError('Meta validation leakage or a missing training class')
    return train, valid, {key: value for key, value in plan.items()
                          if key not in {'train_indices', 'valid_indices'}}


def fit_stacker(features, labels, deadline, threads=2):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    if time.monotonic() >= deadline:
        raise TimeoutError('No time for stacker fit')
    features, labels = np.asarray(features, dtype=float), np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError('Invalid stacker features')
    weights = sqrt_weights(labels)
    model = LogisticRegression(**PARAMETERS)
    with threadpool_limits(limits=threads):
        model.fit(features, labels, sample_weight=weights)
    if time.monotonic() >= deadline:
        raise TimeoutError('Stacker fit exceeded the research deadline')
    return model, dict(mean=float(weights.mean()), sha256=array_hash(weights),
                        class_counts=np.bincount(labels).tolist())


def stack_probability(model, features, count):
    if list(model.classes_) != list(range(count)):
        raise ValueError('Meta-model class order is inconsistent')
    return validate_probability(model.predict_proba(features), count)


def use_stacker(control_score, meta_score):
    if not np.isfinite(control_score) or not np.isfinite(meta_score):
        raise ValueError('Meta selection scores must be finite')
    return bool(meta_score > control_score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, started = args.root.resolve(), time.monotonic()
    scratch, deadline = root / '.test-temp', started + 120
    protocol_path, report_path = (scratch / f'oof-stacker-{name}.json' for name in ('protocol', 'report'))
    if protocol_path.exists() or report_path.exists():
        raise FileExistsError('Do not overwrite a frozen stacker experiment')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    import joblib
    import pandas as pd
    import ml_backend as backend

    protocol = dict(parameters=PARAMETERS, sample_weight='SqrtBalanced, mean normalized to1 per fitting subset',
                    features='Concatenated logclip(probabilities,1e-6,1) from LGB63/XGB6/standaloneET100',
                    folds=FOLDS, meta_split='Explicit group planner on ORIGINAL feature vectors within inner_valid; '
                                          'fixed planner seed42,20% validation, preserve every class in meta train.',
                    selection='Fit meta_train only; compare MacroF1 against uniform average on meta_valid; '
                              'strict improvement selects stacker, ties/worse keep control. '
                              'If selected, fresh exact preset fit on all inner OOS rows, then predict outer.',
                    continuation_gate='Both folds must select stacker on meta validation and '
                                      'paired outer MacroF1 delta>=0.002 EACH. Engineering threshold, not significance.',
                    warning='Exploratory reused folds and historical base. Cached inner predictions are out of '
                            'sample for base fitting, but base tree-count/leaf selection used ALL inner labels, '
                            'including meta_valid. Thus meta_valid is NOT independent of base model selection. '
                            'A gate pass requires a separate fully nested/crossfit confirmation before any runtime change. '
                            'Outer base refits include inner-validation '
                            'rows; this standard stacking refit introduces a probability-distribution shift. '
                            'Not a blind test or measured runtime improvement; no C/lambda/bias search.',
                    api_calls=0, threads=2, max_seconds=120, production_changed=False,
                    source_sha256=sha256(__file__), backend_sha256=sha256(root / 'ml_backend.py'),
                    validation_source_sha256=sha256(root / 'ml_validation.py'),
                    guard_source_sha256=sha256(root / 'qa' / 'research_attack_specialist.py'),
                    historical_cache_producer_sha256={name: sha256(scratch / name)
                                                       for name in ('night-classification.py', 'late-extra100.py')})
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, protocol_sha256=sha256(protocol_path), trials=[], completed=False)

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    data = scratch / 'organizer' / 'multiclass'
    problem = backend.load_problem(dict(task_path=str(data / 'task.md'), task_dir=str(data), request={}))
    raw_features = problem['train'][problem['features']].copy()
    labels, classes = problem['y'], problem['classes']
    if len(classes) != 10:
        raise ValueError('This experiment requires the known ten-class organizer data')
    groups = pd.util.hash_pandas_object(raw_features, index=False).to_numpy()
    sources = [problem[name] for name in ('train_path', 'test_path', 'sample_path') if problem[name]]
    input_hashes = {str(path): sha256(path) for path in sources}
    report.update(classes=classes, input_sha256=input_hashes, feature_names=list(raw_features.columns))
    save()
    for fold in FOLDS:
        print('START', fold, flush=True)
        fold_started = time.monotonic()
        index_path = scratch / 'night-classification-results' / f'{fold}-indices.npz'
        with np.load(index_path, allow_pickle=False) as indices:
            train, valid, inner_train, inner_valid = (indices[name].copy() for name in
                                                      ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, inner_train, inner_valid)
        inner_parts, outer_parts, hashes = [], [], {str(index_path): sha256(index_path)}
        for relative in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                         f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                         f'late-extra100-results/{fold}-raw100.npz'):
            path = scratch / relative
            hashes[str(path)] = sha256(path)
            with np.load(path, allow_pickle=False) as cache:
                valid_key = 'validation_indices' if 'validation_indices' in cache else 'val_indices'
                if not (np.array_equal(cache[valid_key], valid)
                        and np.array_equal(cache['inner_indices'], inner_valid)):
                    raise ValueError('Cached probability indices mismatch')
                inner_parts.append(validate_probability(cache['inner_probability'].copy(), 10))
                outer_parts.append(validate_probability(cache['probability'].copy(), 10))
        inner_features, outer_features = meta_features(inner_parts), meta_features(outer_parts)
        inner_base, outer_base = np.mean(inner_parts, axis=0), np.mean(outer_parts, axis=0)
        inner_y = labels.iloc[inner_valid].to_numpy()
        meta_train, meta_valid, meta_plan = meta_group_split(raw_features.iloc[inner_valid], inner_y, classes)
        meta_train_rows, meta_valid_rows = inner_valid[meta_train], inner_valid[meta_valid]
        if (set(meta_train_rows) | set(meta_valid_rows)) & (set(inner_train) | set(valid)):
            raise ValueError('Meta rows overlap base fitting rows or outer validation')
        probe, probe_weights = fit_stacker(inner_features[meta_train], inner_y[meta_train], deadline)
        probe_probability = stack_probability(probe, inner_features[meta_valid], 10)
        meta_control = metrics(inner_y[meta_valid], inner_base[meta_valid], classes)
        meta_result = metrics(inner_y[meta_valid], probe_probability, classes)
        selected = use_stacker(meta_control['macro_f1'], meta_result['macro_f1'])
        probe_path = scratch / f'oof-stacker-{fold}-probe.joblib'
        joblib.dump(dict(model=probe, classes=classes, clip=CLIP, parameters=PARAMETERS), probe_path)
        record = dict(fold=fold, guards=guards, cache_sha256=hashes, meta_validation_plan=meta_plan,
                      selected_stacker=selected, meta_control=meta_control, meta_stacker=meta_result,
                      fit_index_sha256={name: array_hash(rows) for name, rows in
                                        [('meta_train', meta_train_rows), ('meta_valid', meta_valid_rows),
                                         ('all_meta_fit', inner_valid), ('base_inner_fit', inner_train),
                                         ('outer_valid', valid)]},
                      rows=dict(meta_train=len(meta_train), meta_valid=len(meta_valid),
                                all_meta_fit=len(inner_valid), outer_valid=len(valid)),
                      probe_weights=probe_weights, probe_iterations=probe.n_iter_.tolist(),
                      probe_model_path=str(probe_path), probe_model_sha256=sha256(probe_path),
                      outer_labels_used_for_selection=False,
                      meta_valid_independent_of_base_hyperparameter_selection=False)
        report['trials'].append(record)
        save()  # Persist selection before final meta fit or any outer scoring.
        outer_selected = outer_base.copy()
        final_probability = np.empty((0, 10))
        if selected:
            final_model, final_weights = fit_stacker(inner_features, inner_y, deadline)
            final_probability = stack_probability(final_model, outer_features, 10)
            outer_selected = final_probability
            final_path = scratch / f'oof-stacker-{fold}-final.joblib'
            joblib.dump(dict(model=final_model, classes=classes, clip=CLIP, parameters=PARAMETERS), final_path)
            record.update(final_weights=final_weights, final_iterations=final_model.n_iter_.tolist(),
                          final_model_path=str(final_path), final_model_sha256=sha256(final_path))
        probability_path = scratch / f'oof-stacker-{fold}-probabilities.npz'
        np.savez_compressed(probability_path, train_indices=train, valid_indices=valid,
                            inner_train_indices=inner_train, inner_valid_indices=inner_valid,
                            meta_train_indices=meta_train_rows, meta_valid_indices=meta_valid_rows,
                            inner_features=inner_features, outer_features=outer_features,
                            meta_probe_probability=probe_probability, inner_base=inner_base,
                            outer_base=outer_base, outer_stacker=final_probability, outer_selected=outer_selected)
        # First outer-label access: every selection and all model fitting are done.
        outer_y = labels.iloc[valid].to_numpy()
        record.update(control=metrics(outer_y, outer_base, classes),
                      selected=metrics(outer_y, outer_selected, classes),
                      probability_path=str(probability_path), probability_sha256=sha256(probability_path),
                      seconds=round(time.monotonic() - fold_started, 3))
        record['delta'] = record['selected']['macro_f1'] - record['control']['macro_f1']
        save()
        print(json.dumps(dict(fold=fold, selected_stacker=selected,
                              meta_control=meta_control['macro_f1'], meta_stacker=meta_result['macro_f1'],
                              control=record['control']['macro_f1'], selected=record['selected']['macro_f1'],
                              delta=record['delta'])), flush=True)
    if input_hashes != {str(path): sha256(path) for path in sources}:
        raise RuntimeError('Original inputs changed')
    report.update(completed=True, input_hashes_unchanged=True,
                  qualifies_for_runtime_followup=all(trial['selected_stacker'] and trial['delta'] >= .002
                                                      for trial in report['trials']))
    save()
    print('COMPLETE', report_path, flush=True)


if __name__ == '__main__':
    main()
