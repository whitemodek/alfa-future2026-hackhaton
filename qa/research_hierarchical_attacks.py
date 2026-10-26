"""Local-only UNSW hierarchy research; excluded from the submission runtime.

Normal versus attack gate followed by nine-way conditional attack prediction.
All choices use nested training folds. Previously used folds and historical
caches make this exploratory, not a blind test or production-quality proof.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from ml_models import fit as fit_model
from qa.research_attack_specialist import (
    FOLDS, array_hash, guarded_splits, metrics, sha256, validate_probability,
)


GATE_REQUEST = dict(iterations=1000, num_leaves=31, depth=-1, learning_rate=.05,
                    random_seed=42, early_stopping_rounds=70,
                    auto_class_weights='None', encoding='native')
TYPE_REQUEST = dict(iterations=1000, num_leaves=63, depth=-1, learning_rate=.05,
                    random_seed=42, early_stopping_rounds=70,
                    auto_class_weights='SqrtBalanced', encoding='native')
LAMBDAS = (0., .25, .5, 1.)


def joint_probabilities(attack_probability, conditional_types, normal_index):
    conditional_types = validate_probability(conditional_types)
    attack_probability = np.asarray(attack_probability, dtype=float)
    count = conditional_types.shape[1] + 1
    if (not isinstance(normal_index, (int, np.integer)) or not 0 <= normal_index < count
            or attack_probability.shape != (len(conditional_types),)
            or not np.isfinite(attack_probability).all()
            or (attack_probability < 0).any() or (attack_probability > 1).any()):
        raise ValueError('Invalid gate probabilities or Normal class index')
    result = np.zeros((len(conditional_types), count), dtype=float)
    attacks = [index for index in range(count) if index != normal_index]
    result[:, normal_index] = 1 - attack_probability
    result[:, attacks] = attack_probability[:, None] * conditional_types
    return validate_probability(result, count)


def blend_probability(base, hierarchy, strength):
    base, hierarchy = validate_probability(base), validate_probability(hierarchy)
    if base.shape != hierarchy.shape or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Invalid hierarchy blend')
    return validate_probability((1 - strength) * base + strength * hierarchy)


def select_strength(base, hierarchy, truth):
    from sklearn.metrics import f1_score
    scores = [float(f1_score(truth, blend_probability(base, hierarchy, strength).argmax(axis=1),
                             labels=range(base.shape[1]), average='macro', zero_division=0))
              for strength in LAMBDAS]
    return float(LAMBDAS[int(np.argmax(scores))]), scores


class HierarchicalAttackStrategy:
    """Two saved fold-only adapters; refit has no validation-label argument."""
    def __init__(self, classes):
        self.classes = list(classes)
        if len(self.classes) < 3 or len(set(self.classes)) != len(self.classes) or self.classes.count('Normal') != 1:
            raise ValueError('Expected distinct classes including exactly one Normal')
        self.normal_index = self.classes.index('Normal')
        self.attack_indices = [i for i in range(len(self.classes)) if i != self.normal_index]
        self.type_mapping = {original: local for local, original in enumerate(self.attack_indices)}
        self.gate = self.types = None
        self.fit_indices = None
        self.stop_indices = None

    @staticmethod
    def _indices(rows, total, forbidden=()):
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
                or (rows < 0).any() or (rows >= total).any()
                or len(np.unique(rows)) != len(rows) or set(rows) & set(forbidden)):
            raise ValueError('Invalid or forbidden training/validation row indices')
        return rows

    def _targets(self, y, indices):
        labels = np.asarray(y.iloc[indices])
        if not np.issubdtype(labels.dtype, np.integer) or set(labels) != set(range(len(self.classes))):
            raise ValueError('Each fitting/stopping fold must contain every declared class')
        mask = labels != self.normal_index
        attacks = indices[mask]
        types = np.asarray([self.type_mapping[int(value)] for value in labels[mask]])
        return (labels != self.normal_index).astype(int), attacks, types

    def fit(self, x, y, train_indices, valid_indices, categorical, deadline, threads=2,
            forbidden_indices=()):
        train_indices = self._indices(train_indices, len(x), forbidden_indices)
        valid_indices = self._indices(valid_indices, len(x), forbidden_indices)
        if set(train_indices) & set(valid_indices):
            raise ValueError('Training and stopping rows overlap')
        gate_y, attack_train, types_y = self._targets(y, train_indices)
        gate_valid_y, attack_valid, types_valid_y = self._targets(y, valid_indices)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No time for hierarchical training')
        gate = fit_model('lightgbm', x.iloc[train_indices], gate_y,
                         x.iloc[valid_indices], gate_valid_y, categorical, GATE_REQUEST,
                         time.monotonic() + remaining * .4, threads, 'classification', 'log_loss')
        types = fit_model('lightgbm', x.iloc[attack_train], types_y,
                          x.iloc[attack_valid], types_valid_y, categorical, TYPE_REQUEST,
                          deadline, threads, 'classification', 'macro_f1')
        self.gate, self.types = gate, types
        self.fit_indices, self.stop_indices = train_indices.copy(), valid_indices.copy()
        self.attack_fit_indices, self.attack_stop_indices = attack_train.copy(), attack_valid.copy()
        return self

    def refit(self, x, y, train_indices, deadline, threads=2, forbidden_indices=()):
        if self.gate is None or self.types is None:
            raise ValueError('Cannot refit an unfitted hierarchy')
        train_indices = self._indices(train_indices, len(x), forbidden_indices)
        gate_y, attack_train, types_y = self._targets(y, train_indices)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No time for hierarchical refit')
        result = HierarchicalAttackStrategy(self.classes)
        result.gate = self.gate.refit(x.iloc[train_indices], gate_y,
                                      time.monotonic() + remaining * .4, threads)
        result.types = self.types.refit(x.iloc[attack_train], types_y, deadline, threads)
        if (result.gate.tree_count_ != self.gate.tree_count_
                or result.types.tree_count_ != self.types.tree_count_):
            raise RuntimeError('Hierarchy refit did not preserve both selected counts')
        result.fit_indices, result.attack_fit_indices = train_indices.copy(), attack_train.copy()
        return result

    def predict_proba(self, x):
        if self.gate is None or self.types is None:
            raise ValueError('Cannot predict with an unfitted hierarchy')
        gate = validate_probability(self.gate.predict_proba(x), 2)
        types = validate_probability(self.types.predict_proba(x), len(self.attack_indices))
        if list(self.gate.classes_) != [0, 1] or list(self.types.classes_) != list(range(len(self.attack_indices))):
            raise ValueError('Saved stage class order does not match the hierarchy mapping')
        return joint_probabilities(gate[:, 1], types, self.normal_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, started = args.root.resolve(), time.monotonic()
    scratch = root / '.test-temp'
    protocol_path, report_path = (scratch / f'hierarchical-attacks-{name}.json' for name in ('protocol', 'report'))
    if protocol_path.exists() or report_path.exists():
        raise FileExistsError('Existing experiment must not be silently overwritten')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['MPLCONFIGDIR'] = str(scratch / 'matplotlib')
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import ml_backend as backend

    protocol = dict(folds=FOLDS, gate=GATE_REQUEST, types=TYPE_REQUEST, blend_strengths=LAMBDAS,
                    threads=2, max_seconds=300, api_calls=0, production_changed=False,
                    primary='Inner-only blend-strength selection against fixed historical cached baseline.',
                    secondary='Pure hierarchy evaluated as predeclared descriptive secondary; never outer-selected.',
                    continuation_gate='Inner-selected strength>0 and outer paired MacroF1 delta>=0.002 EACH fold. '
                                      'Engineering threshold, not significance; no post-outer tuning.',
                    architecture='Normal versus Attack gate (unweighted log loss); nine-way attack classifier '
                                 '(SqrtBalanced MacroF1). P(Normal)=1-P(Attack); P(type)=P(Attack)*P(type|Attack).',
                    selection='Stage counts on inner validation only; both stages exact-count outer-train refit; '
                              'strength selected on full10class inner F1 before any outer labels are accessed.',
                    warning='Exploratory previously used folds/historical base; no blind-test or runtime-gain claim.',
                    source_sha256=sha256(__file__), adapter_sha256=sha256(root / 'ml_models.py'),
                    backend_sha256=sha256(root / 'ml_backend.py'),
                    guard_source_sha256=sha256(root / 'qa' / 'research_attack_specialist.py'),
                    historical_cache_producer_sha256={name: sha256(scratch / name)
                                                       for name in ('night-classification.py', 'late-extra100.py')})
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, protocol_sha256=sha256(protocol_path), trials=[], completed=False)

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    deadline = started + 300
    data = scratch / 'organizer' / 'multiclass'
    problem = backend.load_problem(dict(task_path=str(data / 'task.md'), task_dir=str(data), request={}))
    x, _, categorical = backend._features(problem)
    y, classes = problem['y'], problem['classes']
    if len(classes) != 10 or 'Normal' not in classes:
        raise ValueError('Research protocol requires the known ten-class organizer task')
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    sources = [problem[name] for name in ('train_path', 'test_path', 'sample_path') if problem[name]]
    input_hashes = {str(path): sha256(path) for path in sources}
    report.update(classes=classes, input_sha256=input_hashes, feature_names=list(x.columns))
    save()
    for fold_no, fold in enumerate(FOLDS):
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
                    raise ValueError('Historical cache indices mismatch')
                inner_parts.append(validate_probability(cache['inner_probability'].copy(), 10))
                outer_parts.append(validate_probability(cache['probability'].copy(), 10))
        inner_base, outer_base = np.mean(inner_parts, axis=0), np.mean(outer_parts, axis=0)
        allowance = (deadline - time.monotonic()) / (len(FOLDS) - fold_no)
        if allowance < 15:
            raise TimeoutError('Insufficient remaining hierarchy budget')
        with threadpool_limits(limits=2):
            model = HierarchicalAttackStrategy(classes).fit(
                x, y, inner_train, inner_valid, categorical,
                time.monotonic() + allowance * .55, 2, forbidden_indices=valid)
            inner_hierarchy = model.predict_proba(x.iloc[inner_valid])
            strength, scores = select_strength(inner_base, inner_hierarchy, y.iloc[inner_valid])
            record = dict(fold=fold, guards=guards, cache_sha256=hashes, selected_strength=strength,
                          stage_rounds=dict(gate=model.gate.tree_count_, types=model.types.tree_count_),
                          inner_scores=[dict(strength=value, macro_f1=score)
                                        for value, score in zip(LAMBDAS, scores)],
                          training_parameters=dict(gate=model.gate.parameters, types=model.types.parameters),
                          fit_index_sha256={name: array_hash(rows) for name, rows in
                                            [('inner_train', inner_train), ('inner_stop', inner_valid),
                                             ('inner_attack_train', model.attack_fit_indices),
                                             ('inner_attack_stop', model.attack_stop_indices),
                                             ('outer_train', train), ('outer_valid', valid)]},
                          rows=dict(inner_train=len(inner_train), inner_stop=len(inner_valid),
                                    inner_attack_train=len(model.attack_fit_indices),
                                    inner_attack_stop=len(model.attack_stop_indices),
                                    outer_train=len(train), outer_valid=len(valid)),
                          outer_labels_used_for_selection=False, exact_outer_refit=False)
            report['trials'].append(record)
            save()  # Stage counts and strength pinned before outer refit/evaluation.
            full = model.refit(x, y, train, min(deadline, fold_started + allowance),
                               2, forbidden_indices=valid)
            outer_hierarchy = full.predict_proba(x.iloc[valid])
        selected = blend_probability(outer_base, outer_hierarchy, strength)
        probability_path = scratch / f'hierarchical-attacks-{fold}-probabilities.npz'
        np.savez_compressed(probability_path, train_indices=train, valid_indices=valid,
                            inner_train_indices=inner_train, inner_valid_indices=inner_valid,
                            inner_attack_fit_indices=model.attack_fit_indices,
                            inner_attack_stop_indices=model.attack_stop_indices,
                            outer_attack_fit_indices=full.attack_fit_indices,
                            inner_base=inner_base, outer_base=outer_base,
                            inner_hierarchy=inner_hierarchy, outer_hierarchy=outer_hierarchy,
                            outer_selected=selected)
        # First outer-label access, after models, counts, strength and predictions.
        outer_y = y.iloc[valid].to_numpy()
        record.update(control=metrics(outer_y, outer_base, classes),
                      selected=metrics(outer_y, selected, classes),
                      pure_hierarchy_secondary=metrics(outer_y, outer_hierarchy, classes),
                      exact_outer_refit=True, outer_attack_fit_rows=len(full.attack_fit_indices),
                      probability_path=str(probability_path), probability_sha256=sha256(probability_path),
                      seconds=round(time.monotonic() - fold_started, 3))
        record['delta'] = record['selected']['macro_f1'] - record['control']['macro_f1']
        save()
        print(json.dumps(dict(fold=fold, strength=strength, rounds=record['stage_rounds'],
                              control=record['control']['macro_f1'], selected=record['selected']['macro_f1'],
                              pure_hierarchy=record['pure_hierarchy_secondary']['macro_f1'],
                              delta=record['delta'])), flush=True)
    if input_hashes != {str(path): sha256(path) for path in sources}:
        raise RuntimeError('Original data changed during hierarchy experiment')
    report.update(completed=True, input_hashes_unchanged=True,
                  improves_both=all(trial['delta'] > 0 for trial in report['trials']),
                  qualifies_for_runtime_followup=all(trial['selected_strength'] > 0 and trial['delta'] >= .002
                                                      for trial in report['trials']))
    save()
    print('COMPLETE', report_path, flush=True)


if __name__ == '__main__':
    main()
