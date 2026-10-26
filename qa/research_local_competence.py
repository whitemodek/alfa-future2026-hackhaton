"""Public-data pilot of a fixed, input-dependent local-competence ensemble.

Research only, not a scientific novelty or leaderboard claim. Loss targets are
strict three-fold group OOF errors; a separate selection fold freezes policy.
No OOF/selection labels enter an expert fit or early stopping.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import numpy as np

from qa.research_attack_specialist import array_hash, sha256, validate_probability


CASES = ('energy_efficiency', 'wine_quality_red', 'diabetes',
         'bank_marketing_no_duration', 'wine_quality_white', 'digits')
EXPERTS = ('linear', 'lightgbm', 'kernel')
OUTER_SEED = 20261028
LGB_ROUNDS = 180
MAX_CELLS = 50_000_000


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Research computation deadline reached')


def sparse_preprocessor(x, categorical):
    """Reuse fold-only numeric/OHE preprocessing; dummy y is never informative."""
    from ml_models import _adapter
    parameters = dict(native={}, weighting='None', encoding='linear_sparse')
    return _adapter('linear', x, np.zeros(len(x)), categorical, parameters, 'regression', 'rmse', 2)


class Expert:
    def __init__(self, kind, classification, count, metric):
        self.kind, self.classification, self.count, self.metric = kind, classification, count, metric
        self.adapter = self.kernel = self.model = None
        self.observed = self.constant = None

    def fit(self, x, y, categorical, deadline):
        from ml_models import fit, _sample_weights
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression, Ridge
        from threadpoolctl import threadpool_limits
        check_time(deadline)
        if not 1 <= len(x) <= 50_000:
            raise ValueError('Expert training row cap is50000')
        y = np.asarray(y)
        if len(y) != len(x):
            raise ValueError('Expert target length differs')
        local_y = y
        if self.classification:
            self.observed, local_y = np.unique(y, return_inverse=True)
            if (not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= self.count).any()):
                raise ValueError('Invalid global class labels')
            if len(self.observed) == 1:
                self.constant = int(self.observed[0])
                return self
        if self.kind == 'linear':
            self.adapter = fit('linear', x, local_y, x, local_y, categorical, {}, deadline, 2,
                               'classification' if self.classification else 'regression', self.metric)
        elif self.kind == 'lightgbm':
            request = dict(iterations=1, num_leaves=31, depth=-1, learning_rate=.05,
                           random_seed=42, early_stopping_rounds=70)
            # The one-round factory receives TRAIN as validation. It only supplies
            # fixed preprocessing/parameters; no held-out labels or fitted tree is retained.
            prototype = fit('lightgbm', x, local_y, x, local_y, categorical, request,
                            deadline, 2, 'classification' if self.classification else 'regression', self.metric)
            prototype.tree_count_ = LGB_ROUNDS
            self.adapter = prototype.refit(x, local_y, deadline, 2)  # select=False, exactly180 rounds.
            if self.adapter.tree_count_ != LGB_ROUNDS:
                raise RuntimeError('Fixed LightGBM round count changed')
        elif self.kind == 'kernel':
            self.adapter = sparse_preprocessor(x, categorical)
            transformed = self.adapter._transform(x)
            width = transformed.shape[1]
            components = min(256, len(x))
            if width > 200_000 or len(x) * components > MAX_CELLS:
                raise ValueError('Nystroem encoded width or dense expansion exceeds cap')
            self.kernel = Nystroem(kernel='rbf', gamma=1 / max(1, width),
                                    n_components=components, random_state=42, n_jobs=2)
            with threadpool_limits(limits=2):
                encoded = self.kernel.fit_transform(transformed)
                self.model = (LogisticRegression(C=1., solver='lbfgs', max_iter=200, random_state=42)
                              if self.classification else Ridge(alpha=1.))
                weighting = ('SqrtBalanced' if self.metric in {'macro_f1', 'balanced_accuracy'} else 'None')
                weights = _sample_weights(local_y, weighting) if self.classification else None
                self.model.fit(encoded, local_y, sample_weight=weights)
        else:
            raise ValueError('Unknown expert family')
        check_time(deadline)
        return self

    def predict(self, x):
        from threadpoolctl import threadpool_limits
        if self.constant is not None:
            result = np.zeros((len(x), self.count))
            result[:, self.constant] = 1.
            return result
        if self.kind == 'kernel':
            if len(x) * self.kernel.n_components > MAX_CELLS:
                raise ValueError('Nystroem query expansion exceeds cap')
            with threadpool_limits(limits=2):
                encoded = self.kernel.transform(self.adapter._transform(x))
                result = self.model.predict_proba(encoded) if self.classification else self.model.predict(encoded)
        else:
            result = self.adapter.predict_proba(x) if self.classification else self.adapter.predict(x)
        if not self.classification:
            result = np.asarray(result, dtype=float).reshape(-1)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite expert prediction')
            return result
        aligned = np.zeros((len(x), self.count))
        aligned[:, self.observed] = result
        return validate_probability(aligned, self.count)

    def bundle(self):
        return dict(kind=self.kind, classification=self.classification, count=self.count, metric=self.metric,
                    adapter=self.adapter, kernel=self.kernel, model=self.model,
                    observed=self.observed, constant=self.constant)


def fit_experts(x, y, categorical, classification, count, metric, deadline):
    return [Expert(kind, classification, count, metric).fit(x, y, categorical, deadline) for kind in EXPERTS]


def honest_oof(x, y, groups, categorical, classification, count, metric, deadline):
    """Every row is predicted by experts fitted without its entire feature group."""
    from sklearn.model_selection import GroupKFold
    if len(np.unique(groups)) < 3:
        raise ValueError('Three independent feature groups are required')
    shape = (len(x), 3, count) if classification else (len(x), 3)
    predictions = np.full(shape, np.nan)
    folds, assignment = [], np.full(len(x), -1, dtype=int)
    for fold, (train, valid) in enumerate(GroupKFold(3).split(x, groups=groups)):
        check_time(deadline)
        if set(train) & set(valid) or set(groups[train]) & set(groups[valid]):
            raise ValueError('An OOF feature group crossed the fit/prediction boundary')
        models = fit_experts(x.iloc[train], np.asarray(y)[train], categorical,
                             classification, count, metric, deadline)
        for column, model in enumerate(models):
            predictions[valid, column] = model.predict(x.iloc[valid])
        assignment[valid] = fold
        folds.append(dict(fold=fold, train=x.index.to_numpy()[train].tolist(),
                          valid=x.index.to_numpy()[valid].tolist()))
    if not np.isfinite(predictions).all() or (assignment < 0).any():
        raise ValueError('OOF predictions are incomplete')
    return predictions, assignment, folds


def row_losses(predictions, truth, classification):
    truth = np.asarray(truth)
    if classification:
        if predictions.ndim != 3 or len(predictions) != len(truth):
            raise ValueError('Invalid classification OOF shape')
        losses = -np.log(np.clip(predictions[np.arange(len(truth))[:, None],
                                             np.arange(3)[None, :], truth[:, None]], 1e-6, 1.))
    else:
        losses = (predictions - truth[:, None]) ** 2
    if losses.shape != (len(truth), 3) or not np.isfinite(losses).all():
        raise ValueError('Invalid per-row OOF loss targets')
    return losses


def competence_weights(predicted_losses):
    predicted_losses = np.asarray(predicted_losses, dtype=float)
    if predicted_losses.ndim != 2 or predicted_losses.shape[1] != 3 or not np.isfinite(predicted_losses).all():
        raise ValueError('Expected three finite predicted losses per row')
    weights = np.full_like(predicted_losses, 1 / 6)
    weights[np.arange(len(weights)), predicted_losses.argmin(axis=1)] += .5
    return weights


def combine(predictions, weights):
    predictions, weights = np.asarray(predictions), np.asarray(weights)
    if weights.shape != predictions.shape[:2] or not np.allclose(weights.sum(axis=1), 1):
        raise ValueError('Prediction and normalized weight shapes differ')
    return ((predictions * weights[:, :, None]).sum(axis=1) if predictions.ndim == 3
            else (predictions * weights).sum(axis=1))


def fit_gate(x, oof_predictions, truth, categorical, classification, deadline):
    from sklearn.ensemble import ExtraTreesRegressor
    check_time(deadline)
    preprocessor = sparse_preprocessor(x, categorical)
    losses = row_losses(oof_predictions, truth, classification)
    model = ExtraTreesRegressor(n_estimators=100, max_depth=5,
                                 min_samples_leaf=max(10, len(x) // 100), max_features=.75,
                                 random_state=42, n_jobs=2)
    model.fit(preprocessor._transform(x), losses)
    check_time(deadline)
    return preprocessor, model


def local_prediction(gate, x, predictions):
    preprocessor, model = gate
    weights = competence_weights(model.predict(preprocessor._transform(x)))
    result = combine(predictions, weights)
    if result.ndim == 2:
        result = validate_probability(result)
    return result, weights


def score(metric, truth, prediction):
    from sklearn.metrics import f1_score, roc_auc_score
    if metric == 'rmse':
        return float(np.sqrt(np.mean((np.asarray(truth) - prediction) ** 2)))
    if metric == 'macro_f1':
        return float(f1_score(truth, prediction.argmax(axis=1), labels=range(prediction.shape[1]),
                              average='macro', zero_division=0))
    if metric == 'roc_auc':
        return float(roc_auc_score(truth, prediction[:, 1]))
    raise ValueError('Unsupported pilot metric')


def choose_policy(selection_scores, metric):
    names = [*EXPERTS, 'uniform']
    higher = metric != 'rmse'
    control = max(names, key=lambda name: selection_scores[name]) if higher else min(
        names, key=lambda name: selection_scores[name])
    improved = (selection_scores['local'] > selection_scores[control] if higher
                else selection_scores['local'] < selection_scores[control])
    return 'local' if improved else control, control


def inner_split(x, y, classification, count):
    import pandas as pd
    from ml_validation import plan_validation
    problem = dict(train=x, test=x.iloc[:0], y=pd.Series(np.asarray(y)), features=list(x.columns),
                   classes=list(range(count)) if classification else None, text='', target='__target__')
    plan = plan_validation(problem, dict(split='group', group_by_features=True, validation_fraction=.2))
    return plan['train_indices'], plan['valid_indices'], {k: v for k, v in plan.items()
                                                          if k not in {'train_indices', 'valid_indices'}}


def promotion_gate(results):
    if len(results) != 6 or any(record.get('status') != 'ok' for record in results):
        return dict(passes=False, reason='All six completed cases are required')
    nonworse, gains, gain_types, safe = 0, 0, set(), True
    for record in results:
        control, selected = record['outer_scores'][record['control']], record['outer_scores']['selected']
        delta = control - selected if record['metric'] == 'rmse' else selected - control
        nonworse += delta >= -1e-12
        if delta > 1e-12:
            gains += 1
            gain_types.add(record['problem_type'])
        safe &= ((selected - control) / max(abs(control), 1e-12) <= .01 if record['metric'] == 'rmse'
                 else control - selected <= .005)
    return dict(passes=bool(nonworse >= 4 and gains >= 2 and len(gain_types) == 2 and safe),
                nonworse=int(nonworse), strict_gains=int(gains), gain_problem_types=sorted(gain_types),
                degradation_limits_satisfied=bool(safe))


def run_case(spec, root, out, deadline, report, save):
    import joblib
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import qa.transfer_benchmark as transfer
    started, cpu_started = time.monotonic(), time.process_time()
    frame = pd.read_csv(spec['path'])
    target = frame[spec['target']]
    raw = frame.drop(columns=[spec['target']])
    ids = [name for name in raw if re.search(r'(^id$|^id_|_id$|^index$|^row_?id$)', name, re.I)]
    raw = raw.drop(columns=ids)
    classification, metric = spec['problem_type'] == 'classification', spec['metric']
    with patch.object(transfer, 'SEED', OUTER_SEED):
        outer_train, outer_valid, outer_plan = transfer.outer_split(raw, target, classification)
    # Labels used for stratification above; model class encoding and all weights
    # are fitted on outer training only, never on outer class counts.
    encoder = LabelEncoder().fit(target.iloc[outer_train]) if classification else None
    train_y = (encoder.transform(target.iloc[outer_train]) if classification
               else target.iloc[outer_train].to_numpy(dtype=float))
    count = len(encoder.classes_) if classification else None
    x = raw.copy()
    categorical = []
    for name in x:
        if pd.api.types.is_numeric_dtype(x[name]):
            x[name] = pd.to_numeric(x[name], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(name)
            x[name] = x[name].fillna('__MISSING__').astype(str)
    train_x, outer_x = x.iloc[outer_train], x.iloc[outer_valid]
    raw_train = raw.iloc[outer_train].reset_index(drop=True)
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    if set(groups[outer_train]) & set(groups[outer_valid]):
        raise ValueError('Feature group crosses outer split')
    pool, selection, inner_plan = inner_split(raw_train, train_y, classification, count)
    if pool is None or selection is None or set(groups[outer_train[pool]]) & set(groups[outer_train[selection]]):
        raise ValueError('Invalid independent selection split')
    pool_x, pool_y = train_x.iloc[pool], train_y[pool]
    oof, assignment, oof_folds = honest_oof(pool_x, pool_y, groups[outer_train[pool]],
                                          categorical, classification, count, metric, deadline)
    gate = fit_gate(pool_x, oof, pool_y, categorical, classification, deadline)
    inner_models = fit_experts(pool_x, pool_y, categorical, classification, count, metric, deadline)
    selection_predictions = np.stack([model.predict(train_x.iloc[selection]) for model in inner_models], axis=1)
    selection_local, selection_weights = local_prediction(gate, train_x.iloc[selection], selection_predictions)
    selection_candidates = {name: selection_predictions[:, i] for i, name in enumerate(EXPERTS)}
    selection_candidates.update(uniform=selection_predictions.mean(axis=1), local=selection_local)
    selection_scores = {name: score(metric, train_y[selection], value)
                         for name, value in selection_candidates.items()}
    policy, control = choose_policy(selection_scores, metric)
    record = dict(name=spec['name'], metric=metric, problem_type=spec['problem_type'], status='selected',
                  source_sha256=sha256(spec['path']), source_url=spec.get('source_url'), dropped_ids=ids,
                  classes=encoder.classes_.tolist() if classification else None,
                  outer_plan=outer_plan, inner_plan=inner_plan, selected_policy=policy, control=control,
                  selection_scores=selection_scores, outer_labels_used_for_policy=False,
                  outer_train_sha256=array_hash(outer_train), outer_valid_sha256=array_hash(outer_valid),
                  original_feature_names=list(raw.columns), oof_folds=oof_folds,
                  training_rows=len(outer_train), outer_rows=len(outer_valid))
    report['cases'].append(record)
    save()  # Freeze chosen policy/control before outer refits, prediction or scoring.
    selected_gate = None
    full_oof, full_assignment, full_folds = np.empty((0,)), np.empty((0,), dtype=int), []
    if policy == 'local':
        full_oof, full_assignment, full_folds = honest_oof(train_x, train_y, groups[outer_train],
                                                         categorical, classification, count, metric, deadline)
        selected_gate = fit_gate(train_x, full_oof, train_y, categorical, classification, deadline)
    final_models = fit_experts(train_x, train_y, categorical, classification, count, metric, deadline)
    outer_predictions = np.stack([model.predict(outer_x) for model in final_models], axis=1)
    outer_candidates = {name: outer_predictions[:, i] for i, name in enumerate(EXPERTS)}
    outer_candidates['uniform'] = outer_predictions.mean(axis=1)
    outer_weights = np.empty((0, 3))
    if policy == 'local':
        outer_candidates['selected'], outer_weights = local_prediction(selected_gate, outer_x, outer_predictions)
    else:
        outer_candidates['selected'] = outer_candidates[policy].copy()
    model_path = out / f'{spec["name"]}-models.joblib'
    joblib.dump(dict(experts=[model.bundle() for model in final_models], gate=selected_gate,
                     policy=policy, control=control, classes=record['classes']), model_path)
    prediction_path = out / f'{spec["name"]}-predictions.npz'
    np.savez_compressed(prediction_path, outer_train=outer_train, outer_valid=outer_valid,
                        pool=outer_train[pool], selection=outer_train[selection],
                        oof=oof, oof_assignment=assignment, pool_truth=pool_y,
                        selection_predictions=selection_predictions, selection_weights=selection_weights,
                        selection_local=selection_local, selection_truth=train_y[selection],
                        full_oof=full_oof, full_oof_assignment=full_assignment,
                        outer_experts=outer_predictions, outer_selected=outer_candidates['selected'],
                        outer_weights=outer_weights)
    # First access to outer labels since stratification: policy and every fit are complete.
    outer_truth = (encoder.transform(target.iloc[outer_valid]) if classification
                   else target.iloc[outer_valid].to_numpy(dtype=float))
    outer_scores = {name: score(metric, outer_truth, value) for name, value in outer_candidates.items()}
    record.update(status='ok', outer_scores=outer_scores, full_oof_folds=full_folds,
                  model_path=str(model_path), model_sha256=sha256(model_path),
                  prediction_path=str(prediction_path), prediction_sha256=sha256(prediction_path),
                  seconds=round(time.monotonic() - started, 3), cpu_seconds=round(time.process_time() - cpu_started, 3))
    save()
    print(json.dumps(dict(name=spec['name'], policy=policy, control=control,
                          selection=selection_scores, outer=outer_scores, seconds=record['seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, started, cpu_started = args.root.resolve(), time.monotonic(), time.process_time()
    out = root / '.test-temp' / 'local-competence'
    out.mkdir(exist_ok=True)
    protocol_path, report_path = out / 'protocol.json', out / 'report.json'
    if protocol_path.exists() or report_path.exists():
        raise FileExistsError('Do not overwrite a frozen pilot')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['MPLCONFIGDIR'] = str(root / '.test-temp' / 'matplotlib')
    from threadpoolctl import threadpool_limits
    specs = json.loads((root / '.test-temp' / 'transfer-specs.json').read_text(encoding='utf-8'))
    specs = [next(spec for spec in specs if spec['name'] == name) for name in CASES]
    protocol = dict(cases=CASES, outer_seed=OUTER_SEED,
                    outer='Exact qa.transfer_benchmark.outer_split with scoped SEED=20261028; same parent native comparison.',
                    inner='Explicit group20% selection with seed42 planner. GroupKFold3 honest OOF on remaining80%. '
                          'No expert has OOF/selection labels in fitting or early stopping.',
                    experts=dict(linear='Existing fixed linear adapter; observed-fold-only class encoding.',
                                 lightgbm='Fixed180 rounds,31 leaves,lr.05; factory1round TRAIN-as-validation then exact refit select=False.',
                                 kernel='Train-only sparse median/OHE/scaler, RBF Nystroem min(256,ntrain) landmarks, '
                                        'gamma=1/encodedwidth, Ridgealpha1 or LogisticC1max200.'),
                    gate='ExtraTreesRegressor100,depth5,minleaf=max(10,nOOF//100),maxfeatures.75,njobs2,seed42; '
                         'targets actual per-expert group-OOF squared error or per-row true-class negative log probability. '
                         'Input raw-feature train-only sparse preprocessing; weights .5uniform+.5onehotargmin predicted loss.',
                    selection='On separate selection choose best fixed single/uniform as CONTROL; local policy chosen only '
                              'on strict improvement. Freeze policy, refit experts on ALLoutertrain. '
                              'If local selected, regenerate honest3fold OOF on ALLoutertrain and refit loss gate.',
                    promotion='All6 completed; >=4nonworse, >=2strict gains including public classification+regression; '
                              'no relative RMSE degradation>1% or score loss>0.005 versus selected global CONTROL.',
                    novelty='New for this repository; not a claim of scientific novelty or SOTA.',
                    limitations='One fixed split per public dataset, some previously inspected datasets; exploratory pilot, '
                                'not leaderboard proof. Native comparison is separate, with different model set and compute budget '
                                '(parent90s warmup,2threads), not a matched compute-budget experiment.',
                    threads=2, case_seconds=240, total_seconds=900, api_calls=0, production_changed=False,
                    source_sha256=sha256(__file__), adapter_sha256=sha256(root / 'ml_models.py'),
                    validation_sha256=sha256(root / 'ml_validation.py'),
                    outer_helper_sha256=sha256(root / 'qa' / 'transfer_benchmark.py'),
                    input_sha256={spec['name']: sha256(spec['path']) for spec in specs}, specs=specs)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, protocol_sha256=sha256(protocol_path), cases=[], completed=False)

    def save():
        report.update(seconds=round(time.monotonic() - started, 3),
                      cpu_seconds=round(time.process_time() - cpu_started, 3))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for spec in specs:
        check_time(started + 900)
        print('START', spec['name'], flush=True)
        try:
            with threadpool_limits(limits=2):
                run_case(spec, root, out, min(started + 900, time.monotonic() + 240), report, save)
        except Exception as exc:
            existing = next((case for case in report['cases'] if case['name'] == spec['name']), None)
            if existing is None:
                existing = dict(name=spec['name'])
                report['cases'].append(existing)
            existing.update(status='error', error=type(exc).__name__ + ': ' + str(exc))
            save()
            print('ERROR', spec['name'], existing['error'], flush=True)
    report.update(completed=True, promotion=promotion_gate(report['cases']),
                  inputs_unchanged=all(sha256(spec['path']) == protocol['input_sha256'][spec['name']] for spec in specs))
    save()
    print('COMPLETE', report_path, json.dumps(report['promotion']), flush=True)


if __name__ == '__main__':
    main()
