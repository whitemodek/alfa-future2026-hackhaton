"""Group-safe AutoGluon portfolio on two external organizer-training holdouts.

Only original train labels are used. No leaderboard/test labels, API calls,
pretrained downloads or random-fold mixing of duplicate feature groups.
Internal stacked OOF scores are not treated as independent quality evidence.
"""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from qa.research_attack_specialist import FOLDS, sha256, guarded_splits, metrics, array_hash


def main(diverse=False):
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / ('ag-pilot-diverse' if diverse else 'ag-pilot')
    out.mkdir(exist_ok=False)
    import torch
    from autogluon.tabular import TabularPredictor
    from ml_automl import grouped_bag_labels, hyperparameters
    import ml_backend as b
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol = dict(folds=FOLDS, time_limit_per_fold=1200, bag_folds=3, stack_levels=1, threads=4,
        weights=('SqrtBalanced computed INSIDE EACH fit fold, including neural CrossEntropyLoss' if diverse else
                 'SqrtBalanced computed INSIDE EACH model fit fold; NN_TORCH unweighted'),
        diverse_allocation=diverse,
        per_child_time_caps=dict(catboost=45, neural=120) if diverse else {},
        grouping='StratifiedGroupKFold3 assignments supplied as 3 LeaveOneGroupOut fold IDs, '
                 'not a group per unique vector; marker excluded from model features',
        gate='External Macro F1 gain >=.002 on BOTH folds before exact runtime follow-up',
        api_calls=0, platform_submissions=0, pretrained_downloads=False,
        source_sha256={n: sha256(root / n) for n in ('ml_automl.py', 'qa/research_autogluon.py')},
        versions={name: importlib.metadata.version(name) for name in
                  ('autogluon.tabular', 'autogluon.core', 'torch', 'numpy', 'pandas', 'scikit-learn')},
        limitation='Known dataset/historical outer splits; internal stacked OOF scores can be optimistic '
                   'and are not independent validation. Different compute/model pool than historical cached control.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    started = time.monotonic()
    report = dict(protocol=protocol, trials=[], completed=False)
    data = scratch / 'organizer/multiclass'
    p = b.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request={}))
    x, _, _ = b._features(p)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    report['input_sha256'] = b._input_hashes(p)

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    for fold in FOLDS:
        with np.load(scratch / f'night-classification-results/{fold}-indices.npz', allow_pickle=False) as a:
            train, valid, it, iv = (a[k].copy() for k in ('train', 'valid', 'inner_train', 'inner_valid'))
        guards = guarded_splits(groups, train, valid, it, iv)
        frame = x.iloc[train].reset_index(drop=True)
        y = p['y'].iloc[train].to_numpy()
        assignment = grouped_bag_labels(frame, y)
        if '__target__' in frame or '__cv_fold__' in frame:
            raise ValueError('Reserved AutoML columns collide')
        frame['__target__'] = y
        frame['__cv_fold__'] = assignment
        row = dict(fold=fold, guards=guards, source_rows=len(train), outer_rows=len(valid),
            bag_assignment_sha256=array_hash(assignment), completed=False)
        report['trials'].append(row)
        np.savez_compressed(out / f'{fold}-split.npz', train=train, valid=valid, bag_assignment=assignment)
        save()
        print('START', fold, flush=True)
        predictor = TabularPredictor(label='__target__', problem_type='multiclass', eval_metric='f1_macro',
            groups='__cv_fold__', path=str(out / fold), verbosity=2,
            log_to_file=True, log_file_path=str(out / f'{fold}-training.log')).fit(
                frame, time_limit=1200, hyperparameters=hyperparameters(diverse=diverse),
                num_bag_folds=3, num_bag_sets=1, num_stack_levels=1, dynamic_stacking=False,
                num_cpus=4, num_gpus=0, fit_strategy='sequential', memory_limit=12,
                ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'})
        row['models'] = predictor.model_names()
        row['best_model'] = predictor.model_best
        predictor.leaderboard().to_csv(out / f'{fold}-internal-leaderboard.csv', index=False)
        probability = predictor.predict_proba(x.iloc[valid])
        if set(probability.columns) != set(range(len(p['classes']))):
            raise ValueError('AutoML dropped or changed a target class')
        probability = probability[list(range(len(p['classes'])))].to_numpy()
        if '__cv_fold__' in predictor.features() or '__target__' in predictor.features():
            raise ValueError('Reserved splitting/target marker leaked into model features')
        np.savez_compressed(out / f'{fold}-frozen.npz', probability=probability, valid_indices=valid)
        # The fitted predictor, chosen model and probabilities are now frozen.
        parts = []
        for rel in (f'night-classification-results/{fold}-lightgbm63_sqrt.npz',
                    f'night-classification-results/{fold}-xgboost6_sqrt.npz',
                    f'late-extra100-results/{fold}-raw100.npz'):
            with np.load(scratch / rel, allow_pickle=False) as a:
                key = 'validation_indices' if 'validation_indices' in a else 'val_indices'
                if not np.array_equal(a[key], valid):
                    raise ValueError('Cached control alignment changed')
                parts.append(a['probability'].copy())
        row['base'] = metrics(p['y'].iloc[valid].to_numpy(), np.mean(parts, axis=0), p['classes'])
        row['candidate'] = metrics(p['y'].iloc[valid].to_numpy(), probability, p['classes'])
        row['gain'] = row['candidate']['macro_f1'] - row['base']['macro_f1']
        row['completed'] = True
        save()
        print('RESULT', fold, row['base']['macro_f1'], row['candidate']['macro_f1'], row['gain'], flush=True)
        del predictor
        gc.collect()
    report.update(completed=True, passed=all(r['gain'] >= .002 for r in report['trials']))
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diverse', action='store_true')
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '4'
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    main(diverse=args.diverse)
