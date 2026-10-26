"""No-network smoke check of the installed platform AutoGluon version."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(diverse=False):
    import torch
    from autogluon.tabular import TabularPredictor
    from ml_automl import grouped_bag_labels, hyperparameters
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    out = root / ('.test-temp/automl-smoke-diverse' if diverse else '.test-temp/automl-smoke')
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(42)
    x = pd.DataFrame({'x': rng.normal(size=240), 'category': ['a', 'b', 'c'] * 80})
    y = np.arange(240) % 3
    train = x.copy()
    train['target'] = y
    train['group'] = grouped_bag_labels(x, y)
    predictor = TabularPredictor(label='target', problem_type='multiclass', eval_metric='f1_macro',
        groups='group', path=str(out / 'models'), verbosity=2).fit(
            train, time_limit=120, hyperparameters=hyperparameters(smoke=True, diverse=diverse),
            num_bag_folds=3, num_bag_sets=1, num_stack_levels=0, dynamic_stacking=False,
            num_cpus=2, num_gpus=0, fit_strategy='sequential', memory_limit=12,
            ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'})
    p = predictor.predict_proba(x.iloc[:5])
    result = dict(models=predictor.model_names(), columns=list(map(int, p.columns)),
                  shape=list(p.shape), finite=bool(np.isfinite(p.to_numpy()).all()),
                  feature_columns=list(predictor.features()))
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diverse', action='store_true')
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'
    main(diverse=args.diverse)
