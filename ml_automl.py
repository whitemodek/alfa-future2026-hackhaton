"""Optional AutoGluon models with class weights computed INSIDE each fit fold.

AutoGluon/Torch are supplied by the platform requirements. Import this module
only for the optional AutoML path; existing native startup does not import it.
"""
import numpy as np

from autogluon.tabular.models import LGBModel, CatBoostModel, XGBoostModel, RFModel, XTModel, TabularNeuralNetTorchModel


class _FoldSqrtWeights:
    def _fit(self, X, y, sample_weight=None, **kwargs):
        if self.problem_type in {'binary', 'multiclass'}:
            from ml_models import _sample_weights
            values = np.asarray(y)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError('Fold class weights require encoded integer training labels')
            weights = _sample_weights(values, 'SqrtBalanced')
            sample_weight = weights if sample_weight is None else weights * np.asarray(sample_weight)
            sample_weight = sample_weight / np.mean(sample_weight)
        return super()._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)


class SqrtLGB(_FoldSqrtWeights, LGBModel):
    pass


class SqrtCB(_FoldSqrtWeights, CatBoostModel):
    pass


class SqrtXGB(_FoldSqrtWeights, XGBoostModel):
    pass


class SqrtRF(_FoldSqrtWeights, RFModel):
    pass


class SqrtXT(_FoldSqrtWeights, XTModel):
    pass


class SqrtNN(TabularNeuralNetTorchModel):
    def _fit(self, X, y, **kwargs):
        values = np.asarray(y)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Weighted neural training requires encoded integer fold targets')
        counts = np.bincount(values).astype(float)
        if not len(counts) or (counts <= 0).any():
            raise ValueError('Each neural training fold must preserve every class')
        self._sqrt_class_weights = np.sqrt(counts.max() / counts)
        return super()._fit(X=X, y=y, **kwargs)

    def _get_default_loss_function(self):
        import torch
        if self.problem_type in {'binary', 'multiclass'}:
            return torch.nn.CrossEntropyLoss(weight=torch.tensor(self._sqrt_class_weights,
                dtype=torch.float32, device=self.device))
        return super()._get_default_loss_function()


def grouped_bag_labels(x, y, folds=3):
    """Return small fold IDs, never one AutoGluon group per unique entity."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    assigned = np.full(len(x), -1, dtype=int)
    expected = set(y)
    for fold, (train, valid) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(x, y, groups)):
        if set(groups[train]) & set(groups[valid]) or set(y[train]) != expected:
            raise ValueError('Cannot preserve classes in group-disjoint AutoML folds')
        assigned[valid] = fold
    if (assigned < 0).any() or set(assigned) != set(range(folds)):
        raise ValueError('Incomplete AutoML fold assignments')
    return assigned


def hyperparameters(smoke=False, diverse=False):
    rounds = 10 if smoke else 1200
    trees = 10 if smoke else 100
    result = {
        SqrtLGB: dict(num_boost_round=rounds, num_leaves=63, learning_rate=.05,
                     lambda_l2=3, feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
                     ag_args=dict(name='SqrtLGB', priority=100)),
        SqrtCB: dict(iterations=rounds, depth=6, learning_rate=.05,
                    ag_args=dict(name='SqrtCB', priority=90)),
        SqrtXGB: dict(n_estimators=rounds, max_depth=6, learning_rate=.05,
                     min_child_weight=3, reg_lambda=3, subsample=.9, colsample_bytree=.9,
                     ag_args=dict(name='SqrtXGB', priority=80)),
        'NN_TORCH': dict(num_epochs=2 if smoke else 200, ag_args=dict(priority=70)),
        SqrtRF: dict(n_estimators=trees, max_leaf_nodes=10000, min_samples_leaf=3,
                     max_features='sqrt', ag_args=dict(name='SqrtRF', priority=50)),
        SqrtXT: dict(n_estimators=trees, max_leaf_nodes=10000, min_samples_leaf=3,
                     max_features=1., ag_args=dict(name='SqrtXT', priority=40)),
    }
    if diverse:
        # Fixed per-child ceilings prevent one slow custom-metric model from
        # exhausting the complete L1 allocation before other families run.
        result[SqrtLGB]['ag_args']['priority'] = 100
        result[SqrtXGB]['ag_args']['priority'] = 95
        result[SqrtRF]['ag_args']['priority'] = 90
        result[SqrtXT]['ag_args']['priority'] = 85
        result[SqrtCB]['iterations'] = 10 if smoke else 600
        result[SqrtCB]['ag_args']['priority'] = 50
        result[SqrtCB]['ag_args_fit'] = dict(max_time_limit=45)
        result.pop('NN_TORCH')
        result[SqrtNN] = dict(num_epochs=2 if smoke else 100,
            ag_args=dict(name='SqrtNN', priority=80), ag_args_fit=dict(max_time_limit=120))
    return result
