"""CPU model adapters with fold-only preprocessing and bounded fitting.

Only load bundles produced by this worker from the private artifact directory:
joblib, like native generated Python, is not an untrusted-input sandbox.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

MAX_DENSE_CELLS = 50_000_000
MAX_LINEAR_COLUMNS = 200_000
MAX_TEXT_CHARACTERS = 100_000_000


def _linear_text_columns(frame, categorical):
    """Identify natural-language columns from this training fold only."""
    result = []
    for name in categorical:
        values = frame[name].fillna('').astype(str)
        unique = values.nunique()
        if (unique >= 20 and unique / max(1, len(values)) >= .2
                and values.str.len().mean() >= 25
                and values.str.split().str.len().mean() >= 4):
            result.append(name)
            if len(result) == 4:
                break
    return result


def _number(request, name, default, low, high, integer=False):
    value = request.get(name, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high
            or (integer and int(value) != value)):
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value) if integer else float(value)


def _check_deadline(deadline):
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('deadline must be finite')
    if time.monotonic() >= deadline:
        raise TimeoutError('Model deadline reached before training')


class ModelAdapter:
    def __init__(self, family, columns, categorical, vocabulary, parameters,
                 problem_type, metric, classes, threads):
        self.family = family
        self.feature_names_ = list(columns)
        self.categorical = list(categorical)
        self.vocabulary = vocabulary
        self.parameters = parameters
        self.problem_type = problem_type
        self.metric = metric
        self.classes_ = classes
        self.threads = threads
        self.tree_count_ = 0
        self.model = None
        self.encoder = None
        self.imputer = None

    def _transform(self, frame):
        import numpy as np
        import pandas as pd
        if not set(self.feature_names_).issubset(frame.columns):
            raise ValueError('Prediction features differ from training')
        if self.parameters.get('encoding') == 'linear_sparse':
            from scipy import sparse
            numeric = [name for name in self.feature_names_ if name not in self.categorical]
            text_columns = getattr(self, 'text_columns', [])
            categorical = [name for name in self.categorical if name not in text_columns]
            indicators = len(self.imputer.indicator_.features_) if self.imputer is not None else 0
            if len(frame) * (len(numeric) + indicators + len(categorical)) > MAX_DENSE_CELLS:
                raise ValueError('Linear feature expansion exceeds the nonzero-entry limit')
            blocks = []
            if numeric:
                number = frame[numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
                number[~np.isfinite(number)] = np.nan
                number = sparse.csr_matrix(self.imputer.transform(number))
                blocks.append(self.scaler.transform(number))
            if self.encoder is not None:
                blocks.append(self.encoder.transform(frame[categorical].fillna('__MISSING__').astype(str)))
            text_characters = 0
            for name, vectorizer in getattr(self, 'text_vectorizers', {}).items():
                values = frame[name].fillna('').astype(str)
                text_characters += int(values.str.len().sum())
                if text_characters > MAX_TEXT_CHARACTERS:
                    raise ValueError('Linear text input exceeds the character limit')
                blocks.append(vectorizer.transform(values))
                if sum(block.nnz for block in blocks) > MAX_DENSE_CELLS:
                    raise ValueError('Linear feature expansion exceeds the nonzero-entry limit')
            return sparse.hstack(blocks, format='csr') if blocks else sparse.csr_matrix((len(frame), 1))
        if self.parameters.get('encoding') == 'onehot':
            numeric = [name for name in self.feature_names_ if name not in self.categorical]
            imputer = getattr(self, 'imputer', None)
            indicators = len(imputer.indicator_.features_) if imputer is not None else 0
            width = len(numeric) + indicators + sum(len(values) for values in self.vocabulary.values())
            if len(frame) * width > MAX_DENSE_CELLS:
                raise ValueError('Dense one-hot expansion exceeds the 50-million-cell limit')
            number = frame[numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
            number[~np.isfinite(number)] = np.nan
            if imputer is not None:
                number = imputer.transform(number)
            if self.encoder is None:
                return number
            category = self.encoder.transform(frame[self.categorical].fillna('__MISSING__').astype(str))
            # Dense storage is intentional: sparse XGBoost interprets omitted zero
            # entries as missing, which changes zero-valued numeric features.
            return np.concatenate([number, category], axis=1)
        result = frame[self.feature_names_].copy()
        for name in self.feature_names_:
            if name in self.categorical:
                values = result[name].fillna('__MISSING__').astype(str)
                result[name] = pd.Categorical(values, categories=self.vocabulary[name])
            else:
                result[name] = pd.to_numeric(result[name], errors='coerce').astype(float).replace(
                    [np.inf, -np.inf], np.nan)
        # Both native engines restrict characters in feature names.
        result.columns = [f'f{i}' for i in range(len(result.columns))]
        return result

    def _raw_predict(self, frame):
        transformed = self._transform(frame)
        if self.family == 'linear':
            return (self.model.predict(transformed) if self.classes_ is None
                    else self.model.predict_proba(transformed))
        if self.family == 'extra_trees':
            return self.model.predict_proba(transformed)
        if self.family == 'lightgbm':
            return self.model.predict(transformed, num_iteration=self.tree_count_,
                                      num_threads=self.threads)
        import xgboost as xgb
        data = xgb.DMatrix(transformed, enable_categorical=True, nthread=self.threads)
        return self.model.predict(data)

    def predict_proba(self, frame):
        import numpy as np
        if self.classes_ is None:
            raise ValueError('Regression model has no class probabilities')
        return _probability(self._raw_predict(frame), len(self.classes_))

    def predict(self, frame):
        import numpy as np
        if self.classes_ is None:
            return np.asarray(self._raw_predict(frame), dtype=float).reshape(-1)
        return np.asarray(self.classes_)[self.predict_proba(frame).argmax(axis=1)]

    def save_model(self, path):
        import joblib
        joblib.dump(self, Path(path))

    @classmethod
    def load_model(cls, path):
        import joblib
        model = joblib.load(Path(path))
        if not isinstance(model, cls) or model.family not in {'lightgbm', 'xgboost', 'extra_trees', 'linear'}:
            raise ValueError('Invalid model adapter bundle')
        return model

    def refit(self, x, y, deadline, threads):
        """Return a new full-data model; never mutate the validated adapter."""
        _check_deadline(deadline)
        threads = _threads(threads)
        adapter = _adapter(self.family, x, y, self.categorical, dict(self.parameters),
                           self.problem_type, self.metric, threads)
        adapter._train(x, y, None, None, self.tree_count_, deadline, select=False)
        if adapter.tree_count_ != self.tree_count_:
            raise TimeoutError('Full-data refit did not complete the selected tree count')
        return adapter

    def _train(self, x, y, valid_x, valid_y, rounds, deadline, select=True):
        import numpy as np
        train_x = self._transform(x)
        valid = self._transform(valid_x) if valid_x is not None else None
        weights = _sample_weights(y, self.parameters['weighting']) if self.classes_ is not None else None
        parameters = dict(self.parameters['native'])
        if self.family == 'linear':
            from sklearn.linear_model import LogisticRegression, Ridge
            from threadpoolctl import threadpool_limits
            _check_deadline(deadline)
            model = Ridge(**parameters) if self.classes_ is None else LogisticRegression(**parameters)
            with threadpool_limits(limits=self.threads):
                model.fit(train_x, y, sample_weight=weights)
            _check_deadline(deadline)
            self.model, self.tree_count_ = model, 1
            return
        if self.family == 'extra_trees':
            self._train_extra_trees(train_x, y, weights, rounds, deadline)
            return
        history = {}
        higher = _higher(self.metric)

        def primary(predictions, data):
            value = _score(self.metric, data.get_label(), predictions, self.classes_)
            return 'primary', value if value is not None else float('nan'), higher

        _check_deadline(deadline)
        if self.family == 'lightgbm':
            import lightgbm as lgb
            parameters['num_threads'] = self.threads
            train = lgb.Dataset(train_x, label=y, weight=weights,
                                categorical_feature=([f'f{self.feature_names_.index(n)}' for n in self.categorical]
                                                     if self.parameters.get('encoding') != 'onehot' else []))
            valid_set = lgb.Dataset(valid, label=valid_y, reference=train) if valid is not None else None

            def timeout(env):
                if time.monotonic() >= deadline:
                    raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
            timeout.order = 50
            callbacks = [lgb.record_evaluation(history), timeout]
            if select:
                callbacks.append(lgb.early_stopping(self.parameters['patience'],
                                                    first_metric_only=True, verbose=False))
            booster = lgb.train(parameters, train, num_boost_round=rounds,
                                valid_sets=[valid_set] if valid_set is not None else None,
                                valid_names=['valid'] if valid_set is not None else None,
                                feval=primary if select else None,
                                callbacks=callbacks, keep_training_booster=True)
            trained = booster.current_iteration()
            chosen = _select(history, trained, higher) if select else trained
            self.model = lgb.Booster(model_str=booster.model_to_string(num_iteration=chosen))
        else:
            import xgboost as xgb
            parameters['nthread'] = self.threads
            train = xgb.DMatrix(train_x, label=y, weight=weights, enable_categorical=True,
                                nthread=self.threads)
            valid_set = xgb.DMatrix(valid, label=valid_y, enable_categorical=True,
                                    nthread=self.threads) if valid is not None else None

            class Timeout(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    return time.monotonic() >= deadline

            callbacks = [Timeout()]
            if select:
                callbacks.append(xgb.callback.EarlyStopping(
                    rounds=self.parameters['patience'], metric_name=parameters['eval_metric'],
                    data_name='valid', maximize=False, save_best=False))
            _check_deadline(deadline)
            booster = xgb.train(parameters, train, num_boost_round=rounds,
                                evals=[(valid_set, 'valid')] if valid_set is not None else [],
                                custom_metric=(lambda pred, data: primary(pred, data)[:2]) if select else None,
                                evals_result=history, callbacks=callbacks, verbose_eval=False)
            trained = booster.num_boosted_rounds()
            chosen = _select(history, trained, higher) if select else trained
            self.model = booster[:chosen]
        if chosen < 1:
            raise TimeoutError('No complete model iteration was trained')
        self.tree_count_ = int(chosen)

    def _train_extra_trees(self, train_x, y, weights, rounds, deadline):
        """Stop between complete batches; refit verifies the exact selected count."""
        from sklearn.ensemble import ExtraTreesClassifier
        _check_deadline(deadline)
        model = ExtraTreesClassifier(**self.parameters['native'], n_estimators=1,
                                     warm_start=True, n_jobs=self.threads)
        completed = 0
        seconds_per_tree = None
        while completed < rounds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            batch = min(10 if completed else 1, rounds - completed)
            if seconds_per_tree is not None:
                # Account for fit overhead and variation in individual tree size.
                affordable = int(remaining / max(seconds_per_tree * 1.5, .001))
                if affordable < 1:
                    break
                batch = min(batch, affordable)
            started = time.monotonic()
            model.set_params(n_estimators=completed + batch)
            model.fit(train_x, y, sample_weight=weights)
            elapsed = time.monotonic() - started
            seconds_per_tree = max(elapsed / batch, seconds_per_tree or 0)
            completed = len(model.estimators_)
        if not completed:
            raise TimeoutError('No complete ExtraTrees batch was trained')
        self.model = model
        self.tree_count_ = completed


def _probability(raw, count):
    import numpy as np
    raw = np.asarray(raw, dtype=float)
    return np.column_stack([1 - raw, raw]) if count == 2 and raw.ndim == 1 else raw.reshape(-1, count)


def _score(metric, y, raw, classes):
    import numpy as np
    from ml_backend import _score as score
    probability = _probability(raw, len(classes)) if classes is not None else None
    prediction = probability.argmax(axis=1) if probability is not None else np.asarray(raw)
    return score({'metric': metric, 'classes': classes}, y, prediction, probability)


def _higher(metric):
    from ml_backend import metric_direction
    return metric_direction(metric) == 'maximize'


def _select(history, trained, higher):
    import numpy as np
    curve = np.asarray(history.get('valid', {}).get('primary', [])[:trained], dtype=float)
    valid = np.flatnonzero(np.isfinite(curve))
    if not len(valid):
        return int(trained)
    best = valid[np.argmax(curve[valid]) if higher else np.argmin(curve[valid])]
    return int(best) + 1


def _sample_weights(y, mode):
    import numpy as np
    if mode == 'None':
        return None
    counts = np.bincount(np.asarray(y, dtype=int))
    weights = counts.max() / counts
    if mode == 'SqrtBalanced':
        weights = np.sqrt(weights)
    rows = weights[np.asarray(y, dtype=int)]
    # Keep regularization and minimum-child thresholds comparable across
    # weighting schemes: only relative class weights should change.
    return rows / rows.mean()


def _threads(threads):
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError('threads must be an integer between 1 and 8')
    return threads


def _adapter(family, x, y, categorical, parameters, problem_type, metric, threads):
    import numpy as np
    text_columns = (_linear_text_columns(x, categorical)
                    if parameters.get('encoding') == 'linear_sparse' else [])
    category_columns = [name for name in categorical if name not in text_columns]
    vocabulary = {name: list(dict.fromkeys(x[name].fillna('__MISSING__').astype(str)))
                  for name in category_columns}
    classes = sorted(np.unique(y).tolist()) if problem_type == 'classification' else None
    if classes is not None and (len(classes) < 2 or classes != list(range(len(classes)))):
        raise ValueError('Classification labels must be contiguous encoded integers starting at zero')
    adapter = ModelAdapter(family, list(x.columns), categorical, vocabulary, parameters,
                           problem_type, metric, classes, threads)
    if parameters.get('encoding') == 'linear_sparse':
        from scipy import sparse
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.feature_extraction.text import TfidfVectorizer
        numeric = [name for name in x.columns if name not in categorical]
        width = len(numeric) + sum(map(len, vocabulary.values()))
        if width > MAX_LINEAR_COLUMNS or len(x) * (2 * len(numeric) + len(category_columns)) > MAX_DENSE_CELLS:
            raise ValueError('Linear feature expansion exceeds the sparse resource limit')
        adapter.scaler = None
        if numeric:
            import pandas as pd
            number = x[numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
            number[~np.isfinite(number)] = np.nan
            adapter.imputer = SimpleImputer(strategy='median', add_indicator=True,
                                             keep_empty_features=True)
            number = sparse.csr_matrix(adapter.imputer.fit_transform(number))
            width += len(adapter.imputer.indicator_.features_)
            if width > MAX_LINEAR_COLUMNS:
                raise ValueError('Linear feature expansion exceeds the sparse resource limit')
            adapter.scaler = StandardScaler(with_mean=False).fit(number)
        if category_columns:
            adapter.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
            adapter.encoder.fit(x[category_columns].fillna('__MISSING__').astype(str))
        adapter.text_columns, adapter.text_vectorizers = text_columns, {}
        text_entries, text_characters = 0, 0
        for name in text_columns:
            values = x[name].fillna('').astype(str)
            text_characters += int(values.str.len().sum())
            if text_characters > MAX_TEXT_CHARACTERS:
                raise ValueError('Linear text input exceeds the character limit')
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                         max_features=20_000, dtype=np.float64)
            try:
                encoded = vectorizer.fit_transform(values)
            except ValueError as exc:
                if not any(fragment in str(exc) for fragment in (
                        'empty vocabulary', 'After pruning, no terms remain',
                        'max_df corresponds to < documents than min_df')):
                    raise
                # No recurring words supply no learned lexical features.
                continue
            width += encoded.shape[1]
            text_entries += encoded.nnz
            if width > MAX_LINEAR_COLUMNS or text_entries > MAX_DENSE_CELLS:
                raise ValueError('Linear feature expansion exceeds the sparse resource limit')
            adapter.text_vectorizers[name] = vectorizer
        return adapter
    if parameters.get('encoding') == 'onehot':
        width = len(x.columns) - len(categorical) + sum(map(len, vocabulary.values()))
        if len(x) * width > MAX_DENSE_CELLS:
            raise ValueError('Dense one-hot expansion exceeds the 50-million-cell limit')
        if categorical:
            from sklearn.preprocessing import OneHotEncoder
            adapter.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            adapter.encoder.fit(x[categorical].fillna('__MISSING__').astype(str))
        if family == 'extra_trees':
            from sklearn.impute import SimpleImputer
            numeric = [name for name in x.columns if name not in categorical]
            if numeric:
                import pandas as pd
                number = x[numeric].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
                number[~np.isfinite(number)] = np.nan
                indicators = int(np.isnan(number).any(axis=0).sum())
                if len(x) * (width + indicators) > MAX_DENSE_CELLS:
                    raise ValueError('Dense one-hot expansion exceeds the 50-million-cell limit')
                adapter.imputer = SimpleImputer(strategy='median', add_indicator=True,
                                                 keep_empty_features=True).fit(number)
    return adapter


def fit(family, x_train, y_train, x_valid, y_valid, categorical, request,
        deadline, threads, problem_type, metric):
    """Fit on one pinned fold; custom primary scoring never controls patience."""
    _check_deadline(deadline)
    if family not in {'lightgbm', 'xgboost', 'extra_trees', 'linear'}:
        raise ValueError('family must be lightgbm, xgboost, extra_trees or linear')
    if problem_type not in {'classification', 'regression'}:
        raise ValueError('Unknown problem type')
    if not isinstance(request, dict) or any(name not in x_train for name in categorical):
        raise ValueError('Invalid request or categorical columns')
    request = dict(request)
    for alias, canonical in (('max_depth', 'depth'), ('reg_lambda', 'l2_leaf_reg')):
        if alias in request:
            if canonical in request and request[alias] != request[canonical]:
                raise ValueError(f'Conflicting {alias} and {canonical}')
            request[canonical] = request[alias]
    threads = _threads(threads)
    regression = problem_type == 'regression'
    if family == 'linear':
        weighting = request.get('auto_class_weights', 'SqrtBalanced' if metric.removeprefix('neg_') in {
            'macro_f1', 'balanced_accuracy'} else 'None')
        if weighting not in {'None', 'Balanced', 'SqrtBalanced'}:
            raise ValueError('Invalid auto_class_weights')
        native = (dict(alpha=1., solver='lsqr', tol=1e-6) if regression else
                  dict(C=1., solver='lbfgs', max_iter=200, tol=1e-4, random_state=42))
        parameters = dict(native=native, weighting=weighting, encoding='linear_sparse')
        adapter = _adapter(family, x_train, y_train, categorical, parameters,
                           problem_type, metric, threads)
        adapter._train(x_train, y_train, x_valid, y_valid, 1, deadline)
        return adapter
    if family == 'extra_trees':
        if regression:
            raise ValueError('extra_trees currently supports classification only')
        if request.get('encoding', 'onehot') != 'onehot':
            raise ValueError('extra_trees requires onehot encoding')
        rounds = _number(request, 'iterations', 100, 1, 1000, True)
        weighting = request.get('auto_class_weights', 'SqrtBalanced' if metric.removeprefix('neg_') in {
            'macro_f1', 'balanced_accuracy'} else 'None')
        if weighting not in {'None', 'Balanced', 'SqrtBalanced'}:
            raise ValueError('Invalid auto_class_weights')
        native = dict(min_samples_leaf=_number(request, 'min_samples_leaf', 3, 1, 1000, True),
                      max_features=_number(request, 'max_features', 1., .05, 1.),
                      random_state=_number(request, 'random_seed', 42, 0, 2 ** 31 - 1, True))
        parameters = dict(native=native, weighting=weighting, encoding='onehot')
        adapter = _adapter(family, x_train, y_train, categorical, parameters,
                           problem_type, metric, threads)
        adapter._train(x_train, y_train, x_valid, y_valid, rounds, deadline)
        return adapter
    xgb_regression = family == 'xgboost' and regression
    encoding = request.get('encoding', 'onehot' if xgb_regression else 'native')
    if encoding not in {'native', 'onehot'}:
        raise ValueError('encoding must be native or onehot')
    rounds = _number(request, 'iterations', 1200, 1, 4000, True)
    depth = _number(request, 'depth', -1 if family == 'lightgbm' else (4 if xgb_regression else 6),
                    -1 if family == 'lightgbm' else 2, 12, True)
    if depth in (0, 1):
        raise ValueError('depth must be -1 for LightGBM or an integer between 2 and 12')
    rate = _number(request, 'learning_rate', .04 if xgb_regression else .05, .001, 1)
    seed = _number(request, 'random_seed', 42, 0, 2 ** 31 - 1, True)
    patience = _number(request, 'early_stopping_rounds', 60, 1, 300, True)
    weighting = request.get('auto_class_weights', 'SqrtBalanced' if metric.removeprefix('neg_') in {
        'macro_f1', 'balanced_accuracy'} else 'None')
    if weighting not in {'None', 'Balanced', 'SqrtBalanced'}:
        raise ValueError('Invalid auto_class_weights')
    mae = metric.removeprefix('neg_') == 'mae'
    count = len(set(y_train)) if not regression else 0
    subsample = _number(request, 'subsample', .85 if xgb_regression else (.9 if not regression else 1), .3, 1)
    colsample = _number(request, 'colsample_bytree', .9 if xgb_regression or not regression else 1, .3, 1)
    regularization = _number(request, 'l2_leaf_reg', 10 if xgb_regression else (2 if family == 'lightgbm' and not regression else 3), 0, 1000)
    if family == 'lightgbm':
        native = dict(objective=('regression_l1' if mae else 'regression') if regression else (
            'binary' if count == 2 else 'multiclass'),
            metric=('l1' if mae else 'l2') if regression else ('binary_logloss' if count == 2 else 'multi_logloss'),
            learning_rate=rate, max_depth=depth,
            num_leaves=_number(request, 'num_leaves', (63 if count > 2 else 31) if depth == -1 else min(31, 2 ** depth), 2, 512, True),
            min_data_in_leaf=_number(request, 'min_child_samples', 30 if not regression else 20, 2, 1000, True),
            lambda_l2=regularization, bagging_fraction=subsample,
            bagging_freq=1 if subsample < 1 else 0, feature_fraction=colsample,
            seed=seed, num_threads=threads, verbosity=-1, deterministic=True, force_col_wise=True)
    else:
        native = dict(objective=('reg:absoluteerror' if mae else 'reg:squarederror') if regression else (
            'binary:logistic' if count == 2 else 'multi:softprob'),
            eval_metric=('mae' if mae else 'rmse') if regression else ('logloss' if count == 2 else 'mlogloss'),
            eta=rate, max_depth=depth,
            min_child_weight=_number(request, 'min_child_weight', 5 if xgb_regression else 3, 0, 1000),
            reg_lambda=regularization, subsample=subsample, colsample_bytree=colsample,
            seed=seed, nthread=threads, tree_method='hist', device='cpu',
            max_cat_to_onehot=4, max_cat_threshold=64)
    if count > 2:
        native['num_class'] = count
    parameters = dict(native=native, weighting=weighting, patience=patience, encoding=encoding)
    adapter = _adapter(family, x_train, y_train, categorical, parameters,
                       problem_type, metric, threads)
    adapter._train(x_train, y_train, x_valid, y_valid, rounds, deadline)
    return adapter
