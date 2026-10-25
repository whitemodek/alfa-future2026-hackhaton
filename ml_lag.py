"""Causal entity-lag features complement an independently validated history model.

The XGBoost configuration is fixed: 1800 rounds, patience100, depth4. Each
training row sees targets only from strictly earlier periods. Refit retains
the original holdout model, selected tree count and exact recorded score.
"""
from pathlib import Path
import json
import math
import numbers
import time


LAG_COLUMNS = ('__history_target_lag', '__history_available')
XGB_REQUEST = dict(iterations=1800, early_stopping_rounds=100, depth=4,
                   learning_rate=.04, min_child_weight=5, l2_leaf_reg=10,
                   subsample=.85, colsample_bytree=.9, random_seed=42,
                   encoding='onehot')
POLICY = 'strict_past_entity_lag_missing_collision_fallback_v1'
CONFIG_FIELDS = ('model', 'target', 'features', 'categorical_features', 'classes',
                 'request', 'validation_id', 'score', 'metric', 'direction', 'input_hashes',
                 'fit_scope', 'training_rows', 'selected_iterations',
                 'training_parameters', 'model_path', 'model_sha256',
                 'validation_model_path', 'validation_model_sha256',
                 'validation_path', 'validation_sha256', 'history_manifest_path',
                 'history_manifest_sha256', 'entity_columns', 'time_column',
                 'history_weight', 'lag_weight', 'lag_request', 'lag_feature_names')


def _deadline(deadline):
    import ml_backend as b
    if time.monotonic() >= deadline:
        raise b.MLRequestError('Insufficient time for causal lag features.')


def causal_lags(train, target, query, columns, time_column, deadline=math.inf):
    """Return train/query lags without own-period or future-target leakage.

    Collision detection is causal too: an entity is disabled only after a
    duplicate entity-period has appeared in its available training history.
    Query rows must all occur after training. Missing identity never matches.
    Sorting is O(n log n), rather than one full-data scan per timestamp.
    """
    import numpy as np
    import pandas as pd
    import ml_backend as b
    _deadline(deadline)
    if (not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
            or time_column in columns
            or any(c not in train or c not in query for c in [*columns, time_column])):
        raise b.MLRequestError('Lag features need distinct entity columns and a separate time column.')
    if not len(train) or not len(query):
        raise b.MLRequestError('Lag features require nonempty training and query rows.')
    for frame in (train, query):
        periods = frame[time_column]
        if (not pd.api.types.is_numeric_dtype(periods) or periods.isna().any()
                or not np.isfinite(periods.to_numpy(dtype=float)).all()):
            raise b.MLRequestError('Lag periods must be numeric, finite and nonmissing.')
    periods = train[time_column].tolist()
    query_periods = query[time_column].tolist()
    if min(query_periods) <= max(periods):
        raise b.MLRequestError('Lag query periods must be strictly later than training.')
    values = np.asarray(target, dtype=float).reshape(-1)
    if len(values) != len(train) or not np.isfinite(values).all():
        raise b.MLRequestError('Lag training targets must be finite and aligned.')

    def identity(row):
        for value in row:
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                return None
            if isinstance(value, numbers.Number):
                try:
                    if not math.isfinite(value):
                        return None
                except (TypeError, ValueError, OverflowError):
                    return None
        try:
            hash(row)
        except TypeError:
            return None
        return row

    keys = [identity(row) for row in train[columns].itertuples(index=False, name=None)]
    query_keys = [identity(row) for row in query[columns].itertuples(index=False, name=None)]
    train_lag = np.full(len(train), np.nan)
    query_lag = np.full(len(query), np.nan)
    latest, ambiguous = {}, set()
    ordered = sorted(range(len(train)), key=periods.__getitem__)
    position = 0
    while position < len(ordered):
        _deadline(deadline)
        stop = position + 1
        period = periods[ordered[position]]
        while stop < len(ordered) and periods[ordered[stop]] == period:
            stop += 1
        current = ordered[position:stop]
        # Predict every row before adding any targets from its own period.
        counts = {}
        for i in current:
            key = keys[i]
            if key is not None:
                if key in latest and key not in ambiguous:
                    train_lag[i] = latest[key]
                counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            if count > 1:
                ambiguous.add(key)
                latest.pop(key, None)
        for i in current:
            key = keys[i]
            if key is not None and key not in ambiguous:
                latest[key] = values[i]
        position = stop
    _deadline(deadline)
    for i, key in enumerate(query_keys):
        if key is not None and key in latest:
            query_lag[i] = latest[key]
    return train_lag, query_lag, dict(
        training_coverage=float(np.isfinite(train_lag).mean()),
        query_coverage=float(np.isfinite(query_lag).mean()),
        ambiguous_keys=len(ambiguous), usable_entity_keys=len(latest),
        missing_training_rows=sum(key is None for key in keys))


def _history_predictions(envelope, history, started, budget):
    """Reconstruct both prediction vectors from hashed native models, never CSV."""
    import ml_backend as b
    import ml_history as h
    import ml_portfolio as p
    p._check_time(started, budget, 2)
    problem, ti, vi = p.checked_problem(envelope, history)
    if history.get('model') != 'history':
        raise b.MLRequestError('Lag features require a validated history pipeline.')
    config_path, config = p.read_manifest(history.get('history_config_path'), problem['root'])
    history_fields = ('entity_columns', 'time_column', 'history_weight',
                      'base_manifest_path', 'base_manifest_sha256')
    if (b._file_hash(config_path) != history.get('history_config_sha256')
            or config.get('version') != 1
            or config.get('policy') != 'latest_strictly_past_missing_or_ambiguous_fallback_v1'
            or any(config.get(k) != history.get(k) for k in history_fields)):
        raise b.MLRequestError('Lag history configuration changed.')
    base_path, base = p.read_manifest(history.get('base_manifest_path'), problem['root'])
    if b._file_hash(base_path) != history.get('base_manifest_sha256'):
        raise b.MLRequestError('Lag history base manifest changed.')
    if (history.get('fit_scope') != base.get('fit_scope')
            or history.get('training_rows') != base.get('training_rows')
            or history.get('validation_id') != base.get('validation_id')):
        raise b.MLRequestError('Lag history and its base have different training scopes.')
    own, own_ti, own_vi, vp, tp = h._base_predictions(envelope, base, started, budget)
    if (history.get('input_hashes') != b._input_hashes(own)
            or any(history.get(k) != own[k] for k in ('target', 'features', 'classes', 'metric', 'direction'))
            or list(ti) != list(own_ti) or list(vi) != list(own_vi)):
        raise b.MLRequestError('Lag history problem or validation changed.')
    columns, time_column, weight = (history.get('entity_columns'), history.get('time_column'),
                                     history.get('history_weight'))
    if (problem['classes'] is not None or problem['validation']['strategy'] != 'time'
            or problem['metric'].removeprefix('neg_') not in {'rmse', 'mse', 'mae', 'r2'}
            or time_column != problem['validation'].get('diagnostics', {}).get('time_column')
            or time_column not in problem['train'] or time_column == problem['target']):
        raise b.MLRequestError('Lag features require forward temporal regression.')
    h.entity_columns(problem, dict(entity_columns=columns))
    train, valid = problem['train'].iloc[ti], problem['train'].iloc[vi]
    if (valid[time_column].min() <= train[time_column].max()
            or problem['test'][time_column].min() <= problem['train'][time_column].max()):
        raise b.MLRequestError('Lag validation/test periods must be strictly in the future.')
    vp, _ = h.blend_past(train, problem['y'].iloc[ti], valid, vp, columns, time_column, weight)
    score = b._score(problem, problem['y'].iloc[vi], vp, None)
    if score is None or not math.isclose(score, history['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Lag history no longer reproduces its original score.')
    full = history['fit_scope'] == 'full_train'
    tp, _ = h.blend_past(problem['train'] if full else train,
                         problem['y'] if full else problem['y'].iloc[ti],
                         problem['test'], tp, columns, time_column, weight)
    return problem, ti, vi, vp, tp


def _augment(frame, lag):
    import numpy as np
    import ml_backend as b
    if any(c in frame for c in LAG_COLUMNS):
        raise b.MLRequestError('Input features collide with reserved lag feature names.')
    result = frame.copy()
    result[LAG_COLUMNS[0]] = lag
    result[LAG_COLUMNS[1]] = np.isfinite(lag).astype(np.int8)
    return result


def _fold_frames(problem, ti, vi, columns, time_column, deadline):
    import pandas as pd
    import ml_backend as b
    x, tx, categorical = b._features(problem)
    # The query frame deliberately excludes all other columns, including target.
    raw_columns = [*columns, time_column]
    train = problem['train'].iloc[ti][raw_columns]
    query = pd.concat([problem['train'].iloc[vi][raw_columns], problem['test'][raw_columns]],
                      ignore_index=True)
    tr, qr, diagnostics = causal_lags(train, problem['y'].iloc[ti], query, columns, time_column, deadline)
    return (_augment(x.iloc[ti], tr), _augment(x.iloc[vi], qr[:len(vi)]),
            _augment(tx, qr[len(vi):]), categorical, diagnostics)


def _full_frames(problem, columns, time_column, deadline):
    import ml_backend as b
    x, tx, _ = b._features(problem)
    raw_columns = [*columns, time_column]
    tr, qr, diagnostics = causal_lags(problem['train'][raw_columns], problem['y'],
                                     problem['test'][raw_columns], columns, time_column, deadline)
    return _augment(x, tr), _augment(tx, qr), diagnostics


def _configuration(manifest):
    return dict(version=1, policy=POLICY, **{key: manifest[key] for key in CONFIG_FIELDS})


def _check_configuration(problem, manifest):
    import ml_backend as b
    import ml_portfolio as p
    path, config = p.read_manifest(manifest.get('lag_config_path'), problem['root'])
    if (b._file_hash(path) != manifest.get('lag_config_sha256')
            or any(key not in manifest for key in CONFIG_FIELDS)
            or config != _configuration(manifest)
            or manifest['model'] != 'lag' or manifest['lag_request'] != XGB_REQUEST
            or manifest['lag_weight'] != .5 or manifest['lag_feature_names'] != list(LAG_COLUMNS)
            or manifest['fit_scope'] not in {'holdout_train', 'full_train'}):
        raise b.MLRequestError('Lag configuration changed.')
    validation = p.private_path(manifest['validation_path'], problem['root'], '.npz')
    if b._file_hash(validation) != manifest['validation_sha256']:
        raise b.MLRequestError('Lag validation artifact changed.')


def _load_model(problem, manifest, validation=False):
    import ml_backend as b
    import ml_portfolio as p
    from ml_models import ModelAdapter
    prefix = 'validation_model' if validation else 'model'
    path = p.private_path(manifest[prefix + '_path'], problem['root'], '.joblib')
    if b._file_hash(path) != manifest[prefix + '_sha256']:
        raise b.MLRequestError('Lag model artifact changed.')
    model = ModelAdapter.load_model(path)
    if (model.family != 'xgboost' or model.classes_ is not None
            or model.parameters != manifest['training_parameters']
            or model.tree_count_ != manifest['selected_iterations']
            or model.feature_names_ != [*problem['features'], *LAG_COLUMNS]
            or model.categorical != manifest['categorical_features']):
        raise b.MLRequestError('Lag model parameters differ from the configuration.')
    return model


def _publish(problem, history_path, history, model, validation_model, validation_model_path,
             validation_prediction, prediction, score, categorical, scope, diagnostics, started):
    import ml_backend as b
    import ml_portfolio as p
    candidate, schema = b._output(problem, prediction, None, history['request'])
    model_path = candidate.with_suffix('.lag.joblib')
    model.save_model(model_path)
    if validation_model_path is None:
        validation_model_path = model_path
    validation_path, validation_hash = p.save_validation(candidate, validation_prediction, None)
    manifest = dict(model='lag', target=problem['target'], features=problem['features'],
                    categorical_features=categorical, classes=None, request=history['request'],
                    validation_id=history['validation_id'], validation=problem['validation'],
                    score=score, metric=problem['metric'], direction=problem['direction'],
                    fit_scope=scope, training_rows=len(problem['train']) if scope == 'full_train'
                    else problem['validation']['diagnostics']['training_rows'],
                    selected_iterations=validation_model.tree_count_,
                    training_parameters=validation_model.parameters, input_hashes=b._input_hashes(problem),
                    model_path=str(model_path), model_sha256=b._file_hash(model_path),
                    validation_model_path=str(validation_model_path),
                    validation_model_sha256=b._file_hash(validation_model_path),
                    validation_path=validation_path, validation_sha256=validation_hash,
                    history_manifest_path=str(history_path), history_manifest_sha256=b._file_hash(history_path),
                    entity_columns=history['entity_columns'], time_column=history['time_column'],
                    history_weight=history['history_weight'], lag_weight=.5,
                    lag_request=dict(XGB_REQUEST), lag_feature_names=list(LAG_COLUMNS),
                    lag_diagnostics=diagnostics)
    config_path = candidate.with_suffix('.lag-config.json')
    config_path.write_text(json.dumps(_configuration(manifest), ensure_ascii=False, allow_nan=False,
                                     indent=2), encoding='utf-8')
    manifest.update(lag_config_path=str(config_path), lag_config_sha256=b._file_hash(config_path))
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = p._response(problem, 'lag', candidate, schema, manifest_path, manifest, started,
                         'Fixed50/50 history pipeline plus causal-lag XGBoost; original holdout models retained.')
    result.update(entity_columns=history['entity_columns'], lag_diagnostics=diagnostics)
    if scope == 'full_train':
        result['final'] = True
    return result


def run_lag(envelope, started, budget, threads):
    import numpy as np
    import ml_backend as b
    import ml_portfolio as p
    from ml_models import fit
    p._check_time(started, budget, 2)
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    history_path, history = p.read_manifest(envelope['request'].get('manifest_path'), root)
    problem, ti, vi, history_vp, history_tp = _history_predictions(envelope, history, started, budget)
    deadline = started + budget - min(15, max(2, budget*.08))
    xt, xv, tx, categorical, diagnostics = _fold_frames(
        problem, ti, vi, history['entity_columns'], history['time_column'], deadline)
    p._check_time(started, budget, 2)
    model = fit('xgboost', xt, problem['y'].iloc[ti], xv, problem['y'].iloc[vi],
                categorical, dict(XGB_REQUEST), deadline, threads, 'regression', problem['metric'])
    vp = .5*history_vp + .5*np.asarray(model.predict(xv)).reshape(-1)
    prediction = .5*history_tp + .5*np.asarray(model.predict(tx)).reshape(-1)
    score = b._score(problem, problem['y'].iloc[vi], vp, None)
    if score is None:
        raise b.MLRequestError('Lag pipeline has no finite validation score.')
    p._check_time(started, budget, 1)
    return _publish(problem, history_path, history, model, model, None, vp, prediction,
                     score, categorical, 'holdout_train', diagnostics, started)


def refit_lag(envelope, manifest, started, budget, threads):
    import numpy as np
    import ml_backend as b
    import ml_portfolio as p
    p._check_time(started, budget, 2)
    problem, ti, vi = p.checked_problem(envelope, manifest)
    _check_configuration(problem, manifest)
    history_path, history = p.read_manifest(manifest['history_manifest_path'], problem['root'])
    if b._file_hash(history_path) != manifest['history_manifest_sha256']:
        raise b.MLRequestError('Lag history manifest changed.')
    own, own_ti, own_vi, history_vp, _ = _history_predictions(envelope, history, started, budget)
    if (history['validation_id'] != manifest['validation_id']
            or list(ti) != list(own_ti) or list(vi) != list(own_vi)
            or any(manifest[k] != history[k] for k in ('entity_columns', 'time_column', 'history_weight'))):
        raise b.MLRequestError('Lag history validation or entity configuration changed.')
    deadline = started + budget - min(15, max(2, budget*.08))
    _, xv, _, categorical, _ = _fold_frames(
        own, ti, vi, manifest['entity_columns'], manifest['time_column'], deadline)
    if categorical != manifest['categorical_features']:
        raise b.MLRequestError('Lag categorical feature representation changed.')
    validation_model = _load_model(problem, manifest, validation=True)
    vp = .5*history_vp + .5*np.asarray(validation_model.predict(xv)).reshape(-1)
    score = b._score(problem, problem['y'].iloc[vi], vp, None)
    if score is None or not math.isclose(score, manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Saved lag pipeline does not reproduce its original holdout score.')
    # Verify the current artifact even when only the original holdout model is
    # needed for scoring, so a corrupt full model cannot be silently reused.
    current = _load_model(problem, manifest)
    if history['fit_scope'] != 'full_train':
        p._check_time(started, budget, 3)
        left = deadline - time.monotonic()
        final_history = b.run(dict(envelope, request=dict(action='refit', manifest_path=str(history_path)),
                                    time_budget_seconds=left))
        if final_history.get('fit_scope') != 'full_train' or final_history.get('final') is not True:
            raise b.MLRequestError('Lag history base did not complete its full refit.')
        history_path, full_history = p.read_manifest(final_history['manifest_path'], problem['root'])
        if (full_history['validation_id'] != history['validation_id']
                or full_history['score'] != history['score']
                or any(full_history[k] != history[k] for k in ('entity_columns', 'time_column', 'history_weight'))):
            raise b.MLRequestError('Lag history refit changed the selected pipeline.')
        history = full_history
    _, _, _, full_history_vp, history_tp = _history_predictions(envelope, history, started, budget)
    if not np.allclose(full_history_vp, history_vp, rtol=1e-10, atol=1e-10):
        raise b.MLRequestError('Lag full history changed the original validation predictions.')
    full_x, full_tx, diagnostics = _full_frames(
        problem, manifest['entity_columns'], manifest['time_column'], deadline)
    p._check_time(started, budget, 2)
    model = (current if manifest['fit_scope'] == 'full_train'
             else validation_model.refit(full_x, problem['y'], deadline, threads))
    if model.tree_count_ != manifest['selected_iterations']:
        raise b.MLRequestError('Lag refit did not complete the selected tree count.')
    prediction = .5*history_tp + .5*np.asarray(model.predict(full_tx)).reshape(-1)
    p._check_time(started, budget, 1)
    return _publish(problem, history_path, history, model, validation_model,
                     Path(manifest['validation_model_path']), vp, prediction,
                     manifest['score'], categorical, 'full_train', diagnostics, started)
