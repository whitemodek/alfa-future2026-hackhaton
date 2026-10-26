"""Optional fixed blend of direct and strictly causal residual forecasts."""
import json
import math
from pathlib import Path
import time

import numpy as np

from ml_lag import XGB_REQUEST, LAG_COLUMNS, _fold_frames, _full_frames


POLICY = 'causal_residual_half_blend_v1'


def combine(base, history, lag, residual):
    base, history, lag = [np.asarray(a, dtype=float).reshape(-1) for a in (base, history, lag)]
    residual = np.asarray(residual, dtype=float).reshape(-1)
    available = np.isfinite(lag)
    if (base.shape != history.shape or base.shape != lag.shape or len(residual) != available.sum()
            or not np.isfinite(base).all() or not np.isfinite(history).all() or not np.isfinite(residual).all()):
        raise ValueError('Finite aligned direct/history/residual predictions are required')
    prediction = base.copy()
    prediction[available] = (.5 * base[available] + .25 * history[available]
                             + .25 * (lag[available] + residual))
    return prediction


def _base(envelope, manifest, started, budget):
    import ml_backend as b
    import ml_portfolio as p
    import ml_lag as lag
    if manifest.get('model') != 'lag':
        raise b.MLRequestError('Residual blend requires a validated causal-lag pipeline')
    problem, ti, vi = p.checked_problem(envelope, manifest)
    if problem['metric'] != 'rmse':
        raise b.MLRequestError('Residual blend currently supports temporal RMSE only')
    lag._check_configuration(problem, manifest)
    history_path, history = p.read_manifest(manifest['history_manifest_path'], problem['root'])
    if b._file_hash(history_path) != manifest['history_manifest_sha256']:
        raise b.MLRequestError('Residual history source changed')
    own, _, _, history_vp, history_tp = lag._history_predictions(envelope, history, started, budget)
    if manifest['fit_scope'] != history['fit_scope']:
        raise b.MLRequestError('Residual base and history scopes differ')
    deadline = started + budget
    _, xv, tx, _, _ = _fold_frames(problem, ti, vi, manifest['entity_columns'], manifest['time_column'], deadline)
    validation = lag._load_model(problem, manifest, validation=True)
    current = lag._load_model(problem, manifest)
    vp = .5 * history_vp + .5 * validation.predict(xv)
    if not math.isclose(b._score(problem, problem['y'].iloc[vi], vp, None), manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Residual base no longer reproduces its score')
    if manifest['fit_scope'] == 'full_train':
        _, tx, _ = _full_frames(problem, manifest['entity_columns'], manifest['time_column'], deadline)
    tp = .5 * history_tp + .5 * current.predict(tx)
    return problem, ti, vi, history_vp, history_tp, vp, tp


def _configuration(manifest):
    return {key: value for key, value in manifest.items()
            if key not in {'residual_config_path', 'residual_config_sha256'}}


def _publish(envelope, base_path, base, model, validation_model, validation_path, started, budget, full):
    import ml_backend as b
    import ml_portfolio as p
    problem, ti, vi, hvp, htp, bvp, btp = _base(envelope, base, started, budget)
    deadline = started + budget
    xt, xv, tx, cats, _ = _fold_frames(problem, ti, vi, base['entity_columns'], base['time_column'], deadline)
    available = np.isfinite(xv[LAG_COLUMNS[0]].to_numpy())
    residual_vp = validation_model.predict(xv.iloc[np.flatnonzero(available)]) if available.any() else np.empty(0)
    vp = combine(bvp, hvp, xv[LAG_COLUMNS[0]], residual_vp)
    if full:
        _, tx, _ = _full_frames(problem, base['entity_columns'], base['time_column'], deadline)
    query_available = np.isfinite(tx[LAG_COLUMNS[0]].to_numpy())
    residual_tp = model.predict(tx.iloc[np.flatnonzero(query_available)]) if query_available.any() else np.empty(0)
    tp = combine(btp, htp, tx[LAG_COLUMNS[0]], residual_tp)
    score = b._score(problem, problem['y'].iloc[vi], vp, None)
    candidate, schema = b._output(problem, tp, None, base['request'])
    model_path = candidate.with_suffix('.joblib')
    model.save_model(model_path)
    if validation_path is None:
        validation_path = model_path
    vector_path, vector_hash = p.save_validation(candidate, vp, None)
    manifest = dict(model='residual', target=problem['target'], features=problem['features'],
        categorical_features=cats, classes=None, request=base['request'], score=score, metric='rmse', direction='minimize',
        validation=problem['validation'], validation_id=b._validation_id(problem, ti, vi),
        input_hashes=b._input_hashes(problem), fit_scope='full_train' if full else 'holdout_train',
        training_rows=len(problem['y']) if full else len(ti), selected_iterations=validation_model.tree_count_,
        training_parameters=validation_model.parameters, model_path=str(model_path), model_sha256=b._file_hash(model_path),
        validation_model_path=str(validation_path), validation_model_sha256=b._file_hash(validation_path),
        validation_path=vector_path, validation_sha256=vector_hash,
        base_manifest_path=str(base_path), base_manifest_sha256=b._file_hash(base_path),
        entity_columns=base['entity_columns'], time_column=base['time_column'],
        residual_request=dict(XGB_REQUEST), residual_weight=.5, residual_policy=POLICY)
    config = candidate.with_suffix('.residual-config.json')
    config.write_text(json.dumps(_configuration(manifest), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    manifest.update(residual_config_path=str(config), residual_config_sha256=b._file_hash(config))
    path = candidate.with_suffix('.json')
    path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = p._response(problem, 'residual', candidate, schema, path, manifest, started,
                         'Fixed half-blend with a strictly past-target residual model; inner-only tree selection.')
    if full:
        result['final'] = True
    return result


def run_residual(envelope, started, budget, threads):
    import ml_backend as b
    import ml_portfolio as p
    from ml_models import fit
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    path, base = p.read_manifest(envelope['request'].get('manifest_path'), root)
    problem, ti, vi, *_ = _base(envelope, base, started, budget)
    time_column = base['time_column']
    periods = problem['train'][time_column]
    last = periods.iloc[ti].max()
    inner_train = np.asarray([i for i in ti if periods.iloc[i] < last], dtype=int)
    inner_valid = np.asarray([i for i in ti if periods.iloc[i] == last], dtype=int)
    if not len(inner_train) or not len(inner_valid):
        raise b.MLRequestError('Residual selection needs at least two development periods')
    deadline = started + budget - min(15, max(2, budget * .08))
    xt, xv, _, cats, _ = _fold_frames(problem, inner_train, inner_valid, base['entity_columns'], time_column, deadline)
    a, v = np.isfinite(xt[LAG_COLUMNS[0]]), np.isfinite(xv[LAG_COLUMNS[0]])
    outer_x, _, _, _, _ = _fold_frames(problem, ti, vi, base['entity_columns'], time_column, deadline)
    available = np.isfinite(outer_x[LAG_COLUMNS[0]].to_numpy())
    if available.sum() < 10:
        raise b.MLRequestError('Insufficient strictly past training pairs for residual fitting')
    if a.sum() < 10 or v.sum() < 10:
        # Short histories cannot supply a separate residual-selection period.
        # A fixed conservative count avoids using the outer validation answers.
        frame = outer_x.iloc[np.flatnonzero(available)]
        labels = problem['y'].iloc[ti].to_numpy()[available] - outer_x[LAG_COLUMNS[0]].to_numpy()[available]
        model = fit('xgboost', frame, labels, frame, labels, cats, dict(XGB_REQUEST, iterations=1),
                    time.monotonic() + max(1, deadline - time.monotonic()) * .2, threads, 'regression', 'rmse')
        model.tree_count_ = 128
    else:
        model = fit('xgboost', xt.iloc[np.flatnonzero(a)],
            problem['y'].iloc[inner_train].to_numpy()[a] - xt[LAG_COLUMNS[0]].to_numpy()[a],
            xv.iloc[np.flatnonzero(v)], problem['y'].iloc[inner_valid].to_numpy()[v] - xv[LAG_COLUMNS[0]].to_numpy()[v],
            cats, dict(XGB_REQUEST), time.monotonic() + max(1, deadline - time.monotonic()) * .45,
            threads, 'regression', 'rmse')
    trained = model.refit(outer_x.iloc[np.flatnonzero(available)],
        problem['y'].iloc[ti].to_numpy()[available] - outer_x[LAG_COLUMNS[0]].to_numpy()[available], deadline, threads)
    return _publish(envelope, path, base, trained, trained, None, started, budget, full=False)


def refit_residual(envelope, manifest, started, budget, threads):
    import ml_backend as b
    import ml_portfolio as p
    from ml_models import ModelAdapter
    problem, _, _ = p.checked_problem(envelope, manifest)
    config_path, config = p.read_manifest(manifest.get('residual_config_path'), problem['root'])
    if (b._file_hash(config_path) != manifest.get('residual_config_sha256') or config != _configuration(manifest)
            or manifest.get('residual_policy') != POLICY or manifest.get('residual_weight') != .5
            or manifest.get('residual_request') != XGB_REQUEST):
        raise b.MLRequestError('Residual configuration changed')
    base_path, base = p.read_manifest(manifest['base_manifest_path'], problem['root'])
    if b._file_hash(base_path) != manifest['base_manifest_sha256']:
        raise b.MLRequestError('Residual base changed')
    validation_path = p.private_path(manifest['validation_model_path'], problem['root'], '.joblib')
    if b._file_hash(validation_path) != manifest['validation_model_sha256']:
        raise b.MLRequestError('Residual validation model changed')
    selected = ModelAdapter.load_model(validation_path)
    if selected.tree_count_ != manifest['selected_iterations'] or selected.parameters != manifest['training_parameters']:
        raise b.MLRequestError('Residual selected parameters changed')
    if base['fit_scope'] != 'full_train':
        allowance = (budget - (time.monotonic() - started)) * .7
        final_base = b.run(dict(envelope, request=dict(action='refit', manifest_path=str(base_path)),
                                time_budget_seconds=allowance))
        if final_base.get('fit_scope') != 'full_train':
            raise b.MLRequestError('Residual base full refit did not finish')
        base_path, base = p.read_manifest(final_base['manifest_path'], problem['root'])
    x, _, _ = _full_frames(problem, base['entity_columns'], base['time_column'], started + budget)
    available = np.isfinite(x[LAG_COLUMNS[0]].to_numpy())
    full = selected.refit(x.iloc[np.flatnonzero(available)],
        problem['y'].to_numpy()[available] - x[LAG_COLUMNS[0]].to_numpy()[available], started + budget - 1, threads)
    result = _publish(envelope, base_path, base, full, selected, validation_path, started, budget, full=True)
    if not math.isclose(result['score'], manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Residual full refit changed its holdout score')
    # Exact original float is required for the parent's final equal-score replacement.
    path, final = p.read_manifest(result['manifest_path'], problem['root'])
    final['score'] = result['score'] = manifest['score']
    config = p.private_path(final['residual_config_path'], problem['root'], '.json')
    config.write_text(json.dumps(_configuration(final), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    final['residual_config_sha256'] = b._file_hash(config)
    path.write_text(json.dumps(final, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return result
