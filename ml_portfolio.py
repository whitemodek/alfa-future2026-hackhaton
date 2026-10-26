"""Native model comparisons and artifact contracts shared with the agent worker."""
from pathlib import Path
import json
import math
import time


def _check_time(started, budget, reserve=0):
    import ml_backend as b
    if time.monotonic() + reserve >= started + budget:
        raise b.MLRequestError('Insufficient time for the remaining model operation.')


def private_path(value, root, suffix=None):
    import ml_backend as b
    root = Path(root).resolve()
    directory = root / '.ml-artifacts'
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if (directory.is_symlink() or not directory.resolve().is_relative_to(root)
            or path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())
            or (suffix and path.suffix != suffix)):
        raise b.MLRequestError('Model artifacts must be regular files inside .ml-artifacts.')
    return path


def read_manifest(value, root):
    import ml_backend as b
    path = private_path(value, root, '.json')
    if path.stat().st_size > 64000:
        raise b.MLRequestError('Model manifest exceeds 64 KB.')
    result = json.loads(path.read_text(encoding='utf-8'),
                        parse_constant=lambda v: (_ for _ in ()).throw(b.MLRequestError('Nonfinite manifest')))
    if not isinstance(result, dict):
        raise b.MLRequestError('Model manifest must be an object.')
    return path, result


def finite_score(result):
    value = result.get('score')
    return type(value) in (int, float) and math.isfinite(value)


def improves(result, best):
    return (result.get('status') == 'ok' and result.get('candidate_path')
            and finite_score(result) and (not finite_score(best) or
            (all(result.get(k) == best.get(k) for k in ('metric', 'direction', 'validation_id'))
             and (result['score'] > best['score'] if best['direction'] == 'maximize'
                  else result['score'] < best['score']))))


def checkpoint(envelope, best):
    """Persist only a complete worker result before starting another trial."""
    callback = envelope.get('_checkpoint_callback')
    if callable(callback) and best.get('status') == 'ok' and best.get('candidate_path'):
        callback(best)


def _response(problem, model, candidate, schema, manifest_path, manifest, started, notes=''):
    result = dict(version=1, status='ok', candidate_path=str(candidate), schema=schema,
                  model=model, score=manifest['score'], metric=problem['metric'],
                  direction=problem['direction'], validation_id=manifest['validation_id'],
                  validation=problem['validation'], manifest_path=str(manifest_path),
                  fit_scope=manifest['fit_scope'], training_rows=manifest['training_rows'],
                  selected_iterations=manifest.get('selected_iterations'), notes=notes,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    return result


def save_validation(path, prediction, probability):
    import numpy as np
    import ml_backend as b
    path = Path(path).with_suffix('.npz')
    np.savez_compressed(path, prediction=prediction,
                        probability=probability if probability is not None else np.empty((0, 0)))
    return str(path), b._file_hash(path)


def run_single(envelope, started, budget, threads):
    import numpy as np
    import ml_backend as b
    from ml_models import fit
    request = envelope['request']
    family = request['model']
    problem = b.load_problem(envelope)
    train_i, valid_i = b._split(problem, request)
    if train_i is None or (problem['classes'] is not None and len(problem['classes']) < 2):
        return b.run(dict(envelope, request=dict(request, action='baseline')))
    x, test_x, categorical = b._features(problem)
    margin = min(15, max(2, budget * .08))
    available = budget - (time.monotonic() - started) - margin
    if available < 1:
        raise b.MLRequestError('Not enough time to train an alternative model.')
    fit_started = time.monotonic()
    share = .4 if request.get('refit', True) else 1.0
    model = fit(family, x.iloc[train_i], problem['y'].iloc[train_i],
                x.iloc[valid_i], problem['y'].iloc[valid_i], categorical, request,
                time.monotonic() + available * share, threads,
                problem['problem_type'], problem['metric'])
    fit_seconds = time.monotonic() - fit_started
    count = len(problem['classes']) if problem['classes'] is not None else None
    probability = b._aligned_proba(model, x.iloc[valid_i], count) if count else None
    prediction = np.asarray(model.predict(x.iloc[valid_i])).reshape(-1)
    score = b._score(problem, problem['y'].iloc[valid_i], prediction, probability)
    if score is None:
        raise b.MLRequestError('Alternative model has no valid holdout score.')
    test_probability = b._aligned_proba(model, test_x, count) if count else None
    candidate, schema = b._output(problem, model.predict(test_x), test_probability, request)
    model_path = candidate.with_suffix('.joblib')
    model.save_model(model_path)
    validation_path, validation_hash = save_validation(candidate, prediction, probability)
    manifest = dict(model=family, target=problem['target'], features=problem['features'],
                    categorical_features=categorical, classes=problem['classes'], request=request,
                    validation_id=b._validation_id(problem, train_i, valid_i),
                    validation=problem['validation'], score=score, metric=problem['metric'],
                    direction=problem['direction'], fit_scope='holdout_train',
                    training_rows=len(train_i), selected_iterations=model.tree_count_,
                    training_parameters=model.parameters, input_hashes=b._input_hashes(problem),
                    model_path=str(model_path), model_sha256=b._file_hash(model_path),
                    validation_model_path=str(model_path), validation_model_sha256=b._file_hash(model_path),
                    validation_path=validation_path, validation_sha256=validation_hash)
    notes = 'Validated model retained before optional full-data refit.'
    left = budget - (time.monotonic() - started) - margin
    if request.get('refit', True) and left > max(1, fit_seconds * len(x) / len(train_i) * 1.15):
        try:
            full = model.refit(x, problem['y'], time.monotonic() + left, threads)
            p = b._aligned_proba(full, test_x, count) if count else None
            full_candidate, full_schema = b._output(problem, full.predict(test_x), p, request)
            full_path = full_candidate.with_suffix('.joblib')
            full.save_model(full_path)
            model, candidate, schema, model_path = full, full_candidate, full_schema, full_path
            manifest.update(fit_scope='full_train', training_rows=len(x),
                            model_path=str(model_path), model_sha256=b._file_hash(model_path))
            notes = 'Full-data refit completed; score remains the original holdout score.'
        except Exception as exc:
            notes += ' Refit failed: ' + type(exc).__name__
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = _response(problem, family, candidate, schema, manifest_path, manifest, started, notes)
    if request.get('action') == 'finalize' and manifest['fit_scope'] == 'full_train':
        result['final'] = True
    return result


def checked_problem(envelope, manifest):
    import ml_backend as b
    original = manifest.get('request')
    if not isinstance(original, dict) or not finite_score(manifest):
        raise b.MLRequestError('Invalid model manifest request or score.')
    problem = b.load_problem(dict(envelope, request=original))
    train_i, valid_i = b._split(problem, original)
    if (train_i is None or manifest.get('input_hashes') != b._input_hashes(problem)
            or manifest.get('validation_id') != b._validation_id(problem, train_i, valid_i)
            or any(manifest.get(k) != problem[k] for k in ('target', 'features', 'classes', 'metric', 'direction'))):
        raise b.MLRequestError('Model data or validation changed.')
    return problem, train_i, valid_i


def refit_single(envelope, manifest, started, budget, threads):
    import numpy as np
    import ml_backend as b
    from ml_models import ModelAdapter
    problem, train_i, valid_i = checked_problem(envelope, manifest)
    path = private_path(manifest.get('model_path'), problem['root'], '.joblib')
    if (manifest.get('fit_scope') != 'holdout_train'
            or manifest.get('model_sha256') != b._file_hash(path)):
        raise b.MLRequestError('Selected alternative model is missing or changed.')
    selected = ModelAdapter.load_model(path)
    if (selected.family != manifest.get('model') or selected.parameters != manifest.get('training_parameters')
            or selected.tree_count_ != manifest.get('selected_iterations')):
        raise b.MLRequestError('Selected model parameters differ from the manifest.')
    x, test_x, _ = b._features(problem)
    count = len(problem['classes']) if problem['classes'] is not None else None
    probability = b._aligned_proba(selected, x.iloc[valid_i], count) if count else None
    score = b._score(problem, problem['y'].iloc[valid_i], selected.predict(x.iloc[valid_i]), probability)
    if score is None or not math.isclose(score, manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Saved model does not reproduce its holdout score.')
    left = budget - (time.monotonic() - started) - min(15, max(2, budget * .08))
    if left < 1:
        raise b.MLRequestError('Not enough time for full-data refit.')
    full = selected.refit(x, problem['y'], time.monotonic() + left, threads)
    p = b._aligned_proba(full, test_x, count) if count else None
    candidate, schema = b._output(problem, full.predict(test_x), p, manifest['request'])
    full_path = candidate.with_suffix('.joblib')
    full.save_model(full_path)
    final = dict(manifest, fit_scope='full_train', training_rows=len(x),
                 model_path=str(full_path), model_sha256=b._file_hash(full_path))
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(final, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = _response(problem, full.family, candidate, schema, manifest_path, final, started,
                       'Full-data refit of selected parameters and tree count; original score retained.')
    result['final'] = True
    return result


def _load_model(manifest, root, validation=False):
    import ml_backend as b
    key = 'validation_model_path' if validation else 'model_path'
    hash_key = 'validation_model_sha256' if validation else 'model_sha256'
    path = private_path(manifest.get(key), root)
    if b._file_hash(path) != manifest.get(hash_key):
        raise b.MLRequestError('Component model is missing or changed.')
    if manifest.get('model') == 'catboost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = CatBoostClassifier() if manifest['classes'] is not None else CatBoostRegressor()
        model.load_model(str(path))
    elif manifest.get('model') in {'lightgbm', 'xgboost', 'extra_trees', 'linear'}:
        from ml_models import ModelAdapter
        model = ModelAdapter.load_model(path)
    else:
        raise b.MLRequestError('Only independent native models can be ensemble components.')
    if model.tree_count_ != manifest.get('selected_iterations'):
        raise b.MLRequestError('Component tree count differs from its manifest.')
    return model


def _blend(problem, models, weights, x):
    import numpy as np
    import ml_backend as b
    frames = x if isinstance(x, list) else [x] * len(models)
    if len(frames) != len(models):
        raise b.MLRequestError('Every ensemble component needs its own input frame.')
    if problem['classes'] is not None:
        probability = sum(w * b._aligned_proba(m, frame, len(problem['classes']))
                          for w, m, frame in zip(weights, models, frames))
        return np.argmax(probability, axis=1), probability
    return sum(w * np.asarray(m.predict(frame)).reshape(-1)
               for w, m, frame in zip(weights, models, frames)), None


def run_ensemble(envelope, started, budget, threads):
    import numpy as np
    import ml_backend as b
    _check_time(started, budget)
    request = envelope['request']
    paths = request.get('manifest_paths')
    if not isinstance(paths, list) or not 2 <= len(paths) <= 3 or len(set(map(str, paths))) != len(paths):
        raise b.MLRequestError('An ensemble needs two or three distinct native manifests.')
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    weights = request.get('weights', [1.0] * len(paths))
    if (not isinstance(weights, list) or len(weights) != len(paths)
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)):
        raise b.MLRequestError('Ensemble weights must be positive finite numbers.')
    weights = np.asarray(weights, dtype=float)
    weights /= weights.max()  # Avoid overflow for valid large finite inputs.
    weights /= weights.sum()
    components, models, validation_models = [], [], []
    validation_frames, test_frames = [], []
    problem = train_i = valid_i = None
    for index, value in enumerate(paths):
        _check_time(started, budget, 1)
        path, manifest = read_manifest(value, root)
        p, ti, vi = checked_problem(envelope, manifest)
        if problem is None:
            problem, train_i, valid_i = p, ti, vi
        elif any(manifest.get(k) != components[0]['manifest'].get(k)
                 for k in ('validation_id', 'metric', 'direction', 'features', 'classes')):
            raise b.MLRequestError('Ensemble components must use identical data, targets and validation.')
        models.append(_load_model(manifest, root))
        validation_models.append(_load_model(manifest, root, validation=True))
        # Representation is part of each model's saved request. A shared raw
        # feature list does not mean that numeric/categorical conversions match.
        component_x, component_test_x, _ = b._features(p)
        validation_frames.append(component_x.iloc[vi])
        test_frames.append(component_test_x)
        components.append(dict(manifest_path=str(path), manifest_sha256=b._file_hash(path),
                               weight=float(weights[index]), manifest=manifest))
    # Always use the saved HOLDOUT models for scoring, even if test predictions
    # use full-data refits. Scoring full refits on these rows would leak labels.
    validation_predictions, validation_probabilities = [], []
    for model, component, frame in zip(validation_models, components, validation_frames):
        _check_time(started, budget, 1)
        p, prob = _blend(problem, [model], [1.0], frame)
        actual = b._score(problem, problem['y'].iloc[valid_i], p, prob)
        if actual is None or not math.isclose(actual, component['manifest']['score'], rel_tol=1e-10, abs_tol=1e-10):
            raise b.MLRequestError('Component does not reproduce its saved holdout score.')
        validation_predictions.append(p)
        validation_probabilities.append(prob)
    if problem['classes'] is not None:
        probability = sum(w * p for w, p in zip(weights, validation_probabilities))
        prediction = np.argmax(probability, axis=1)
    else:
        probability = None
        prediction = sum(w * p for w, p in zip(weights, validation_predictions))
    score = b._score(problem, problem['y'].iloc[valid_i], prediction, probability)
    _check_time(started, budget, 1)
    test_prediction, test_probability = _blend(problem, models, weights, test_frames)
    _check_time(started, budget)
    candidate, schema = b._output(problem, test_prediction, test_probability, components[0]['manifest']['request'])
    full = all(c['manifest']['fit_scope'] == 'full_train' for c in components)
    manifest = dict(model='ensemble', target=problem['target'], features=problem['features'],
                    classes=problem['classes'], request=components[0]['manifest']['request'],
                    metric=problem['metric'], direction=problem['direction'], score=score,
                    validation_id=b._validation_id(problem, train_i, valid_i), validation=problem['validation'],
                    input_hashes=b._input_hashes(problem), fit_scope='full_train' if full else 'holdout_train',
                    training_rows=len(problem['y']) if full else len(train_i),
                    components=[{k: v for k, v in c.items() if k != 'manifest'} for c in components])
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return _response(problem, 'ensemble', candidate, schema, manifest_path, manifest, started,
                     'Fixed weighted average; holdout scores use only original holdout models.')


def refit_ensemble(envelope, manifest, started, budget, threads):
    import ml_backend as b
    _check_time(started, budget)
    problem, _, _ = checked_problem(envelope, manifest)
    components = manifest.get('components')
    if not isinstance(components, list) or not 2 <= len(components) <= 3:
        raise b.MLRequestError('Invalid ensemble components.')
    pending = []
    for component in components:
        _check_time(started, budget)
        path, part = read_manifest(component.get('manifest_path'), problem['root'])
        if b._file_hash(path) != component.get('manifest_sha256'):
            raise b.MLRequestError('Ensemble component manifest changed.')
        pending.append((component, path, part))
    paths, weights = [], []
    for index, (component, path, part) in enumerate(pending):
        if part.get('fit_scope') != 'full_train':
            left = budget - (time.monotonic() - started) - min(20, budget * .1)
            remaining_fits = sum(p.get('fit_scope') != 'full_train' for _, _, p in pending[index:])
            allowance = left / remaining_fits
            if allowance < 2:
                raise b.MLRequestError('Insufficient time to refit every ensemble component.')
            result = b.run(dict(envelope, request=dict(action='refit', manifest_path=str(path)),
                                time_budget_seconds=allowance))
            if result.get('fit_scope') != 'full_train' or result.get('final') is not True:
                raise b.MLRequestError('An ensemble component did not complete full-data refit.')
            path = Path(result['manifest_path'])
        paths.append(str(path))
        weights.append(component['weight'])
    result = run_ensemble(dict(envelope, request=dict(action='ensemble', manifest_paths=paths,
                                                     weights=weights)), started, budget, threads)
    if (result.get('fit_scope') != 'full_train'
            or not math.isclose(result['score'], manifest['score'], rel_tol=1e-10, abs_tol=1e-10)
            or any(result.get(k) != manifest.get(k) for k in ('metric', 'direction', 'validation_id'))):
        raise b.MLRequestError('Final ensemble no longer describes the selected validation.')
    # Preserve the exact original score for the parent's equal-score replacement.
    result['score'] = manifest['score']
    final_path, final_manifest = read_manifest(result['manifest_path'], problem['root'])
    final_manifest['score'] = manifest['score']
    final_path.write_text(json.dumps(final_manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result['final'] = True
    return result


def _try_seed_bag(envelope, candidates, started, budget, threads):
    """Use a second cheap seed only inside a fixed conservative regression blend."""
    import ml_backend as b
    cb = next((r for r in candidates if r.get('model') == 'catboost'), None)
    xgb = next((r for r in candidates if r.get('model') == 'xgboost'), None)
    if not cb or not xgb or xgb.get('elapsed_seconds', budget) > min(10, budget / 10):
        return None
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    _, selected = read_manifest(xgb['manifest_path'], root)
    if selected.get('classes') is not None or any(cb.get(k) != xgb.get(k)
            for k in ('metric', 'direction', 'validation_id')):
        return None
    left = budget - (time.monotonic() - started) - 20
    if left < 30:
        return None
    params = dict(selected['request'], action='experiment', refit=False)
    params['random_seed'] = 17 if params.get('random_seed', 42) != 17 else 42
    allowance = min(left, max(30, xgb.get('elapsed_seconds', 0) * 2.5))
    extra = b.run(dict(envelope, request=params, time_budget_seconds=allowance))
    if not finite_score(extra) or any(extra.get(k) != cb.get(k)
            for k in ('metric', 'direction', 'validation_id')):
        return None
    # Never promote the single lucky seed: assess the predefined seed mixture.
    return run_ensemble(dict(envelope, request=dict(action='ensemble',
        manifest_paths=[cb['manifest_path'], xgb['manifest_path'], extra['manifest_path']],
        weights=[.5, .25, .25])), started, budget, threads)


def run_portfolio(envelope, started, budget, threads):
    """Compare different learners without paying LLM tokens for standard trials."""
    import ml_backend as b
    request = envelope['request']
    if not isinstance(request.get('seed_bagging', True), bool):
        raise b.MLRequestError('seed_bagging must be a boolean.')
    for key in ('entity_history', 'entity_lag', 'extra_trees', 'linear'):
        if not isinstance(request.get(key, True), bool):
            raise b.MLRequestError(key + ' must be a boolean.')
    families = request.get('families', ['catboost', 'lightgbm', 'xgboost'])
    if (not isinstance(families, list) or not families or len(families) > 4
            or len(set(families)) != len(families)
            or any(f not in {'catboost', 'lightgbm', 'xgboost', 'extra_trees', 'linear'} for f in families)):
        raise b.MLRequestError('families must list distinct supported native models.')
    best, candidates, trials = {}, [], []
    for index, family in enumerate(families):
        left = budget - (time.monotonic() - started) - min(20, budget * .1)
        if left < 3:
            break
        allowance = left / (len(families) - index)
        params = {k: v for k, v in request.items() if k not in {'families', 'manifest_paths', 'weights'}}
        params.update(action='search' if family == 'catboost' else 'experiment', model=family, refit=False)
        try:
            if family == 'catboost' and 'depth' not in params:
                # Start with the regression control that survived both temporal
                # backtests. A busy CPU may disable neighbor search completely.
                problem = b.load_problem(envelope)
                params['depth'] = 8 if problem['classes'] is None else 6
            result = b.run(dict(envelope, request=params, time_budget_seconds=allowance))
            accepted = bool(improves(result, best))
            trials.append(dict(model=family, score=result.get('score'), accepted=accepted,
                               selected_iterations=result.get('selected_iterations')))
            if result.get('model') != 'dummy' and finite_score(result) and result.get('manifest_path'):
                candidates.append(result)
            if accepted or not best:
                best = result
                checkpoint(envelope, best)
        except Exception as exc:
            trials.append(dict(model=family, accepted=False, error=type(exc).__name__, message=str(exc)[:300]))
    if not best:
        left = max(1, budget - (time.monotonic() - started) - 1)
        best = b.run(dict(envelope, request=dict(request, action='baseline'), time_budget_seconds=left))
        checkpoint(envelope, best)
    if request.get('seed_bagging', True):
        try:
            bag = _try_seed_bag(envelope, candidates, started, budget, threads)
            if bag is not None:
                accepted = bool(improves(bag, best))
                trials.append(dict(model='regression_seed_bag', score=bag['score'], accepted=accepted))
                if accepted:
                    best = bag
                    checkpoint(envelope, best)
        except Exception as exc:
            trials.append(dict(model='regression_seed_bag', accepted=False, error=type(exc).__name__))
    pair_seconds = 0.0
    if len(candidates) >= 2 and budget - (time.monotonic() - started) > 20:
        compatible = [r for r in candidates if all(r.get(k) == best.get(k)
                      for k in ('metric', 'direction', 'validation_id'))]
        compatible.sort(key=lambda r: r['score'], reverse=best.get('direction') == 'maximize')
        if len(compatible) >= 2:
            pair_started = time.monotonic()
            try:
                blended = run_ensemble(dict(envelope, request=dict(action='ensemble',
                    manifest_paths=[r['manifest_path'] for r in compatible[:2]])), started, budget, threads)
                accepted = bool(improves(blended, best))
                trials.append(dict(model='ensemble', score=blended['score'], accepted=accepted))
                if accepted:
                    best = blended
                    checkpoint(envelope, best)
            except Exception as exc:
                trials.append(dict(model='ensemble', accepted=False, error=type(exc).__name__))
            finally:
                pair_seconds = time.monotonic() - pair_started
    # The established native ensemble is saved before optional work. A forest
    # gets a small training allowance and leaves time for loading and prediction.
    if request.get('extra_trees', True) and 'families' not in request:
        try:
            problem = b.load_problem(envelope)
            left = budget - (time.monotonic() - started)
            allowance = min(45.0, left - 60.0)
            if problem['classes'] is not None and allowance >= 15:
                params = dict(request, action='experiment', model='extra_trees', refit=False)
                # Boosting iteration/depth settings do not define a forest size.
                for key in ('iterations', 'depth', 'learning_rate'):
                    params.pop(key, None)
                forest = b.run(dict(envelope, request=params, time_budget_seconds=allowance))
                accepted = bool(improves(forest, best))
                trials.append(dict(model='extra_trees', score=forest.get('score'), accepted=accepted,
                                   selected_iterations=forest.get('selected_iterations')))
                if finite_score(forest) and forest.get('manifest_path'):
                    candidates.append(forest)
                if accepted:
                    best = forest
                    checkpoint(envelope, best)
        except Exception as exc:
            trials.append(dict(model='extra_trees', accepted=False, error=type(exc).__name__,
                               message=str(exc)[:300]))
    compatible = [r for r in candidates if all(r.get(k) == best.get(k)
                  for k in ('metric', 'direction', 'validation_id'))]
    boosting = [r for r in compatible if r.get('model') != 'extra_trees']
    forests = [r for r in compatible if r.get('model') == 'extra_trees']
    boosting.sort(key=lambda r: r['score'], reverse=best.get('direction') == 'maximize')
    if forests and boosting:
        forest = forests[0]
        for count, label, floor in ((1, 'forest_pair_ensemble', 30),
                                    (2, 'three_model_ensemble', 45)):
            minimum = max(floor, pair_seconds * (count + 1) + 10)
            if len(boosting) < count or budget - (time.monotonic() - started) < minimum:
                continue
            ensemble_started = time.monotonic()
            try:
                blended = run_ensemble(dict(envelope, request=dict(action='ensemble',
                    manifest_paths=[r['manifest_path'] for r in boosting[:count]] +
                                   [forest['manifest_path']])), started, budget, threads)
                accepted = bool(improves(blended, best))
                trials.append(dict(model=label, score=blended['score'], accepted=accepted))
                if accepted:
                    best = blended
                    checkpoint(envelope, best)
            except Exception as exc:
                trials.append(dict(model=label, accepted=False, error=type(exc).__name__))
            finally:
                pair_seconds = max(pair_seconds, time.monotonic() - ensemble_started)
    # A different inductive bias: trees cannot extrapolate a continuous trend.
    # Run after the established tree portfolio, preserving its checkpoint and
    # forest allowance. The existing worker enforces the hard process deadline.
    if request.get('linear', True) and 'families' not in request:
        left = budget - (time.monotonic() - started)
        allowance = min(20.0, left - 30.0)
        if allowance >= 5:
            try:
                params = dict(request, action='experiment', model='linear', refit=False)
                result = b.run(dict(envelope, request=params, time_budget_seconds=allowance))
                accepted = bool(improves(result, best))
                trials.append(dict(model='linear', score=result.get('score'), accepted=accepted))
                if accepted:
                    best = result
                    checkpoint(envelope, best)
            except Exception as exc:
                trials.append(dict(model='linear', accepted=False, error=type(exc).__name__))
    if request.get('entity_history', True) and finite_score(best) and best.get('model') != 'dummy':
        try:
            from ml_history import entity_columns, run_history
            problem = b.load_problem(envelope)
            b._split(problem, request)
            if (problem['classes'] is None and problem['validation']['strategy'] == 'time'
                    and entity_columns(problem, request)[0]):
                history_request = dict(action='history', manifest_path=best['manifest_path'])
                for key in ('entity_columns', 'history_weight'):
                    if key in request:
                        history_request[key] = request[key]
                history = run_history(dict(envelope, request=history_request), started, budget, threads)
                accepted = bool(improves(history, best))
                trials.append(dict(model='history', score=history['score'], accepted=accepted,
                                   entity_columns=history['entity_columns'],
                                   coverage=history['history_validation']['coverage']))
                if accepted:
                    best = history
                    checkpoint(envelope, best)
        except Exception as exc:
            trials.append(dict(model='history', accepted=False, error=type(exc).__name__,
                               message=str(exc)[:300]))
    if (best.get('model') == 'history' and request.get('entity_lag', True)
            and budget - (time.monotonic() - started) >= 30):
        try:
            from ml_lag import run_lag
            lag = run_lag(dict(envelope, request=dict(action='lag',
                          manifest_path=best['manifest_path'])), started, budget, threads)
            accepted = bool(improves(lag, best))
            trials.append(dict(model='lag', score=lag['score'], accepted=accepted,
                               selected_iterations=lag.get('selected_iterations')))
            if accepted:
                best = lag
                checkpoint(envelope, best)
        except Exception as exc:
            trials.append(dict(model='lag', accepted=False, error=type(exc).__name__,
                               message=str(exc)[:300]))
    return dict(best, portfolio_results=trials, elapsed_seconds=round(time.monotonic() - started, 3))
