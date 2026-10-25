"""Conservative past-target blending for repeated entities in temporal regression."""
from pathlib import Path
import json
import math
import re
import time


def entity_columns(problem, request):
    """Resolve explicit keys, or a conservative structural building signature.

    A structural signature is a hypothesis, never a certified building ID. It
    is considered only on a forward time split and still needs a better score.
    """
    import ml_backend as b
    explicit = request.get('entity_columns')
    if explicit is not None:
        if (not isinstance(explicit, list) or not explicit or len(explicit) > 8
                or any(not isinstance(c, str) for c in explicit)
                or len(set(explicit)) != len(explicit)
                or any(c not in problem['features'] for c in explicit)):
            raise b.MLRequestError('entity_columns must name 1–8 distinct model features.')
        return explicit, 'Explicit entity signature supplied by caller'
    # Require all four semantic roles. Do not guess from row IDs, weather or
    # arbitrary high-cardinality measurements, nor search keys using targets.
    roles = {
        'location': r'^(?:state|region|country|city)(?:_factor|_code)?$',
        'type': r'^(?:building|facility|property)_(?:class|type)$',
        'size': r'^(?:floor|building|property)?_?area$',
        'construction': r'^(?:year_built|construction_year|built_year)$',
    }
    matched = {role: [c for c in problem['features'] if re.fullmatch(pattern, c, re.I)]
               for role, pattern in roles.items()}
    if not all(matched.values()):
        return [], 'No complete structural entity signature found'
    columns = [c for names in matched.values() for c in names]
    if len(columns) > 8:
        return [], 'Ambiguous structural signature'
    return columns, 'Heuristic structural signature: location, type, area, construction year'


def blend_past(train, target, query, base, columns, time_column, weight=.5):
    """Use latest strictly earlier observations; missing/ambiguous keys fall back.

    Reject a key altogether if it identifies multiple rows in any training
    period. Query targets are neither an argument nor part of a join.
    """
    import numpy as np
    import pandas as pd
    import numbers
    import ml_backend as b
    if (type(weight) not in (int, float) or not math.isfinite(weight)
            or not 0 < weight < 1):
        raise b.MLRequestError('history_weight must be finite and strictly between 0 and 1.')
    if (not columns or time_column in columns or len(set(columns)) != len(columns)
            or any(c not in train or c not in query for c in [*columns, time_column])):
        raise b.MLRequestError('History requires distinct entity columns and a separate time column.')
    if (not pd.api.types.is_numeric_dtype(train[time_column])
            or not pd.api.types.is_numeric_dtype(query[time_column])):
        raise b.MLRequestError('History currently requires numeric ordered periods.')
    if (not np.isfinite(train[time_column].to_numpy(dtype=float)).all()
            or not np.isfinite(query[time_column].to_numpy(dtype=float)).all()):
        raise b.MLRequestError('History periods must be finite and nonmissing.')
    prediction = np.asarray(base, dtype=float).reshape(-1).copy()
    values = np.asarray(target, dtype=float).reshape(-1)
    if (len(prediction) != len(query) or len(values) != len(train)
            or not np.isfinite(prediction).all() or not np.isfinite(values).all()):
        raise b.MLRequestError('History needs finite aligned targets and predictions.')
    # Numeric equality survives float/int changes; tuples avoid delimiter
    # collisions, and keys with any missing cell are never matched.
    def usable_rows(frame):
        result = frame[columns].notna().all(axis=1).to_numpy()
        for column in columns:
            # Numeric sentinels are missing entity identity, including in an
            # object column mixed with strings. Literal string codes stay valid.
            def finite_key(value):
                if not isinstance(value, numbers.Number):
                    return True
                try:
                    return math.isfinite(value)
                except (TypeError, ValueError, OverflowError):
                    return False
            result &= np.fromiter((finite_key(value) for value in frame[column]),
                                  dtype=bool, count=len(frame))
        return result

    usable = usable_rows(train)
    duplicated = train.duplicated([*columns, time_column], keep=False).to_numpy() & usable
    ambiguous = set(train.loc[duplicated, columns].itertuples(index=False, name=None))
    history = {}
    for key, period, value, ok in zip(train[columns].itertuples(index=False, name=None),
                                      train[time_column], values, usable):
        if ok and key not in ambiguous:
            history.setdefault(key, []).append((period, value))
    for rows in history.values():
        rows.sort(key=lambda row: row[0])
    matched = 0
    query_ok = usable_rows(query)
    for i, (key, period, ok) in enumerate(zip(query[columns].itertuples(index=False, name=None),
                                             query[time_column], query_ok)):
        if not ok:
            continue
        earlier = [row for row in history.get(key, ()) if row[0] < period]
        if earlier:
            prediction[i] = (1 - weight) * prediction[i] + weight * earlier[-1][1]
            matched += 1
    return prediction, dict(matched_rows=matched, coverage=matched / max(1, len(query)),
                            ambiguous_keys=len(ambiguous), usable_entity_keys=len(history),
                            missing_training_rows=int((~usable).sum()))


def _base_predictions(envelope, manifest, started, budget):
    """Reproduce the original score with original models and their own frames."""
    import numpy as np
    import ml_backend as b
    import ml_portfolio as p
    p._check_time(started, budget, 1)
    problem, ti, vi = p.checked_problem(envelope, manifest)
    if manifest.get('model') == 'ensemble':
        parts = manifest.get('components', [])
        if not 2 <= len(parts) <= 3:
            raise b.MLRequestError('Invalid history base ensemble.')
        loaded = []
        for part in parts:
            path, item = p.read_manifest(part.get('manifest_path'), problem['root'])
            if b._file_hash(path) != part.get('manifest_sha256'):
                raise b.MLRequestError('History base component changed.')
            loaded.append((float(part['weight']), item))
        if (any(not math.isfinite(w) or w <= 0 for w, _ in loaded)
                or not math.isclose(sum(w for w, _ in loaded), 1.0)):
            raise b.MLRequestError('Invalid history base ensemble weights.')
    elif manifest.get('model') in {'catboost', 'lightgbm', 'xgboost'}:
        loaded = [(1.0, manifest)]
    else:
        raise b.MLRequestError('History requires an independent native model or ensemble.')
    valid_prediction = np.zeros(len(vi))
    test_prediction = np.zeros(len(problem['test']))
    for weight, part in loaded:
        p._check_time(started, budget, 1)
        own, own_ti, own_vi = p.checked_problem(envelope, part)
        if part['classes'] is not None or part['validation_id'] != manifest['validation_id']:
            raise b.MLRequestError('History base components must share one regression validation.')
        x, tx, _ = b._features(own)
        model = p._load_model(part, problem['root'], validation=True)
        val = np.asarray(model.predict(x.iloc[own_vi])).reshape(-1)
        actual = b._score(own, own['y'].iloc[own_vi], val, None)
        if actual is None or not math.isclose(actual, part['score'], rel_tol=1e-10, abs_tol=1e-10):
            raise b.MLRequestError('History component no longer reproduces its score.')
        test_model = p._load_model(part, problem['root'])
        valid_prediction += weight * val
        test_prediction += weight * np.asarray(test_model.predict(tx)).reshape(-1)
    actual = b._score(problem, problem['y'].iloc[vi], valid_prediction, None)
    if actual is None or not math.isclose(actual, manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('History base no longer reproduces its score.')
    return problem, ti, vi, valid_prediction, test_prediction


def run_history(envelope, started, budget, threads):
    import ml_backend as b
    import ml_portfolio as p
    request = envelope['request']
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    base_path, base = p.read_manifest(request.get('manifest_path'), root)
    problem, ti, vi, validation_prediction, prediction = _base_predictions(envelope, base, started, budget)
    time_column = problem['validation'].get('diagnostics', {}).get('time_column')
    if (problem['classes'] is not None or problem['validation']['strategy'] != 'time'
            or not time_column or time_column == problem['target']
            or time_column not in problem['train']
            or problem['metric'].removeprefix('neg_') not in {'rmse', 'mse', 'mae', 'r2'}):
        raise b.MLRequestError('History requires forward time validation for regression.')
    columns, rationale = entity_columns(problem, request)
    if not columns:
        raise b.MLRequestError(rationale)
    if time_column in columns:
        raise b.MLRequestError('Entity signature cannot include the time column.')
    weight = request.get('history_weight', .5)
    train, query = problem['train'].iloc[ti], problem['train'].iloc[vi]
    if query[time_column].min() <= train[time_column].max():
        raise b.MLRequestError('History validation must be strictly later than training.')
    vp, diagnostics = blend_past(train, problem['y'].iloc[ti], query, validation_prediction,
                                 columns, time_column, weight)
    if not diagnostics['matched_rows']:
        raise b.MLRequestError('Entity history has no unambiguous past matches on validation.')
    if request.get('entity_columns') is None:
        import pandas as pd
        sizes = [c for c in columns if re.fullmatch(r'(?:floor|building|property)?_?area', c, re.I)]
        construction = [c for c in columns if re.fullmatch(r'(?:year_built|construction_year|built_year)', c, re.I)]
        unique_keys = len(train[columns].dropna().drop_duplicates())
        if (any(not pd.api.types.is_numeric_dtype(train[c]) for c in sizes + construction)
                or train[time_column].nunique() < 2
                or diagnostics['usable_entity_keys'] < 20
                or diagnostics['coverage'] < .1
                or diagnostics['ambiguous_keys'] / max(1, unique_keys) > .05):
            raise b.MLRequestError('Automatic entity signature lacks sufficient unambiguous temporal support.')
    score = b._score(problem, problem['y'].iloc[vi], vp, None)
    full = base['fit_scope'] == 'full_train'
    full_train = problem['train'] if full else train
    full_y = problem['y'] if full else problem['y'].iloc[ti]
    if problem['test'][time_column].min() <= problem['train'][time_column].max():
        raise b.MLRequestError('History test periods must be strictly later than all training periods.')
    prediction, test_diagnostics = blend_past(full_train, full_y, problem['test'], prediction,
                                               columns, time_column, weight)
    p._check_time(started, budget, 1)
    candidate, schema = b._output(problem, prediction, None, base['request'])
    manifest = dict(base, model='history', base_manifest_path=str(base_path),
                    base_manifest_sha256=b._file_hash(base_path), score=score,
                    entity_columns=columns, time_column=time_column, history_weight=weight,
                    history_rationale=rationale, history_validation=diagnostics,
                    history_test=test_diagnostics)
    configuration = dict(version=1, entity_columns=columns, time_column=time_column,
                         history_weight=weight, base_manifest_path=str(base_path),
                         base_manifest_sha256=b._file_hash(base_path),
                         policy='latest_strictly_past_missing_or_ambiguous_fallback_v1')
    configuration_path = candidate.with_suffix('.history.json')
    configuration_path.write_text(json.dumps(configuration, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    manifest.update(history_config_path=str(configuration_path),
                    history_config_sha256=b._file_hash(configuration_path))
    # A history pipeline is reconstructed from its immutable base manifest;
    # it does not pretend to be the base's independent serialized estimator.
    for key in ('components', 'model_path', 'model_sha256', 'validation_model_path',
                'validation_model_sha256', 'validation_path', 'validation_sha256'):
        manifest.pop(key, None)
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = p._response(problem, 'history', candidate, schema, manifest_path, manifest, started,
                         'Fixed past-target blend; incomplete or ambiguous entity keys use the base model.')
    result.update(history_validation=diagnostics, history_test=test_diagnostics,
                  entity_columns=columns, history_rationale=rationale)
    if full:
        result['final'] = True
    return result


def refit_history(envelope, manifest, started, budget, threads):
    import ml_backend as b
    import ml_portfolio as p
    problem, _, _ = p.checked_problem(envelope, manifest)
    config_path, config = p.read_manifest(manifest.get('history_config_path'), problem['root'])
    if (b._file_hash(config_path) != manifest.get('history_config_sha256')
            or config.get('version') != 1
            or config.get('policy') != 'latest_strictly_past_missing_or_ambiguous_fallback_v1'
            or any(config.get(k) != manifest.get(k) for k in
                   ('entity_columns', 'time_column', 'history_weight',
                    'base_manifest_path', 'base_manifest_sha256'))):
        raise b.MLRequestError('History configuration changed.')
    base_path, base = p.read_manifest(manifest.get('base_manifest_path'), problem['root'])
    if b._file_hash(base_path) != manifest.get('base_manifest_sha256'):
        raise b.MLRequestError('History base manifest changed.')
    request = dict(action='history', manifest_path=str(base_path),
                   entity_columns=manifest['entity_columns'], history_weight=manifest['history_weight'])
    reproduced = run_history(dict(envelope, request=request), started, budget, threads)
    if (not math.isclose(reproduced['score'], manifest['score'], rel_tol=1e-10, abs_tol=1e-10)
            or reproduced['validation_id'] != manifest['validation_id']):
        raise b.MLRequestError('History no longer reproduces its original score.')
    if base.get('fit_scope') != 'full_train':
        left = budget - (time.monotonic() - started) - min(20, budget * .1)
        p._check_time(started, budget, 2)
        final_base = b.run(dict(envelope, request=dict(action='refit', manifest_path=str(base_path)),
                                time_budget_seconds=left))
        if final_base.get('fit_scope') != 'full_train' or final_base.get('final') is not True:
            raise b.MLRequestError('History base did not complete full refit.')
        request['manifest_path'] = final_base['manifest_path']
    result = run_history(dict(envelope, request=request), started, budget, threads)
    if (result['fit_scope'] != 'full_train'
            or not math.isclose(result['score'], manifest['score'], rel_tol=1e-10, abs_tol=1e-10)):
        raise b.MLRequestError('Final history pipeline changed its validation score.')
    # Keep the exact float for equal-score final replacement in ArtifactStore.
    path, final = p.read_manifest(result['manifest_path'], problem['root'])
    final['score'] = result['score'] = manifest['score']
    path.write_text(json.dumps(final, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result['final'] = True
    return result
