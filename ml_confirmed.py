"""Nested model selection with a single, separately held-out confirmation.

The confirmation split is reserved before choosing models or tree counts.
Only a frozen challenger is compared with the fixed LightGBM control there.
This action is optional and does not alter the established startup portfolio.
"""
import json
import math
import time

import numpy as np


def select_confirmed(control, challenger, direction, metric):
    """Return a frozen challenger only when both comparisons clear a margin."""
    sign = 1 if direction == 'maximize' else -1
    if direction not in {'maximize', 'minimize'}:
        raise ValueError('Unknown metric direction')
    values = [r[k] for r in (control, challenger) for k in ('selection_score', 'confirmation_score')]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Cannot confirm nonfinite scores')
    margins = [(.002 if metric == 'macro_f1' else max(abs(control[k]) * .005, 1e-8))
               for k in ('selection_score', 'confirmation_score')]
    gains = [sign * (challenger[k] - control[k]) for k in ('selection_score', 'confirmation_score')]
    accepted = all(g >= m for g, m in zip(gains, margins))
    return bool(accepted), dict(selection_gain=gains[0], confirmation_gain=gains[1],
                               minimum_gains=margins, accepted=bool(accepted))


def nested_split(problem, request):
    """Map development-only inner indices back to the original training rows."""
    import ml_backend as b
    outer_train, confirmation = b._split(problem, request)
    if outer_train is None:
        raise b.MLRequestError('Confirmed selection requires an outer holdout')
    outer_validation = dict(problem['validation'])
    development = dict(problem, train=problem['train'].iloc[outer_train].reset_index(drop=True),
                       y=problem['y'].iloc[outer_train].reset_index(drop=True))
    inner_request = dict(request)
    strategy = outer_validation['strategy']
    inner_request['split'] = strategy
    if strategy == 'time' and outer_validation.get('diagnostics', {}).get('time_column'):
        inner_request['time_column'] = outer_validation['diagnostics']['time_column']
    if strategy == 'group':
        inner_request['group_columns'] = outer_validation['diagnostics']['group_columns']
    inner_train, selection = b._split(development, inner_request)
    if inner_train is None:
        raise b.MLRequestError('Confirmed selection requires an inner holdout')
    fit_rows, select_rows = outer_train[inner_train], outer_train[selection]
    if any(set(a) & set(c) for a, c in ((fit_rows, select_rows), (fit_rows, confirmation),
                                        (select_rows, confirmation))):
        raise b.MLRequestError('Nested validation rows overlap')
    if problem['classes'] is not None:
        expected = set(range(len(problem['classes'])))
        if any(set(problem['y'].iloc[rows]) != expected for rows in (fit_rows, select_rows, confirmation)):
            raise b.MLRequestError('Too few independent examples to evaluate every class in all three parts')
    # load_problem's original validation contract remains authoritative outside this helper.
    problem['validation'] = outer_validation
    return outer_train, confirmation, fit_rows, select_rows


def run_confirmed(envelope, started, budget, threads):
    import ml_backend as b
    from ml_models import fit
    from ml_portfolio import _response, save_validation, checkpoint

    request = envelope['request']
    problem = b.load_problem(envelope)
    if problem['metric'] not in {'macro_f1', 'rmse', 'mae'}:
        raise b.MLRequestError('Confirmed selection supports macro_f1, rmse and mae')
    # No tuning budget or model-family list from an LLM: keep the proposal set fixed.
    outer_train, confirmation, fit_rows, selection = nested_split(problem, request)
    x, test_x, categorical = b._features(problem)
    count = len(problem['classes']) if problem['classes'] is not None else None
    deadline = started + budget
    configs = (
        ('lightgbm', dict(iterations=360, num_leaves=31, learning_rate=.05)),
        ('xgboost', dict(iterations=360, depth=6, learning_rate=.05)),
        ('linear', {}),
    )
    trials = []
    models = {}
    # Half the budget is reserved for confirmation fits, artifact persistence and full refit.
    selection_deadline = started + budget * .5
    for ordinal, (family, parameters) in enumerate(configs):
        available = selection_deadline - time.monotonic()
        if available < 2:
            break
        trial_deadline = time.monotonic() + available / (len(configs) - ordinal)
        params = dict(parameters, action='experiment', model=family, refit=False)
        try:
            model = fit(family, x.iloc[fit_rows], problem['y'].iloc[fit_rows],
                        x.iloc[selection], problem['y'].iloc[selection], categorical, params,
                        trial_deadline, threads, problem['problem_type'], problem['metric'])
            probability = b._aligned_proba(model, x.iloc[selection], count) if count else None
            score = b._score(problem, problem['y'].iloc[selection], model.predict(x.iloc[selection]), probability)
            if score is None:
                raise ValueError('No finite selection score')
            trials.append(dict(model=family, selection_score=score, selected_iterations=model.tree_count_))
            models[family] = model
        except Exception as exc:
            trials.append(dict(model=family, error_type=type(exc).__name__))
    if 'lightgbm' not in models:
        raise b.MLRequestError('Fixed control did not finish; preserve the existing startup candidate')
    eligible = [r for r in trials if r['model'] in models]
    control = next(r for r in eligible if r['model'] == 'lightgbm')
    challenger = (max if problem['direction'] == 'maximize' else min)(eligible, key=lambda r: r['selection_score'])
    selected_families = list(dict.fromkeys(('lightgbm', challenger['model'])))
    fitted, predictions = {}, {}
    # Fit and predict BOTH frozen proposals before the first read of confirmation answers.
    confirmation_deadline = started + budget * .85
    for ordinal, family in enumerate(selected_families):
        available = confirmation_deadline - time.monotonic()
        if available < 1:
            raise b.MLRequestError('No time for the complete paired confirmation')
        fitted[family] = models[family].refit(x.iloc[outer_train], problem['y'].iloc[outer_train],
            time.monotonic() + available / (len(selected_families) - ordinal), threads)
        model = fitted[family]
        predictions[family] = (np.asarray(model.predict(x.iloc[confirmation])).reshape(-1),
            b._aligned_proba(model, x.iloc[confirmation], count) if count else None)
    for record in eligible:
        if record['model'] in fitted:
            prediction, probability = predictions[record['model']]
            record['confirmation_score'] = b._score(problem, problem['y'].iloc[confirmation], prediction, probability)
    accepted, verdict = select_confirmed(control, challenger, problem['direction'], problem['metric'])
    chosen = challenger if accepted else control
    family = chosen['model']
    model = fitted[family]
    prediction, probability = predictions[family]
    params = dict(next(parameters for name, parameters in configs if name == family),
                  action='experiment', model=family, refit=False)
    # Keep the exact original split settings for later refit/reproduction checks.
    params.update({k: request[k] for k in ('split', 'time_column', 'group_columns',
                   'group_by_features', 'validation_fraction') if k in request})
    test_probability = b._aligned_proba(model, test_x, count) if count else None
    candidate, schema = b._output(problem, model.predict(test_x), test_probability, params)
    path = candidate.with_suffix('.joblib')
    model.save_model(path)
    validation_path, validation_hash = save_validation(candidate, prediction, probability)
    manifest = dict(model=family, target=problem['target'], features=problem['features'],
        categorical_features=categorical, classes=problem['classes'], request=params,
        validation_id=b._validation_id(problem, outer_train, confirmation), validation=problem['validation'],
        score=chosen['confirmation_score'], metric=problem['metric'], direction=problem['direction'],
        fit_scope='holdout_train', training_rows=len(outer_train), selected_iterations=model.tree_count_,
        training_parameters=model.parameters, input_hashes=b._input_hashes(problem),
        model_path=str(path), model_sha256=b._file_hash(path), validation_model_path=str(path),
        validation_model_sha256=b._file_hash(path), validation_path=validation_path,
        validation_sha256=validation_hash, confirmation=dict(verdict, trials=trials,
            inner_training_rows=len(fit_rows), selection_rows=len(selection), confirmation_rows=len(confirmation)))
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    result = _response(problem, family, candidate, schema, manifest_path, manifest, started,
                       'Nested selection; one confirmation against a fixed LightGBM control.')
    result['confirmation'] = manifest['confirmation']
    checkpoint(envelope, result)
    # Reuse the established refit implementation, including its verification of the holdout model.
    if request.get('refit', True) and deadline - time.monotonic() > 2:
        try:
            from ml_portfolio import refit_single
            refitted = refit_single(envelope, manifest, started, budget, threads)
            refitted['confirmation'] = manifest['confirmation']
            checkpoint(envelope, refitted)
            return refitted
        except Exception as exc:
            result['notes'] += ' Full-data refit deferred: ' + type(exc).__name__
    return result
