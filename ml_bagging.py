"""Optional, fixed-seed averaging of existing honestly validated candidates."""
import copy
import json
import math
from pathlib import Path
import time

import numpy as np


def _single(envelope, manifest, started, budget, threads):
    import ml_backend as b
    from ml_models import SeedBagAdapter, _adapter
    from ml_portfolio import checked_problem, _load_model, save_validation, _response
    p, train, valid = checked_problem(envelope, manifest)
    if p['classes'] is None:
        raise b.MLRequestError('Seed bags currently support classification only')
    if manifest.get('model') not in {'lightgbm', 'xgboost', 'extra_trees'}:
        raise b.MLRequestError('Only independent LightGBM, XGBoost and ExtraTrees models can be bagged')
    base = _load_model(manifest, p['root'], validation=True)
    if isinstance(base, SeedBagAdapter) or base.family not in {'lightgbm', 'xgboost', 'extra_trees'}:
        raise b.MLRequestError('Cannot bag this model family or an already bagged model')
    x, test_x, _ = b._features(p)
    count = len(p['classes'])
    original_score = b._score(p, p['y'].iloc[valid], base.predict(x.iloc[valid]),
                             b._aligned_proba(base, x.iloc[valid], count))
    if original_score is None or not math.isclose(original_score, manifest['score'], rel_tol=1e-10, abs_tol=1e-10):
        raise b.MLRequestError('Base holdout model does not reproduce its saved score')
    members = [base]
    deadline = started + budget - min(15, max(2, budget * .08))
    # Fixed, distinct seeds; preserve an unusual original seed instead of duplicating it.
    original_seed = base.parameters['native'].get('seed', base.parameters['native'].get('random_state', 42))
    seeds = [seed for seed in (17, 29, 43) if seed != original_seed][:2]
    for index, seed in enumerate(seeds):
        remaining = deadline - time.monotonic()
        if remaining < 2:
            raise b.MLRequestError('Not enough time to finish the complete seed bag')
        parameters = copy.deepcopy(base.parameters)
        parameters['native']['random_state' if base.family == 'extra_trees' else 'seed'] = seed
        model = _adapter(base.family, x.iloc[train], p['y'].iloc[train], base.categorical,
                         parameters, base.problem_type, base.metric, threads)
        model._train(x.iloc[train], p['y'].iloc[train], None, None, base.tree_count_,
                     time.monotonic() + remaining / (len(seeds) - index), select=False)
        if model.tree_count_ != base.tree_count_:
            raise TimeoutError('A seed replica did not finish the frozen tree count')
        members.append(model)
    bag = SeedBagAdapter(members)
    probability = b._aligned_proba(bag, x.iloc[valid], count)
    prediction = bag.predict(x.iloc[valid])
    score = b._score(p, p['y'].iloc[valid], prediction, probability)
    candidate, schema = b._output(p, bag.predict(test_x), b._aligned_proba(bag, test_x, count), manifest['request'])
    path = candidate.with_suffix('.joblib')
    bag.save_model(path)
    validation_path, validation_hash = save_validation(candidate, prediction, probability)
    result_manifest = dict(manifest, model_path=str(path), model_sha256=b._file_hash(path),
        validation_model_path=str(path), validation_model_sha256=b._file_hash(path),
        validation_path=validation_path, validation_sha256=validation_hash,
        score=score, fit_scope='holdout_train', training_rows=len(train),
        selected_iterations=bag.tree_count_, training_parameters=bag.parameters,
        seed_bag=dict(original_seed=int(original_seed), new_seeds=seeds, frozen_count=base.tree_count_))
    manifest_path = candidate.with_suffix('.json')
    manifest_path.write_text(json.dumps(result_manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return _response(p, bag.family, candidate, schema, manifest_path, result_manifest, started,
                     'Three fixed seeds; original holdout rows never enter replica fitting or stopping.')


def run_seed_bag(envelope, started, budget, threads):
    import ml_backend as b
    from ml_portfolio import read_manifest, checked_problem, run_ensemble, checkpoint
    root = Path(envelope.get('task_dir') or Path(envelope['task_path']).parent).resolve()
    _, manifest = read_manifest(envelope['request'].get('manifest_path'), root)
    problem, _, _ = checked_problem(envelope, manifest)
    if problem['classes'] is None:
        raise b.MLRequestError('Seed bags currently support classification only')
    if manifest.get('model') != 'ensemble':
        result = _single(envelope, manifest, started, budget, threads)
        checkpoint(envelope, result)
        return result
    components = manifest.get('components')
    if not isinstance(components, list) or not 2 <= len(components) <= 3:
        raise b.MLRequestError('Invalid seed bag source ensemble')
    pending = []
    for component in components:
        path, part = read_manifest(component.get('manifest_path'), root)
        if b._file_hash(path) != component.get('manifest_sha256'):
            raise b.MLRequestError('Source ensemble component changed')
        pending.append((component, path, part))
    paths, weights, added = [], [], 0
    for index, (component, path, part) in enumerate(pending):
        if part.get('model') in {'lightgbm', 'xgboost', 'extra_trees'}:
            left = budget - (time.monotonic() - started) - 5
            remaining_models = sum(m.get('model') in {'lightgbm', 'xgboost', 'extra_trees'}
                                   for _, _, m in pending[index:])
            allowance = left / max(remaining_models, 1)
            if allowance < 3:
                raise b.MLRequestError('No time for all source ensemble replicas')
            result = _single(envelope, part, time.monotonic(), allowance, threads)
            paths.append(result['manifest_path'])
            added += 1
        else:
            paths.append(str(path))
        weights.append(component['weight'])
    if not added:
        raise b.MLRequestError('No supported independent source models to bag')
    result = run_ensemble(dict(envelope, request=dict(action='ensemble', manifest_paths=paths, weights=weights)),
                          started, budget, threads)
    checkpoint(envelope, result)
    return result
