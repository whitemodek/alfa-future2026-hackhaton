"""Confirm a predeclared causal-stats candidate in fixed history blends.

No training or configuration selection occurs here. This compares both saved
nested candidates and the actual prior runtime, which used a different
outer-early-stopping protocol and must not be called the same control.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from qa.research_causal_stats import _sha, _save


def confirm(root, study, release, output):
    import ml_backend as b
    import ml_history as h
    from ml_lag import causal_lags

    root, study, release, output = map(Path, (root, study, release, output))
    report = json.loads(study.read_text(encoding='utf-8'))
    if not report.get('complete') or not report.get('passes_continuation_gate'):
        raise ValueError('The predeclared standalone gate was not passed')
    protocol = dict(weight_history=.5, weight_lag=.5, variants=['control', 'stats6'],
        outer_periods=[5, 4], source_sha256=_sha(__file__), study_sha256=_sha(study),
        release_sha256=_sha(release),
        continuation='Both fixed stats6 pipelines must beat their corresponding actual '
                     'current runtime before any integration; otherwise research only.',
        limitation='Outer folds were previously inspected; exploratory evidence only. '
                   'Actual runtime used outer early stopping, unlike nested research candidates.')
    protocol_path = output.with_name(output.stem + '-protocol.json')
    _save(protocol_path, protocol)
    envelope = dict(task_dir=str(root), task_path=str(root / 'task.md'), request={})
    p = b.load_problem(envelope)
    columns, _ = h.entity_columns(p, {})
    time_column = 'Year_Factor'
    raw, y = p['train'], p['y']
    records = []
    artifacts = output.with_name(output.stem + '-artifacts')
    artifacts.mkdir(exist_ok=True)

    release_record = json.loads(release.read_text(encoding='utf-8'))[0]
    release_root = Path(release_record['work_dir']) / 'data'
    journal = next(release_root.glob('.agent-*/events.jsonl'))
    events = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
    results = [e['result'].get('response', {}) for e in events if e['event'] == 'backend']
    lag_result = [r for r in results if r.get('model') == 'lag' and r.get('fit_scope') == 'full_train'][-1]
    lag_path = Path(lag_result['manifest_path'])
    lag = json.loads(lag_path.read_text(encoding='utf-8'))
    history_path = Path(lag['history_manifest_path'])
    history = json.loads(history_path.read_text(encoding='utf-8'))
    base_path = Path(history['base_manifest_path'])
    base = json.loads(base_path.read_text(encoding='utf-8'))
    own_env = dict(task_dir=str(release_root), task_path=str(release_root / 'task.md'), request={})
    own, train_i, valid_i, base_prediction, _ = h._base_predictions(own_env, base,
                                                                  time.monotonic(), 150)
    primary_history, primary_diag = h.blend_past(own['train'].iloc[train_i], own['y'].iloc[train_i],
        own['train'].iloc[valid_i], base_prediction, columns, time_column, .5)
    primary_history_rmse = float(np.sqrt(np.mean((primary_history - own['y'].iloc[valid_i]) ** 2)))
    if not np.isclose(primary_history_rmse, history['score'], rtol=1e-10, atol=1e-10):
        raise ValueError('History holdout models did not reproduce saved score')
    if b._file_hash(own['train_path']) != b._file_hash(p['train_path']):
        raise ValueError('Primary and research training data differ')

    for period in (5, 4):
        ti = np.flatnonzero(raw[time_column].to_numpy() < period)
        vi = np.flatnonzero(raw[time_column].to_numpy() == period)
        truth = y.iloc[vi].to_numpy()
        hashes = {}
        if period == 5:
            np.testing.assert_array_equal(vi, valid_i)
            fixed_history = primary_history
            actual_current = float(release_record['best']['score'])
            hashes.update(history_manifest=dict(path=str(history_path), sha256=_sha(history_path)),
                          base_manifest=dict(path=str(base_path), sha256=_sha(base_path)),
                          lag_manifest=dict(path=str(lag_path), sha256=_sha(lag_path)))
        else:
            cache = root.parents[1] / 'night-regression'
            weights = {'cb_d8': .5, 'adapter_d4': .25, 'adapter_d4_s17': .25}
            baseline = np.zeros(len(vi))
            for name, weight in weights.items():
                path = cache / f'{name}-period{period}.npy'
                baseline += weight * np.load(path, allow_pickle=False)
                hashes[name] = dict(path=str(path), sha256=_sha(path))
            fixed_history, diag = h.blend_past(raw.iloc[ti], y.iloc[ti], raw.iloc[vi],
                                             baseline, columns, time_column, .5)
            actual_lag_path = root.parents[1] / 'late-lag-feature' / 'xgboost-lagTrue-period4.npy'
            actual_lag = np.load(actual_lag_path, allow_pickle=False)
            actual_current = float(np.sqrt(np.mean((.5 * fixed_history + .5 * actual_lag - truth) ** 2)))
            hashes['actual_lag'] = dict(path=str(actual_lag_path), sha256=_sha(actual_lag_path))
            if not np.isclose(actual_current, 32.64325416847114, rtol=1e-9, atol=1e-9):
                raise ValueError('Secondary frozen runtime no longer reproduces expected score')
        record = dict(outer_period=period, actual_current_runtime_rmse=actual_current,
                      fixed_history_rmse=float(np.sqrt(np.mean((fixed_history - truth) ** 2))),
                      provenance=hashes, variants={})
        arrays = dict(outer_indices=vi, truth=truth, history_prediction=fixed_history)
        for variant in ('control', 'stats6'):
            source = next(r for r in report['records']
                          if r['outer_period'] == period and r['variant'] == variant)
            artifact = source['artifacts']['arrays']
            path = Path(artifact['path'])
            if _sha(path) != artifact['sha256']:
                raise ValueError('Saved research arrays changed')
            saved = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(saved['outer_validation_indices'], vi)
            np.testing.assert_allclose(saved['outer_truth'], truth, rtol=0, atol=0)
            prediction = .5 * fixed_history + .5 * saved['prediction']
            score = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            record['variants'][variant] = dict(pipeline_rmse=score,
                delta_vs_actual_runtime=score - actual_current,
                arrays_sha256=artifact['sha256'], selected_iterations=source['selected_iterations'])
            arrays[variant + '_pipeline_prediction'] = prediction
        record['paired_delta'] = (record['variants']['stats6']['pipeline_rmse']
                                  - record['variants']['control']['pipeline_rmse'])
        artifact_path = artifacts / f'period{period}.npz'
        np.savez_compressed(artifact_path, **arrays)
        record['arrays'] = dict(path=str(artifact_path), sha256=_sha(artifact_path))
        records.append(record)
    result = dict(protocol_sha256=_sha(protocol_path), records=records, complete=True,
        production_changed=False, external_api_requests=0,
        paired_improves_both=all(r['paired_delta'] < 0 for r in records),
        beats_current_both=all(r['variants']['stats6']['delta_vs_actual_runtime'] < 0 for r in records))
    result['recommendation'] = ('Eligible for a separate integration review, not automatic promotion'
        if result['beats_current_both'] else 'Research only: candidate fails the frozen current-runtime gate')
    _save(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.test-temp/organizer/regression'))
    parser.add_argument('--study', type=Path, default=Path('.test-temp/causal-stats-replay-report.json'))
    parser.add_argument('--release', type=Path, default=Path('.test-temp/platform-upgrade-report.json'))
    parser.add_argument('--output', type=Path, default=Path('.test-temp/causal-stats-pipeline-report.json'))
    args = parser.parse_args()
    print(json.dumps(confirm(args.root.resolve(), args.study.resolve(), args.release.resolve(),
                             args.output.resolve()), indent=2))
