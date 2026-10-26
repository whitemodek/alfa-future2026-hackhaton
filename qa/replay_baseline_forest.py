"""Offline replay of the live baseline's fixed RF300 proposal, not its score."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from qa.live_baseline_comparison import digest, evaluate, save


def main():
    root = Path(__file__).resolve().parents[1]
    trial = root / '.test-temp/live-baseline-comparison'
    out = trial / 'rf-candidate'
    out.mkdir(exist_ok=False)
    data = trial / 'native'
    sys.path.insert(0, str(trial / 'native-code'))
    import ml_backend as backend
    from sklearn.ensemble import RandomForestClassifier
    from threadpoolctl import threadpool_limits
    import joblib
    request = dict(action='inspect', problem_type='classification', metric='macro_f1', target='quality')
    problem = backend.load_problem(dict(task_dir=str(data), task_path=str(data / 'task.md'), request=request))
    ti, vi = backend._split(problem, request)
    x, tx, cats = backend._features(problem)
    if cats or x.isna().any().any() or tx.isna().any().any():
        raise ValueError('The proposed bare RF requires this fixed numeric complete case')
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    assert not set(groups[ti]) & set(groups[vi])
    parameters = dict(n_estimators=300, random_state=42, n_jobs=2)
    save(out / 'protocol.json', dict(parameters=parameters,
        source='Exact model preset proposed in baseline profiled-followup trace; no outer tuning',
        trace_sha256=digest(trial / 'profiled-followup/baseline-trace.json'),
        validation=problem['validation'], validation_id=backend._validation_id(problem, ti, vi),
        train_sha256=digest(data / 'train.csv'), external_api_requests=0,
        limitation='Offline model replay after a non-completing live baseline; not official baseline B_F1.'))
    started = time.monotonic()
    with threadpool_limits(limits=2):
        model = RandomForestClassifier(**parameters).fit(x.iloc[ti].to_numpy(), problem['y'].iloc[ti])
        prediction = model.predict(x.iloc[vi].to_numpy())
        probability = model.predict_proba(x.iloc[vi].to_numpy())
        validation_score = backend._score(problem, problem['y'].iloc[vi], prediction, probability)
        final = RandomForestClassifier(**parameters).fit(x.to_numpy(), problem['y'])
        encoded = final.predict(tx.to_numpy())
    decoded = np.asarray(problem['classes'])[encoded.astype(int)]
    template = pd.read_csv(data / 'sample_submission.csv')
    template.to_csv(out / 'sample_submission.csv', index=False)
    result = template.copy()
    result['quality'] = decoded
    result.to_csv(out / 'submission.csv', index=False)
    joblib.dump(final, out / 'model.joblib')
    # Outer answers enter evaluation only after all model fitting has completed.
    truth = np.load(trial / 'final-checker.npz', allow_pickle=False)['truth']
    original_report = json.loads((trial / 'report.json').read_text())
    old = next(r['evaluation'] for r in original_report['arms'] if r['arm'] == 'native')
    report = dict(validation_score=validation_score, outer=evaluate(out, truth, problem['classes']),
        prior_native_outer=old, seconds=round(time.monotonic() - started, 3),
        model_sha256=digest(out / 'model.joblib'), protocol_sha256=digest(out / 'protocol.json'),
        external_api_requests=0, platform_submissions=0)
    save(out / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
