"""Offline paired pilot of optional nested selection; no API or platform calls."""
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

import qa.transfer_benchmark as transfer
from qa.transfer_benchmark import file_hash


CASES = ('energy_efficiency', 'wine_quality_red', 'diabetes', 'wine_quality_white', 'digits')
SEED = 20261102


def metric(spec, truth, prediction):
    from sklearn.metrics import f1_score, mean_squared_error
    if spec['metric'] == 'rmse':
        return float(np.sqrt(mean_squared_error(truth, prediction.astype(float))))
    return float(f1_score(truth.astype(str), prediction.astype(str), labels=np.unique(truth.astype(str)),
                          average='macro', zero_division=0))


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / '.test-temp/confirmed-pilot'
    out.mkdir(exist_ok=False)
    specs = json.loads((root / '.test-temp/transfer-specs.json').read_text())
    specs = [next(s for s in specs if s['name'] == name) for name in CASES]
    protocol = dict(seed=SEED, cases=CASES, selection='Augment current native portfolio only if confirmation '
        'score improves on the SAME original split; never select with outer answers',
        budgets=dict(native=90, confirmed=90, refit=30), threads=2, api_calls=0, platform_submissions=0,
        gate='At least 4/5 nonworse outer scores; 2 gains >=.002 F1 or .5% RMSE; '
             'no loss >.01 F1 or >2% RMSE. Otherwise no default enablement.',
        limitation='New split on previously explored public datasets; unequal extra compute; not a blind leaderboard estimate',
        source_sha256={name: file_hash(root / name) for name in
            ('ml_confirmed.py', 'ml_backend.py', 'ml_models.py', 'ml_portfolio.py', 'qa/benchmark_confirmed.py')},
        dataset_sha256={s['name']: file_hash(s['path']) for s in specs})
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, completed=False, cases=[])
    started = time.monotonic()

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    import ml_backend as b
    from ml_portfolio import improves
    from threadpoolctl import threadpool_limits
    for spec in specs:
        print('START', spec['name'], flush=True)
        case_started = time.monotonic()
        with patch.object(transfer, 'SEED', SEED):
            folder = transfer.prepare_case(spec, out)
        env = dict(task_dir=str(folder), task_path=str(folder / 'task.md'), threads=2, time_budget_seconds=90)
        with threadpool_limits(limits=2):
            control = b.run(dict(env, request=dict(action='portfolio')))
            if control.get('fit_scope') == 'holdout_train':
                control = b.run(dict(env, time_budget_seconds=30,
                                     request=dict(action='refit', manifest_path=control['manifest_path'])))
            control_file = out / (spec['name'] + '-control.csv')
            control_file.write_bytes(Path(control['candidate_path']).read_bytes())
            error = None
            try:
                candidate = b.run(dict(env, request=dict(action='confirmed')))
                if candidate.get('fit_scope') == 'holdout_train':
                    candidate = b.run(dict(env, time_budget_seconds=30,
                        request=dict(action='refit', manifest_path=candidate['manifest_path'])))
                if any(candidate[k] != control[k] for k in ('metric', 'direction', 'validation_id')):
                    raise ValueError('Candidate and control validation contracts differ')
                chosen = candidate if improves(candidate, control) else control
            except Exception as exc:
                candidate, chosen, error = None, control, type(exc).__name__ + ': ' + str(exc)[:200]
        # Freeze predictions and selection BEFORE opening outer answers.
        decision = dict(control=control, candidate=candidate, selected='candidate' if chosen is candidate else 'control', error=error)
        (out / (spec['name'] + '-decision.json')).write_text(json.dumps(decision, indent=2), encoding='utf-8')
        predictions = {}
        template = pd.read_csv(folder / 'sample_submission.csv')
        for label, response in (('control', control), ('candidate', candidate), ('selected', chosen)):
            if response is None:
                continue
            frame = pd.read_csv(response['candidate_path'])
            if list(frame) != list(template) or not np.array_equal(frame['id'], template['id']):
                raise ValueError('Prediction schema or row alignment changed')
            predictions[label] = frame[spec['target']].to_numpy()
        truth = pd.read_csv(out / 'tasks/_answers' / (spec['name'] + '.csv'))[spec['target']].to_numpy()
        scores = {label: metric(spec, truth, prediction) for label, prediction in predictions.items()}
        gain = ((scores['selected'] - scores['control']) if spec['metric'] == 'macro_f1'
                else (scores['control'] - scores['selected']) / scores['control'])
        case = dict(name=spec['name'], metric=spec['metric'], scores=scores,
            selected=decision['selected'], gain=gain, confirmation=candidate.get('confirmation') if candidate else None,
            error=error, seconds=round(time.monotonic() - case_started, 3))
        report['cases'].append(case)
        save()
        print(json.dumps(case), flush=True)
    cases = report['cases']
    report['passed'] = (sum(c['gain'] >= -1e-12 for c in cases) >= 4 and
        sum(c['gain'] >= (.002 if c['metric'] == 'macro_f1' else .005) for c in cases) >= 2 and
        all(c['gain'] >= (-.01 if c['metric'] == 'macro_f1' else -.02) for c in cases) and
        all(c['error'] is None for c in cases))
    report['completed'] = True
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    main()
