"""Fixed .75 tree / .25 linear candidate on fresh public regression splits."""
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import pandas as pd

import qa.transfer_benchmark as transfer
from qa.benchmark_confirmed import metric


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / '.test-temp/linear-blend-pilot'
    out.mkdir(exist_ok=False)
    names = ('energy_efficiency', 'wine_quality_red', 'diabetes')
    all_specs = json.loads((root / '.test-temp/transfer-specs.json').read_text())
    specs = [next(s for s in all_specs if s['name'] == name) for name in names]
    protocol = dict(seed=20261103, cases=names, weights=[.75, .25], api_calls=0, platform_submissions=0,
        selection='One fixed blend, accepted only on the existing validation split before any outer answers are read',
        gate='At least two relative outer RMSE gains >=.5%; no degradation >2%; all three completed',
        limitation='New splits on already researched datasets, exploratory pilot; not an official score',
        source_sha256={name: transfer.file_hash(root / name) for name in ('ml_portfolio.py', 'qa/benchmark_linear_blend.py')},
        dataset_sha256={s['name']: transfer.file_hash(s['path']) for s in specs})
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = dict(protocol=protocol, completed=False, cases=[])
    started = time.monotonic()
    import ml_backend as b
    from threadpoolctl import threadpool_limits

    def save():
        report['seconds'] = round(time.monotonic() - started, 3)
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    for spec in specs:
        print('START', spec['name'], flush=True)
        with patch.object(transfer, 'SEED', protocol['seed']):
            folder = transfer.prepare_case(spec, out)
        env = dict(task_dir=str(folder), task_path=str(folder / 'task.md'), threads=2, time_budget_seconds=90)
        with threadpool_limits(limits=2):
            response = b.run(dict(env, request=dict(action='portfolio', linear_blend=True)))
            control = response['linear_blend_control']
            same = response['manifest_path'] == control['manifest_path']
            original = dict(control)
            if control['fit_scope'] == 'holdout_train':
                control = b.run(dict(env, time_budget_seconds=30,
                    request=dict(action='refit', manifest_path=control['manifest_path'])))
            if same:
                response = control
            elif response['fit_scope'] == 'holdout_train':
                response = b.run(dict(env, time_budget_seconds=30,
                    request=dict(action='refit', manifest_path=response['manifest_path'])))
        # Freeze selected models before evaluation.
        (out / (spec['name'] + '-decision.json')).write_text(json.dumps(dict(
            control=control, selected=response, original_control=original), indent=2), encoding='utf-8')
        truth = pd.read_csv(out / 'tasks/_answers' / (spec['name'] + '.csv'))[spec['target']].to_numpy()
        scores = {name: metric(spec, truth, pd.read_csv(record['candidate_path'])[spec['target']].to_numpy())
                  for name, record in (('control', control), ('selected', response))}
        row = dict(name=spec['name'], scores=scores, accepted=not same,
                   relative_gain=(scores['control'] - scores['selected']) / scores['control'])
        report['cases'].append(row)
        save()
        print(json.dumps(row), flush=True)
    report['passed'] = (sum(r['relative_gain'] >= .005 for r in report['cases']) >= 2
                        and all(r['relative_gain'] >= -.02 for r in report['cases']))
    report['completed'] = True
    save()
    print('PASSED', report['passed'], flush=True)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    main()
