"""Paired offline regression: plain assistant text must not skip real training.

This exercises a scripted provider, not LLM quality or the platform leaderboard.
Only agent.py differs between the frozen prior ZIP and the challenger.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from pack_solution import build
from qa.run_local import run_task


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def challenger(base, output, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        source = Path(folder).resolve()
        assert source.is_relative_to(scratch.resolve())
        with zipfile.ZipFile(base) as archive:
            names = archive.namelist()
            if (len(set(names)) != len(names) or 'agent.py' not in names
                    or any('/' in n or '\\' in n or ':' in n
                           or not (n.endswith('.py') or n == 'requirements.txt') for n in names)):
                raise ValueError('Expected a flat runtime archive')
            for name in names:
                (source / name).write_bytes(archive.read(name))
        (source / 'agent.py').write_bytes((Path(__file__).resolve().parents[1] / 'agent.py').read_bytes())
        result = build(output, source)
    with zipfile.ZipFile(base) as old, zipfile.ZipFile(output) as new:
        assert set(old.namelist()) == set(new.namelist())
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        assert changed == ['agent.py'], changed
    return result


def check(base, candidate, root):
    # Reuse the existing local HTTP fixture; no provider key is read.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    from test_cli import fake_api, tool_call
    root.mkdir(parents=True, exist_ok=True)
    folder = root / 'reg_completion'
    folder.mkdir()
    rng = np.random.default_rng(20261026)
    x = rng.normal(size=550)
    y = 6 * x + .2 * rng.normal(size=550)
    pd.DataFrame(dict(id=np.arange(400), x=x[:400], target=y[:400])).to_csv(folder / 'train.csv', index=False)
    pd.DataFrame(dict(id=np.arange(400, 550), x=x[400:])).to_csv(folder / 'test.csv', index=False)
    pd.DataFrame(dict(id=np.arange(400, 550), target=0.)).to_csv(folder / 'sample_submission.csv', index=False)
    (folder / 'task.md').write_text('Regression. Target: target. Metric: RMSE.', encoding='utf-8')
    answers = root / '_answers'
    answers.mkdir()
    pd.DataFrame(dict(id=np.arange(400, 550), target=y[400:])).to_csv(answers / 'reg_completion.csv', index=False)
    hashes = {path.name: digest(path) for path in folder.iterdir()}
    records = []
    for label, archive in [('prior', base), ('challenger', candidate)]:
        responses = [dict(role='assistant', content='I will train a model after inspecting the baseline.'),
            tool_call('run_backend', dict(request=dict(action='experiment', model='lightgbm',
                iterations=100, learning_rate=.1, early_stopping_rounds=20), timeout_seconds=30)),
            tool_call('finish', {})]
        with fake_api(responses) as (url, requests):
            env = dict(OPENAI_API_KEY='local-only', OPENAI_MODEL='scripted-local', OPENAI_BASE_URL=url,
                NO_PROXY='127.0.0.1,localhost', AGENT_TOTAL_SECONDS='120', AGENT_RESERVE_SECONDS='15',
                AGENT_BOOTSTRAP_SECONDS='15', AGENT_WARMUP_SECONDS='0', AGENT_THREADS='2',
                AGENT_CALL_SECONDS='30', AGENT_API_SECONDS='5', AGENT_MAX_STEPS='5', AGENT_RETRIES='0')
            with patch.dict(os.environ, env):
                result = run_task(archive, folder, 180)
        assert result['success'], result
        data = Path(result['work_dir']) / 'data'
        assert all(digest(folder / name) == value and digest(data / name) == value
                   for name, value in hashes.items())
        result.update(release=label, zip_sha256=digest(archive), local_requests=len(requests), external_api_requests=0)
        records.append(result)
    assert records[0]['local_requests'] == 1
    assert records[1]['local_requests'] == 3
    assert records[1]['check']['metric']['rmse'] < records[0]['check']['metric']['rmse'] * .3
    (root / 'report.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('dist/solution-agent-fix.zip'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(challenger(args.base, args.output, args.root / 'build'), indent=2), flush=True)
    for result in check(args.base.resolve(), args.output.resolve(), args.root / 'check'):
        print(json.dumps({k: result[k] for k in ('release', 'success', 'local_requests', 'check')}), flush=True)
