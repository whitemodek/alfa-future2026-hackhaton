"""Offline ZIP checks for forward-date extrapolation and unseen text phrasing."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.run_local import run_task
from qa.transfer_benchmark import finish_provider, file_hash


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    answers = root / '_answers'
    answers.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261026)
    for name in ('reg_future_dates', 'text_phrasing'):
        folder = root / name
        folder.mkdir(exist_ok=True)
        if any(folder.iterdir()):
            raise ValueError('Use fresh inputs; do not overwrite a frozen benchmark')
        if name.startswith('reg'):
            dates = pd.date_range('2021-01-01', periods=360, freq='D')
            frame = pd.DataFrame(dict(id=np.arange(360), event_date=dates.astype(str),
                                      target=5 + 2.5 * np.arange(360)))
            train, test = frame.iloc[:280].copy(), frame.iloc[280:].copy()
            task = 'Regression. Target: target. Primary metric: RMSE. Predict future event_date.'
        else:
            positive = ['excellent', 'helpful', 'pleasant', 'reliable', 'friendly', 'wonderful']
            negative = ['terrible', 'broken', 'rude', 'unhelpful', 'awful', 'unreliable']
            neutral = ['service', 'delivery', 'today', 'package', 'support', 'order', 'experience', 'product']
            labels = rng.integers(0, 2, 1600)
            texts = []
            for label in labels:
                words = rng.choice(positive if label else negative, size=3).tolist()
                words += rng.choice(neutral, size=5).tolist()
                rng.shuffle(words)
                texts.append(' '.join(words))
            # Fixed label noise, unrelated to phrase structure.
            noisy = labels.copy()
            flipped = rng.random(len(labels)) < .08
            noisy[flipped] = 1 - noisy[flipped]
            frame = pd.DataFrame(dict(id=np.arange(1600), review=texts, target=noisy))
            train, test = frame.iloc[:1280].copy(), frame.iloc[1280:].copy()
            task = 'Binary classification. Target: target. Primary metric: Macro F1. Predict review labels.'
        train.to_csv(folder / 'train.csv', index=False)
        test.drop(columns='target').to_csv(folder / 'test.csv', index=False)
        test[['id', 'target']].to_csv(answers / (name + '.csv'), index=False)
        (answers / (name + '.json')).write_text(json.dumps(dict(kind=name)), encoding='utf-8')
        pd.DataFrame(dict(id=test.id, target=0.)).to_csv(folder / 'sample_submission.csv', index=False)
        (folder / 'task.md').write_text(task, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prepare(root / 'inputs')
    rows = []
    for name in ('reg_future_dates', 'text_phrasing'):
        folder = root / 'inputs' / name
        hashes = {p.name: file_hash(p) for p in folder.iterdir()}
        for label, archive in [('old', args.old), ('new', args.new)]:
            with finish_provider() as (url, requests):
                env = dict(OPENAI_API_KEY='local-only', OPENAI_MODEL='local', OPENAI_BASE_URL=url,
                           NO_PROXY='127.0.0.1,localhost', AGENT_TOTAL_SECONDS='1200',
                           AGENT_RESERVE_SECONDS='180', AGENT_CALL_SECONDS='300',
                           AGENT_BOOTSTRAP_SECONDS='30', AGENT_WARMUP_SECONDS='90',
                           AGENT_THREADS='2', AGENT_API_SECONDS='5', AGENT_MAX_STEPS='2', AGENT_RETRIES='0')
                with patch.dict(os.environ, env):
                    result = run_task(archive.resolve(), folder, 600)
            data = Path(result['work_dir']) / 'data'
            assert {p.name for p in folder.iterdir()} == set(hashes)
            assert all(file_hash(folder / name) == digest and file_hash(data / name) == digest
                       for name, digest in hashes.items())
            result.update(name=name, release=label, zip_sha256=file_hash(archive),
                          external_api_requests=0, local_api_requests=len(requests), inputs_unchanged=True)
            rows.append(result)
            (root / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps({key: result.get(key) for key in
                              ('name', 'release', 'success', 'minutes', 'check')}), flush=True)


if __name__ == '__main__':
    main()
