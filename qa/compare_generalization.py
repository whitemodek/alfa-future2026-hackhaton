"""Paired, offline ZIP comparison on frozen synthetic and optional public tasks.

Outer labels stay outside each worker. Both releases see identical inputs and
budgets. Selection never uses the outer score. No external API requests.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1, make_regression

from qa.transfer_benchmark import run_case


def synthetic_specs(root):
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    specs = []
    for name in ('linear_numeric', 'linear_mixed', 'nonlinear', 'imbalanced'):
        classification = name == 'imbalanced'
        if classification:
            x, y = make_classification(n_samples=1800, n_features=14, n_informative=8,
                                        n_redundant=2, weights=[.9, .1], class_sep=1., random_state=20261026)
        elif name == 'nonlinear':
            x, y = make_friedman1(n_samples=1200, n_features=10, noise=1., random_state=20261026)
        else:
            x, y = make_regression(n_samples=1200, n_features=14, n_informative=10,
                                   noise=8., random_state=20261026)
        frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])
        if name == 'linear_mixed':
            category = np.random.default_rng(20261026).integers(0, 30, size=len(frame))
            frame['category'] = ['kind' + str(v) for v in category]
            y = y + 12 * category
            frame.loc[frame.index[::19], 'x0'] = np.nan
        frame['target'] = y
        path = raw / (name + '.csv')
        if path.exists():
            raise ValueError('Use a new root; benchmark sources must stay frozen')
        frame.to_csv(path, index=False)
        specs.append(dict(name=name, path=str(path.resolve()), target='target',
                          problem_type='classification' if classification else 'regression',
                          metric='roc_auc' if classification else 'rmse',
                          source_url='synthetic:sklearn:seed20261026'))
    return specs


def compare(root, releases, specs, budget, threads):
    rows = []
    for spec in specs:
        for label, archive in releases:
            output = root / label / 'results' / (spec['name'] + '.json')
            if output.exists():
                raise ValueError('Do not re-evaluate an already observed outer holdout')
            result = run_case(spec, root / label, archive.resolve(), budget, threads)
            row = {key: result.get(key) for key in ('name', 'success', 'minutes', 'check', 'zip_sha256')}
            row['release'] = label
            rows.append(row)
            (root / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--specs', type=Path)
    parser.add_argument('--budget', type=int, default=90)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 30 <= args.budget <= 300 or not 1 <= args.threads <= 4:
        parser.error('budget must be30..300 and threads1..4')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.specs.read_text(encoding='utf-8-sig')) if args.specs else synthetic_specs(root)
    compare(root, [('old', args.old), ('new', args.new)], specs, args.budget, args.threads)


if __name__ == '__main__':
    main()
