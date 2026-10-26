"""One fixed half-blend of existing direct-target and residual-target forecasts.

Both original models used strictly past entity history. A residual-only
replacement previously failed; this distinct variance-reduction proposal is
fixed before scoring and needs exact-runtime follow-up on success.
"""
import json
from pathlib import Path

import numpy as np

from qa.research_attack_specialist import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.test-temp'
    out = scratch / 'residual-blend-study'
    out.mkdir(exist_ok=False)
    paths = [scratch / f'building-delta-report-artifacts/period{period}.npz' for period in (5, 4)]
    protocol = dict(periods=[5, 4], residual_candidate_weight=.5, grid_search=False,
        gate='RMSE improves on BOTH periods by >=.1% before exact-runtime follow-up',
        api_calls=0, platform_submissions=0, source_sha256=sha256(__file__),
        cache_sha256={str(p.relative_to(scratch)): sha256(p) for p in paths},
        limitation='Historical temporal folds; exploratory combination of frozen previously inspected forecasts, '
                   'not a blind score and not a retrospectively passed residual-only experiment.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    forecasts = []
    for period, path in zip((5, 4), paths):
        with np.load(path, allow_pickle=False) as a:
            base, candidate = a['baseline'].copy(), a['candidate'].copy()
            if base.shape != candidate.shape or not np.isfinite(base).all() or not np.isfinite(candidate).all():
                raise ValueError('Nonfinite or misaligned frozen forecasts')
            prediction = .5 * base + .5 * candidate
            np.savez_compressed(out / f'period{period}-frozen.npz', prediction=prediction)
            forecasts.append((period, path, base, prediction))
    report = dict(protocol=protocol, records=[])
    # First outer truth access: all forecasts and the weight have already been frozen.
    for period, path, base, prediction in forecasts:
        with np.load(path, allow_pickle=False) as a:
            truth = a['truth'].copy()
        old = float(np.sqrt(np.mean((base - truth) ** 2)))
        new = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        report['records'].append(dict(period=period, control_rmse=old, candidate_rmse=new, relative_gain=(old - new) / old))
    report['passed'] = all(r['relative_gain'] >= .001 for r in report['records'])
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
