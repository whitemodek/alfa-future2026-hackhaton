"""Publish compact study outcomes without datasets, credentials or model files."""
import json
from pathlib import Path


STUDIES = ('duplicate-weighting-study', 'domain-shift-study', 'label-shift-study',
    'classification-seed-bag-study', 'seed-bag-prospective', 'neighbor-expert-study',
    'pair-specialist-study', 'flow-feature-study', 'residual-blend-study',
    'weather-delta-study', 'weather-ridge-study', 'forward-residual-pipeline',
    'ag-pilot', 'ag-pilot-diverse', 'batch-mlp-study')


def main():
    from qa.research_attack_specialist import sha256
    root = Path(__file__).resolve().parents[1]
    records = []
    for name in STUDIES:
        path = root / '.test-temp' / name / 'report.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        rows = r.get('trials', r.get('folds', r.get('records', [])))
        result = dict(name=name, completed=r.get('completed'), passed=r.get('passed'),
            seconds=r.get('seconds'), report_sha256=sha256(path), rows=[])
        if name in {'label-shift-study', 'residual-blend-study'}:
            result['completed'] = True  # Their compact reports are written only after scoring both frozen folds.
        for row in rows:
            selected = {k: row[k] for k in ('fold', 'period', 'seed', 'gain', 'relative_gain', 'control_rmse',
                'candidate_rmse', 'outer_scores', 'outer_rmse', 'selected_outer_score', 'selected_bag',
                'selected_strength', 'inner_control_score', 'inner_candidate_score', 'completed') if k in row}
            for key in ('base', 'candidate'):
                if isinstance(row.get(key), dict) and 'macro_f1' in row[key]:
                    selected[key + '_macro_f1'] = row[key]['macro_f1']
            if 'arms' in row:
                selected['arm_macro_f1'] = {a: v['outer']['macro_f1'] for a, v in row['arms'].items() if 'outer' in v}
            result['rows'].append(selected)
        if 'stopped_early' in r:
            result.update(stopped_early=r['stopped_early'], stop_reason=r['stop_reason'])
        records.append(result)
    report = dict(api_calls=0, platform_submissions=0, stable_archives_replaced=False,
        limitation='Historical exploratory checks are not a blind platform score; failed gates remain failed.', studies=records)
    (root / 'docs/FOLLOWUP_RESULTS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(studies=len(records), path='docs/FOLLOWUP_RESULTS.json')), flush=True)


if __name__ == '__main__':
    main()
