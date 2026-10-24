"""Report cumulative charges, including reservations for ambiguous API failures.

python qa/check_tokens.py TASK_DIR [RUB_PER_MILLION] [--max-tokens N] [--json]
"""
import argparse
import json
from pathlib import Path


def summarize(task_dir):
    runs = []
    for state in sorted(Path(task_dir).glob('.agent-*')):
        log = state / 'events.jsonl'
        if not log.is_file():
            continue
        events = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()
                  if line.strip()]
        calls = [e for e in events if e['event'] in ('api', 'api_error')]
        charges = [e['charged_tokens'] for e in calls if 'charged_tokens' in e]
        if any(type(value) is not int or value < 0 for value in charges):
            raise ValueError(f'Invalid token accounting in {log}')
        unknown_errors = sum(e['event'] == 'api_error' and 'charged_tokens' not in e
                             for e in calls)
        last = events[-1] if events else {}
        runs.append({'state': state.name, 'api_attempts': len(calls),
                     'api_errors': sum(e['event'] == 'api_error' for e in calls),
                     'charged_tokens': max(charges, default=0),
                     'unknown_errors': unknown_errors,
                     'elapsed_seconds': last.get('elapsed', 0),
                     'last_event': last.get('event'), 'reason': last.get('reason')})
    return {'task_dir': str(Path(task_dir).resolve()), 'runs': runs,
            'charged_tokens': sum(r['charged_tokens'] for r in runs),
            'accounting_complete': not any(r['unknown_errors'] for r in runs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_dir', type=Path)
    parser.add_argument('price', nargs='?', type=float)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if not args.task_dir.is_dir():
        parser.error('task_dir must be an existing directory')
    report = summarize(args.task_dir)
    if args.price is not None:
        report['estimated_rubles'] = report['charged_tokens'] * args.price / 1_000_000
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for run in report['runs']:
            print(f"{run['state']}: API {run['api_attempts']}, errors {run['api_errors']}, "
                  f"tokens {run['charged_tokens']:,}, {run['last_event']}")
        print(f"Total: {report['charged_tokens']:,} tokens across {len(report['runs'])} runs")
        if not report['accounting_complete']:
            print('Some old API errors lack a charge; the total is a lower bound.')
        if args.price is not None:
            print(f"Estimated cost: {report['estimated_rubles']:.2f} RUB")
    if args.max_tokens is not None:
        return int(not report['accounting_complete'] or report['charged_tokens'] > args.max_tokens)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
