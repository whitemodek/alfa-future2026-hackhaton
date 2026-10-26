"""Two small public-metadata requests; records counts only, never reasoning text."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

from qa.live_baseline_comparison import MeteredClient, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hard', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    trial = root / '.test-temp/live-baseline-comparison'
    first = json.loads((trial / 'ledger.json').read_text())
    lean = json.loads((trial / 'lean-followup/ledger.json').read_text())
    assert first['completed'] and lean['completed']
    spent_native = first['arms']['native']['charged_tokens'] + lean['charged_tokens']
    prior = lean['cumulative_charged_tokens']
    if args.hard:
        soft = json.loads((trial / 'thinking-probe/ledger.json').read_text())
        assert soft['completed']
        spent_native += soft['charged_tokens']
        prior = soft['cumulative_charged_tokens']
    cap = min(2200, 15000 - spent_native)
    if args.hard:
        cap = min(cap, 1600)
    assert cap > 0 and prior + cap <= 160000
    out = trial / ('thinking-probe-hard' if args.hard else 'thinking-probe')
    out.mkdir(exist_ok=False)
    ledger = dict(allocated=cap, charged_tokens=0, completed=False, prior_native=spent_native)
    save(out / 'ledger.json', ledger)
    environment = {}
    for line in (root / '.env').read_text(encoding='utf-8-sig').splitlines():
        name, sep, value = line.strip().removeprefix('export ').partition('=')
        if sep and name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
            environment[name] = value.strip().strip('\"\'')
    assert urlparse(environment['OPENAI_BASE_URL']).hostname == 'changellenge.up.railway.app'
    tools = [dict(type='function', function=dict(name='finish', description='Publish the saved candidate.',
             parameters=dict(type='object', properties={}, additionalProperties=False)))]
    system = 'You operate an ML pipeline. Use tools. Never reveal private data.'
    instruction = ('The public wine-quality task has seven classes and uses Macro F1. '
        'Three tree models have been compared honestly and no justified new experiment remains. '
        'The best validated CSV is saved. End the task by calling finish.')
    def updated(client):
        ledger.update(charged_tokens=client.spent, events=client.events)
        save(out / 'ledger.json', ledger)
    modes = ('hard_none',) if args.hard else ('default', 'soft_no_think')
    save(out / 'protocol.json', dict(order=modes, completion_tokens=256,
         only_difference='Prefix /no_think in the user instruction', cap=cap, platform_submissions=0,
         warning='Small operational probe, not an ML-quality comparison; no reasoning text retained.'))
    from openai import OpenAI
    reports = []
    with OpenAI(api_key=environment['OPENAI_API_KEY'], base_url=environment['OPENAI_BASE_URL'], max_retries=0) as remote:
        metered = MeteredClient(remote, cap, updated, time.monotonic() + 120)
        try:
            for mode in modes:
                prompt = '/no_think\n' + instruction if mode == 'soft_no_think' else instruction
                start = time.monotonic()
                extra = dict(extra_body=dict(reasoning=dict(effort='none'))) if args.hard else {}
                response = metered.create(model=environment['OPENAI_MODEL'], tools=tools,
                    parallel_tool_calls=False, max_tokens=256,
                    messages=[dict(role='system', content=system), dict(role='user', content=prompt)], **extra)
                report = dict(mode=mode, seconds=round(time.monotonic() - start, 3), **metered.events[-1])
                reports.append(report)
                save(out / 'report.json', reports)
                print(json.dumps(report), flush=True)
        finally:
            ledger.update(completed=True, cumulative_charged_tokens=prior + metered.spent,
                          combined_native_charged_tokens=spent_native + metered.spent)
            save(out / 'ledger.json', ledger)


if __name__ == '__main__':
    main()
