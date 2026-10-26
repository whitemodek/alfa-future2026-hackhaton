"""One budgeted real-LLM pilot. No leaderboard submissions or runtime edits.

Baseline prompt/executor are extracted from the official source. MCP transport
is replaced by synchronous direct tool calls and shared resource limits.
This is a baseline-style adaptation, not an official baseline score.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd


class BudgetLimit(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class MeteredClient:
    """Same conservative reservation and cap for both arms; persist before send."""
    def __init__(self, remote, cap, callback, deadline):
        self.remote, self.cap, self.callback, self.deadline = remote, cap, callback, deadline
        self.spent, self.calls, self.events = 0, 0, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        completion = min(int(kwargs.get('max_tokens', 1536)), 1536)
        bound = len(json.dumps([kwargs['messages'], kwargs['tools']], ensure_ascii=False).encode()) + 512
        reservation = bound + completion
        if self.calls >= 6 or self.spent + reservation > self.cap or time.monotonic() + 2 >= self.deadline:
            raise BudgetLimit('Common API budget or deadline reached')
        self.calls += 1
        self.spent += reservation
        event = dict(call=self.calls, reserved=reservation, charged=reservation, status='reserved')
        self.events.append(event)
        self.callback(self)
        kwargs.update(max_tokens=completion, timeout=min(45, self.deadline - time.monotonic()))
        try:
            result = self.remote.chat.completions.create(**kwargs)
        except Exception as exc:
            event.update(status='error', error_type=type(exc).__name__,
                         status_code=getattr(exc, 'status_code', None))
            self.callback(self)
            raise RuntimeError('Provider request failed: ' + type(exc).__name__) from None
        actual = getattr(getattr(result, 'usage', None), 'total_tokens', None)
        if type(actual) is int and actual >= 0:
            self.spent += actual - reservation
            event['charged'] = actual
        event.update(status='ok', tool_names=[c.function.name for c in
                     (getattr(result.choices[0].message, 'tool_calls', None) or [])],
                     content_characters=len(result.choices[0].message.content or ''),
                     finish_reason=result.choices[0].finish_reason if hasattr(result.choices[0], 'finish_reason') else None,
                     prompt_tokens=getattr(getattr(result, 'usage', None), 'prompt_tokens', None),
                     completion_tokens=getattr(getattr(result, 'usage', None), 'completion_tokens', None),
                     reasoning_tokens=getattr(getattr(getattr(result, 'usage', None), 'completion_tokens_details', None),
                                              'reasoning_tokens', None))
        self.callback(self)
        if self.spent > self.cap:
            raise BudgetLimit('Reported provider usage exceeded conservative cap; stop')
        return result


def baseline_parts(source, folder, deadline):
    """Extract original prompt/function, limiting subprocess wall time equally."""
    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_python')
    function.decorator_list = []
    agent = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_agent')
    messages = next(n for n in ast.walk(agent) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'messages' for t in n.targets)).value
    system = ast.literal_eval(messages.elts[0].values[1])

    def bounded_run(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(args[0], 0)
        kwargs['timeout'] = min(kwargs.get('timeout', 60), 60, remaining)
        kwargs['cwd'] = str(folder)
        kwargs['env'] = dict(kwargs.get('env', {}), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                             MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        # The official Linux executor clears the environment. Windows requires
        # SYSTEMROOT for asyncio/socket initialization; no provider credentials pass.
        for name in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'MPLCONFIGDIR'):
            if name in os.environ:
                kwargs['env'][name] = os.environ[name]
        return subprocess.run(*args, **kwargs)

    namespace = dict(sys=sys, os=os, json=json,
        subprocess=SimpleNamespace(run=bounded_run, TimeoutExpired=subprocess.TimeoutExpired))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    execute = namespace['run_python']
    tools = [dict(type='function', function=dict(name='run_python', description=execute.__doc__,
        parameters=dict(type='object', properties=dict(code=dict(type='string')), required=['code'])))]
    return system, tools, execute


def run_baseline(folder, source, client, model, deadline, trace):
    system, tools, execute = baseline_parts(source, folder, deadline)
    messages = [dict(role='system', content=system),
                dict(role='user', content=(folder / 'task.md').read_text(encoding='utf-8'))]
    for step in range(6):
        response = client.create(model=model, messages=messages, tools=tools, parallel_tool_calls=False)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return dict(finished=True, reason='plain_completion', steps=step + 1)
        for call in message.tool_calls:
            if call.function.name != 'run_python':
                raise ValueError('Unknown baseline tool')
            try:
                arguments = json.loads(call.function.arguments)
                output = execute(**arguments)
                trace.append(dict(step=step + 1, code=arguments.get('code'), output=output))
            except ValueError as exc:
                output = 'Invalid tool arguments: ' + type(exc).__name__
            messages.append(dict(role='tool', tool_call_id=call.id, content=output))
    return dict(finished=False, reason='step_limit', steps=6)


def evaluate(folder, truth, classes):
    from sklearn.metrics import f1_score
    path = folder / 'submission.csv'
    if not path.is_file():
        return dict(format_ok=False, error='submission.csv missing')
    try:
        frame = pd.read_csv(path)
        template = pd.read_csv(folder / 'sample_submission.csv')
        if list(frame) != list(template) or len(frame) != len(template):
            raise ValueError('Schema or row count differs')
        if frame['id'].tolist() != template['id'].tolist():
            raise ValueError('ID order differs')
        prediction = pd.to_numeric(frame['quality'], errors='raise').to_numpy()
        if not np.isfinite(prediction).all() or not np.isin(prediction, classes).all():
            raise ValueError('Unknown or invalid original numeric label')
        return dict(format_ok=True, macro_f1=float(f1_score(truth, prediction,
                    labels=classes, average='macro', zero_division=0)), rows=len(frame))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        return dict(format_ok=False, error_type=type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly enable real provider calls')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from qa.check_tokens import summarize
    previous = [summarize(p) for batch in ('final-e2e', 'final-e2e2', 'final-e2e3', 'final-e2e4')
                for p in (root / '.test-temp' / batch).glob('*')
                if p.is_dir() and p.name in ('regression', 'multiclass')]
    if not all(r['accounting_complete'] for r in previous):
        raise ValueError('Prior token accounting incomplete')
    prior = sum(r['charged_tokens'] for r in previous)
    cap = 15000
    if prior + 2 * cap > 160000:
        raise ValueError('The existing cumulative authorization would be exceeded')
    out = root / '.test-temp/live-baseline-comparison'
    if out.exists():
        raise FileExistsError('A live trial/ledger already exists; do not rerun')
    out.mkdir()
    source = root / '.test-temp/organizer-baseline.py'
    if digest(source) != 'ecf66e8d7c054624eeca710c13e2b82c1ac22a03314db8a45ff1778fa16e372d':
        raise ValueError('Original baseline source differs')
    specs = json.loads((root / '.test-temp/transfer-specs.json').read_text(encoding='utf-8-sig'))
    spec = next(s for s in specs if s['name'] == 'wine_quality_white')
    frame = pd.read_csv(spec['path'])
    from qa import transfer_benchmark as benchmark
    with patch.object(benchmark, 'SEED', 20261031):
        ti, vi, split = benchmark.outer_split(frame.drop(columns='quality'), frame.quality, True)
    truth = frame.quality.iloc[vi].to_numpy().copy()  # Not written to either task directory.
    classes = sorted(frame.quality.iloc[ti].unique().tolist())
    task = ('Tabular multiclass classification. Target quality; metric Macro F1. '
        'Inputs: train.csv, test.csv, sample_submission.csv. '
        'ID is not a feature. Write submission.csv in template ID order with original integer quality labels. '
        'Use only provided files; no network, package installation or unrelated host files. '
        'Resource limit:2 CPU,32GB RAM.\n')
    inputs = {}
    for arm in ('baseline', 'native'):
        folder = out / arm
        folder.mkdir()
        train = frame.iloc[ti].copy()
        train.insert(0, 'id', ti)
        test = frame.iloc[vi].drop(columns='quality').copy()
        test.insert(0, 'id', vi)
        train.to_csv(folder / 'train.csv', index=False)
        test.to_csv(folder / 'test.csv', index=False)
        pd.DataFrame(dict(id=vi, quality=classes[0])).to_csv(folder / 'sample_submission.csv', index=False)
        (folder / 'task.md').write_text(task, encoding='utf-8')
        inputs[arm] = {p.name: digest(p) for p in folder.iterdir()}
    assert inputs['baseline'] == inputs['native']
    code = out / 'native-code'
    code.mkdir()
    archive = root / 'dist/solution-agent-fix.zip'
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != 13 or any('/' in n or '\\' in n for n in z.namelist()):
            raise ValueError('Invalid frozen native ZIP')
        z.extractall(code)
    protocol = dict(case='wine_quality_white', source_url=spec['source_url'], source_sha256=digest(spec['path']),
        source_baseline_sha256=digest(source), native_zip_sha256=digest(archive), split=split,
        train_indices_sha256=hashlib.sha256(ti.tobytes()).hexdigest(),
        test_indices_sha256=hashlib.sha256(vi.tobytes()).hexdigest(), inputs=inputs,
        cap_per_arm=cap, prior_charged_tokens=prior, maximum_cumulative_tokens=prior + 2 * cap,
        response_tokens=1536, max_api_calls=6, total_seconds_per_arm=300, tool_seconds=60,
        threads=2, baseline_transport='Direct tool-calling adaptation of original prompt/loop/executor; MCP absent',
        limitations='One case/seed, stochastic LLM, small context-limited API budget; not official B_F1 or leaderboard proof.',
        platform_submissions=0)
    save(out / 'protocol.json', protocol)
    if not args.run:
        print('Prepared only; real API disabled')
        return
    environment = {}
    for line in (root / '.env').read_text(encoding='utf-8-sig').splitlines():
        name, sep, value = line.strip().removeprefix('export ').partition('=')
        if sep and name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
            environment[name] = value.strip().strip('\"\'')
    if urlparse(environment.get('OPENAI_BASE_URL', '')).hostname != 'changellenge.up.railway.app':
        raise ValueError('Unexpected provider destination')
    if not environment.get('OPENAI_MODEL') or not environment.get('OPENAI_API_KEY'):
        raise ValueError('Explicit provider configuration required')
    import psutil
    process = psutil.Process()
    allowed = process.cpu_affinity()
    process.cpu_affinity(allowed[:2])
    os.environ.update(OPENAI_MODEL=environment['OPENAI_MODEL'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    from openai import OpenAI
    sys.path.insert(0, str(code))
    from agent import Agent
    from agent_config import Config
    report = dict(protocol_sha256=digest(out / 'protocol.json'), arms=[], model=environment['OPENAI_MODEL'],
                  api_destination='changellenge.up.railway.app', platform_submissions=0)
    # Reserve the entire batch persistently before the first paid request.
    ledger = dict(prior=prior, allocated=2 * cap, maximum=prior + 2 * cap, arms={})
    save(out / 'ledger.json', ledger)

    def updated(arm, client):
        ledger['arms'][arm] = dict(charged_tokens=client.spent, calls=client.calls, events=client.events)
        save(out / 'ledger.json', ledger)

    try:
        with OpenAI(api_key=environment['OPENAI_API_KEY'], base_url=environment['OPENAI_BASE_URL'], max_retries=0) as remote:
            for arm in ('baseline', 'native'):
                started = time.monotonic()
                deadline = started + 300
                metered = MeteredClient(remote, cap, lambda c, a=arm: updated(a, c), deadline)
                folder = out / arm
                trace = []
                print('START', arm, flush=True)
                try:
                    if arm == 'baseline':
                        result = run_baseline(folder, source, metered, environment['OPENAI_MODEL'], deadline, trace)
                    else:
                        result = Agent(folder / 'task.md', metered, Config(total_seconds=300, reserve_seconds=45,
                            call_seconds=60, api_seconds=45, max_steps=6, max_api_attempts=6,
                            max_tokens=500000, response_tokens=1536, retries=0,
                            bootstrap_seconds=15, warmup_seconds=45, threads=2)).run(bootstrap=True)
                except Exception as exc:
                    result = dict(finished=False, reason=type(exc).__name__)
                unchanged = all(digest(folder / name) == value for name, value in inputs[arm].items())
                record = dict(arm=arm, seconds=round(time.monotonic() - started, 3), result=result,
                    inputs_unchanged=unchanged, charged_tokens=metered.spent, calls=metered.calls,
                    evaluation=evaluate(folder, truth, classes))
                report['arms'].append(record)
                save(out / (arm + '-tool-trace.json'), trace)
                save(out / 'report.json', report)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        process.cpu_affinity(allowed)
        charged = sum(r['charged_tokens'] for r in ledger['arms'].values())
        ledger.update(charged_this_batch=charged, cumulative_charged_tokens=prior + charged, completed=True)
        save(out / 'ledger.json', ledger)
    # Answers enter a diagnostic artifact only after both agents have stopped.
    np.savez_compressed(out / 'final-checker.npz', train_indices=ti, test_indices=vi, truth=truth)


if __name__ == '__main__':
    main()
