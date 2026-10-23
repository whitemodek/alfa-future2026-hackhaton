"""Bounded tool-calling orchestrator, independent of ML implementation."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from agent_config import Config
from agent_prompts import SYSTEM_PROMPT, TOOLS
from artifacts import ArtifactStore, SubmissionSchema
from process_runner import run_python


class BudgetExceeded(RuntimeError):
    pass


def redact(text):
    for name, value in os.environ.items():
        if len(value) >= 6 and any(part in name.upper() for part in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            text = text.replace(value, '[REDACTED]')
    return re.sub(r'(?i)Bearer\s+[^\s"\x27]+', 'Bearer [REDACTED]', text)


def inventory(root):
    entries = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(base)/d).is_symlink())
        for name in sorted(files):
            path = Path(base) / name
            if name.startswith('.') or path.is_symlink():
                continue
            entries.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
            if len(entries) >= 200:
                return entries
    return entries


class Agent:
    def __init__(self, task_path, client, config=None, *, clock=time.monotonic, sleep=time.sleep):
        self.config = config or Config()
        self.clock, self.sleep = clock, sleep
        self.started = clock()
        self.task_path = Path(task_path).resolve(strict=True)
        self.task_dir = self.task_path.parent
        if self.task_path.stat().st_size > 128000:
            raise ValueError('task exceeds 128 KB')
        self.task = self.task_path.read_text(encoding='utf-8-sig')
        self.client = client
        self.model = os.environ.get('OPENAI_MODEL', 'qwen/qwen3.8-27b')
        self.state_dir = Path(tempfile.mkdtemp(prefix='.agent-', dir=self.task_dir))
        self.store = ArtifactStore(self.task_dir, self.state_dir, SubmissionSchema())
        self.configured = False
        self.finished = False
        self.attempts = self.tokens = self.step = 0
        self.messages = [ {'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': 'TASK:\n' + self.task + '\nFILES:\n' +
                          json.dumps(inventory(self.task_dir), ensure_ascii=False)} ]
        self.journal('start', config=asdict(self.config))

    def remaining(self):
        return max(0.0, self.config.total_seconds - (self.clock() - self.started))

    def journal(self, event, **fields):
        entry = {'event': event, 'step': self.step, 'elapsed': self.clock() - self.started, **fields}
        with (self.state_dir / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(redact(json.dumps(entry, ensure_ascii=False, allow_nan=False)) + '\n')

    def _request(self):
        # Drop only complete old assistant/tool exchanges, keeping initial instructions.
        while len(json.dumps(self.messages).encode()) > self.config.context_bytes and len(self.messages) > 3:
            end = 3
            while end < len(self.messages) and self.messages[end]['role'] == 'tool':
                end += 1
            del self.messages[2:end]
        payload = self.messages + [{'role': 'user', 'content': json.dumps({
            'time_left_seconds': round(self.remaining(), 2),
            'phase': 'finalize' if (self.remaining() <= self.config.reserve_seconds or
                                   self.step >= self.config.max_steps - 2 or
                                   self.tokens >= self.config.max_tokens * 0.8) else 'explore',
            'best': self.store.best})}]
        payload = json.loads(redact(json.dumps(payload, ensure_ascii=False)))
        # UTF-8 bytes overestimate typical token counts; include schema and framing overhead.
        input_bound = len(json.dumps([payload, TOOLS]).encode('utf-8')) + 512
        if input_bound > self.config.context_bytes + 20000:
            raise BudgetExceeded('context limit reached')
        for retry in range(self.config.retries + 1):
            reservation = input_bound + self.config.response_tokens
            if self.attempts >= self.config.max_api_attempts or self.tokens + reservation > self.config.max_tokens:
                raise BudgetExceeded('API budget reached')
            if self.remaining() <= 2:
                raise BudgetExceeded('time limit reached')
            self.attempts += 1
            self.tokens += reservation  # Keep full charge on ambiguous network failures.
            try:
                result = self.client.chat.completions.create(
                    model=self.model, messages=payload, tools=TOOLS, parallel_tool_calls=False,
                    max_tokens=self.config.response_tokens,
                    timeout=min(self.config.api_seconds, self.remaining() - 1))
                usage = getattr(result, 'usage', None)
                actual = getattr(usage, 'total_tokens', None)
                if isinstance(actual, int) and actual >= 0:
                    self.tokens += actual - reservation
                self.journal('api', attempt=self.attempts, charged_tokens=self.tokens)
                return result.choices[0].message
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                transient = status == 429 or (isinstance(status, int) and status >= 500) or type(exc).__name__ in ('APITimeoutError', 'APIConnectionError')
                self.journal('api_error', error_type=type(exc).__name__, status=status)
                delay = min(2 ** retry, 8)
                if not transient or retry == self.config.retries or self.remaining() <= delay + 2:
                    raise
                self.sleep(delay)

    def _timeout(self, requested=None):
        limit = self.remaining() - 2
        if limit <= 0:
            raise BudgetExceeded('time limit reached')
        if self.remaining() > self.config.reserve_seconds:
            limit = min(limit, self.remaining() - self.config.reserve_seconds)
        value = self.config.call_seconds if requested is None else requested
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('timeout_seconds must be positive and finite')
        return min(value, self.config.call_seconds, limit)

    def execute(self, name, args):
        spec = next((t['function'] for t in TOOLS if t['function']['name'] == name), None)
        if not spec or not isinstance(args, dict):
            raise ValueError('unknown tool or invalid arguments')
        if set(args) - set(spec['parameters']['properties']) or set(spec['parameters']['required']) - set(args):
            raise ValueError('unexpected or missing arguments')
        if name == 'configure_submission':
            if self.store.best:
                raise ValueError('schema is locked after first candidate')
            if not args.get('sample_path') and (not args.get('columns') or args.get('expected_rows') is None):
                raise ValueError('provide sample_path or columns and expected_rows')
            schema = SubmissionSchema(**args)
            self.store = ArtifactStore(self.task_dir, self.state_dir, schema)
            self.configured = True
            self.journal('schema', schema=asdict(schema))
            return {'configured': True}
        if name == 'register_candidate':
            if not self.configured:
                raise ValueError('configure_submission first')
            result = self.store.consider(**args)
            self.journal('candidate', result=result)
            return result
        if name == 'finish':
            path = self.store.publish()
            self.finished = True
            return {'submission': str(path)}
        timeout = self._timeout(args.get('timeout_seconds'))
        if name == 'run_backend':
            return self._backend(args['request'], timeout)
        code = args['code']
        if not isinstance(code, str):
            raise ValueError('code must be a string')
        # Save reproducible experiment source, without secrets, separately from bounded output.
        (self.state_dir / f'experiment-{self.step:03d}.py').write_text(redact(code), encoding='utf-8')
        result = run_python(code, self.task_dir, timeout, self.config.memory_mb, self.config.threads)
        self.journal('execution', result=result)
        return result

    def _backend(self, request, timeout):
        if not isinstance(request, dict):
            raise ValueError('request must be an object')
        module = Path(__file__).resolve().with_name('ml_backend.py')
        if not module.is_file():
            raise ValueError('ml_backend.py is not installed; use run_python')
        invocation = uuid.uuid4().hex
        request_path = self.state_dir / f'request-{invocation}.json'
        response_path = self.state_dir / f'response-{invocation}.json'
        payload = {'version': 1, 'task_path': str(self.task_path), 'task_dir': str(self.task_dir),
                   'time_budget_seconds': timeout, 'threads': self.config.threads,
                   'memory_mb': self.config.memory_mb, 'request': request}
        request_path.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
        # Execute module as __main__ in the same limited child; never import plugin in parent.
        code = ('import runpy, sys\n' + f'sys.path.insert(0, {str(module.parent)!r})\n' +
                f'sys.argv = [{str(module)!r}, {str(request_path)!r}, {str(response_path)!r}]\n' +
                f'runpy.run_path({str(module)!r}, run_name="__main__")\n')
        result = run_python(code, self.task_dir, timeout, self.config.memory_mb, self.config.threads)
        if result['exit_code'] == 0 and response_path.is_file():
            if response_path.is_symlink() or response_path.stat().st_size > 64000:
                raise ValueError('invalid backend response file')
            response = json.loads(response_path.read_text(encoding='utf-8'),
                                  parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if not isinstance(response, dict):
                raise ValueError('backend response must be an object')
            result['response'] = response
        elif result['exit_code'] == 0:
            result['error'] = 'backend did not write response JSON'
        self.journal('backend', result=result)
        return result

    def run(self):
        reason = 'step_limit'
        try:
            for self.step in range(1, self.config.max_steps + 1):
                message = self._request()
                calls = getattr(message, 'tool_calls', None) or []
                if len(calls) > 8:
                    raise ValueError('too many tool calls')
                self.messages.append({'role': 'assistant', 'content': redact(message.content or ''),
                    **({'tool_calls': [c.model_dump(exclude_none=True) for c in calls]} if calls else {})})
                if not calls:
                    if self.store.best:
                        reason = 'completed'
                        break
                    self.messages.append({'role': 'user', 'content': 'No registered candidate. Produce, validate and register a submission before finishing.'})
                    continue
                for call in calls:
                    try:
                        if len(call.function.arguments) > 140000:
                            raise ValueError('tool arguments too large')
                        args = json.loads(call.function.arguments, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                        result = self.execute(call.function.name, args)
                    except Exception as exc:
                        result = {'error': redact(str(exc))[:2000], 'type': type(exc).__name__}
                        self.journal('tool_error', tool=call.function.name, **result)
                    self.messages.append({'role': 'tool', 'tool_call_id': call.id,
                                          'content': redact(json.dumps(result, ensure_ascii=False, default=str))})
                    if self.finished:
                        reason = 'completed'
                        break
                if self.finished:
                    break
        except (Exception, KeyboardInterrupt) as exc:
            reason = type(exc).__name__
            self.journal('stopped', reason=reason)
        if self.store.best:
            path = self.store.publish()
            self.journal('published', reason=reason, submission=str(path), best=self.store.best)
            return {'success': True, 'reason': reason, 'submission': str(path), 'state_dir': str(self.state_dir)}
        self.journal('failed', reason=reason)
        return {'success': False, 'reason': reason, 'state_dir': str(self.state_dir)}
