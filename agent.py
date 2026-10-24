"""Bounded tool-calling orchestrator, independent of ML implementation."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

from agent_config import Config
from agent_prompts import SYSTEM_PROMPT, TOOLS
from artifacts import ArtifactStore, SubmissionSchema
from process_runner import run_python


class BudgetExceeded(RuntimeError):
    pass


# Метрики с известным направлением: защищает от RMSE с direction=maximize.
LOWER_IS_BETTER = ('rmse', 'mse', 'mae', 'mape', 'rmsle', 'msle', 'logloss', 'log_loss',
                   'cross_entropy', 'crossentropy', 'error', 'loss', 'wape', 'medae', 'deviance')
HIGHER_IS_BETTER = ('f1', 'accuracy', 'auc', 'r2', 'precision', 'recall', 'kappa', 'mcc',
                    'ndcg', 'gini', 'iou', 'dice', 'balanced_acc', 'average_precision', 'map@')


def metric_direction(metric):
    name = re.sub(r'[\s-]+', '_', str(metric or '').lower())
    if name.startswith('neg_'):
        underlying = metric_direction(name[4:])
        return {'minimize': 'maximize', 'maximize': 'minimize'}.get(underlying)
    if name in LOWER_IS_BETTER or name in ('root_mean_squared_error', 'mean_squared_error',
                                           'mean_absolute_error', 'mean_absolute_percentage_error'):
        return 'minimize'
    if name in HIGHER_IS_BETTER or name in ('macro_f1', 'f1_macro', 'weighted_f1', 'f1_weighted',
                                            'roc_auc', 'roc_auc_ovr', 'balanced_accuracy', 'r2_score'):
        return 'maximize'
    return None


def shrink(text, limit):
    if not isinstance(text, str):
        return text
    encoded = text.encode('utf-8')
    if len(encoded) <= limit:
        return text
    marker = f'\n...[truncated from {len(encoded)} bytes]...\n'
    if len(marker) >= limit:
        return encoded[:limit].decode('utf-8', errors='ignore')
    available = limit - len(marker)
    head = available * 3 // 10
    tail = available - head
    return (encoded[:head].decode('utf-8', errors='ignore') + marker +
            encoded[-tail:].decode('utf-8', errors='ignore'))


def output_names_from_task(task, task_dir):
    names = []
    for match in re.findall(r'[\w\-./]+\.csv', task):
        name = match.lstrip('./')
        lowered = Path(name).name.lower()
        if (name and not (task_dir / name).exists() and '..' not in Path(name).parts
                and not any(word in lowered for word in ('train', 'test', 'sample', 'dictionary'))
                and name not in names):
            names.append(name)
    return names


class TextToolCall:

    def __init__(self, identifier, name, arguments):
        self.id = identifier
        self.function = SimpleNamespace(name=name, arguments=(
            arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)))

    def model_dump(self, **kwargs):
        return {'id': self.id, 'type': 'function',
                'function': {'name': self.function.name, 'arguments': self.function.arguments}}


def text_tool_calls(content, step):
    calls = []
    for number, block in enumerate(re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content or '', re.S)):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get('name'), str):
            calls.append(TextToolCall(f'text_{step}_{number}', data['name'], data.get('arguments', {})))
    return calls


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
        # The platform always receives submission.csv; explicit task aliases are additional.
        self.outputs = ['submission.csv'] + [n for n in output_names_from_task(self.task, self.task_dir)
                                             if n != 'submission.csv']
        self.finished = False
        self.attempts = self.tokens = self.step = 0
        self.experiments = []
        self._native_best = None
        self.finalization_attempted = False
        self.messages = [ {'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': 'TASK:\n' + self.task + '\nFILES:\n' +
                          json.dumps(inventory(self.task_dir), ensure_ascii=False)} ]
        self.journal('start', config=asdict(self.config))

    def publish(self):
        primary = self.store.publish()
        for name in self.outputs[1:]:
            try:
                self.store.publish(name)
            except Exception as exc:
                self.journal('alias_publish_error', name=name, error_type=type(exc).__name__)
        return primary

    def remaining(self):
        return max(0.0, self.config.total_seconds - (self.clock() - self.started))

    def journal(self, event, **fields):
        entry = {'event': event, 'step': self.step, 'elapsed': self.clock() - self.started, **fields}
        with (self.state_dir / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(redact(json.dumps(entry, ensure_ascii=False, allow_nan=False)) + '\n')

    def _discard_oldest_exchange(self):
        if len(self.messages) <= 3:
            return False
        end = 3
        while end < len(self.messages) and self.messages[end]['role'] == 'tool':
            end += 1
        if end == len(self.messages):
            return False  # Keep the most recent result, especially an unregistered candidate.
        del self.messages[2:end]
        return True

    def _request_payload(self):
        payload = self.messages + [{'role': 'user', 'content': json.dumps({
            'time_left_seconds': round(self.remaining(), 2),
            'phase': 'finalize' if (self.remaining() <= self.config.reserve_seconds or
                                   self.step >= self.config.max_steps - 2 or
                                   self.tokens >= self.config.max_tokens * 0.8) else 'explore',
            'best': self.store.best,
            'recent_experiments': self.experiments})}]
        payload = json.loads(redact(json.dumps(payload, ensure_ascii=False)))
        input_bound = len(json.dumps([payload, TOOLS], ensure_ascii=False).encode('utf-8')) + 512
        return payload, input_bound

    def _request(self):
        # Remove complete old exchanges, preserving system/task and latest results.
        while len(json.dumps(self.messages, ensure_ascii=False).encode()) > self.config.context_bytes:
            if not self._discard_oldest_exchange():
                break
        payload, input_bound = self._request_payload()
        if input_bound > self.config.context_bytes + 20000:
            raise BudgetExceeded('context limit reached')
        for retry in range(self.config.retries + 1):
            if self.attempts >= self.config.max_api_attempts:
                raise BudgetExceeded('API attempt limit reached')
            while self.tokens + input_bound + self.config.response_tokens > self.config.max_tokens:
                if not self._discard_oldest_exchange():
                    break
                payload, input_bound = self._request_payload()
            response_tokens = min(self.config.response_tokens,
                                  self.config.max_tokens - self.tokens - input_bound)
            # A short final tool call can fit even when the usual completion allowance cannot.
            if response_tokens < min(256, self.config.response_tokens):
                raise BudgetExceeded('API token budget reached')
            # Reserve a conservative UTF-8 byte bound. Refund with exact reported usage.
            reservation = input_bound + response_tokens
            if self.remaining() <= 2:
                raise BudgetExceeded('time limit reached')
            self.attempts += 1
            self.tokens += reservation  # Keep full charge on ambiguous network failures.
            try:
                result = self.client.chat.completions.create(
                    model=self.model, messages=payload, tools=TOOLS, parallel_tool_calls=False,
                    max_tokens=response_tokens,
                    timeout=min(self.config.api_seconds, self.remaining() - 1))
                usage = getattr(result, 'usage', None)
                actual = getattr(usage, 'total_tokens', None)
                if type(actual) is int and actual >= 0:
                    self.tokens += actual - reservation
                self.journal('api', attempt=self.attempts, charged_tokens=self.tokens)
                return result.choices[0].message
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                transient = status == 429 or (isinstance(status, int) and status >= 500) or type(exc).__name__ in ('APITimeoutError', 'APIConnectionError')
                self.journal('api_error', error_type=type(exc).__name__, status=status,
                             charged_tokens=self.tokens)
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
        else:
            # Финализация: оставить время на вызов LLM, register_candidate и publish.
            margin = min(self.config.finalize_margin_seconds, self.remaining() / 2)
            limit = min(limit, self.remaining() - margin)
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
            args = dict(args)
            output = args.pop('output_path', None)
            output_name = None
            if output is not None:
                if not isinstance(output, str):
                    raise ValueError('output_path must be a string')
                raw_target = self.task_dir / output
                if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                       for p in (raw_target, *raw_target.parents) if p != self.task_dir):
                    raise ValueError('output_path must not use links')
                target = raw_target.resolve()
                if (not target.is_relative_to(self.task_dir) or target.suffix.lower() != '.csv'
                        or (target.exists() and str(target.relative_to(self.task_dir)) not in self.outputs)):
                    raise ValueError('output_path must be a new .csv inside the task directory (never an input file)')
                output_name = str(target.relative_to(self.task_dir))
            schema = SubmissionSchema(**args)
            store = ArtifactStore(self.task_dir, self.state_dir, schema)
            self.store = store
            if output_name and output_name not in self.outputs:
                self.outputs.append(output_name)
            self.configured = True
            self.journal('schema', schema=asdict(schema))
            return {'configured': True}
        if name == 'register_candidate':
            if not self.configured:
                raise ValueError('configure_submission first')
            if args.get('score') is not None:
                known = metric_direction(args.get('metric'))
                if 'direction' not in args:
                    if known is None:
                        raise ValueError('direction (maximize/minimize) is required for this metric')
                    args = {**args, 'direction': known}
                elif known and args['direction'] != known:
                    raise ValueError(f"metric {args.get('metric')!r} must be {known}d, got {args['direction']}")
            result = self.store.consider(**args)
            self.journal('candidate', result=result)
            if result.get('accepted'):
                self.publish()  # файл на диске сразу: kill на 180-й минуте его не отнимет
            return result
        if name == 'finish':
            path = self.publish()
            self.finished = True
            return {'submission': str(path)}
        timeout = self._timeout(args.get('timeout_seconds'))
        if name == 'run_backend':
            result = self._backend(args['request'], timeout)
            try:
                registration = self._accept_native_result(result)
                if registration is not None:
                    result['registration'] = registration
            except Exception as exc:
                result['registration'] = {'accepted': False, 'error': redact(str(exc))[:2000],
                                          'type': type(exc).__name__}
                self.journal('native_registration_error', error_type=type(exc).__name__)
            self._remember_experiment(name, args, result)
            return result
        code = args['code']
        if not isinstance(code, str):
            raise ValueError('code must be a string')
        # Save reproducible experiment source, without secrets, separately from bounded output.
        experiment_path = self.state_dir / f'experiment-{self.step:03d}-{uuid.uuid4().hex[:8]}.py'
        experiment_path.write_text(redact(code), encoding='utf-8')
        result = run_python(code, self.task_dir, timeout, self.config.memory_mb, self.config.threads)
        result['experiment_path'] = str(experiment_path.relative_to(self.task_dir))
        self.journal('execution', result=result)
        self._remember_experiment(name, args, result)
        return result

    def _remember_experiment(self, name, args, result):
        """Retain bounded factual outcomes when expensive full exchanges are discarded."""
        response = result.get('response', {})
        entry = {'tool': name, 'step': self.step, 'exit_code': result.get('exit_code')}
        if name == 'run_backend':
            entry['request'] = shrink(json.dumps(args['request'], ensure_ascii=False), 700)
            entry.update({key: response[key] for key in (
                'model', 'score', 'metric', 'direction', 'validation_id', 'fit_scope',
                'candidate_path', 'manifest_path') if key in response})
            validation = response.get('validation')
            if isinstance(validation, dict):
                entry['validation'] = {key: shrink(str(validation[key]), 300) for key in
                                       ('strategy', 'rationale') if key in validation}
            if 'registration' in result:
                entry['registration'] = {key: result['registration'][key] for key in
                                         ('accepted', 'reason', 'error') if key in result['registration']}
        else:
            entry['experiment_path'] = result['experiment_path']
            entry['output_summary'] = shrink(result.get('stdout', '') + '\n' +
                                             result.get('stderr', ''), 400)
        entry = {key: shrink(value, 700) if isinstance(value, str) else value
                 for key, value in entry.items()}
        self.experiments.append(entry)
        # Long paths/validation identifiers cannot grow status context without bound.
        while len(self.experiments) > 8 or (len(self.experiments) > 1 and
                len(json.dumps(self.experiments, ensure_ascii=False).encode('utf-8')) > 6000):
            self.experiments.pop(0)

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

    def _accept_native_result(self, result):
        response = result.get('response', {})
        if (result.get('exit_code') != 0 or response.get('status') != 'ok'
                or not response.get('candidate_path')):
            return None
        if not self.configured:
            self.execute('configure_submission', response['schema'])
        candidate = {key: response[key] for key in ('candidate_path', 'score', 'metric',
                     'direction', 'validation_id', 'final') if key in response}
        # Existing schema and validation identity remain authoritative.
        registration = self.execute('register_candidate', candidate)
        if registration.get('accepted'):
            self._native_best = {'response': dict(response),
                                 'snapshot': registration['snapshot']}
        return registration

    def _finalize_native(self):
        """Spend remaining compute on the selected model, even after API exhaustion."""
        if self.finalization_attempted or not self._native_best or not self.store.best:
            return
        response = self._native_best['response']
        if (self.store.best['snapshot'] != self._native_best['snapshot'] or
                response.get('fit_scope') != 'holdout_train' or
                not response.get('manifest_path') or self.store.best.get('final')):
            return
        timeout = min(self.config.call_seconds, self.remaining() - 2)
        if timeout < 30:
            return
        self.finalization_attempted = True
        request = {'action': 'refit', 'manifest_path': response['manifest_path']}
        result = self._backend(request, timeout)
        refitted = result.get('response', {})
        # Final equal-score replacement must describe this exact selected validation.
        if (refitted.get('fit_scope') != 'full_train' or refitted.get('final') is not True or
                any(refitted.get(key) != response.get(key) for key in
                    ('score', 'metric', 'direction', 'validation_id'))):
            self.journal('native_finalization_rejected', reason='incomplete_or_different_validation')
            return
        registration = self._accept_native_result(result)
        self.journal('native_finalization', registration=registration)

    def _bootstrap(self):
        module = Path(__file__).resolve().with_name('ml_backend.py')
        if not module.is_file() or self.config.bootstrap_seconds == 0:
            return
        for action, allowance in (('baseline', self.config.bootstrap_seconds),
                                  ('experiment', self.config.warmup_seconds)):
            if allowance == 0 or self.remaining() <= self.config.reserve_seconds + 5:
                continue
            # Native ML runs use a modest fraction of short smoke-test budgets.
            timeout = min(allowance, self.remaining() * 0.2,
                          self.remaining() - self.config.reserve_seconds - 2)
            if timeout < 2:
                continue
            try:
                result = self._backend({'action': action}, timeout)
                response = result.get('response', {})
                if result.get('exit_code') != 0 or response.get('status') != 'ok':
                    self.journal('bootstrap_error', action=action,
                                 error_type=response.get('error_type', 'backend_failed'))
                    if not self.store.best:
                        break
                    continue
                registration = self._accept_native_result(result)
                if registration is None:
                    raise ValueError('native training returned no candidate')
                result['registration'] = registration
                self._remember_experiment('run_backend', {'request': {'action': action}}, result)
                self.messages.append({'role': 'user', 'content':
                    'Native ML startup result (already evaluated and saved if better): ' +
                    redact(json.dumps({'response': response, 'registration': registration}, ensure_ascii=False))})
            except Exception as exc:
                self.journal('bootstrap_error', action=action, error_type=type(exc).__name__)
                if not self.store.best:
                    break

    def run(self, *, bootstrap=False):
        reason = 'step_limit'
        cancelled = False
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            def on_sigterm(signum, frame):
                raise KeyboardInterrupt('SIGTERM')
            previous_handler = signal.signal(signal.SIGTERM, on_sigterm)
        try:
            if bootstrap:
                self._bootstrap()
            for self.step in range(1, self.config.max_steps + 1):
                message = self._request()
                calls = getattr(message, 'tool_calls', None) or text_tool_calls(
                    getattr(message, 'content', None), self.step)
                if len(calls) > 8:
                    raise ValueError('too many tool calls')
                self.messages.append({'role': 'assistant', 'content': redact(message.content or ''),
                    **({'tool_calls': [c.model_dump(exclude_none=True) for c in calls]} if calls else {})})
                assistant_message = self.messages[-1]
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
                        if call.function.name == 'run_python' and result.get('experiment_path'):
                            # Repeatedly retransmitting generated scripts wastes the paid context.
                            # The exact source persists and can be read by the next child process.
                            for saved_call in assistant_message.get('tool_calls', []):
                                if saved_call['id'] == call.id:
                                    saved_call['function']['arguments'] = json.dumps({
                                        **args, 'code': '# Executed source saved at ' + result['experiment_path']})
                    except Exception as exc:
                        result = {'error': redact(str(exc))[:2000], 'type': type(exc).__name__}
                        self.journal('tool_error', tool=call.function.name, **result)
                    if isinstance(result, dict):  # полный вывод уже в events.jsonl
                        result = {key: shrink(value, self.config.tool_output_bytes)
                                  if key in ('stdout', 'stderr') else value for key, value in result.items()}
                    self.messages.append({'role': 'tool', 'tool_call_id': call.id,
                                          'content': redact(json.dumps(result, ensure_ascii=False, default=str))})
                    if self.finished:
                        reason = 'completed'
                        break
                if self.finished:
                    break
        except (Exception, KeyboardInterrupt) as exc:
            reason = type(exc).__name__
            cancelled = isinstance(exc, KeyboardInterrupt)
            self.journal('stopped', reason=reason)
        finally:
            try:
                if not cancelled:
                    self._finalize_native()
            except (Exception, KeyboardInterrupt) as exc:
                self.journal('native_finalization_error', error_type=type(exc).__name__)
                if isinstance(exc, KeyboardInterrupt):
                    reason = type(exc).__name__
            finally:
                if previous_handler is not None:
                    signal.signal(signal.SIGTERM, previous_handler)
        if self.store.best:
            path = self.publish()
            self.journal('published', reason=reason, submission=str(path), best=self.store.best)
            return {'success': True, 'reason': reason, 'submission': str(path), 'state_dir': str(self.state_dir)}
        self.journal('failed', reason=reason)
        return {'success': False, 'reason': reason, 'state_dir': str(self.state_dir)}
