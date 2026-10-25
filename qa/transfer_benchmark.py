"""Evaluate a frozen solution ZIP on new public datasets with hidden outer labels.

Input: a JSON list of dataset specs (name/path/target/problem_type/metric/source).
Data and reports belong in an ignored scratch directory. Only the generated
train/test/template/task files enter the platform runner. Outer labels are
copied to the checker after the solution exits, never to its task directory.
"""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from qa.run_local import run_task


SEED = 20261025


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def outer_split(features, target, classification):
    """Target 20% outer rows while keeping ALL duplicate feature groups intact."""
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    indices = np.arange(len(features))
    duplicates = len(np.unique(groups)) < len(groups)
    if classification and duplicates:
        candidates = StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            features, target, groups)
    elif classification:
        a, b = train_test_split(indices, test_size=.2, random_state=SEED, stratify=target)
        candidates = [(a, b)]
    else:
        candidates = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(
            features, target, groups)
    labels = set(target) if classification else None
    eligible = []
    for train, test in candidates:
        # This is stratification, not a choice based on model/test performance.
        if classification and (set(target.iloc[train]) != labels or set(target.iloc[test]) != labels):
            continue
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Duplicate feature group crosses the outer split')
        eligible.append((train, test))
    if not eligible:
        raise ValueError('Cannot preserve every class on both sides of an independent grouped split')
    # Large duplicate groups can make 20% impossible. Choose by row count,
    # never model performance or prediction errors; stable ties keep fold order.
    train, test = min(eligible, key=lambda pair: abs(len(pair[1]) / len(features) - .2))
    return train, test, dict(seed=SEED, method=('stratified_group' if classification and duplicates
                                              else 'stratified' if classification else 'group_shuffle'),
                             outer_fraction=len(test) / len(features),
                             duplicate_rows=len(groups) - len(np.unique(groups)))


INPUT_FILES = frozenset({'train.csv', 'test.csv', 'sample_submission.csv', 'task.md'})


def verify_case(folder, metadata_path):
    """Verify a frozen source, exact worker inputs and separate outer answers."""
    folder, metadata_path = Path(folder), Path(metadata_path)

    def linked(path):
        return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()

    def regular(path, message):
        if linked(path) or not path.is_file():
            raise ValueError(message)

    regular(metadata_path, 'Frozen dataset/split metadata is missing or linked')
    frozen = json.loads(metadata_path.read_text(encoding='utf-8'))
    name = frozen['spec']['name']
    if name != folder.name or name == '_answers':
        raise ValueError('Frozen dataset/split names do not match')
    source = Path(frozen['spec']['path'])
    regular(source, 'Frozen dataset/split source is missing or linked')
    if file_hash(source) != frozen.get('source_sha256'):
        raise ValueError('Frozen dataset/split source changed')
    if (linked(folder) or not folder.is_dir()
            or {p.name for p in folder.iterdir()} != INPUT_FILES
            or set(frozen.get('input_sha256', {})) != INPUT_FILES):
        raise ValueError('Frozen benchmark input file set changed')
    for filename in INPUT_FILES:
        path = folder / filename
        regular(path, 'Frozen benchmark input is not a regular unlinked file')
        if file_hash(path) != frozen['input_sha256'][filename]:
            raise ValueError('Frozen benchmark input changed')
    answers = folder.parent / '_answers'
    names = {name + '.csv', name + '.json'}
    if (linked(answers) or not answers.is_dir()
            or set(frozen.get('hidden_answer_sha256', {})) != names):
        raise ValueError('Frozen hidden answer metadata is missing or changed')
    for filename in names:
        path = answers / filename
        regular(path, 'Frozen hidden answer is missing or linked')
        if file_hash(path) != frozen['hidden_answer_sha256'][filename]:
            raise ValueError('Frozen hidden answer changed')
    return frozen


def prepare_case(spec, root):
    name = spec['name']
    if not re.fullmatch(r'[a-z0-9_]+', name) or name == '_answers':
        raise ValueError('Dataset names must be lowercase safe identifiers')
    if (spec['problem_type'], spec['metric']) not in {
            ('regression', 'rmse'), ('classification', 'macro_f1'), ('classification', 'roc_auc')}:
        raise ValueError('Pilot evaluator supports regression RMSE, classification Macro F1 and binary ROC AUC')
    frame = pd.read_csv(spec['path'], low_memory=False)
    target = spec['target']
    y, features = frame[target], frame.drop(columns=target)
    if y.isna().any() or not len(features.columns) or 'id' in features:
        raise ValueError('Expected a nonmissing single target, features, and no preexisting row ID')
    if spec['metric'] == 'roc_auc' and y.nunique() != 2:
        raise ValueError('ROC AUC pilot cases must be binary')
    train_i, test_i, diagnostics = outer_split(features, y, spec['problem_type'] == 'classification')
    folder = root / 'tasks' / name
    metadata_path = root / 'splits' / (name + '.json')
    expected = dict(spec=spec, source_sha256=file_hash(spec['path']),
                    train_indices=train_i.tolist(), test_indices=test_i.tolist(), **diagnostics)
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(old.get(k) != value for k, value in expected.items()):
            raise ValueError('Frozen dataset/split changed; use a new benchmark directory')
        verify_case(folder, metadata_path)
        return folder
    folder.mkdir(parents=True, exist_ok=False)
    train = frame.iloc[train_i].copy()
    train.insert(0, 'id', train_i)
    test = features.iloc[test_i].copy()
    test.insert(0, 'id', test_i)
    train.to_csv(folder / 'train.csv', index=False)
    test.to_csv(folder / 'test.csv', index=False)
    probability = spec['metric'] in {'roc_auc', 'log_loss', 'average_precision'}
    fill = 0 if probability or spec['problem_type'] == 'regression' else sorted(y.iloc[train_i].unique())[0]
    pd.DataFrame({'id': test_i, target: fill}).to_csv(folder / 'sample_submission.csv', index=False)
    (folder / 'task.md').write_text(
        f"Tabular {spec['problem_type']}. Target: {target}. Primary metric: {spec['metric']}.\n"
        'Inputs: train.csv, test.csv, sample_submission.csv. Use the template row order.\n'
        'Output submission.csv beside task.md. Row ID is not a model feature.\n'
        + ('Submit probability of the greater/sorted-last training label.\n' if probability else '')
        + f"Dataset source: {spec['source_url']}\n", encoding='utf-8')
    answers = root / 'tasks' / '_answers'
    answers.mkdir(exist_ok=True)
    pd.DataFrame({'id': test_i, target: y.iloc[test_i].to_numpy()}).to_csv(answers / f'{name}.csv', index=False)
    kind = ('regression' if spec['problem_type'] == 'regression' else 'binary_auc' if probability else 'multiclass')
    (answers / f'{name}.json').write_text(json.dumps(dict(kind=kind)), encoding='utf-8')
    expected['input_sha256'] = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
    expected['hidden_answer_sha256'] = {filename: file_hash(answers / filename)
                                      for filename in (name + '.csv', name + '.json')}
    metadata_path.parent.mkdir(exist_ok=True)
    metadata_path.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    verify_case(folder, metadata_path)
    return folder


@contextmanager
def finish_provider():
    """One local finish response; no external LLM/provider is called."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 2_000_000:
                self.send_error(400)
                return
            self.rfile.read(size)
            requests.append(1)
            result = dict(id='benchmark', object='chat.completion', created=0, model='local',
                          choices=[dict(index=0, finish_reason='tool_calls', message=dict(
                              role='assistant', content=None, tool_calls=[dict(id='finish-1', type='function',
                                  function=dict(name='finish', arguments='{}'))]))],
                          usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_case(spec, root, archive, budget, threads):
    folder = prepare_case(spec, root)
    metadata_path = root / 'splits' / (spec['name'] + '.json')
    frozen = verify_case(folder, metadata_path)
    metadata_hash = file_hash(metadata_path)
    output = root / 'results' / (spec['name'] + '.json')
    if output.exists():
        raise ValueError('Case already evaluated; keep outer test results frozen')
    with finish_provider() as (url, requests):
        environment = dict(OPENAI_API_KEY='local-benchmark-only', OPENAI_MODEL='local',
                           OPENAI_BASE_URL=url, NO_PROXY='127.0.0.1,localhost',
                           AGENT_TOTAL_SECONDS='1200', AGENT_RESERVE_SECONDS='180',
                           AGENT_CALL_SECONDS='300', AGENT_BOOTSTRAP_SECONDS='30',
                           AGENT_WARMUP_SECONDS=str(budget), AGENT_THREADS=str(threads),
                           AGENT_API_SECONDS='5', AGENT_MAX_STEPS='2', AGENT_RETRIES='0',
                           AGENT_MAX_TOKENS='20000', AGENT_RESPONSE_TOKENS='256')
        with patch.dict(os.environ, environment):
            record = run_task(archive, folder, 600)
    record.update(name=spec['name'], source_url=spec['source_url'],
                  zip_sha256=file_hash(archive), warmup_seconds=budget, threads=threads,
                  external_api_requests=0, local_api_requests=len(requests))
    state = next((Path(record['work_dir']) / 'data').glob('.agent-*'), None)
    if state is not None and (state / 'best.json').exists():
        record['best'] = json.loads((state / 'best.json').read_text(encoding='utf-8'))
        events = [json.loads(l) for l in (state / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        native = [e['result'] for e in events if e['event'] == 'backend']
        record['native'] = [dict(exit_code=x.get('exit_code'), timed_out=x.get('timed_out'),
            checkpoint_recovered=x.get('checkpoint_recovered', False),
            **{k: x.get('response', {}).get(k) for k in
               ('model', 'score', 'fit_scope', 'selected_iterations', 'portfolio_results')}) for x in native]
    if file_hash(metadata_path) != metadata_hash:
        raise ValueError('Frozen split metadata changed during evaluation')
    verify_case(folder, metadata_path)
    worker_data = Path(record['work_dir']) / 'data'
    for name, digest in frozen['input_sha256'].items():
        copied = worker_data / name
        if (copied.is_symlink() or not copied.is_file()
                or getattr(copied, 'is_junction', lambda: False)() or file_hash(copied) != digest):
            raise ValueError('A copied worker input changed during evaluation')
    record['inputs_unchanged'] = True
    # Evaluate a simple baseline only after model selection and publication end.
    # It never participates in inner model selection or provides outer labels.
    from sklearn.metrics import f1_score
    target = spec['target']
    train_y = pd.read_csv(folder / 'train.csv', usecols=[target], dtype=str, keep_default_na=False)[target]
    truth = pd.read_csv(folder.parent / '_answers' / (spec['name'] + '.csv'),
                        dtype=str, keep_default_na=False)[target]
    if spec['metric'] == 'rmse':
        record['dummy_outer_score'] = float(np.sqrt(np.mean((truth.astype(float) - train_y.astype(float).mean()) ** 2)))
    elif spec['metric'] == 'roc_auc':
        record['dummy_outer_score'] = .5
    else:
        record['dummy_outer_score'] = float(f1_score(truth, np.repeat(train_y.mode().iloc[0], len(truth)),
            labels=sorted(train_y.unique()), average='macro', zero_division=0))
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', type=Path)
    parser.add_argument('--root', type=Path, default=Path('.test-temp/transfer-benchmark'))
    parser.add_argument('--zip', type=Path, default=Path('dist/solution.zip'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--names', nargs='*')
    parser.add_argument('--budget', type=int, default=180)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 30 <= args.budget <= 300 or not 1 <= args.threads <= 4:
        parser.error('Pilot budget must be 30–300 seconds with 1–4 threads')
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.specs.read_text(encoding='utf-8-sig'))
    if len({s['name'] for s in specs}) != len(specs):
        parser.error('Duplicate case names')
    for spec in specs:
        if args.names and spec['name'] not in args.names:
            continue
        if args.prepare_only:
            prepare_case(spec, args.root)
            print('PREPARED', spec['name'], flush=True)
        else:
            print('START', spec['name'], flush=True)
            result = run_case(spec, args.root, args.zip.resolve(), args.budget, args.threads)
            print(json.dumps({k: result.get(k) for k in ('name', 'success', 'minutes', 'check', 'best')}), flush=True)


if __name__ == '__main__':
    main()
