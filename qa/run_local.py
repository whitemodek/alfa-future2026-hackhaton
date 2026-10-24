"""Run a solution ZIP in a clean directory with a platform-style time limit."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import psutil

HERE = Path(__file__).resolve().parent
COMMAND = ["{python}", "solution.py", "{task_md}"]
LOG_LIMIT = 16_384


def _extract(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if sum(entry.file_size for entry in entries) > 100_000_000:
            raise ValueError("ZIP contents exceed 100 MB")
        names = set()
        for entry in entries:
            name = entry.filename
            if (not name or name.casefold() in names or '/' in name or '\\' in name
                    or ':' in name or name in {'.', '..'} or name.endswith(('.', ' '))
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("ZIP must contain unique regular files at its root")
            names.add(name.casefold())
        if not any(entry.filename == 'solution.py' for entry in entries):
            raise ValueError("ZIP does not contain solution.py")
        archive.extractall(destination)


def _drain(stream, chunks):
    while True:
        data = stream.read(4096)
        if not data:
            break
        chunks.append(data)
    stream.close()


def _kill_tree(process, descendants):
    if os.name == 'nt':
        if process.poll() is None:
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(list(descendants.values())):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _safe_tail(chunks):
    result = b''.join(chunks).decode('utf-8', errors='replace')
    for name, value in os.environ.items():
        if value and ('KEY' in name.upper() or 'TOKEN' in name.upper() or 'SECRET' in name.upper()):
            result = result.replace(value, '[REDACTED]')
    return result[-800:]


def run_task(zip_path, task_dir, limit_seconds):
    if not math.isfinite(limit_seconds) or limit_seconds <= 0:
        raise ValueError('Time limit must be positive')
    task_dir = Path(task_dir).resolve()
    work = Path(tempfile.mkdtemp(prefix='platform-'))
    code_dir, data_dir = work / 'code', work / 'data'
    _extract(zip_path, code_dir)
    # A previous local run must never make an unsuccessful run look successful.
    shutil.copytree(task_dir, data_dir, ignore=shutil.ignore_patterns('submission.csv', '.agent-*'))
    command = [part.format(python=sys.executable, task_md=str(data_dir / 'task.md')) for part in COMMAND]
    options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=code_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    out, err = deque(maxlen=LOG_LIMIT // 4096), deque(maxlen=LOG_LIMIT // 4096)
    readers = [threading.Thread(target=_drain, args=(stream, chunks), daemon=True)
               for stream, chunks in ((process.stdout, out), (process.stderr, err))]
    for reader in readers:
        reader.start()
    killed = False
    descendants = {}
    observed = psutil.Process(process.pid)
    try:
        while process.poll() is None:
            try:
                descendants.update({child.pid: child for child in observed.children(recursive=True)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            remaining = limit_seconds - (time.monotonic() - started)
            if remaining <= 0:
                killed = True
                break
            try:
                process.wait(timeout=min(remaining, 0.02))
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Also reap descendants left behind by an exited entry point.
        _kill_tree(process, descendants)
        process.wait(timeout=15)
        for reader in readers:
            reader.join(timeout=2)
    elapsed = time.monotonic() - started
    answers_src = task_dir.parent / '_answers' / f'{task_dir.name}.csv'
    if answers_src.exists():
        (work / '_answers').mkdir(exist_ok=True)
        shutil.copy(answers_src, work / '_answers' / 'data.csv')
        metadata = answers_src.with_suffix('.json')
        if metadata.exists():
            shutil.copy(metadata, work / '_answers' / 'data.json')
        else:
            (work / '_answers' / 'data.json').write_text(json.dumps({'kind': task_dir.name}), encoding='utf-8')
    check = subprocess.run([sys.executable, str(HERE / 'check_submission.py'), str(data_dir)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    try:
        checked = json.loads(check.stdout)
    except (ValueError, TypeError):
        checked = {'format_ok': False, 'problems': ['Submission checker failed'], 'metric': None}
    return {'task': task_dir.name, 'exit_code': process.returncode, 'killed_by_limit': killed,
            'success': process.returncode == 0 and not killed and bool(checked.get('format_ok')),
            'minutes': round(elapsed / 60, 2), 'check': checked,
            'stderr_tail': _safe_tail(err), 'work_dir': str(work)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip')
    parser.add_argument('tasks_root')
    parser.add_argument('--limit-min', type=float, default=180)
    args = parser.parse_args(argv)
    tasks = sorted(p for p in Path(args.tasks_root).iterdir() if (p / 'task.md').exists())
    if not tasks or not math.isfinite(args.limit_min) or args.limit_min <= 0:
        parser.error('Provide at least one task and a positive time limit')
    report = [run_task(Path(args.zip).resolve(), task, args.limit_min * 60) for task in tasks]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item.get('success', False) for item in report) else 1


if __name__ == '__main__':
    sys.exit(main())
