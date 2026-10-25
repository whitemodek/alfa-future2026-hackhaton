"""Bounded Python subprocesses. Resource controls are not a security sandbox.

Each invocation has fresh Python state; files in ``cwd`` persist. psutil is
required to monitor the combined resident memory of the process and children.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    import psutil
except ImportError:  # Report the dependency only when execution is requested.
    psutil = None


MAX_CODE_BYTES = 128 * 1024
_SAFE_ENV = {
    "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
}
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def _child_environment(threads: int) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() in _SAFE_ENV}
    environment.update({name: str(threads) for name in _THREAD_ENV})
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       PYTHONNOUSERSITE="1")
    return environment


class _Capture:
    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()
        self.tail = bytearray()
        self.total = 0
        self.truncated = False

    def drain(self, pipe) -> None:
        try:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                self.total += len(chunk)
                head_limit = (self.limit + 1) // 2
                tail_limit = self.limit - head_limit
                head_room = head_limit - len(self.data)
                self.data.extend(chunk[:head_room])
                remainder = chunk[head_room:]
                if tail_limit:
                    self.tail.extend(remainder)
                    if len(self.tail) > tail_limit:
                        del self.tail[:-tail_limit]
                self.truncated = self.total > self.limit
        except (OSError, ValueError):
            pass
        finally:
            pipe.close()

    def text(self) -> str:
        if not self.truncated:
            return (self.data + self.tail).decode("utf-8", errors="replace")
        # Preserve final metrics and tracebacks after verbose training progress.
        return (self.data.decode("utf-8", errors="replace") + "\n[output truncated]\n" +
                self.tail.decode("utf-8", errors="replace"))


def _cleanup(process: subprocess.Popen, descendants: dict) -> None:
    """Kill retained descendants even if the direct child has already exited."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in reversed(list(descendants.values())):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


def run_python(
    code: str,
    cwd: Path,
    timeout_seconds: float,
    memory_mb: int = 28000,
    threads: int = 8,
    max_output_bytes: int = 24000,
) -> dict:
    """Run code with bounded output, deadline, and process-tree RSS monitoring.

    ``max_output_bytes`` applies separately to stdout and stderr, excluding the
    truncation marker. Memory is sampled every 20 ms; it is not a hard OS limit.
    Windows descendant cleanup is best effort. A process deliberately escaping
    its group or disappearing between samples needs an external OS sandbox.
    Secrets are excluded from the environment, but files and network remain
    accessible. Invalid inputs or missing psutil raise before code is run.
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Python code exceeds 128 KiB")
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    for name, value in (("memory_mb", memory_mb), ("threads", threads),
                        ("max_output_bytes", max_output_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("cwd must be a directory")
    if psutil is None:
        raise RuntimeError("psutil is required for process memory enforcement")

    started = time.monotonic()
    timed_out = False
    memory_exceeded = False
    descendants = {}
    stdout = _Capture(max_output_bytes)
    stderr = _Capture(max_output_bytes)
    readers = []
    process = None
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py",
                                         prefix=".agent_run_", dir=cwd,
                                         delete=False) as script:
            script_path = Path(script.name)
            script.write(code)
        environment = _child_environment(threads)
        # LightGBM imports matplotlib. A credential-free environment has no HOME
        # on Windows; keep its optional cache in this invocation's task folder.
        environment.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(script_path.with_suffix('.cache')))
        process = subprocess.Popen(
            [sys.executable, "-u", str(script_path)], cwd=cwd,
            env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           if os.name == "nt" else 0),
        )
        for capture, pipe in ((stdout, process.stdout), (stderr, process.stderr)):
            reader = threading.Thread(target=capture.drain, args=(pipe,), daemon=True)
            readers.append(reader)
            reader.start()
        try:
            root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            root = None
        while process.poll() is None:
            if time.monotonic() - started >= timeout_seconds:
                timed_out = True
                break
            try:
                children = root.children(recursive=True) if root else []
                for child in children:
                    descendants[child.pid] = child
                resident = 0
                for current in ([root] if root else []) + list(descendants.values()):
                    try:
                        resident += current.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                if resident > memory_mb * 1024 * 1024:
                    memory_exceeded = True
                    break
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as exc:
                raise RuntimeError("Cannot enforce subprocess memory limit") from exc
            time.sleep(min(0.02, max(0, timeout_seconds - (time.monotonic() - started))))
    finally:
        if process is not None:
            _cleanup(process, descendants)
        for reader in readers:
            reader.join(timeout=1)
        if script_path is not None:
            script_path.unlink(missing_ok=True)
    return {
        "stdout": stdout.text(), "stderr": stderr.text(),
        "exit_code": process.returncode, "timed_out": timed_out,
        "memory_exceeded": memory_exceeded,
        "duration_seconds": time.monotonic() - started,
    }
