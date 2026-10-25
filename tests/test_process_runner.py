import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import process_runner
from process_runner import run_python


class InputTests(unittest.TestCase):
    def test_oversized_code_is_rejected(self):
        with self.assertRaises(ValueError):
            run_python("x" * (128 * 1024 + 1), Path.cwd(), 1)

    def test_missing_monitor_fails_closed(self):
        with patch.object(process_runner, "psutil", None):
            with self.assertRaisesRegex(RuntimeError, "psutil"):
                run_python("raise AssertionError", Path.cwd(), 1)

    def test_invalid_limits(self):
        for timeout in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                run_python("", Path.cwd(), timeout)


@unittest.skipIf(process_runner.psutil is None, "psutil must be installed")
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cwd = Path(self.temporary.name)

    def run_code(self, code, **kwargs):
        return run_python(code, self.cwd, **{"timeout_seconds": 5, **kwargs})

    def test_fresh_state_and_persistent_files(self):
        first = self.run_code("from pathlib import Path\nx = 42\nPath('saved').write_text('ok')")
        second = self.run_code("from pathlib import Path\nprint(Path('saved').read_text())\nprint('x' in globals())")
        self.assertEqual(first["exit_code"], 0)
        self.assertEqual(second["stdout"].splitlines(), ["ok", "False"])
        self.assertFalse(list(self.cwd.glob(".agent_run_*.py")))

    def test_secrets_removed_and_threads_capped(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private", "CUSTOM_PASSWORD": "secret",
                                     "PYTHONPATH": "danger", "OMP_NUM_THREADS": "100"}):
            result = self.run_code("import os\nprint([k for k in ('OPENAI_API_KEY', 'CUSTOM_PASSWORD', 'PYTHONPATH') if k in os.environ])\nprint(os.environ['OMP_NUM_THREADS'])", threads=2)
        self.assertEqual(result["stdout"].splitlines(), ["[]", "2"])

    def test_stdout_stderr_and_nonzero_exit(self):
        result = self.run_code("import sys\nprint('hello')\nprint('error', file=sys.stderr)\nsys.exit(7)")
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("hello", result["stdout"])
        self.assertIn("error", result["stderr"])

    def test_output_flood_does_not_deadlock(self):
        result = self.run_code("import sys\nfor _ in range(2000):\n print('x' * 1000)\n print('y' * 1000, file=sys.stderr)", max_output_bytes=100)
        self.assertEqual(result["exit_code"], 0)
        for key in ("stdout", "stderr"):
            self.assertLess(len(result[key]), 150)
            self.assertIn("truncated", result[key])

    def test_stdin_is_closed(self):
        result = self.run_code("import sys\nprint(repr(sys.stdin.read()))")
        self.assertEqual(result["stdout"].strip(), "''")

    def test_optional_model_import_has_writable_cache_without_home(self):
        import importlib.util
        if importlib.util.find_spec('lightgbm') is None:
            self.skipTest('LightGBM not installed')
        result = self.run_code("import os\nfrom pathlib import Path\nimport lightgbm\n"
                               "assert 'HOME' not in os.environ and 'USERPROFILE' not in os.environ\n"
                               "cache=Path(os.environ['MPLCONFIGDIR'])\n"
                               "assert cache.is_relative_to(Path.cwd())\nprint('model imported')",
                               timeout_seconds=20)
        self.assertEqual(result['exit_code'], 0, result)
        self.assertIn('model imported', result['stdout'])

    def test_long_training_output_keeps_final_metric_and_error(self):
        result = self.run_code("import sys\nprint('START')\nprint('x' * 50000)\n"
                               "print('FINAL_SCORE=0.75')\n"
                               "print('y' * 50000, file=sys.stderr)\n"
                               "raise RuntimeError('final diagnostic')", max_output_bytes=200)
        self.assertNotEqual(result['exit_code'], 0)
        self.assertTrue(result['stdout'].startswith('START'))
        self.assertIn('FINAL_SCORE=0.75', result['stdout'])
        self.assertIn('final diagnostic', result['stderr'])
        self.assertLess(len(result['stdout']), 240)

    def test_short_capture_preserves_all_bytes_in_order(self):
        result = self.run_code("import sys\nsys.stdout.write('abcdefghijk')", max_output_bytes=12)
        self.assertEqual(result['stdout'], 'abcdefghijk')

    def test_timeout(self):
        result = self.run_code("import time\ntime.sleep(60)", timeout_seconds=0.2)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["duration_seconds"], 3)

    def test_memory_limit(self):
        result = self.run_code("import time\nx=bytearray(96 * 1024 * 1024)\ntime.sleep(10)", memory_mb=48)
        self.assertTrue(result["memory_exceeded"])
        self.assertFalse(result["timed_out"])

    def test_descendant_cleanup_on_timeout_and_normal_exit(self):
        for timeout in (2, 8):
            with self.subTest(timeout=timeout):
                marker = self.cwd / "child_pid"
                marker.unlink(missing_ok=True)
                result = self.run_code(
                    "import subprocess, sys, time\nfrom pathlib import Path\n"
                    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                    "Path('child_pid').write_text(str(child.pid))\ntime.sleep(3)",
                    timeout_seconds=timeout)
                self.assertEqual(result["timed_out"], timeout < 3)
                pid = int(marker.read_text())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        child = process_runner.psutil.Process(pid)
                        if child.status() == process_runner.psutil.STATUS_ZOMBIE:
                            break
                    except process_runner.psutil.NoSuchProcess:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"Child {pid} survived process cleanup")


if __name__ == "__main__":
    unittest.main()
