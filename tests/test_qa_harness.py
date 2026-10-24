"""Independent checks for ZIP execution and the external QA scorer."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import psutil
from pack_solution import build
from qa.run_local import _extract, _safe_tail, run_task
from qa.check_submission import check_format, score
from test_cli import fake_api, tool_call


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / 'reg_smoke'
        self.task.mkdir()
        (self.task / 'task.md').write_text('Write predictions matching sample_submission.csv.', encoding='utf-8')
        (self.task / 'sample_submission.csv').write_text('id,prediction\na,0.0\nb,0.0\n')
        (self.root / '_answers').mkdir()
        (self.root / '_answers' / 'reg_smoke.csv').write_text('id,prediction\na,0.2\nb,0.8\n')
        self.work_dirs = []
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.cleanup_runs)

    def cleanup_runs(self):
        for work in self.work_dirs:
            shutil.rmtree(work, ignore_errors=True)

    def run_archive(self, source, limit=10):
        archive = self.root / 'toy.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('solution.py', source)
        result = run_task(archive, self.task, limit)
        self.work_dirs.append(result['work_dir'])
        return result

    def test_existing_submission_cannot_mask_no_output(self):
        (self.task / 'submission.csv').write_text('id,prediction\na,0.2\nb,0.8\n')
        result = self.run_archive('pass')
        self.assertEqual(result['exit_code'], 0)
        self.assertFalse(result['check']['format_ok'])
        self.assertIsNone(result['check']['metric'])

    def test_noisy_timeout_kills_descendant_and_bounds_logs(self):
        source = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).with_name('child.pid').write_text(str(child.pid))
for i in range(500):
    sys.stderr.write('noise' * 1000)
sys.stderr.flush()
time.sleep(60)
"""
        result = self.run_archive(source, limit=1.5)
        self.assertTrue(result['killed_by_limit'])
        self.assertLessEqual(len(result['stderr_tail']), 800)
        child_pid = int((Path(result['work_dir']) / 'data' / 'child.pid').read_text())
        try:
            child = psutil.Process(child_pid)
            child.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            child.kill()
            self.fail('Timed out solution left a descendant running')

    def test_zip_rejects_unsafe_names_before_extracting(self):
        archive = self.root / 'unsafe.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('solution.py', 'pass')
            output.writestr('../escape.py', 'pass')
        with self.assertRaises(ValueError):
            _extract(archive, self.root / 'extract')
        self.assertFalse((self.root / 'escape.py').exists())

    def test_zip_rejects_windows_filename_aliases_on_every_platform(self):
        for alias in ('SOLUTION.PY', 'solution.py.', 'solution.py '):
            with self.subTest(alias=alias):
                archive = self.root / 'alias.zip'
                destination = self.root / 'unpacked'
                with zipfile.ZipFile(archive, 'w') as output:
                    output.writestr('solution.py', 'pass')
                    output.writestr(alias, "raise RuntimeError('overwritten')")
                with self.assertRaises(ValueError):
                    _extract(archive, destination)
                self.assertFalse(destination.exists())

    def test_invalid_csv_never_receives_metric(self):
        for value in ('id,prediction\na,0.2\n', 'id,prediction\na,inf\nb,0.8\n',
                      'id,prediction\nb,0.2\na,0.8\n'):
            with self.subTest(value=value):
                (self.task / 'submission.csv').write_text(value)
                problems, submission = check_format(self.task)
                self.assertTrue(problems)
                self.assertIsNone(score(self.task, submission))

    def test_scoring_matches_ids_and_preserves_text_ids(self):
        (self.task / 'sample_submission.csv').write_text('id,prediction\n001,0.0\n002,0.0\n')
        (self.task / 'submission.csv').write_text('id,prediction\n001,0.2\n002,0.8\n')
        (self.root / '_answers' / 'reg_smoke.csv').write_text('id,prediction\n002,0.8\n001,0.2\n')
        problems, submission = check_format(self.task)
        self.assertFalse(problems)
        self.assertEqual(score(self.task, submission), {'rmse': 0.0})

    def test_log_tail_redacts_secret_before_truncating(self):
        with patch.dict(os.environ, {'TEST_API_KEY': 'secret-value'}):
            result = _safe_tail([b'secret-value' + b'x' * 795])
        self.assertNotIn('value', result)

    def test_current_final_zip_runs_cli_and_is_scored(self):
        archive = self.root / 'final.zip'
        build(archive)
        steps = [tool_call('configure_submission', {
                    'columns': ['id', 'prediction'], 'expected_rows': 2,
                    'id_columns': ['id'], 'numeric_columns': ['prediction']}),
                 tool_call('run_python', {'code':
                     "from pathlib import Path\nPath('candidate.csv').write_text('id,prediction\\na,0.2\\nb,0.8\\n')"}),
                 tool_call('register_candidate', {'candidate_path': 'candidate.csv'}),
                 tool_call('finish', {})]
        with fake_api(steps) as (url, requests):
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'offline-key', 'OPENAI_BASE_URL': url, 'OPENAI_MODEL': 'offline-model',
                'AGENT_TOTAL_SECONDS': '40', 'AGENT_RESERVE_SECONDS': '5', 'AGENT_CALL_SECONDS': '5',
                'AGENT_API_SECONDS': '5', 'AGENT_MAX_STEPS': '8', 'AGENT_RETRIES': '0',
                'NO_PROXY': '127.0.0.1,localhost', 'AGENT_ML_BASELINE_SECONDS': '0',
            }):
                result = run_task(archive, self.task, 50)
                self.work_dirs.append(result['work_dir'])
        self.assertEqual(result['exit_code'], 0, result)
        self.assertTrue(result['check']['format_ok'], result)
        self.assertEqual(result['check']['metric'], {'rmse': 0.0})
        self.assertGreater(len(requests), 0)


if __name__ == '__main__':
    unittest.main()
