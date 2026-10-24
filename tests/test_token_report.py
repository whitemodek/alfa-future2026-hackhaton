import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from qa.check_tokens import main, summarize


class TokenReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_run(self, name, events):
        folder = self.root / name
        folder.mkdir()
        (folder / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)), encoding='utf-8')

    def test_failed_reservations_and_repeated_runs_are_counted(self):
        self.write_run('.agent-first', [{'event': 'api', 'charged_tokens': 100},
                                       {'event': 'api_error', 'charged_tokens': 900},
                                       {'event': 'published', 'elapsed': 12}])
        self.write_run('.agent-second', [{'event': 'api', 'charged_tokens': 250}])
        report = summarize(self.root)
        self.assertEqual(report['charged_tokens'], 1150)
        self.assertTrue(report['accounting_complete'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), '--max-tokens', '1149']), 1)
            self.assertEqual(main([str(self.root), '--max-tokens', '1150']), 0)

    def test_old_unknown_failure_does_not_pass_a_budget_check(self):
        self.write_run('.agent-old', [{'event': 'api_error'}])
        self.assertFalse(summarize(self.root)['accounting_complete'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), '--max-tokens', '10000']), 1)

    def test_empty_log_and_no_api_cost_zero(self):
        self.write_run('.agent-empty', [])
        self.assertEqual(summarize(self.root)['charged_tokens'], 0)
