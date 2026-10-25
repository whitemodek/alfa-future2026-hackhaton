"""Regression tests for metric handling and native ML startup."""
import json
from pathlib import Path
import tempfile
import unittest

from agent import Agent
from agent_config import Config
from test_agent import ScriptedProvider, reply, Call


class NativeAgentTests(unittest.TestCase):
    def test_failed_native_worker_retains_bounded_error_response(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Regression. Metric RMSE.')
            runner = Agent(task, ScriptedProvider(), Config(total_seconds=60, reserve_seconds=5))
            result = runner._backend({'action': 'unsupported-action'}, 15)
            self.assertNotEqual(result['exit_code'], 0)
            self.assertEqual(result['response']['status'], 'error')
            self.assertIn('Unsupported action', result['response']['error'])
            self.assertIsNone(runner._accept_native_result(result))

    def test_native_baseline_survives_first_provider_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Regression. Predict target, metric RMSE. Use train.csv, test.csv and sample_submission.csv.')
            (root / 'train.csv').write_text('id,x,target\n' + ''.join(f'{i},{i},{i * 2}\n' for i in range(20)))
            (root / 'test.csv').write_text('id,x\n21,21\n22,22\n')
            (root / 'sample_submission.csv').write_text('id,target\n21,0\n22,0\n')
            provider = ScriptedProvider(RuntimeError('provider unavailable'))
            runner = Agent(task, provider, Config(total_seconds=60, reserve_seconds=5, warmup_seconds=0))
            result = runner.run(bootstrap=True)
            self.assertTrue(result['success'], result)
            self.assertEqual(result['reason'], 'RuntimeError')
            self.assertEqual(runner.store.best['rows'], 2)
            self.assertEqual(len(provider.requests), 1)
            self.assertEqual((root / 'submission.csv').read_text().splitlines()[0], 'id,target')

    def test_negative_sklearn_scores_keep_the_larger_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Regression task')
            (root / 'a.csv').write_text('prediction\n2\n')
            (root / 'b.csv').write_text('prediction\n3\n')
            provider = ScriptedProvider(
                reply(Call('configure_submission', {'columns': ['prediction'], 'expected_rows': 1})),
                reply(Call('register_candidate', {'candidate_path': 'a.csv', 'score': -5,
                      'metric': 'neg_root_mean_squared_error', 'direction': 'maximize', 'validation_id': 'fixed'})),
                reply(Call('register_candidate', {'candidate_path': 'b.csv', 'score': -2,
                      'metric': 'neg_root_mean_squared_error', 'validation_id': 'fixed'})),
                reply(Call('finish', {})))
            result = Agent(task, provider).run()
            self.assertTrue(result['success'])
            self.assertEqual((root / 'submission.csv').read_text(), 'prediction\n3\n')

    def test_invalid_schema_does_not_add_broken_output_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Return predictions')
            (root / 'candidate.csv').write_text('prediction\n2\n')
            (root / 'blocked').write_text('This is a file, not directory')
            provider = ScriptedProvider(
                reply(Call('configure_submission', {'columns': ['x', 'x'], 'expected_rows': 1,
                                                   'output_path': 'blocked/result.csv'})),
                reply(Call('configure_submission', {'columns': ['prediction'], 'expected_rows': 1})),
                reply(Call('register_candidate', {'candidate_path': 'candidate.csv'})),
                reply(Call('finish', {})))
            result = Agent(task, provider).run()
            self.assertTrue(result['success'])
            self.assertFalse((root / 'blocked' / 'result.csv').exists())


if __name__ == '__main__':
    unittest.main()
