"""A startup fallback must not make an ordinary chat response terminate work."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from agent import Agent
from agent_config import Config
from test_agent import ScriptedProvider, Call, reply


class CompletionProtocolTests(unittest.TestCase):
    def test_plain_text_after_startup_does_not_skip_pending_improvement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Classification. Primary metric: accuracy.', encoding='utf-8')
            (root / 'baseline.csv').write_text('prediction\n0\n', encoding='utf-8')
            (root / 'better.csv').write_text('prediction\n1\n', encoding='utf-8')
            provider = ScriptedProvider(
                reply(content='I will inspect the startup model and try an improvement.'),
                reply(Call('register_candidate', dict(candidate_path='better.csv', score=.9,
                    metric='accuracy', direction='maximize', validation_id='fixed'))),
                reply(Call('finish', {})))
            runner = Agent(task, provider, Config(total_seconds=60, reserve_seconds=5, max_steps=5))

            def startup():
                runner.execute('configure_submission', dict(columns=['prediction'], expected_rows=1))
                runner.execute('register_candidate', dict(candidate_path='baseline.csv', score=.5,
                    metric='accuracy', direction='maximize', validation_id='fixed'))

            with patch.object(runner, '_bootstrap', side_effect=startup):
                result = runner.run(bootstrap=True)
            self.assertTrue(result['success'])
            self.assertEqual(runner.store.best['score'], .9)
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual((root / 'submission.csv').read_text(), 'prediction\n1\n')

    def test_repeated_plain_text_is_bounded_and_preserves_saved_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Predict target.', encoding='utf-8')
            (root / 'candidate.csv').write_text('target\n2\n', encoding='utf-8')
            provider = ScriptedProvider(*(reply(content='Done') for _ in range(5)))
            runner = Agent(task, provider, Config(total_seconds=60, reserve_seconds=5, max_steps=8))
            runner.execute('configure_submission', dict(columns=['target'], expected_rows=1))
            runner.execute('register_candidate', dict(candidate_path='candidate.csv'))
            result = runner.run()
            self.assertTrue(result['success'])
            self.assertEqual(result['reason'], 'invalid_completion')
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual((root / 'submission.csv').read_text(), 'target\n2\n')

    def test_explicit_finish_does_not_force_additional_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Predict target.', encoding='utf-8')
            (root / 'candidate.csv').write_text('target\n2\n', encoding='utf-8')
            provider = ScriptedProvider(reply(Call('finish', {})))
            runner = Agent(task, provider, Config(total_seconds=60, reserve_seconds=5))
            runner.execute('configure_submission', dict(columns=['target'], expected_rows=1))
            runner.execute('register_candidate', dict(candidate_path='candidate.csv'))
            self.assertEqual(runner.run()['reason'], 'completed')
            self.assertEqual(len(provider.requests), 1)


if __name__ == '__main__':
    unittest.main()
