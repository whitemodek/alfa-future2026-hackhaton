"""A timed-out portfolio must retain its completed, validated best model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from agent_config import Config
from test_agent import SCHEMA, GOOD, ScriptedProvider


class NativeCheckpointTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Return predictions.', encoding='utf-8')
        self.candidate = self.root / 'candidate.csv'
        self.candidate.write_text(GOOD, encoding='utf-8')

    def result(self):
        return dict(version=1, status='ok', model='lightgbm', schema=SCHEMA,
                    candidate_path=str(self.candidate), score=.8, metric='accuracy',
                    direction='maximize', validation_id='same-split', fit_scope='holdout_train')

    def test_timed_out_native_process_registers_complete_checkpoint(self):
        # A real limited subprocess writes a complete candidate response and
        # hangs in its next experiment. Prior code discarded the entire result.
        worker = self.root / 'ml_backend.py'
        worker.write_text('import json,sys,time\nfrom pathlib import Path\n'
                          'Path(sys.argv[2]).write_text(' + repr(json.dumps(dict(self.result(), checkpoint=True))) + ')\n'
                          'time.sleep(20)\n', encoding='utf-8')
        agent = Agent(self.task, ScriptedProvider(), Config(total_seconds=30, reserve_seconds=5))
        with patch('agent.__file__', str(self.root / 'agent.py')):
            result = agent._backend({'action': 'portfolio'}, .5)
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['exit_code'], 0)
        registration = agent._accept_native_result(result)
        self.assertIsNotNone(registration)
        self.assertTrue(registration['accepted'])
        self.assertEqual(agent.store.best['score'], .8)
        self.assertEqual((self.root / 'submission.csv').read_text(), GOOD)

    def test_unmarked_failure_and_invalid_checkpoint_never_register(self):
        agent = Agent(self.task, ScriptedProvider(), Config(total_seconds=30, reserve_seconds=5))
        result = dict(exit_code=1, response=self.result())
        self.assertIsNone(agent._accept_native_result(result))
        result['checkpoint_recovered'] = True
        result['response']['status'] = 'error'
        self.assertIsNone(agent._accept_native_result(result))

    def test_startup_recovers_checkpoint_and_revalidates_csv(self):
        agent = Agent(self.task, ScriptedProvider(), Config(total_seconds=30, reserve_seconds=5))
        result = dict(exit_code=1, timed_out=True, checkpoint_recovered=True,
                      response=dict(self.result(), checkpoint=True))
        with patch.object(agent, '_backend', return_value=result):
            agent._bootstrap()
        self.assertEqual(agent.store.best['score'], .8)
        self.assertEqual((self.root / 'submission.csv').read_text(), GOOD)
        # Recovery is no exemption from the ordinary finite-number contract.
        self.candidate.write_text('id,prediction\na,inf\nb,2\n')
        fresh = Agent(self.task, ScriptedProvider(), Config(total_seconds=30, reserve_seconds=5))
        with self.assertRaises(ValueError):
            fresh._accept_native_result(result)

    def test_worker_keeps_atomic_checkpoint_if_a_later_trial_crashes(self):
        import ml_backend as b
        request = self.root / 'request.json'
        response = self.root / 'response.json'
        request.write_text(json.dumps(dict(version=1, request={'action': 'portfolio'})))

        def crash_after_success(envelope):
            envelope['_checkpoint_callback'](self.result())
            raise RuntimeError('later experiment failed')

        with patch('ml_backend.run', crash_after_success):
            code = b.main([str(request), str(response)])
        self.assertEqual(code, 0)
        saved = json.loads(response.read_text())
        self.assertEqual(saved['score'], .8)
        self.assertTrue(saved['checkpoint'])
        self.assertFalse(response.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
