"""Recovery refits only the selected native model, without another provider call."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from agent_config import Config
from test_agent import Call, ScriptedProvider, reply


class NativeFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Regression. Metric RMSE. Return id,prediction.', encoding='utf-8')
        (self.root / 'sample_submission.csv').write_text('id,prediction\na,0\nb,0\n')
        self.now = 0

    def native(self, name, score=2, *, full=False):
        candidate = self.root / f'{name}.csv'
        candidate.write_text(f'id,prediction\na,{10 if full else 1}\nb,2\n')
        return {'exit_code': 0, 'response': {
            'status': 'ok', 'candidate_path': str(candidate),
            'schema': {'columns': ['id', 'prediction'], 'expected_rows': 2,
                       'id_columns': ['id'], 'numeric_columns': ['prediction'],
                       'sample_path': str(self.root / 'sample_submission.csv')},
            'score': score, 'metric': 'rmse', 'direction': 'minimize',
            'validation_id': 'fixed-holdout', 'model': 'catboost',
            'manifest_path': str(self.root / f'{name}.json'),
            'fit_scope': 'full_train' if full else 'holdout_train', 'final': full}}

    def runner(self, provider, **options):
        return Agent(self.task, provider, Config(total_seconds=90, reserve_seconds=10,
                     call_seconds=60, max_steps=4, **options), clock=lambda: self.now)

    def test_provider_failure_refits_selected_native_and_publishes_equal_score(self):
        provider = ScriptedProvider(reply(Call('run_backend', {'request': {'action': 'train'}})),
                                    RuntimeError('provider unavailable'))
        runner = self.runner(provider)
        trained, final = self.native('trained'), self.native('refitted', full=True)
        with patch.object(runner, '_backend', side_effect=[trained, final]) as backend:
            result = runner.run()
            runner._finalize_native()
        self.assertTrue(result['success'])
        self.assertEqual(result['reason'], 'RuntimeError')
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(backend.call_count, 2)
        request, timeout = backend.call_args.args
        self.assertEqual(request, {'action': 'refit', 'manifest_path': trained['response']['manifest_path']})
        self.assertLessEqual(timeout, runner.remaining() - 2)
        self.assertTrue(runner.store.best['final'])
        self.assertIn('a,10', Path(result['submission']).read_text())

    def test_api_attempt_exhaustion_does_not_prevent_local_refit(self):
        provider = ScriptedProvider(reply(Call('run_backend', {'request': {'action': 'train'}})))
        runner = self.runner(provider, max_api_attempts=1)
        with patch.object(runner, '_backend', side_effect=[self.native('trained'),
                                                         self.native('refitted', full=True)]) as backend:
            result = runner.run()
        self.assertEqual(result['reason'], 'BudgetExceeded')
        self.assertTrue(runner.store.best['final'])
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(backend.call_count, 2)

    def test_failed_or_incomparable_refit_preserves_original_once(self):
        for outcome in ('different_score', 'different_validation', 'partial', 'invalid_csv', 'exception'):
            with self.subTest(outcome=outcome):
                provider = ScriptedProvider(reply(Call('run_backend', {'request': {'action': 'train'}})),
                                            RuntimeError('provider unavailable'))
                runner = self.runner(provider)
                final = self.native('bad-final', full=True)
                if outcome == 'different_score':
                    final['response']['score'] = 1
                elif outcome == 'different_validation':
                    final['response']['validation_id'] = 'different-split'
                elif outcome == 'partial':
                    final['response']['fit_scope'] = 'holdout_train'
                elif outcome == 'invalid_csv':
                    Path(final['response']['candidate_path']).write_text('id,prediction\na,nan\nb,2\n')
                else:
                    final = RuntimeError('refit failed')
                with patch.object(runner, '_backend', side_effect=[self.native('trained'), final]) as backend:
                    result = runner.run()
                    runner._finalize_native()
                self.assertTrue(result['success'])
                self.assertFalse(runner.store.best['final'])
                self.assertTrue(runner.finalization_attempted)
                self.assertEqual(backend.call_count, 2)
                self.assertIn('a,1', Path(result['submission']).read_text())

    def test_stronger_custom_candidate_is_not_replaced_by_native_refit(self):
        (self.root / 'custom.csv').write_text('id,prediction\na,99\nb,2\n')
        provider = ScriptedProvider(
            reply(Call('run_backend', {'request': {'action': 'train'}})),
            reply(Call('register_candidate', {'candidate_path': 'custom.csv', 'score': 1,
                  'metric': 'rmse', 'direction': 'minimize', 'validation_id': 'fixed-holdout'})),
            RuntimeError('provider unavailable'))
        runner = self.runner(provider)
        with patch.object(runner, '_backend', return_value=self.native('trained')) as backend:
            result = runner.run()
        self.assertEqual(backend.call_count, 1)
        self.assertFalse(runner.finalization_attempted)
        self.assertIn('a,99', Path(result['submission']).read_text())

    def test_rejected_native_experiment_does_not_change_selected_refit(self):
        provider = ScriptedProvider(reply(Call('run_backend', {'request': {'action': 'train'}}, 1),
                                          Call('run_backend', {'request': {'action': 'experiment'}}, 2)),
                                    RuntimeError('provider unavailable'))
        runner = self.runner(provider)
        trained = self.native('selected', score=2)
        with patch.object(runner, '_backend', side_effect=[trained, self.native('weaker', score=3),
                                                         self.native('refitted', full=True)]) as backend:
            runner.run()
        self.assertEqual(backend.call_args.args[0]['manifest_path'], trained['response']['manifest_path'])
        self.assertTrue(runner.store.best['final'])

    def test_cancellation_skips_additional_training(self):
        provider = ScriptedProvider()
        runner = self.runner(provider)
        first = reply(Call('run_backend', {'request': {'action': 'train'}}))
        with patch.object(provider, 'create', side_effect=[first, KeyboardInterrupt]), \
                patch.object(runner, '_backend', return_value=self.native('trained')) as backend:
            result = runner.run()
        self.assertTrue(result['success'])
        self.assertEqual(result['reason'], 'KeyboardInterrupt')
        self.assertEqual(backend.call_count, 1)
        self.assertFalse(runner.finalization_attempted)

    def test_insufficient_remaining_time_skips_refit(self):
        provider = ScriptedProvider(reply(Call('run_backend', {'request': {'action': 'train'}})),
                                    RuntimeError('provider unavailable'))
        runner = self.runner(provider)
        def train(request, timeout):
            self.now = 75
            return self.native('trained')
        with patch.object(runner, '_backend', side_effect=train) as backend:
            result = runner.run()
        self.assertTrue(result['success'])
        self.assertEqual(backend.call_count, 1)
        self.assertFalse(runner.finalization_attempted)
