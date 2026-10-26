"""Optional forest work must not erase the established native ensemble."""
import unittest
from unittest.mock import patch

from ml_portfolio import run_portfolio


class PortfolioScheduleTests(unittest.TestCase):
    def schedule(self, *, budget=300, pair_seconds=5, forest_seconds=20,
                 forest_error=None, extended_pair_seconds=5, linear=False):
        clock = [0.0]
        calls, ensembles, saved = [], [], []

        def result(model, score):
            return dict(status='ok', model=model, score=score, metric='macro_f1',
                        direction='maximize', validation_id='fixed-fold',
                        manifest_path=model + '.json', candidate_path=model + '.csv')

        def train(envelope):
            request = envelope['request']
            family = request['model']
            calls.append((family, envelope['time_budget_seconds'], dict(request)))
            if family == 'extra_trees':
                clock[0] += forest_seconds
                if forest_error:
                    raise forest_error('optional forest interrupted')
                return result(family, .64)
            clock[0] += 10
            return result(family, {'catboost': .60, 'lightgbm': .65, 'xgboost': .64, 'linear': .70}[family])

        def ensemble(envelope, started, allowance, threads):
            paths = envelope['request']['manifest_paths']
            ensembles.append(list(paths))
            if 'extra_trees.json' not in paths:
                clock[0] += pair_seconds
                return result('legacy_ensemble', .67)
            clock[0] += extended_pair_seconds
            return result('forest_pair' if len(paths) == 2 else 'forest_triple',
                          .68 if len(paths) == 2 else .69)

        def checkpoint(envelope, best):
            saved.append(dict(best))

        envelope = {'request': {'action': 'portfolio', 'seed_bagging': False,
                                'entity_history': False, 'linear': linear}}
        patches = (patch('ml_portfolio.time.monotonic', side_effect=lambda: clock[0]),
                   patch('ml_backend.run', side_effect=train),
                   patch('ml_backend.load_problem', return_value={'classes': [0, 1]}),
                   patch('ml_portfolio.run_ensemble', side_effect=ensemble),
                   patch('ml_portfolio.checkpoint', side_effect=checkpoint))
        return envelope, patches, calls, ensembles, saved, clock, budget

    def execute(self, case):
        envelope, patches, *_ = case
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            return run_portfolio(envelope, 0, case[-1], 2)

    def test_optional_forest_overrun_returns_original_ensemble_and_checkpoints_it_first(self):
        case = self.schedule(forest_seconds=300, forest_error=TimeoutError)
        result = self.execute(case)
        _, _, calls, ensembles, saved, _, _ = case
        self.assertEqual(result['model'], 'legacy_ensemble')
        self.assertEqual(ensembles, [['lightgbm.json', 'xgboost.json']])
        self.assertEqual(saved[-1]['model'], 'legacy_ensemble')
        self.assertEqual([name for name, _, _ in calls],
                         ['catboost', 'lightgbm', 'xgboost', 'extra_trees'])
        self.assertEqual(calls[-1][1], 45)

    def test_worker_interruption_after_native_pair_still_has_its_checkpoint(self):
        case = self.schedule(forest_seconds=300, forest_error=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(case)
        self.assertEqual(case[4][-1]['model'], 'legacy_ensemble')

    def test_fixed_forest_pair_and_triple_follow_the_original_pair(self):
        case = self.schedule()
        result = self.execute(case)
        self.assertEqual(case[3], [['lightgbm.json', 'xgboost.json'],
                                   ['lightgbm.json', 'extra_trees.json'],
                                   ['lightgbm.json', 'xgboost.json', 'extra_trees.json']])
        self.assertEqual(result['model'], 'forest_triple')
        self.assertEqual([item['model'] for item in case[4]],
                         ['catboost', 'lightgbm', 'legacy_ensemble', 'forest_pair', 'forest_triple'])
        self.assertNotIn('iterations', case[2][-1][2])

    def test_slow_original_pair_prevents_further_expensive_ensembles(self):
        case = self.schedule(budget=180, pair_seconds=40, forest_seconds=45)
        result = self.execute(case)
        self.assertEqual(result['model'], 'legacy_ensemble')
        self.assertEqual(len(case[3]), 1)
        self.assertEqual(case[2][-1][1], 45)

    def test_forest_is_skipped_without_training_and_ensemble_reserve(self):
        case = self.schedule(budget=100)
        result = self.execute(case)
        self.assertEqual(result['model'], 'legacy_ensemble')
        self.assertEqual(len(case[2]), 3)

    def test_slow_forest_pair_prevents_triple_using_its_measured_cost(self):
        case = self.schedule(budget=180, extended_pair_seconds=50)
        result = self.execute(case)
        self.assertEqual(len(case[3]), 2)
        self.assertEqual(result['model'], 'forest_pair')

    def test_linear_runs_after_tree_checkpoints_and_can_improve_them(self):
        case = self.schedule(linear=True)
        result = self.execute(case)
        self.assertEqual(result['model'], 'linear')
        self.assertEqual(case[2][-1][0], 'linear')
        self.assertEqual(case[2][-1][1], 20)
        self.assertEqual(case[4][-2]['model'], 'forest_triple')
        self.assertEqual(case[4][-1]['model'], 'linear')

    def test_linear_skipped_when_optional_work_exhausts_budget(self):
        case = self.schedule(linear=True, forest_seconds=300, forest_error=TimeoutError)
        self.assertEqual(self.execute(case)['model'], 'legacy_ensemble')
        self.assertNotIn('linear', [name for name, _, _ in case[2]])


if __name__ == '__main__':
    unittest.main()
