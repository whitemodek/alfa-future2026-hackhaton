"""Search schedules cheap neighbors without weakening candidate comparability."""
import unittest
from unittest.mock import patch

from ml_backend import _search


class SearchTests(unittest.TestCase):
    def run_search(self, scores, elapsed=5, budget=180):
        clock = [0.0]
        calls = []

        def trial(envelope):
            calls.append(envelope)
            clock[0] += elapsed
            value = scores[len(calls) - 1]
            if isinstance(value, Exception):
                raise value
            depth = envelope['request']['depth']
            result = dict(status='ok', model='catboost', score=value,
                          metric='rmse', direction='minimize', validation_id='fixed-fold',
                          candidate_path=f'depth-{depth}.csv', manifest_path=f'depth-{depth}.json',
                          elapsed_seconds=elapsed, fit_scope='full_train')
            if isinstance(value, dict):
                result.update(value)
            return result

        envelope = dict(request=dict(action='search', split='group', random_seed=77),
                        time_budget_seconds=budget)
        with patch('ml_backend.time.monotonic', side_effect=lambda: clock[0]), \
                patch('ml_backend.run', side_effect=trial):
            result = _search(envelope, started=0, budget=budget)
        return result, calls

    def test_fast_runs_compare_neighbors_and_preserve_winning_manifest(self):
        result, calls = self.run_search([4.0, 5.0, 3.0])
        self.assertEqual([call['request']['depth'] for call in calls], [6, 4, 8])
        self.assertEqual(result['selected_depth'], 8)
        self.assertEqual(result['manifest_path'], 'depth-8.json')
        self.assertEqual(result['candidate_path'], 'depth-8.csv')
        self.assertEqual(result['score'], 3.0)
        self.assertEqual([entry['accepted'] for entry in result['search_results']], [True, False, True])
        for call in calls:
            self.assertEqual(call['request']['action'], 'experiment')
            self.assertEqual(call['request']['random_seed'], 77)
            self.assertEqual(call['request']['split'], 'group')
            self.assertLessEqual(call['time_budget_seconds'], 180)

    def test_slow_initial_run_does_not_start_neighbors(self):
        result, calls = self.run_search([.6], elapsed=80, budget=180)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['selected_depth'], 6)

    def test_failure_and_incomparable_score_preserve_first_valid_candidate(self):
        result, calls = self.run_search([4.0, RuntimeError('failed'),
                                        {'score': 0.0, 'validation_id': 'different-fold'}])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['candidate_path'], 'depth-6.csv')
        self.assertEqual(result['score'], 4.0)
        self.assertEqual(result['search_results'][1]['reason'], 'RuntimeError')
        self.assertEqual(result['search_results'][2]['reason'], 'different_validation')

    def test_maximize_metric_selects_larger_score(self):
        result, _ = self.run_search([{'score': .5, 'direction': 'maximize'},
                                     {'score': .7, 'direction': 'maximize'},
                                     {'score': .6, 'direction': 'maximize'}])
        self.assertEqual(result['score'], .7)
        self.assertEqual(result['selected_depth'], 4)

    def test_short_budget_does_not_start_underfunded_neighbors(self):
        result, calls = self.run_search([4.0], elapsed=1, budget=20)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result['search_results']), 1)
