"""Strict causality checks for the offline history-statistics hypothesis."""
import inspect
import unittest

import numpy as np
import pandas as pd

from ml_lag import causal_lags
from qa.research_causal_stats import causal_stats


class CausalStatsTests(unittest.TestCase):
    def frame(self, periods, keys=None):
        return pd.DataFrame({'entity': ['a'] * len(periods) if keys is None else keys, 'period': periods})

    def test_irregular_gap_stats_and_no_query_label_parameter(self):
        train, query = self.frame([1, 3, 4]), self.frame([6, 8])
        a, b, _ = causal_stats(train, [10, 14, 20], query, ['entity'], 'period')
        np.testing.assert_allclose(a.iloc[2], [14, 1, 12, 2, 1, 2])
        np.testing.assert_allclose(b.iloc[0], [20, 1, 17, 6, 2, 3])
        np.testing.assert_allclose(b.iloc[1], [20, 1, 17, 6, 4, 3])
        self.assertNotIn('query_target', inspect.signature(causal_stats).parameters)

    def test_changing_own_and_future_targets_never_changes_earlier_features(self):
        train, query = self.frame([4, 1, 3, 2]), self.frame([5])
        original = np.array([40., 10., 30., 20.])
        before, _, _ = causal_stats(train, original, query, ['entity'], 'period')
        for cutoff in [1, 2, 3, 4]:
            changed = original.copy()
            changed[train.period.to_numpy() >= cutoff] += 10000
            after, _, _ = causal_stats(train, changed, query, ['entity'], 'period')
            np.testing.assert_allclose(before[train.period <= cutoff], after[train.period <= cutoff],
                                       equal_nan=True)

    def test_identical_collision_disables_only_after_available_period(self):
        train, query = self.frame([1, 2, 2, 3]), self.frame([4])
        a, b, d = causal_stats(train, [10, 20, 20, 30], query, ['entity'], 'period')
        np.testing.assert_allclose(a.iloc[1], [10, 1, 10, np.nan, 1, 1], equal_nan=True)
        np.testing.assert_allclose(a.iloc[2], a.iloc[1], equal_nan=True)
        self.assertEqual(a.iloc[3, 1], 0)
        self.assertTrue(np.isnan(a.iloc[3, 0]))
        self.assertEqual(b.iloc[0, 5], 0)
        self.assertEqual(d['ambiguous_keys'], 1)

    def test_missing_nonfinite_and_unknown_keys_fall_back(self):
        train = self.frame([1, 2, 3], ['a', None, np.inf])
        query = self.frame([4, 4, 4, 4], ['a', None, np.inf, 'unknown'])
        a, b, _ = causal_stats(train, [10, 20, 30], query, ['entity'], 'period')
        np.testing.assert_array_equal(b.iloc[:, 1], [1, 0, 0, 0])
        np.testing.assert_array_equal(b.iloc[:, 5], [1, 0, 0, 0])
        self.assertTrue(np.isnan(a.iloc[1, 0]))
        self.assertTrue(np.isnan(b.iloc[1:, 2:5]).all().all())

    def test_first_two_features_match_runtime_on_shuffled_duplicate_rows(self):
        train = self.frame([3, 1, 2, 2, 4, 1, 3], ['a', 'a', 'a', 'a', 'a', 'b', 'b'])
        query = self.frame([5, 6, 5], ['a', 'b', None])
        y = [30, 10, 20, 21, 40, 50, 70]
        a, b, _ = causal_stats(train, y, query, ['entity'], 'period')
        control_a, control_b, _ = causal_lags(train, y, query, ['entity'], 'period')
        np.testing.assert_allclose(a.iloc[:, 0], control_a, equal_nan=True)
        np.testing.assert_allclose(b.iloc[:, 0], control_b, equal_nan=True)
        np.testing.assert_array_equal(a.iloc[:, 1], np.isfinite(control_a).astype(int))

    def test_query_target_cells_are_ignored_and_overlap_rejected(self):
        train, query = self.frame([1, 2]), self.frame([3])
        query['target'] = 999
        _, before, _ = causal_stats(train, [10, 20], query, ['entity'], 'period')
        query['target'] = -10000
        _, after, _ = causal_stats(train, [10, 20], query, ['entity'], 'period')
        np.testing.assert_allclose(before, after, equal_nan=True)
        with self.assertRaises(ValueError):
            causal_stats(train, [10, 20], self.frame([2]), ['entity'], 'period')


if __name__ == '__main__':
    unittest.main()
