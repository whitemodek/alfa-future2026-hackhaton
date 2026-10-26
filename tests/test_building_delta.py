"""Strict history and unchanged-fallback checks for offline residual forecasting."""
import unittest

import numpy as np
import pandas as pd

from qa.research_building_delta import delta_pipeline, residual_history


class BuildingDeltaTests(unittest.TestCase):
    def frame(self, periods, keys=None):
        return pd.DataFrame({'entity': ['a'] * len(periods) if keys is None else keys, 'period': periods})

    def test_residual_uses_last_strictly_past_value(self):
        result = residual_history(self.frame([1, 3, 4]), [10, 14, 20], self.frame([5]), ['entity'], 'period')
        np.testing.assert_allclose(result['residual_target'], [np.nan, 4, 6], equal_nan=True)
        np.testing.assert_allclose(result['query_lag'], [20])
        np.testing.assert_array_equal(result['train_available'], [False, True, True])

    def test_future_labels_cannot_change_earlier_lags_or_residuals(self):
        train = self.frame([1, 2, 3, 4])
        original = residual_history(train, [10, 20, 30, 40], self.frame([5]), ['entity'], 'period')
        changed = residual_history(train, [10, 20, 3000, 4000], self.frame([5]), ['entity'], 'period')
        np.testing.assert_allclose(original['train_lag'][:3], changed['train_lag'][:3], equal_nan=True)
        np.testing.assert_allclose(original['residual_target'][:2], changed['residual_target'][:2], equal_nan=True)

    def test_query_labels_are_ignored_and_collision_disables_future(self):
        train = self.frame([1, 2, 2, 3])
        query = self.frame([4])
        query['target'] = 100000
        a = residual_history(train, [10, 20, 20, 30], query, ['entity'], 'period')
        query['target'] = -100000
        b = residual_history(train, [10, 20, 20, 30], query, ['entity'], 'period')
        np.testing.assert_allclose(a['query_lag'], b['query_lag'], equal_nan=True)
        np.testing.assert_array_equal(a['train_available'], [False, True, True, False])
        self.assertFalse(a['query_available'][0])

    def test_missing_and_ambiguous_keys_keep_exact_full_baseline(self):
        existing = np.array([11.3, 222.25, -7.2, 1e6])
        result = delta_pipeline(existing, [10, 20, 30, 40], [8, np.nan, np.nan, 5], [2, -1])
        np.testing.assert_allclose(result, [10, 222.25, -7.2, 22])
        np.testing.assert_array_equal(result[[1, 2]], existing[[1, 2]])

    def test_no_available_history_accepts_empty_delta_predictions(self):
        before = np.array([12.5, -1.3])
        after = delta_pipeline(before, [1, 2], [np.nan, np.nan], [])
        np.testing.assert_array_equal(before, after)

    def test_overlapping_query_and_misaligned_delta_are_rejected(self):
        with self.assertRaises(ValueError):
            residual_history(self.frame([1, 2]), [10, 20], self.frame([2]), ['entity'], 'period')
        with self.assertRaises(ValueError):
            delta_pipeline([1, 2], [1, 2], [1, np.nan], [1, 2])


if __name__ == '__main__':
    unittest.main()
