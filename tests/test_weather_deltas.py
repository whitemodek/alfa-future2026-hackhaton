"""Weather changes depend only on past feature snapshots, never targets."""
import time
import unittest

import numpy as np
import pandas as pd

from qa.research_weather_delta import causal_feature_deltas


class WeatherDeltaTests(unittest.TestCase):
    def test_unsorted_history_uses_last_strictly_earlier_period(self):
        train = pd.DataFrame(dict(entity=['a', 'a', 'b', 'a'], period=[3, 1, 1, 2], weather=[30, 10, 50, 20]))
        query = pd.DataFrame(dict(entity=['a', 'b', 'new'], period=[4, 4, 4], weather=[34, 55, 12]))
        a, b = causal_feature_deltas(train, query, ['entity'], 'period', ['weather'])
        np.testing.assert_allclose(a.ravel(), [10, np.nan, np.nan, 10], equal_nan=True)
        np.testing.assert_allclose(b.ravel(), [4, 5, np.nan], equal_nan=True)

    def test_same_period_collisions_disable_only_later_snapshots(self):
        train = pd.DataFrame(dict(entity=['a'] * 4, period=[1, 2, 2, 3], weather=[10, 20, 21, 30]))
        query = pd.DataFrame(dict(entity=['a'], period=[4], weather=[40]))
        a, b = causal_feature_deltas(train, query, ['entity'], 'period', ['weather'])
        np.testing.assert_allclose(a.ravel(), [np.nan, 10, 11, np.nan], equal_nan=True)
        self.assertTrue(np.isnan(b).all())

    def test_unused_targets_cannot_change_features_and_invalid_time_fails(self):
        train = pd.DataFrame(dict(entity=['a', 'a'], period=[1, 2], weather=[10, 20], target=[5, 500]))
        query = pd.DataFrame(dict(entity=['a'], period=[3], weather=[25], target=[-9999]))
        first = causal_feature_deltas(train, query, ['entity'], 'period', ['weather'])
        train['target'] = 0
        query['target'] = 123456
        second = causal_feature_deltas(train, query, ['entity'], 'period', ['weather'])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            causal_feature_deltas(train, query.assign(period=2), ['entity'], 'period', ['weather'])
        with self.assertRaises(TimeoutError):
            causal_feature_deltas(train, query, ['entity'], 'period', ['weather'], deadline=time.monotonic() - 1)
