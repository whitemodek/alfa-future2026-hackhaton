"""Unlabeled-domain diagnostics and group-disjoint split construction."""
import time
import unittest

import numpy as np
import pandas as pd

from qa.domain_weights import estimate_domain_weights
from qa.research_domain_shift import target_like_folds, weighted_training


class DomainTests(unittest.TestCase):
    def test_target_like_splits_keep_groups_and_preserve_every_training_class(self):
        groups = np.repeat(np.arange(100), 3)
        y = np.tile(np.arange(3), 100)
        weights = np.repeat(np.linspace(.25, 4., 100), 3)
        folds = target_like_folds(groups, y, weights)
        seen = set()
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            self.assertFalse(seen & set(groups[valid]))
            self.assertEqual(set(y[train]), {0, 1, 2})
            self.assertGreaterEqual(len(valid), 60)
            seen.update(groups[valid])
        replay = target_like_folds(groups, y, weights)
        for first, second in zip(folds, replay):
            np.testing.assert_array_equal(first[1], second[1])

    def test_invalid_density_weights_fail_before_split(self):
        for weights in ([0, 1], [np.nan, 1], [1]):
            with self.assertRaises(ValueError):
                target_like_folds([0, 1], [0, 1], weights)

    def test_training_weight_adapter_rejects_a_different_subset(self):
        frame = pd.DataFrame({'x': [0, 1, 2]}, index=[10, 20, 30])
        adapter = weighted_training(lambda *args: np.ones(3), frame, np.array([.5, 1, 1.5]))
        np.testing.assert_allclose(adapter(frame, np.array([0, 1, 0]), 'None'), [.5, 1, 1.5])
        with self.assertRaises(ValueError):
            adapter(frame.iloc[::-1], np.array([0, 1, 0]), 'None')

    def test_no_target_arguments_are_needed_for_finite_group_oof_diagnostic(self):
        source = pd.DataFrame({'x': np.tile(np.arange(120), 4).astype(float)})
        target = pd.DataFrame({'x': np.tile(np.arange(120) + 100, 4).astype(float)})
        weights, diagnostic = estimate_domain_weights(source, target, [], time.monotonic() + 20, 1)
        self.assertEqual(weights.shape, (len(source),))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertFalse(diagnostic['target_labels_used'])
        self.assertGreater(diagnostic['group_oof_auc'], .65)


if __name__ == '__main__':
    unittest.main()
