"""AutoML classes use only fit-fold target counts and preserve duplicate groups."""
import importlib.util
import unittest

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('autogluon') is not None, 'Optional platform AutoGluon package is unavailable')
class AutoMLGroupTests(unittest.TestCase):
    def test_all_equal_feature_groups_stay_in_one_bag_fold(self):
        from ml_automl import grouped_bag_labels
        x = pd.DataFrame({'x': np.repeat(np.arange(90), 3)})
        y = np.tile(np.arange(3), 90)
        assignment = grouped_bag_labels(x, y)
        self.assertEqual(set(assignment), {0, 1, 2})
        for start in range(0, len(x), 3):
            self.assertEqual(len(set(assignment[start:start + 3])), 1)
        for fold in range(3):
            self.assertEqual(set(y[assignment != fold]), {0, 1, 2})
        np.testing.assert_array_equal(assignment, grouped_bag_labels(x, y))

    def test_single_group_rare_class_cannot_be_silently_lost(self):
        from ml_automl import grouped_bag_labels
        x = pd.DataFrame({'x': [0] * 10 + list(range(1, 31))})
        y = np.array([1] * 10 + [0] * 30)
        with self.assertRaises(ValueError):
            grouped_bag_labels(x, y)

    def test_class_weights_do_not_depend_on_validation_targets(self):
        from ml_automl import _FoldSqrtWeights

        class Capture:
            problem_type = 'multiclass'

            def _fit(self, X, y, sample_weight=None, **kwargs):
                return np.asarray(sample_weight).copy()

        class Model(_FoldSqrtWeights, Capture):
            pass

        model = Model()
        y = np.array([0, 0, 0, 1])
        a = model._fit(X=np.zeros((4, 1)), y=y, y_val=np.zeros(100, dtype=int))
        b = model._fit(X=np.zeros((4, 1)), y=y, y_val=np.ones(100, dtype=int))
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a.mean(), 1.)
        self.assertAlmostEqual(a[-1] / a[0], np.sqrt(3))
