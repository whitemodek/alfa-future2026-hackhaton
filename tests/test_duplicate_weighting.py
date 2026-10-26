"""Training-only repetition weights and preservation through exact refit."""
import time
import unittest

import numpy as np
import pandas as pd

from ml_models import _training_weights, fit


class DuplicateWeightTests(unittest.TestCase):
    def test_repetitions_reduce_relative_row_influence_and_mean_stays_one(self):
        x = pd.DataFrame({'x': [1, 1, 1, 1, 2, 3]})
        y = np.array([0, 1, 0, 1, 0, 1])
        weights = _training_weights(x, y, 'None', 'inverse_sqrt')
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[0] / weights[4], .5)
        self.assertEqual(weights[0], weights[1])  # groups use features, never labels

    def test_default_and_unique_rows_preserve_existing_class_weights(self):
        x = pd.DataFrame({'x': range(8)})
        y = np.array([0] * 6 + [1] * 2)
        base = _training_weights(x, y, 'SqrtBalanced')
        np.testing.assert_allclose(base, _training_weights(x, y, 'SqrtBalanced', 'inverse_sqrt'))
        self.assertIsNone(_training_weights(x, y, 'None'))

    def test_only_training_frame_determines_repetition_counts(self):
        train = pd.DataFrame({'x': [1, 1, 2, 3]})
        y = np.array([0, 1, 0, 1])
        before = _training_weights(train, y, 'SqrtBalanced', 'inverse_sqrt')
        # Validation/test covariates are deliberately absent from the function interface.
        unrelated_test = pd.DataFrame({'x': [1] * 1000})
        unrelated_test['x'] = 2
        np.testing.assert_array_equal(before, _training_weights(train, y, 'SqrtBalanced', 'inverse_sqrt'))

    def test_refit_retains_policy_and_invalid_modes_fail(self):
        x = pd.DataFrame({'x': np.tile(np.arange(12), 4).astype(float)})
        y = np.tile(np.arange(12) % 2, 4)
        for family in ('lightgbm', 'xgboost', 'extra_trees', 'linear'):
            with self.subTest(family=family):
                model = fit(family, x.iloc[:36], y[:36], x.iloc[36:], y[36:], [],
                    dict(iterations=10, duplicate_weighting='inverse_sqrt'), time.monotonic() + 15,
                    1, 'classification', 'macro_f1')
                full = model.refit(x, y, time.monotonic() + 15, 1)
                self.assertEqual(full.parameters['duplicate_weighting'], 'inverse_sqrt')
                self.assertEqual(full.tree_count_, model.tree_count_)
                self.assertTrue(np.isfinite(full.predict_proba(x)).all())
        with self.assertRaises(ValueError):
            _training_weights(x, y, 'None', 'unexpected')


if __name__ == '__main__':
    unittest.main()
