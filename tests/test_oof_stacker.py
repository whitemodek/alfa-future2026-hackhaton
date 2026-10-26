"""Fixed stacker uses OOS features and a separate group-isolated meta holdout."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.research_oof_stacker import (
    CLIP, PARAMETERS, fit_stacker, meta_features, meta_group_split,
    sqrt_weights, stack_probability, use_stacker,
)


class OOFStackerTests(unittest.TestCase):
    def test_features_are_ordered_log_probabilities_without_learned_statistics(self):
        probabilities = [np.array([[0., .4, .6], [.8, .1, .1]]) for _ in range(3)]
        result = meta_features(probabilities)
        self.assertEqual(result.shape, (2, 9))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:, :3], np.log(np.clip(probabilities[0], CLIP, 1)))
        np.testing.assert_array_equal(result[:, :3], result[:, 6:])
        with self.assertRaises(ValueError):
            meta_features(probabilities[:2])

    def test_normalized_sqrt_balancing_preserves_ratios_and_regularization_scale(self):
        weights = sqrt_weights(np.array([0, 0, 0, 0, 1]))
        self.assertAlmostEqual(float(weights.mean()), 1.)
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 2.)
        with self.assertRaises(ValueError):
            sqrt_weights(np.array([0, 2]))

    def test_meta_split_groups_original_features_and_preserves_training_classes(self):
        frame = pd.DataFrame({'number': np.repeat(np.arange(60), 2),
                              'category': np.tile(['same', 'same'], 60)})
        labels = np.repeat(np.arange(60) % 3, 2)
        train, valid, plan = meta_group_split(frame, labels, ['a', 'b', 'c'])
        groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        self.assertFalse(set(groups[train]) & set(groups[valid]))
        self.assertEqual(set(labels[train]), {0, 1, 2})
        self.assertEqual(plan['strategy'], 'group')
        again = meta_group_split(frame, labels, ['a', 'b', 'c'])
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(valid, again[1])

    def test_real_fixed_meta_fit_aligns_classes_and_never_uses_query_labels(self):
        rng = np.random.default_rng(11)
        labels = np.repeat(np.arange(3), 50)
        # Base one is informative but weakly scaled; no target is part of X.
        probability = np.full((150, 3), .1)
        probability[np.arange(150), labels] = .8
        uninformative = rng.dirichlet(np.ones(3), size=150)
        features = meta_features([probability, uninformative, uninformative])
        train, valid = np.arange(120), np.arange(120, 150)
        # Include all three classes in the fitting subset.
        model, weight_record = fit_stacker(features[train], labels[train], time.monotonic() + 20)
        predicted = stack_probability(model, features[valid], 3)
        self.assertGreater(np.mean(predicted.argmax(axis=1) == labels[valid]), .9)
        np.testing.assert_allclose(predicted.sum(axis=1), 1)
        self.assertEqual(model.C, .1)
        self.assertEqual(model.max_iter, 500)
        self.assertAlmostEqual(weight_record['mean'], 1.)
        self.assertEqual(PARAMETERS['solver'], 'lbfgs')

    def test_selection_is_strict_and_retains_control_on_ties(self):
        self.assertFalse(use_stacker(.6, .6))
        self.assertFalse(use_stacker(.6, .5))
        self.assertTrue(use_stacker(.6, .61))
        with self.assertRaises(ValueError):
            use_stacker(.6, np.nan)

    def test_validation_rows_are_absent_from_fit_labels_and_weights(self):
        features = np.arange(180, dtype=float).reshape(30, 6)
        train = np.array([0, 2, 4, 6, 8, 10])
        labels = np.tile([0, 1, 2], 10)
        seen = []

        class FakeModel:
            def __init__(self, **parameters):
                pass

            def fit(self, x, y, sample_weight):
                seen.append((x.copy(), y.copy(), sample_weight.copy()))
                return self

        with patch('sklearn.linear_model.LogisticRegression', FakeModel):
            fit_stacker(features[train], labels[train], time.monotonic() + 10)
        np.testing.assert_array_equal(seen[0][0], features[train])
        np.testing.assert_array_equal(seen[0][1], labels[train])
        np.testing.assert_allclose(seen[0][2], sqrt_weights(labels[train]))


if __name__ == '__main__':
    unittest.main()
