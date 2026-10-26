"""Bounded transductive label-shift adaptation uses no target answers."""
import unittest
import numpy as np
from qa.label_shift import adapt_label_shift, class_prior


class LabelShiftTests(unittest.TestCase):
    def test_identical_populations_keep_the_decision_probabilities(self):
        source = np.array([[.9, .1], [.4, .6], [.2, .8], [.7, .3]])
        corrected, diagnostic = adapt_label_shift(source, source, [.5, .5], [.5, .5])
        np.testing.assert_allclose(corrected, source)
        np.testing.assert_allclose(diagnostic['clipped_ratio'], [1, 1])
        self.assertFalse(diagnostic['target_labels_used'])

    def test_population_shift_is_bounded_and_normalized(self):
        source = np.array([[.95, .05], [.05, .95]] * 30)
        target = np.array([[.95, .05]] * 6 + [[.05, .95]] * 54)
        corrected, diagnostic = adapt_label_shift(source, target, [.5, .5], [.5, .5])
        np.testing.assert_allclose(corrected.sum(axis=1), 1)
        ratio = np.array(diagnostic['clipped_ratio'])
        self.assertTrue(((ratio >= .25) & (ratio <= 4)).all())
        self.assertGreater(ratio[1], ratio[0])

    def test_absent_training_classes_and_invalid_priors_fail(self):
        with self.assertRaises(ValueError):
            class_prior([0, 0], 2)
        with self.assertRaises(ValueError):
            adapt_label_shift([[.5, .5]], [[.5, .5]], [.5, .5], [0, 1])


if __name__ == '__main__':
    unittest.main()
