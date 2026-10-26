"""Conditional refinements cannot affect confident or unrelated decisions."""
import unittest
import numpy as np
from qa.research_pair_specialists import refine


class PairRefinementTests(unittest.TestCase):
    def test_only_uncertain_top_two_pair_changes_and_pair_mass_is_preserved(self):
        base = np.array([[.4, .35, .25], [.9, .08, .02], [.4, .1, .5]])
        before = base.copy()
        refined, mask = refine(base, np.array([[.1, .9]] * 3), [0, 1])
        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_array_equal(refined[1:], base[1:])
        np.testing.assert_array_equal(refined[:, 2], base[:, 2])
        np.testing.assert_allclose(refined.sum(axis=1), 1.)
        np.testing.assert_array_equal(base, before)

    def test_invalid_pair_and_unaligned_probabilities_fail(self):
        for pair in ([0, 0], [0, 4], [0]):
            with self.assertRaises(ValueError):
                refine([[.5, .3, .2]], [[.5, .5]], pair)
        with self.assertRaises(ValueError):
            refine([[.5, .3, .2]], [[.5, .5], [.6, .4]], [0, 1])
