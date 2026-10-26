"""Probability conservation and fold-isolation contracts for local research only."""
import unittest

import numpy as np

from qa.research_attack_specialist import conditional_blend, guarded_splits, select_strength


class AttackSpecialistTests(unittest.TestCase):
    def test_blend_preserves_subset_mass_and_all_other_probabilities(self):
        base = np.array([[.1, .2, .3, .4], [.5, .1, .2, .2]])
        specialist = np.array([[.8, .2], [.3, .7]])
        result = conditional_blend(base, specialist, [0, 2], .5)
        np.testing.assert_array_equal(result[:, [1, 3]], base[:, [1, 3]])
        np.testing.assert_allclose(result[:, [0, 2]].sum(axis=1), base[:, [0, 2]].sum(axis=1))
        np.testing.assert_allclose(result.sum(axis=1), 1)
        np.testing.assert_allclose(result[:, [0, 2]], .5 * base[:, [0, 2]] +
                                    .5 * base[:, [0, 2]].sum(axis=1, keepdims=True) * specialist)

    def test_zero_strength_is_exact_control_and_zero_mass_stays_zero(self):
        base = np.array([[0., .3, 0., .7], [.2, .3, .1, .4]])
        specialist = np.array([[.1, .9], [.8, .2]])
        np.testing.assert_array_equal(conditional_blend(base, specialist, [0, 2], 0), base)
        np.testing.assert_array_equal(conditional_blend(base, specialist, [0, 2], .5)[0], base[0])

    def test_invalid_probability_subset_and_strength_are_rejected(self):
        base = np.array([[.2, .3, .5]])
        specialist = np.array([[.3, .7]])
        for subset, strength in [([0, 0], .5), ([0, 4], .5), ([0, 1], -1), ([0, 1], np.nan)]:
            with self.subTest(subset=subset, strength=strength), self.assertRaises(ValueError):
                conditional_blend(base, specialist, subset, strength)
        with self.assertRaises(ValueError):
            conditional_blend(base, [[np.nan, .7]], [0, 1], .5)
        with self.assertRaises(ValueError):
            conditional_blend(base, [[.2, .3]], [0, 1], .5)

    def test_nested_group_folds_accept_independent_rows(self):
        result = guarded_splits(np.arange(8), np.array([0, 1, 2, 3]), np.array([4, 5]),
                                np.array([0, 1]), np.array([2, 3]))
        self.assertEqual(result['outer_group_overlap'], 0)
        self.assertTrue(result['inner_folds_partition_outer_train'])

    def test_duplicate_feature_groups_cannot_cross_either_boundary(self):
        for groups in (np.array([0, 1, 2, 3, 0, 5]), np.array([0, 1, 0, 3, 4, 5])):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, 'leakage'):
                guarded_splits(groups, np.array([0, 1, 2, 3]), np.array([4, 5]),
                               np.array([0, 1]), np.array([2, 3]))

    def test_outer_rows_cannot_enter_inner_split_or_overlap_outer_train(self):
        for train, valid, inner_train, inner_valid in (
                ([0, 1, 2, 3], [3, 4], [0, 1], [2, 3]),
                ([0, 1, 2, 3], [4, 5], [0, 1], [2, 4]),
                ([0, 1, 2, 3], [4, 5], [0, 1], [1, 2, 3])):
            with self.subTest(train=train, valid=valid), self.assertRaises(ValueError):
                guarded_splits(np.arange(6), *map(np.asarray, (train, valid, inner_train, inner_valid)))

    def test_inner_selection_retains_control_on_ties_and_worse_specialist(self):
        base = np.array([[.7, .1, .2], [.1, .8, .1], [.1, .1, .8]])
        specialist = np.array([[0., 1.], [1., 0.], [1., 0.]])
        strength, scores = select_strength(base, specialist, [0, 1], [0, 1, 2])
        self.assertEqual(strength, 0.)
        self.assertEqual(scores[0], 1.)
        self.assertEqual(len(scores), 3)


if __name__ == '__main__':
    unittest.main()
