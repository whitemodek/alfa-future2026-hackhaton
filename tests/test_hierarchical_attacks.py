"""Class mapping, probability factorization and hierarchy fitting isolation."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.research_hierarchical_attacks import (
    HierarchicalAttackStrategy, blend_probability, joint_probabilities, select_strength,
)


class FakeAdapter:
    def __init__(self, gate, labels, refit_calls):
        self.gate = gate
        self.classes_ = sorted(np.unique(labels).tolist())
        self.tree_count_ = 7 if gate else 11
        self.refit_calls = refit_calls

    def predict_proba(self, frame):
        return np.tile([.3, .7] if self.gate else [.25, .75], (len(frame), 1))

    def refit(self, frame, labels, deadline, threads):
        self.refit_calls.append((self.gate, frame.index.tolist(), np.asarray(labels).copy()))
        return FakeAdapter(self.gate, labels, self.refit_calls)


class HierarchicalAttackTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'value': np.arange(12)})
        self.target = pd.Series(np.tile([0, 1, 2], 4))
        self.fit_calls, self.refit_calls = [], []

    def fake_fit(self, family, train, labels, valid, valid_labels, categorical,
                 request, deadline, threads, problem_type, metric):
        gate = metric == 'log_loss'
        self.fit_calls.append(dict(gate=gate, train=train.index.tolist(), labels=np.asarray(labels).copy(),
                                   valid=valid.index.tolist(), valid_labels=np.asarray(valid_labels).copy(),
                                   request=dict(request)))
        return FakeAdapter(gate, labels, self.refit_calls)

    def train(self):
        return HierarchicalAttackStrategy(['AttackA', 'Normal', 'AttackB']).fit(
            self.frame, self.target, np.array([0, 1, 2]), np.array([3, 4, 5]), [],
            time.monotonic() + 20, forbidden_indices=np.array([9, 10, 11]))

    def test_joint_probability_places_normal_at_its_actual_index(self):
        result = joint_probabilities([.7, .2], [[.25, .75], [.8, .2]], normal_index=1)
        np.testing.assert_allclose(result, [[.175, .3, .525], [.16, .8, .04]])
        np.testing.assert_allclose(result.sum(axis=1), 1)
        all_normal = joint_probabilities([0.], [[.2, .8]], normal_index=2)
        np.testing.assert_array_equal(all_normal, [[0., 0., 1.]])

    def test_invalid_gate_probabilities_and_mapping_are_rejected(self):
        for gate, normal in [([1.1], 1), ([np.nan], 1), ([.5], 3), ([.5, .5], 1)]:
            with self.subTest(gate=gate, normal=normal), self.assertRaises(ValueError):
                joint_probabilities(gate, [[.25, .75]], normal)
        for classes in (['AttackA', 'AttackB'], ['Normal', 'Normal', 'AttackA']):
            with self.assertRaises(ValueError):
                HierarchicalAttackStrategy(classes)

    def test_gate_sees_binary_labels_and_type_fit_excludes_normal_rows(self):
        with patch('qa.research_hierarchical_attacks.fit_model', side_effect=self.fake_fit):
            model = self.train()
        gate, types = self.fit_calls
        self.assertEqual(gate['train'], [0, 1, 2])
        self.assertEqual(gate['valid'], [3, 4, 5])
        np.testing.assert_array_equal(gate['labels'], [1, 0, 1])
        self.assertEqual(gate['request']['auto_class_weights'], 'None')
        self.assertEqual(types['train'], [0, 2])
        self.assertEqual(types['valid'], [3, 5])
        np.testing.assert_array_equal(types['labels'], [0, 1])
        self.assertEqual(types['request']['auto_class_weights'], 'SqrtBalanced')
        np.testing.assert_allclose(model.predict_proba(self.frame.iloc[:1]), [[.175, .3, .525]])

    def test_future_outer_labels_never_reach_fit_or_refit(self):
        self.target.iloc[9:] = 999  # Invalid labels are harmless when rows remain unseen.
        with patch('qa.research_hierarchical_attacks.fit_model', side_effect=self.fake_fit):
            model = self.train()
        full = model.refit(self.frame, self.target, np.arange(9), time.monotonic() + 20,
                           forbidden_indices=np.array([9, 10, 11]))
        self.assertEqual((full.gate.tree_count_, full.types.tree_count_), (7, 11))
        self.assertEqual(self.refit_calls[0][1], list(range(9)))
        self.assertEqual(self.refit_calls[1][1], [0, 2, 3, 5, 6, 8])
        self.assertTrue(all(not (set(record['train']) | set(record['valid'])) & {9, 10, 11}
                            for record in self.fit_calls))
        self.assertTrue(all(not set(rows) & {9, 10, 11} and 999 not in labels
                            for _, rows, labels in self.refit_calls))
        self.assertIsNot(full.gate, model.gate)
        self.assertIsNot(full.types, model.types)

    def test_forbidden_rows_and_overlap_are_rejected_before_training(self):
        with patch('qa.research_hierarchical_attacks.fit_model') as fit:
            for train, valid in (([0, 1, 2], [2, 3, 4]), ([0, 1, 2], [3, 4, 9]),
                                 ([0, 1, 9], [3, 4, 5])):
                with self.subTest(train=train, valid=valid), self.assertRaises(ValueError):
                    HierarchicalAttackStrategy(['AttackA', 'Normal', 'AttackB']).fit(
                        self.frame, self.target, np.asarray(train), np.asarray(valid), [],
                        time.monotonic() + 20, forbidden_indices=[9, 10, 11])
            fit.assert_not_called()

    def test_missing_or_unseen_training_classes_are_rejected(self):
        for replacement in (1, 999):
            self.target.iloc[0] = replacement
            with patch('qa.research_hierarchical_attacks.fit_model') as fit, self.assertRaises(ValueError):
                self.train()
            fit.assert_not_called()

    def test_saved_stage_class_order_must_match_global_mapping(self):
        with patch('qa.research_hierarchical_attacks.fit_model', side_effect=self.fake_fit):
            model = self.train()
        model.types.classes_ = [1, 0]
        with self.assertRaisesRegex(ValueError, 'class order'):
            model.predict_proba(self.frame)

    def test_blend_control_and_inner_selection_keep_least_intervention_on_ties(self):
        base = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        hierarchy = np.roll(base, 1, axis=1)
        np.testing.assert_array_equal(blend_probability(base, hierarchy, 0), base)
        np.testing.assert_array_equal(blend_probability(base, hierarchy, 1), hierarchy)
        strength, scores = select_strength(base, hierarchy, [0, 1, 2])
        self.assertEqual(strength, 0)
        self.assertEqual(len(scores), 4)
        with self.assertRaises(ValueError):
            blend_probability(base, hierarchy, -1)


if __name__ == '__main__':
    unittest.main()
