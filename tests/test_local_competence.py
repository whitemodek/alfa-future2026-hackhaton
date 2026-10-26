"""Leak boundaries and probability contracts of the research-only local gate."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa.research_local_competence import (
    Expert, choose_policy, combine, competence_weights, honest_oof, row_losses,
)


class LocalCompetenceTests(unittest.TestCase):
    def test_oof_never_predicts_a_training_feature_group(self):
        x = pd.DataFrame({'group': np.repeat(np.arange(6), 2)})
        y = np.arange(12, dtype=float)
        seen = []

        def fit(train, truth, *args):
            np.testing.assert_array_equal(truth, y[train.index])
            groups = set(train.group)
            seen.append(set(train.index))

            class Model:
                def predict(self, query):
                    self_test.assertFalse(groups & set(query.group))
                    return np.full(len(query), np.mean(truth))

            return [Model(), Model(), Model()]

        self_test = self
        with patch('qa.research_local_competence.fit_experts', side_effect=fit):
            predictions, assignment, folds = honest_oof(x, y, x.group.to_numpy(), [], False,
                                                       None, 'rmse', time.monotonic() + 10)
        self.assertEqual(predictions.shape, (12, 3))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(len(seen), 3)
        self.assertEqual(set(assignment), {0, 1, 2})
        self.assertEqual(sum(len(f['valid']) for f in folds), len(x))

    def test_local_weights_and_mixture_keep_probability_mass(self):
        weights = competence_weights([[3, 1, 2], [1, 3, 2]])
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertEqual(weights.argmax(axis=1).tolist(), [1, 0])
        probabilities = np.array([[[.9, .1], [.2, .8], [.5, .5]]] * 2)
        mixture = combine(probabilities, weights)
        np.testing.assert_allclose(mixture.sum(axis=1), 1)
        self.assertTrue((mixture >= 0).all())

    def test_per_row_losses_use_only_supplied_truth(self):
        p = np.array([[[.8, .2], [.4, .6], [.1, .9]]])
        np.testing.assert_allclose(row_losses(p, [1], True), -np.log([[.2, .6, .9]]))
        np.testing.assert_allclose(row_losses(np.array([[1., 2., 3.]]), [2.], False), [[1, 0, 1]])

    def test_selection_falls_back_on_ties_and_worse_local(self):
        scores = dict(linear=.7, lightgbm=.8, kernel=.75, uniform=.77, local=.8)
        self.assertEqual(choose_policy(scores, 'macro_f1'), ('lightgbm', 'lightgbm'))
        scores['local'] = .81
        self.assertEqual(choose_policy(scores, 'macro_f1'), ('local', 'lightgbm'))
        errors = dict(linear=3., lightgbm=2., kernel=1.5, uniform=1.8, local=1.6)
        self.assertEqual(choose_policy(errors, 'rmse'), ('kernel', 'kernel'))

    def test_missing_fold_classes_realign_to_global_columns(self):
        x = pd.DataFrame({'x': np.arange(24, dtype=float)})
        y = np.array([0, 2] * 12)
        model = Expert('linear', True, 3, 'macro_f1').fit(x, y, [], time.monotonic() + 15)
        probability = model.predict(x.iloc[:5])
        self.assertEqual(probability.shape, (5, 3))
        np.testing.assert_allclose(probability[:, 1], 0)
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_too_few_independent_groups_are_rejected(self):
        x = pd.DataFrame({'x': [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            honest_oof(x, np.arange(4), x.x.to_numpy(), [], False, None, 'rmse', time.monotonic() + 10)

    def test_kernel_vocabulary_and_landmarks_use_training_only(self):
        x = pd.DataFrame({'x': np.linspace(-1, 1, 24), 'kind': ['known'] * 24})
        model = Expert('kernel', True, 2, 'roc_auc').fit(
            x, np.array([0, 1] * 12), ['kind'], time.monotonic() + 15)
        query = pd.DataFrame({'x': [4.], 'kind': ['never_seen']})
        probability = model.predict(query)
        self.assertNotIn('never_seen', model.adapter.vocabulary['kind'])
        self.assertTrue((model.kernel.component_indices_ < len(x)).all())
        trained = model.adapter._transform(x)[model.kernel.component_indices_]
        components = model.kernel.components_
        np.testing.assert_allclose(components.toarray(), trained.toarray())
        self.assertEqual(probability.shape, (1, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.)


if __name__ == '__main__':
    unittest.main()
