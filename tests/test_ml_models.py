"""Actual native fits exercise fold preprocessing, serialization and exact refit."""
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_models import ModelAdapter, _sample_weights, fit


@unittest.skipUnless(importlib.util.find_spec('lightgbm') and importlib.util.find_spec('xgboost'),
                     'Optional native tree dependencies are not installed')
class NativeModelTests(unittest.TestCase):
    def test_class_weights_change_ratios_without_rescaling_regularization(self):
        for mode, ratio in [('Balanced', 3), ('SqrtBalanced', np.sqrt(3))]:
            with self.subTest(mode=mode):
                weights = _sample_weights(np.array([0, 0, 0, 1]), mode)
                self.assertAlmostEqual(float(weights.mean()), 1)
                self.assertAlmostEqual(float(weights[-1] / weights[0]), ratio)

    def setUp(self):
        rng = np.random.default_rng(17)
        values = rng.normal(size=240)
        self.frame = pd.DataFrame({'amount [kg]': values, 'kind': np.where(values > 0, 'large', 'small'),
                                   'other': rng.normal(size=240)})
        self.target = 8 * values + rng.normal(size=240) * .1
        self.frame.loc[1, 'amount [kg]'] = np.nan
        self.train, self.valid = self.frame.iloc[:180], self.frame.iloc[180:].copy()
        self.valid.loc[self.valid.index[:5], 'kind'] = 'unseen'

    def fit_adapter(self, family, classification=False, **request):
        y = np.digitize(self.target, [-3, 3]) if classification else self.target
        model = fit(family, self.train, y[:180], self.valid, y[180:], ['kind'],
                    {'iterations': 80, 'learning_rate': .12, 'depth': 4, **request},
                    time.monotonic() + 30, 1,
                    'classification' if classification else 'regression',
                    'macro_f1' if classification else 'rmse')
        return model, y

    def test_regression_beats_constant_and_unknown_categories_do_not_leak(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                model, y = self.fit_adapter(family)
                prediction = model.predict(self.valid)
                error = np.sqrt(np.mean((prediction - y[180:]) ** 2))
                self.assertLess(error, np.std(y[180:]) * .7)
                self.assertNotIn('unseen', model.vocabulary['kind'])
                self.assertEqual(model.feature_names_, list(self.train.columns))
                self.assertTrue(np.isfinite(prediction).all())
                np.testing.assert_allclose(prediction, model.predict(self.valid[list(reversed(self.valid.columns))]))

    def test_multiclass_probabilities_align_and_use_known_encoded_labels(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                model, y = self.fit_adapter(family, classification=True)
                probabilities = model.predict_proba(self.valid)
                self.assertEqual(probabilities.shape, (60, 3))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertEqual(model.classes_, [0, 1, 2])
                self.assertGreater(np.mean(model.predict(self.valid) == y[180:]), .75)

    def test_save_load_and_refit_preserve_selected_iterations(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                model, y = self.fit_adapter(family)
                path = Path(folder) / 'model.joblib'
                model.save_model(path)
                loaded = ModelAdapter.load_model(path)
                np.testing.assert_allclose(model.predict(self.valid), loaded.predict(self.valid))
                before = model.predict(self.valid)
                final = loaded.refit(self.frame, y, time.monotonic() + 30, 1)
                self.assertEqual(final.tree_count_, model.tree_count_)
                self.assertIsNot(final, loaded)
                np.testing.assert_allclose(before, model.predict(self.valid))

    def test_partial_refit_raises_without_mutating_validated_model(self):
        model, y = self.fit_adapter('xgboost')
        self.assertGreater(model.tree_count_, 1)
        before = model.predict(self.valid)

        def incomplete(adapter, *args, **kwargs):
            adapter.tree_count_ = 1

        with patch.object(ModelAdapter, '_train', incomplete):
            with self.assertRaises(TimeoutError):
                model.refit(self.frame, y, time.monotonic() + 30, 1)
        np.testing.assert_allclose(before, model.predict(self.valid))

    def test_deadline_checked_before_fit_or_refit(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family), self.assertRaises(TimeoutError):
                fit(family, self.train, self.target[:180], self.valid, self.target[180:],
                    ['kind'], {}, time.monotonic() - 1, 1, 'regression', 'rmse')
        model, y = self.fit_adapter('lightgbm')
        with self.assertRaises(TimeoutError):
            model.refit(self.frame, y, time.monotonic() - 1, 1)

    def test_native_deadline_callbacks_refuse_partial_full_refits(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                model, y = self.fit_adapter(family)
                self.assertGreater(model.tree_count_, 5)
                before = model.predict(self.valid)
                clock = [0]

                def now():
                    clock[0] += 1
                    return 0 if clock[0] < 6 else 100

                with patch('ml_models.time.monotonic', side_effect=now):
                    with self.assertRaises(TimeoutError):
                        model.refit(self.frame, y, deadline=50, threads=1)
                np.testing.assert_allclose(before, model.predict(self.valid))

    def test_invalid_resources_or_gradient_parameters_are_rejected(self):
        for request, threads in [({'iterations': 4001}, 1), ({'depth': 0}, 1),
                                 ({'learning_rate': float('nan')}, 1), ({}, 9), ({}, True)]:
            with self.subTest(request=request, threads=threads), self.assertRaises(ValueError):
                fit('lightgbm', self.train, self.target[:180], self.valid, self.target[180:],
                    ['kind'], request, time.monotonic() + 30, threads, 'regression', 'rmse')

    def test_binary_probability_column_order(self):
        y = (self.target > 0).astype(int)
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                model = fit(family, self.train, y[:180], self.valid, y[180:], ['kind'],
                            {'iterations': 30}, time.monotonic() + 30, 1,
                            'classification', 'roc_auc')
                self.assertEqual(model.predict_proba(self.valid).shape, (60, 2))
                np.testing.assert_array_equal(model.predict(self.valid), model.predict_proba(self.valid).argmax(axis=1))

    def test_onehot_keeps_true_numeric_zero_and_ignores_unknown_categories(self):
        model, _ = self.fit_adapter('xgboost')
        self.assertEqual(model.parameters['encoding'], 'onehot')
        point = self.valid.iloc[:1].copy()
        point['amount [kg]'] = 0.
        point['kind'] = 'unseen'
        transformed = model._transform(point)
        self.assertEqual(transformed.dtype, np.float32)
        self.assertEqual(transformed[0, 0], 0.)
        self.assertFalse(np.isnan(transformed[0, 0]))
        np.testing.assert_array_equal(transformed[0, -2:], [0, 0])
        self.assertNotIn('unseen', model.encoder.categories_[0])

    def test_dense_expansion_is_bounded_before_allocation(self):
        with patch('ml_models.MAX_DENSE_CELLS', 10):
            with self.assertRaisesRegex(ValueError, 'expansion'):
                self.fit_adapter('xgboost')
        model, _ = self.fit_adapter('xgboost')
        with patch('ml_models.MAX_DENSE_CELLS', 10):
            with self.assertRaisesRegex(ValueError, 'expansion'):
                model.predict(self.valid)

    def test_lightgbm_leafwise_growth_uses_leaf_count_without_depth_limit(self):
        model, _ = self.fit_adapter('lightgbm', depth=-1, num_leaves=31)
        self.assertEqual(model.parameters['native']['max_depth'], -1)
        self.assertEqual(model.parameters['native']['num_leaves'], 31)
        with self.assertRaises(ValueError):
            self.fit_adapter('xgboost', depth=-1)


if __name__ == '__main__':
    unittest.main()

