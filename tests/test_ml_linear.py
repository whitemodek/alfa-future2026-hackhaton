"""Linear candidates cover extrapolation and sparse heterogeneous feature tables."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from ml_models import ModelAdapter, fit


class LinearModelTests(unittest.TestCase):
    def train(self, x, y, categorical=(), classification=False, request=None):
        return fit('linear', x, y, x.iloc[:10], y[:10], list(categorical), request or {},
                   time.monotonic() + 30, 1,
                   'classification' if classification else 'regression',
                   'macro_f1' if classification else 'rmse')

    def test_regression_extrapolates_beyond_training_range(self):
        x = pd.DataFrame({'trend': np.linspace(-1, 1, 200)})
        y = 7 * x.trend.to_numpy() + 4
        model = self.train(x, y)
        future = pd.DataFrame({'trend': [2., 3., 4.]})
        prediction = model.predict(future)
        self.assertLess(float(np.sqrt(np.mean((prediction - (7 * future.trend + 4)) ** 2))), .2)
        self.assertTrue(np.all(prediction > y.max()))
        self.assertEqual(model.tree_count_, 1)

    def test_sparse_high_cardinality_unknown_categories_and_missing_values(self):
        x = pd.DataFrame({'number': np.arange(200, dtype=float),
                          'empty': [np.nan] * 200,
                          'category': [f'v{i}' for i in range(200)]})
        x.loc[0, 'number'] = np.nan
        y = np.arange(200) / 10
        model = self.train(x, y, ['category'])
        transformed = model._transform(x)
        self.assertTrue(sparse.isspmatrix_csr(transformed))
        self.assertLess(transformed.nnz, len(x) * 6)
        new = pd.DataFrame({'number': [np.nan, 201.], 'empty': [np.nan, np.nan],
                            'category': ['previously-unseen', 'v1']})
        self.assertTrue(np.isfinite(model.predict(new)).all())
        self.assertNotIn('previously-unseen', model.vocabulary['category'])
        # Fitted preprocessing belongs to the training fold, including missingness.
        self.assertEqual(float(model.imputer.statistics_[0]), 100.)

    def test_binary_and_multiclass_probabilities_and_label_alignment(self):
        rng = np.random.default_rng(5)
        x = pd.DataFrame(rng.normal(size=(240, 3)), columns=['a', 'b', 'c'])
        for boundaries in ([0.], [-.6, .6]):
            with self.subTest(boundaries=boundaries):
                y = np.digitize(x.a, boundaries)
                model = self.train(x, y, classification=True)
                probability = model.predict_proba(x)
                self.assertEqual(probability.shape, (240, len(boundaries) + 1))
                np.testing.assert_allclose(probability.sum(axis=1), 1)
                np.testing.assert_array_equal(model.predict(x), probability.argmax(axis=1))
                self.assertGreater(np.mean(model.predict(x) == y), .94)
                self.assertEqual(model.classes_, list(range(len(boundaries) + 1)))

    def test_categorical_only_and_numeric_only_models(self):
        category = pd.DataFrame({'kind': ['a', 'b', 'c'] * 30})
        y = np.tile([1., 2., 3.], 30)
        model = self.train(category, y, ['kind'])
        np.testing.assert_allclose(model.predict(category), y, atol=.04)
        self.assertTrue(np.isfinite(model.predict(pd.DataFrame({'kind': ['unknown']}))).all())

    def test_classifier_refit_reorders_features_and_preserves_probabilities(self):
        x = pd.DataFrame({'number': np.linspace(-3, 3, 180), 'kind': ['a', 'b'] * 90})
        y = (x.number.to_numpy() > 1).astype(int)
        model = self.train(x.iloc[:150], y[:150], ['kind'], classification=True)
        before = model.predict_proba(x)
        full = model.refit(x, y, time.monotonic() + 30, 1)
        self.assertEqual(full.tree_count_, 1)
        self.assertEqual(full.classes_, [0, 1])
        np.testing.assert_allclose(full.predict_proba(x), full.predict_proba(x[['kind', 'number']]))
        np.testing.assert_allclose(model.predict_proba(x), before)
        self.assertTrue(np.isfinite(full.predict_proba(x)).all())

    def test_save_load_refit_preserves_validation_model_and_rebuilds_vocabulary(self):
        x = pd.DataFrame({'number': np.arange(100.), 'kind': ['old'] * 100})
        y = 2 * x.number.to_numpy()
        model = self.train(x.iloc[:80], y[:80], ['kind'])
        before = model.predict(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'linear.joblib'
            model.save_model(path)
            loaded = ModelAdapter.load_model(path)
            np.testing.assert_allclose(loaded.predict(x), before)
            full_x = x.copy()
            full_x.loc[90:, 'kind'] = 'new'
            refitted = loaded.refit(full_x, y, time.monotonic() + 30, 1)
            self.assertIn('new', refitted.vocabulary['kind'])
            self.assertNotIn('new', loaded.vocabulary['kind'])
            self.assertEqual(refitted.tree_count_, 1)
            np.testing.assert_allclose(loaded.predict(x), before)

    def test_sparse_expansion_is_rejected_before_estimator_fit(self):
        x = pd.DataFrame({'kind': ['a', 'b', 'c'] * 10})
        with patch('ml_models.MAX_LINEAR_COLUMNS', 2), self.assertRaisesRegex(ValueError, 'resource limit'):
            self.train(x, np.arange(30.), ['kind'])

    def test_deadline_before_fit_and_after_estimator_completion(self):
        x = pd.DataFrame({'x': np.arange(20.)})
        with self.assertRaises(TimeoutError):
            fit('linear', x, np.arange(20.), x, np.arange(20.), [], {},
                time.monotonic() - 1, 1, 'regression', 'rmse')
        # An over-budget fit must not advertise a completed reusable model.
        with patch('ml_models._check_deadline', side_effect=[None, None, TimeoutError('late')]):
            with self.assertRaises(TimeoutError):
                self.train(x, np.arange(20.))

    def test_text_generalizes_to_new_sentences_without_learning_holdout_tokens(self):
        y = np.tile([0, 1], 50)
        sentences = [
            (f'The wonderful helpful service made this customer happy order{i}' if label else
             f'The awful disappointing service made this customer angry order{i}')
            for i, label in enumerate(y)]
        x = pd.DataFrame({'review': sentences})
        model = self.train(x, y, ['review'], classification=True)
        new = pd.DataFrame({'review': [
            'Happy customer thanks wonderful service holdoutonlytoken',
            'Angry customer reports awful service holdoutonlytoken',
            'Helpful wonderful service for a happy customer holdoutonlytoken',
            'Disappointing awful service for an angry customer holdoutonlytoken']})
        np.testing.assert_array_equal(model.predict(new), [1, 0, 1, 0])
        self.assertEqual(model.text_columns, ['review'])
        self.assertNotIn('review', model.vocabulary)
        self.assertIsNone(model.encoder)
        self.assertNotIn('holdoutonlytoken', model.text_vectorizers['review'].vocabulary_)
        self.assertTrue(sparse.isspmatrix_csr(model._transform(new)))
        self.assertLessEqual(len(model.text_vectorizers['review'].vocabulary_), 20_000)
        before = model.predict_proba(new)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'text.joblib'
            model.save_model(path)
            loaded = ModelAdapter.load_model(path)
            np.testing.assert_allclose(loaded.predict_proba(new), before)
            extended_x = pd.concat([x, new], ignore_index=True)
            extended_y = np.concatenate([y, [1, 0, 1, 0]])
            full = loaded.refit(extended_x, extended_y, time.monotonic() + 30, 1)
            self.assertIn('holdoutonlytoken', full.text_vectorizers['review'].vocabulary_)
            self.assertNotIn('holdoutonlytoken', loaded.text_vectorizers['review'].vocabulary_)
            np.testing.assert_allclose(loaded.predict_proba(new), before)

    def test_text_and_numeric_and_categories_share_a_sparse_representation(self):
        x = pd.DataFrame({'amount': np.arange(100.), 'kind': ['a', 'b'] * 50,
                          'review': [f'This useful product works very well for person{i}' for i in range(100)]})
        model = self.train(x, 3 * x.amount.to_numpy(), ['kind', 'review'])
        transformed = model._transform(x)
        self.assertTrue(sparse.isspmatrix_csr(transformed))
        self.assertEqual(set(model.vocabulary), {'kind'})
        self.assertEqual(model.text_columns, ['review'])
        np.testing.assert_allclose(model.predict(x), model.predict(x[['review', 'kind', 'amount']]))

    def test_text_input_resource_limit_and_empty_vocabulary_fallback(self):
        x = pd.DataFrame({'text': [f'Long enough natural language example phrase item{i}' for i in range(30)]})
        with patch('ml_models.MAX_TEXT_CHARACTERS', 10), self.assertRaisesRegex(ValueError, 'character limit'):
            self.train(x, np.arange(30.), ['text'])
        # A text column may satisfy shape heuristics without any eligible tokens.
        empty = pd.DataFrame({'text': [f'{chr(0x400 + i)}     a     b     c     d     e' for i in range(30)]})
        model = self.train(empty, np.arange(30.), ['text'])
        self.assertEqual(model.text_vectorizers, {})
        self.assertTrue(np.isfinite(model.predict(empty)).all())

    def test_linear_bundle_without_text_attributes_remains_compatible(self):
        x = pd.DataFrame({'number': np.arange(30.)})
        model = self.train(x, np.arange(30.))
        before = model.predict(x)
        del model.text_columns
        del model.text_vectorizers
        np.testing.assert_allclose(model.predict(x), before)


if __name__ == '__main__':
    unittest.main()
