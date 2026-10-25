"""Real forest fits cover fold-only schemas, bounded batches and exact refit."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from ml_models import ModelAdapter, _sample_weights, fit


class ExtraTreesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        values = rng.normal(size=180)
        self.frame = pd.DataFrame({'value': values, 'kind': np.where(values > 0, 'a', 'b'),
                                   'empty': np.nan})
        self.target = np.digitize(values, [-.5, .5])
        self.frame.loc[0, 'value'] = np.nan
        self.train = self.frame.iloc[:130]
        self.valid = self.frame.iloc[130:].copy()
        self.valid.loc[self.valid.index[0], 'kind'] = 'new'

    def fit_model(self, **request):
        return fit('extra_trees', self.train, self.target[:130], self.valid,
                   self.target[130:], ['kind'], {'iterations': 23, **request},
                   time.monotonic() + 30, 1, 'classification', 'macro_f1')

    def test_probabilities_schema_and_fold_only_imputation(self):
        model = self.fit_model()
        probability = model.predict_proba(self.valid)
        self.assertEqual(probability.shape, (50, 3))
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertGreater(np.mean(model.predict(self.valid) == self.target[130:]), .8)
        self.assertNotIn('new', model.vocabulary['kind'])
        self.assertAlmostEqual(model.imputer.statistics_[0], self.train.value.median())
        self.assertEqual(model.imputer.statistics_[1], 0)
        self.assertEqual(model.tree_count_, 23)
        point = self.valid.iloc[:1].copy()
        point['value'] = 0.
        transformed = model._transform(point)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed[0, 0], 0)
        np.testing.assert_array_equal(transformed[0, -2:], [0, 0])
        np.testing.assert_array_equal(probability,
                                      model.predict_proba(self.valid[list(reversed(self.valid.columns))]))

    def test_save_load_and_full_refit_preserve_original(self):
        model = self.fit_model()
        before = model.predict_proba(self.valid)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'forest.joblib'
            model.save_model(path)
            loaded = ModelAdapter.load_model(path)
            np.testing.assert_array_equal(before, loaded.predict_proba(self.valid))
            full = self.frame.copy()
            full.loc[150, 'kind'] = 'full-only'
            final = loaded.refit(full, self.target, time.monotonic() + 30, 1)
            self.assertEqual(final.tree_count_, model.tree_count_)
            self.assertIn('full-only', final.vocabulary['kind'])
            self.assertNotIn('full-only', loaded.vocabulary['kind'])
            self.assertIsNot(final, loaded)
            np.testing.assert_array_equal(before, loaded.predict_proba(self.valid))

    def test_warm_start_batches_match_one_shot_forest(self):
        model = self.fit_model()
        reference = ExtraTreesClassifier(**model.parameters['native'], n_estimators=23, n_jobs=1)
        reference.fit(model._transform(self.train), self.target[:130],
                      sample_weight=_sample_weights(self.target[:130], 'SqrtBalanced'))
        np.testing.assert_array_equal(model.predict_proba(self.valid),
                                      reference.predict_proba(model._transform(self.valid)))

    def test_deadline_keeps_only_complete_batches_and_refuses_partial_refit(self):
        original_fit = ExtraTreesClassifier.fit
        clock = [0.]

        def consume_budget(estimator, *args, **kwargs):
            result = original_fit(estimator, *args, **kwargs)
            clock[0] = 2.
            return result

        with patch('ml_models.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(ExtraTreesClassifier, 'fit', consume_budget):
            partial = fit('extra_trees', self.train, self.target[:130], self.valid,
                          self.target[130:], ['kind'], {'iterations': 23}, 1., 1,
                          'classification', 'macro_f1')
        self.assertEqual(partial.tree_count_, 1)
        self.assertEqual(len(partial.model.estimators_), 1)
        self.assertTrue(np.isfinite(partial.predict_proba(self.valid)).all())
        model = self.fit_model()
        before = model.predict_proba(self.valid)
        clock[0] = 0.
        with patch('ml_models.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(ExtraTreesClassifier, 'fit', consume_budget), \
                self.assertRaises(TimeoutError):
            model.refit(self.frame, self.target, deadline=1., threads=1)
        np.testing.assert_array_equal(before, model.predict_proba(self.valid))
        with self.assertRaises(TimeoutError):
            model.refit(self.frame, self.target, deadline=time.monotonic() - 1, threads=1)

    def test_dense_limit_counts_missing_indicators_and_checks_prediction(self):
        # 2 numeric + 2 missing indicators + 2 categorical columns = 6.
        with patch('ml_models.MAX_DENSE_CELLS', len(self.train) * 5), \
                self.assertRaisesRegex(ValueError, 'expansion'):
            self.fit_model()
        model = self.fit_model()
        with patch('ml_models.MAX_DENSE_CELLS', len(self.valid) * 5), \
                self.assertRaisesRegex(ValueError, 'expansion'):
            model.predict(self.valid)

    def test_numeric_only_and_categorical_only_binary(self):
        y = (self.target > 0).astype(int)
        for columns, cats in [(['value', 'empty'], []), (['kind'], ['kind'])]:
            with self.subTest(columns=columns):
                model = fit('extra_trees', self.train[columns], y[:130], self.valid[columns],
                            y[130:], cats, {'iterations': 5}, time.monotonic() + 30, 1,
                            'classification', 'roc_auc')
                probability = model.predict_proba(self.valid[columns])
                self.assertEqual(probability.shape, (50, 2))
                np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_invalid_parameters_and_regression_are_rejected(self):
        for request in ({'iterations': 0}, {'iterations': 1001}, {'min_samples_leaf': True},
                        {'max_features': float('nan')}, {'encoding': 'native'},
                        {'auto_class_weights': 'invented'}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.fit_model(**request)
        with self.assertRaisesRegex(ValueError, 'classification only'):
            fit('extra_trees', self.train, self.target[:130], self.valid,
                self.target[130:], ['kind'], {}, time.monotonic() + 30, 1, 'regression', 'rmse')

    def test_backend_forest_ensemble_and_exact_full_refit(self):
        import json
        from ml_backend import _score, _split, load_problem, run
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            task = root / 'task.md'
            task.write_text('Classification, target: target, macro F1.', encoding='utf-8')
            train = self.frame.copy()
            train['id'] = np.arange(len(train))
            train['target'] = self.target
            train.to_csv(root / 'train.csv', index=False)
            test = self.valid.iloc[:3].copy()
            test['id'] = [1000, 1001, 1002]
            test.to_csv(root / 'test.csv', index=False)
            pd.DataFrame({'id': test.id, 'target': 0}).to_csv(root / 'sample_submission.csv', index=False)

            def envelope(**request):
                return dict(task_path=str(task), task_dir=str(root), threads=1,
                            time_budget_seconds=30, request=request)

            forest = run(envelope(action='experiment', model='extra_trees', iterations=23, refit=False))
            boosting = run(envelope(action='experiment', model='lightgbm', iterations=23, refit=False))
            self.assertEqual(forest['model'], 'extra_trees')
            self.assertEqual(forest['selected_iterations'], 23)
            blend = run(envelope(action='ensemble', manifest_paths=[forest['manifest_path'], boosting['manifest_path']]))
            saved = [json.loads(Path(result['manifest_path']).read_text()) for result in (forest, boosting)]
            probabilities = [np.load(m['validation_path'])['probability'] for m in saved]
            mean = sum(probabilities) / 2
            problem = load_problem(envelope())
            _, vi = _split(problem, {})
            self.assertAlmostEqual(blend['score'], _score(problem, problem['y'].iloc[vi], mean.argmax(axis=1), mean))
            final = run(envelope(action='refit', manifest_path=blend['manifest_path']))
            self.assertTrue(final['final'])
            self.assertEqual(final['fit_scope'], 'full_train')
            self.assertEqual(final['score'], blend['score'])
            self.assertEqual(final['validation_id'], blend['validation_id'])


if __name__ == '__main__':
    unittest.main()
