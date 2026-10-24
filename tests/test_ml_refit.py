"""Saved native model finalization without a second validation training pass."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from catboost import CatBoostRegressor

import test_ml_backend as fixtures
from ml_backend import MLRequestError, run


class MLRefitTests(unittest.TestCase):
    def setUp(self):
        fixtures.MLBackendTests.setUp(self)

    envelope = fixtures.MLBackendTests.envelope
    assert_contract = fixtures.MLBackendTests.assert_contract

    def trained(self):
        result = run(self.envelope(action='experiment', refit=False, iterations=100, depth=4))
        self.assertEqual(result['model'], 'catboost', result['notes'])
        self.assertEqual(result['fit_scope'], 'holdout_train')
        return result

    def refit(self, trained, **extra):
        envelope = self.envelope(action='refit', manifest_path=trained['manifest_path'])
        envelope.update(extra)
        return run(envelope)

    def mutate_manifest(self, result, **changes):
        path = Path(result['manifest_path'])
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest.update(changes)
        path.write_text(json.dumps(manifest), encoding='utf-8')

    def test_real_refit_uses_all_rows_and_no_validation_training(self):
        trained = self.trained()
        original_fit = CatBoostRegressor.fit
        calls = []

        def record_fit(model, data, target, **kwargs):
            calls.append((len(data), kwargs))
            return original_fit(model, data, target, **kwargs)

        with patch.object(CatBoostRegressor, 'fit', record_fit):
            final = self.refit(trained)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 160)
        self.assertNotIn('eval_set', calls[0][1])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertTrue(final['final'])
        self.assertEqual(final['training_rows'], 160)
        self.assertEqual(final['selected_iterations'], trained['selected_iterations'])
        for key in ('score', 'metric', 'direction', 'validation_id'):
            self.assertEqual(final[key], trained[key])
        self.assert_contract(final)

    def test_changed_training_or_test_data_refuses_refit(self):
        trained = self.trained()
        for name in ('train.csv', 'test.csv'):
            path = self.root / name
            original = path.read_bytes()
            frame = pd.read_csv(path)
            frame.loc[2, 'x'] += 1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(MLRequestError, 'no longer matches'):
                self.refit(trained)
            path.write_bytes(original)

    def test_classification_refit_preserves_string_labels_and_class_weights(self):
        frame = pd.read_csv(self.root / 'train.csv')
        frame['target'] = ['high' if value > 0 else 'low' for value in frame['x'].fillna(0)]
        frame.to_csv(self.root / 'train.csv', index=False)
        self.task.write_text('Classification, target target, primary macro F1.', encoding='utf-8')
        trained = self.trained()
        final = self.refit(trained)
        self.assertEqual(final['score'], trained['score'])
        self.assertEqual(final['schema']['allowed_values']['target'], ['high', 'low'])
        self.assert_contract(final)

    def test_tampered_score_is_not_reused(self):
        trained = self.trained()
        self.mutate_manifest(trained, score=trained['score'] + 1)
        with self.assertRaisesRegex(MLRequestError, 'validation score'):
            self.refit(trained)

    def test_tampered_gradient_parameters_are_rejected(self):
        trained = self.trained()
        manifest = json.loads(Path(trained['manifest_path']).read_text(encoding='utf-8'))
        self.mutate_manifest(trained, training_parameters=dict(manifest['training_parameters'], depth=8))
        with self.assertRaisesRegex(MLRequestError, 'parameters differ'):
            self.refit(trained)

    def test_partial_refit_does_not_write_a_final_candidate(self):
        trained = self.trained()
        existing = set((self.root / '.ml-artifacts').glob('candidate-*.csv'))
        original_fit = CatBoostRegressor.fit

        def short_fit(model, *args, **kwargs):
            model.set_params(iterations=1)
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostRegressor, 'fit', short_fit):
            with self.assertRaisesRegex(MLRequestError, 'selected tree count'):
                self.refit(trained)
        self.assertEqual(set((self.root / '.ml-artifacts').glob('candidate-*.csv')), existing)
        self.assertTrue(Path(trained['candidate_path']).is_file())

    def test_exhausted_budget_does_not_start_training(self):
        trained = self.trained()
        with patch.object(CatBoostRegressor, 'fit', side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(MLRequestError, 'Not enough time'):
                self.refit(trained, time_budget_seconds=1)

    def test_manifest_outside_private_ml_directory_is_rejected(self):
        path = self.root / 'manifest.json'
        path.write_text('{}')
        with self.assertRaisesRegex(MLRequestError, 'inside .ml-artifacts'):
            run(self.envelope(action='refit', manifest_path=str(path)))
