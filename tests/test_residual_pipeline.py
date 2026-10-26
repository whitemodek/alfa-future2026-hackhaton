"""Fixed residual blending is causal, preserves fallbacks and survives refit."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_lag as fixtures
import ml_backend as b
from ml_residual import combine


class ResidualTests(unittest.TestCase):
    setUp = fixtures.LagPipelineTests.setUp
    envelope = fixtures.LagPipelineTests.envelope
    history = fixtures.LagPipelineTests.history
    lag = fixtures.LagPipelineTests.lag

    def test_missing_or_ambiguous_history_keeps_the_exact_base(self):
        result = combine([10., 20., 30.], [11., 22., 33.], [5., np.nan, np.nan], [2.])
        np.testing.assert_array_equal(result[1:], [20., 30.])
        self.assertEqual(result[0], .5 * 10 + .25 * 11 + .25 * 7)
        with self.assertRaises(ValueError):
            combine([1.], [2.], [3.], [np.nan])

    def test_full_refit_preserves_original_validation_model_score_and_config(self):
        base = self.lag()
        result = b.run(self.envelope(action='residual', manifest_path=base['manifest_path']))
        before = json.loads(Path(result['manifest_path']).read_text())
        final = b.run(self.envelope(action='refit', manifest_path=result['manifest_path']))
        after = json.loads(Path(final['manifest_path']).read_text())
        self.assertEqual(final['score'], result['score'])
        self.assertEqual(final['validation_id'], result['validation_id'])
        self.assertTrue(final['final'])
        self.assertEqual(after['validation_model_path'], before['validation_model_path'])
        self.assertEqual(after['validation_model_sha256'], before['validation_model_sha256'])
        self.assertEqual(after['selected_iterations'], before['selected_iterations'])
        again = b.run(self.envelope(action='refit', manifest_path=final['manifest_path']))
        self.assertEqual(again['score'], result['score'])

    def test_configuration_drift_fails_closed(self):
        base = self.lag()
        result = b.run(self.envelope(action='residual', manifest_path=base['manifest_path']))
        path = Path(result['manifest_path'])
        manifest = json.loads(path.read_text())
        manifest['residual_weight'] = .9
        path.write_text(json.dumps(manifest))
        with self.assertRaises(b.MLRequestError):
            b.run(self.envelope(action='refit', manifest_path=str(path)))

    def test_short_history_uses_frozen_128_rounds_without_outer_selection(self):
        self.train.loc[self.train['period'] <= 3].to_csv(self.root / 'train.csv', index=False)
        base = self.lag()
        result = b.run(self.envelope(action='residual', manifest_path=base['manifest_path']))
        self.assertEqual(result['selected_iterations'], 128)

    def test_all_new_query_entities_keep_the_direct_pipeline_predictions(self):
        test = pd.read_csv(self.root / 'test.csv')
        test['entity'] = ['new' + str(i) for i in range(len(test))]
        test.to_csv(self.root / 'test.csv', index=False)
        base = self.lag()
        result = b.run(self.envelope(action='residual', manifest_path=base['manifest_path']))
        old = pd.read_csv(base['candidate_path'])
        new = pd.read_csv(result['candidate_path'])
        np.testing.assert_array_equal(old['target'], new['target'])

    def test_outer_answers_cannot_enter_residual_fitting_or_early_stopping(self):
        import ml_models
        base = self.lag()
        problem = b.load_problem(self.envelope())
        _, valid = b._split(problem, dict(split='time', time_column='period'))
        forbidden = set(valid)
        original_fit, original_refit = ml_models.fit, ml_models.ModelAdapter.refit

        def fit(family, x, y, valid_x, valid_y, *args, **kwargs):
            self.assertFalse(set(x.index) & forbidden)
            self.assertFalse(set(valid_x.index) & forbidden)
            return original_fit(family, x, y, valid_x, valid_y, *args, **kwargs)

        def refit(model, x, y, *args, **kwargs):
            self.assertFalse(set(x.index) & forbidden)
            return original_refit(model, x, y, *args, **kwargs)

        with patch('ml_models.fit', fit), patch.object(ml_models.ModelAdapter, 'refit', refit):
            result = b.run(self.envelope(action='residual', manifest_path=base['manifest_path']))
        self.assertEqual(result['status'], 'ok')
