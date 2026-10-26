"""End-to-end contracts for continuous trends and calendar representations."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import test_ml_backend as fixtures
from ml_backend import _features, _split, load_problem, run


class GeneralizationTests(unittest.TestCase):
    setUp = fixtures.MLBackendTests.setUp
    envelope = fixtures.MLBackendTests.envelope

    def dates(self):
        dates = pd.date_range('2020-01-01', periods=260, freq='D')
        train = pd.DataFrame({'id': range(220), 'event_date': dates[:220].astype(str),
                              'target': 3 * np.arange(220) + 10.})
        test = pd.DataFrame({'id': range(220, 260), 'event_date': dates[220:].astype(str)})
        train.to_csv(self.root / 'train.csv', index=False)
        test.to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': test.id, 'target': 0.}).to_csv(self.root / 'sample_submission.csv', index=False)
        self.task.write_text('Regression. Metric: RMSE. Target: target. Predict future event_date.', encoding='utf-8')

    def test_dates_become_calendar_numbers_without_mutating_original_data(self):
        self.dates()
        p = load_problem(self.envelope())
        original = p['train'].copy()
        x, test, cats = _features(p)
        self.assertEqual(cats, [])
        self.assertEqual(len(x.columns), 5)
        self.assertGreater(test.event_date.min(), x.event_date.max())
        self.assertTrue(np.isfinite(x.to_numpy()).all())
        pd.testing.assert_frame_equal(p['train'], original)
        p['categorical_features'] = ['event_date']
        x, _, cats = _features(p)
        self.assertEqual(cats, ['event_date'])
        self.assertEqual(list(x), ['event_date'])

    def test_linear_future_dates_refit_and_ensemble_preserve_validation(self):
        self.dates()
        p = load_problem(self.envelope())
        ti, vi = _split(p, {})
        self.assertEqual(p['validation']['strategy'], 'time')
        self.assertLess(p['train'].event_date.iloc[ti].max(), p['train'].event_date.iloc[vi].min())
        single = run(self.envelope(action='experiment', model='linear', refit=False))
        self.assertLess(single['score'], 3.)
        final = run(self.envelope(action='refit', manifest_path=single['manifest_path']))
        self.assertEqual(final['score'], single['score'])
        self.assertEqual(final['validation_id'], single['validation_id'])
        self.assertTrue(final['final'])
        output = pd.read_csv(final['candidate_path'])
        self.assertLess(float(np.sqrt(np.mean((output.target - (3 * np.arange(220, 260) + 10)) ** 2))), 3.)
        tree = run(self.envelope(action='experiment', model='lightgbm', iterations=20, refit=False))
        ensemble = run(self.envelope(action='ensemble', manifest_paths=[single['manifest_path'], tree['manifest_path']]))
        self.assertEqual(ensemble['validation_id'], single['validation_id'])

    def test_linear_portfolio_and_full_refit_contract(self):
        trained = run(self.envelope(action='portfolio', families=['linear'], entity_history=False))
        self.assertEqual(trained['model'], 'linear')
        final = run(self.envelope(action='refit', manifest_path=trained['manifest_path']))
        manifest = json.loads(Path(final['manifest_path']).read_text())
        self.assertEqual(manifest['fit_scope'], 'full_train')
        self.assertEqual(manifest['selected_iterations'], 1)
        self.assertEqual(final['score'], trained['score'])

    def test_parquet_timestamp_units_use_the_same_trend_scale(self):
        self.dates()
        p = load_problem(self.envelope())
        p['train']['event_date'] = pd.to_datetime(p['train'].event_date).dt.as_unit('us')
        p['test']['event_date'] = pd.to_datetime(p['test'].event_date).dt.as_unit('s')
        x, test, cats = _features(p)
        self.assertEqual(cats, [])
        self.assertAlmostEqual(float(test.event_date.iloc[0] - x.event_date.iloc[-1]), 1.)


if __name__ == '__main__':
    unittest.main()
