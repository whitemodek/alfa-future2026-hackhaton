"""Independent backend routing, causal fit inputs and portfolio failure recovery."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import test_ml_lag as fixtures
from ml_backend import MLRequestError, run
from ml_lag import LAG_COLUMNS


class LagBackendAdversarialTests(unittest.TestCase):
    setUp = fixtures.LagPipelineTests.setUp
    envelope = fixtures.LagPipelineTests.envelope
    history = fixtures.LagPipelineTests.history

    def test_actual_backend_fit_features_exclude_own_and_outer_targets(self):
        from ml_models import fit
        observed = []

        def capture(family, x, y, vx, vy, *args, **kwargs):
            observed.append((x.copy(), y.copy(), vx.copy(), vy.copy()))
            return fit(family, x, y, vx, vy, *args, **kwargs)

        for changed in [False, True]:
            if changed:
                changed_train = self.train.copy()
                changed_train.loc[changed_train.period == 4, 'target'] += 1_000_000
                changed_train.to_csv(self.root / 'train.csv', index=False)
            history = self.history()
            with patch('ml_models.fit', side_effect=capture):
                lag = run(self.envelope(action='lag', manifest_path=history['manifest_path']))
            self.assertEqual(lag['model'], 'lag')
            self.assertEqual(lag['fit_scope'], 'holdout_train')
            actual_x, actual_y, actual_vx, _ = observed[-1]
            self.assertEqual(len(actual_x), 96)
            self.assertEqual(len(actual_vx), 32)
            self.assertNotIn('target', actual_x.columns)
            self.assertEqual(actual_x.columns[-2:].tolist(), list(LAG_COLUMNS))
            self.assertEqual(actual_x.index.tolist(), actual_y.index.tolist())
            for index, row in actual_x.iterrows():
                # Only the original outer train may supply historical targets.
                earlier = self.train.loc[(self.train.entity == row.entity) &
                                         (self.train.period < row.period) &
                                         (self.train.period < 4)]
                if earlier.empty:
                    self.assertTrue(np.isnan(row[LAG_COLUMNS[0]]))
                    self.assertEqual(row[LAG_COLUMNS[1]], 0)
                else:
                    self.assertAlmostEqual(row[LAG_COLUMNS[0]], earlier.iloc[-1].target)
                    self.assertEqual(row[LAG_COLUMNS[1]], 1)
            for _, row in actual_vx.iterrows():
                latest = self.train.loc[(self.train.entity == row.entity) & (self.train.period == 3)]
                self.assertAlmostEqual(row[LAG_COLUMNS[0]], latest.iloc[0].target)
        # Early stopping legitimately sees outer truth; feature construction
        # must be identical even after those labels change by a million.
        pd.testing.assert_frame_equal(observed[0][0], observed[1][0])
        pd.testing.assert_frame_equal(observed[0][2], observed[1][2])
        np.testing.assert_array_equal(observed[0][1], observed[1][1])
        self.assertFalse(np.array_equal(observed[0][3], observed[1][3]))

    def test_backend_refit_reuses_original_holdout_model_and_rejects_manifest_edit(self):
        history = self.history()
        lag = run(self.envelope(action='lag', manifest_path=history['manifest_path']))
        original = json.loads(Path(lag['manifest_path']).read_text(encoding='utf-8'))
        final = run(self.envelope(action='refit', manifest_path=lag['manifest_path']))
        final_manifest = json.loads(Path(final['manifest_path']).read_text(encoding='utf-8'))
        self.assertTrue(final['final'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertEqual(final['score'], lag['score'])
        self.assertEqual(final['validation_id'], lag['validation_id'])
        self.assertEqual(final_manifest['selected_iterations'], original['selected_iterations'])
        self.assertEqual(final_manifest['validation_model_sha256'], original['validation_model_sha256'])
        self.assertEqual(final_manifest['validation_model_path'], original['validation_model_path'])
        self.assertNotEqual(final_manifest['model_path'], final_manifest['validation_model_path'])
        self.assertEqual(pd.read_csv(final['candidate_path']).id.tolist(), [1003, 1002, 1001, 1000])
        changed = dict(original, selected_iterations=original['selected_iterations'] + 1)
        path = self.root / '.ml-artifacts' / 'changed-lag.json'
        path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(MLRequestError, 'Lag configuration changed'):
            run(self.envelope(action='refit', manifest_path=str(path)))

    def test_backend_nonhistory_base_and_reserved_feature_collision_fail_before_training(self):
        history = self.history()
        manifest = json.loads(Path(history['manifest_path']).read_text(encoding='utf-8'))
        with patch('ml_models.fit') as fit:
            with self.assertRaisesRegex(MLRequestError, 'validated history pipeline'):
                run(self.envelope(action='lag', manifest_path=manifest['base_manifest_path']))
            fit.assert_not_called()
        train = pd.read_csv(self.root / 'train.csv')
        test = pd.read_csv(self.root / 'test.csv')
        train[LAG_COLUMNS[0]] = 0
        test[LAG_COLUMNS[0]] = 0
        train.to_csv(self.root / 'train.csv', index=False)
        test.to_csv(self.root / 'test.csv', index=False)
        history = self.history()
        with patch('ml_models.fit') as fit:
            with self.assertRaisesRegex(MLRequestError, 'reserved lag feature names'):
                run(self.envelope(action='lag', manifest_path=history['manifest_path']))
            fit.assert_not_called()


class LagPortfolioFallbackTests(unittest.TestCase):
    def test_lag_failures_preserve_checkpointed_history_and_report_the_failed_trial(self):
        from contextlib import ExitStack
        from ml_portfolio import run_portfolio

        def result(model, score):
            return dict(status='ok', model=model, score=score, metric='rmse',
                        direction='minimize', validation_id='same-temporal-fold',
                        candidate_path=model + '.csv', manifest_path=model + '.json',
                        history_validation={'coverage': 1.0}, entity_columns=['entity'])

        for error in [TimeoutError, MemoryError]:
            with self.subTest(error=error.__name__):
                checkpoints = []
                envelope = {'request': {'action': 'portfolio', 'families': ['catboost'],
                                       'seed_bagging': False, 'entity_columns': ['entity']}}
                problem = {'classes': None, 'features': ['entity', 'period'],
                           'validation': {'strategy': 'time'}}
                with ExitStack() as stack:
                    stack.enter_context(patch('ml_portfolio.time.monotonic', return_value=0))
                    stack.enter_context(patch('ml_backend.run', return_value=result('catboost', 10)))
                    stack.enter_context(patch('ml_backend.load_problem', return_value=problem))
                    stack.enter_context(patch('ml_backend._split'))
                    stack.enter_context(patch('ml_history.entity_columns', return_value=(['entity'], 'explicit')))
                    stack.enter_context(patch('ml_history.run_history', return_value=result('history', 5)))
                    lag = stack.enter_context(patch('ml_lag.run_lag', side_effect=error('lag trial failed')))
                    stack.enter_context(patch('ml_portfolio.checkpoint',
                                              side_effect=lambda env, best: checkpoints.append(dict(best))))
                    final = run_portfolio(envelope, 0, 300, 1)
                lag.assert_called_once()
                self.assertEqual(final['model'], 'history')
                self.assertEqual(final['score'], 5)
                self.assertEqual(checkpoints[-1]['model'], 'history')
                trial = final['portfolio_results'][-1]
                self.assertEqual(trial['model'], 'lag')
                self.assertFalse(trial['accepted'])
                self.assertEqual(trial['error'], error.__name__)


if __name__ == '__main__':
    unittest.main()
