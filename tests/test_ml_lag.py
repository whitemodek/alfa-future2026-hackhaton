"""Causal features and immutable holdout models survive full pipeline refitting."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_backend import MLRequestError, _score, load_problem, run
from ml_lag import (XGB_REQUEST, causal_lags, run_lag, refit_lag,
                    _fold_frames, _full_frames, _history_predictions)


class CausalLagTests(unittest.TestCase):
    def test_own_period_and_future_targets_cannot_enter_training_lags(self):
        train = pd.DataFrame(dict(entity=['a', 'a', 'b', 'a', 'b'], period=[3, 1, 1, 2, 2]))
        query = pd.DataFrame(dict(entity=['b', 'a', 'new'], period=[4, 4, 4]))
        y = np.array([30., 10., 100., 20., 200.])
        tr, qr, info = causal_lags(train, y, query, ['entity'], 'period')
        np.testing.assert_allclose(tr, [20, np.nan, np.nan, 10, 100], equal_nan=True)
        np.testing.assert_allclose(qr, [200, 30, np.nan], equal_nan=True)
        changed = y.copy()
        changed[0] = 999999
        changed_tr, changed_qr, _ = causal_lags(train, changed, query, ['entity'], 'period')
        np.testing.assert_array_equal(tr, changed_tr)
        self.assertEqual(changed_qr[1], 999999)
        self.assertEqual(info['query_coverage'], 2/3)

    def test_collisions_disable_only_after_they_are_in_the_past(self):
        train = pd.DataFrame(dict(entity=['a', 'a', 'a', 'a', None, np.inf, 0],
                                  period=[1, 2, 2, 3, 1, 1, 1]))
        query = pd.DataFrame(dict(entity=['a', None, np.inf, 0], period=[4]*4))
        tr, qr, info = causal_lags(train, [10, 20, 20, 30, 40, 50, 60], query, ['entity'], 'period')
        np.testing.assert_allclose(tr, [np.nan, 10, 10, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(qr, [np.nan, np.nan, np.nan, 60], equal_nan=True)
        self.assertEqual(info['ambiguous_keys'], 1)
        self.assertEqual(info['missing_training_rows'], 2)

    def test_invalid_time_alignment_and_deadline_fail_explicitly(self):
        train = pd.DataFrame(dict(entity=['a'], period=[1]))
        future = pd.DataFrame(dict(entity=['a'], period=[2]))
        bad_queries = [future.assign(period=1), future.assign(period=np.nan),
                       future.assign(period='2')]
        for query in bad_queries:
            with self.subTest(query=query.to_dict()), self.assertRaises(MLRequestError):
                causal_lags(train, [10], query, ['entity'], 'period')
        for targets in ([float('inf')], []):
            with self.assertRaises(MLRequestError):
                causal_lags(train, targets, future, ['entity'], 'period')
        with self.assertRaisesRegex(MLRequestError, 'Insufficient time'):
            causal_lags(train, [10], future, ['entity'], 'period', deadline=time.monotonic()-1)


class LagPipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Temporal regression. Target: target. Primary metric RMSE.', encoding='utf-8')
        rng = np.random.default_rng(29)
        levels = rng.uniform(20, 200, 32)
        rows = []
        for period in range(1, 5):
            for entity, level in enumerate(levels):
                rows.append(dict(id=len(rows), entity=f'e{entity:02}', period=period,
                                 x=entity/32, target=float(level+period*2)))
        self.train = pd.DataFrame(rows)
        self.train.to_csv(self.root / 'train.csv', index=False)
        test = self.train.iloc[:4].drop(columns='target').copy()
        test['period'] = 5
        test['id'] = [1000, 1001, 1002, 1003]
        test.to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame(dict(id=[1003, 1002, 1001, 1000], target=[0]*4)).to_csv(
            self.root / 'sample_submission.csv', index=False)

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root),
                    time_budget_seconds=45, threads=1, request=request)

    def history(self, full=False):
        base = run(self.envelope(action='experiment', model='lightgbm', iterations=20,
                                 depth=3, refit=full, split='time', time_column='period'))
        return run(self.envelope(action='history', manifest_path=base['manifest_path'],
                                 entity_columns=['entity']))

    def lag(self, history=None):
        history = history or self.history()
        envelope = self.envelope(action='lag', manifest_path=history['manifest_path'])
        return run_lag(envelope, time.monotonic(), 45, 1)

    def refit(self, result):
        manifest = json.loads(Path(result['manifest_path']).read_text())
        envelope = self.envelope(action='refit', manifest_path=result['manifest_path'])
        return refit_lag(envelope, manifest, time.monotonic(), 45, 1)

    def test_full_refit_keeps_exact_score_fold_model_and_reversed_sample(self):
        original = self.lag()
        om = json.loads(Path(original['manifest_path']).read_text())
        self.assertEqual(om['lag_request'], XGB_REQUEST)
        self.assertEqual(original['fit_scope'], 'holdout_train')
        final = self.refit(original)
        fm = json.loads(Path(final['manifest_path']).read_text())
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], original['score'])
        self.assertEqual(final['validation_id'], original['validation_id'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertEqual(final['training_rows'], len(self.train))
        self.assertEqual(fm['selected_iterations'], om['selected_iterations'])
        self.assertEqual(fm['validation_model_path'], om['validation_model_path'])
        self.assertEqual(fm['validation_model_sha256'], om['validation_model_sha256'])
        self.assertNotEqual(fm['model_path'], fm['validation_model_path'])
        from ml_models import ModelAdapter
        history = json.loads(Path(fm['history_manifest_path']).read_text())
        problem, ti, vi, history_vp, history_tp = _history_predictions(
            self.envelope(), history, time.monotonic(), 45)
        _, xv, _, _, _ = _fold_frames(problem, ti, vi, fm['entity_columns'], fm['time_column'], time.monotonic()+30)
        holdout = ModelAdapter.load_model(fm['validation_model_path'])
        expected_score = _score(problem, problem['y'].iloc[vi], .5*history_vp+.5*holdout.predict(xv), None)
        self.assertAlmostEqual(expected_score, final['score'], places=10)
        _, tx, _ = _full_frames(problem, fm['entity_columns'], fm['time_column'], time.monotonic()+30)
        model = ModelAdapter.load_model(fm['model_path'])
        expected = .5*history_tp+.5*model.predict(tx)
        submitted = pd.read_csv(final['candidate_path'])
        self.assertEqual(submitted.id.tolist(), [1003, 1002, 1001, 1000])
        np.testing.assert_allclose(submitted.target, expected[::-1])
        repeated = self.refit(final)
        self.assertTrue(repeated['final'])
        self.assertEqual(repeated['score'], final['score'])
        np.testing.assert_allclose(pd.read_csv(repeated['candidate_path']).target, submitted.target)

    def test_config_history_model_and_validation_tampering_are_rejected(self):
        result = self.lag()
        manifest = json.loads(Path(result['manifest_path']).read_text())
        for key in ('lag_config_path', 'history_manifest_path', 'model_path', 'validation_path'):
            path = Path(manifest[key])
            content = path.read_bytes()
            path.write_bytes(content+b'\n')
            try:
                with self.subTest(key=key), self.assertRaises(MLRequestError):
                    self.refit(result)
            finally:
                path.write_bytes(content)
        for key, value in [('score', manifest['score']+1), ('lag_weight', .75),
                           ('selected_iterations', manifest['selected_iterations']+1)]:
            changed = dict(manifest, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(MLRequestError, 'configuration'):
                refit_lag(self.envelope(), changed, time.monotonic(), 45, 1)

    def test_calendar_features_survive_lag_loading_and_refit(self):
        for filename in ('train.csv', 'test.csv'):
            frame = pd.read_csv(self.root / filename)
            frame['event_date'] = (pd.Timestamp('2020-01-01') + pd.to_timedelta(frame.period, unit='D')).astype(str)
            frame.to_csv(self.root / filename, index=False)
        original = self.lag()
        final = self.refit(original)
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], original['score'])

    def test_changed_inputs_and_nonhistory_base_are_rejected(self):
        history = self.history()
        lag = self.lag(history)
        base = json.loads(Path(history['manifest_path']).read_text())
        with self.assertRaisesRegex(MLRequestError, 'validated history pipeline'):
            run_lag(self.envelope(manifest_path=base['base_manifest_path']), time.monotonic(), 45, 1)
        path = self.root / 'train.csv'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(MLRequestError, 'data or validation changed'):
            self.refit(lag)

    def test_failed_training_and_refit_leave_history_untouched(self):
        history = self.history(full=True)
        original_csv = Path(history['candidate_path']).read_bytes()
        with patch('ml_models.fit', side_effect=MemoryError('simulated model allocation failure')):
            with self.assertRaises(MemoryError):
                self.lag(history)
        self.assertEqual(Path(history['candidate_path']).read_bytes(), original_csv)
        lag = self.lag(history)
        with patch('ml_models.ModelAdapter.refit', side_effect=TimeoutError('incomplete refit')):
            with self.assertRaises(TimeoutError):
                self.refit(lag)
        self.assertEqual(Path(history['candidate_path']).read_bytes(), original_csv)

    def test_expired_budget_precedes_any_artifact_load(self):
        with patch('ml_portfolio.read_manifest') as read:
            with self.assertRaisesRegex(MLRequestError, 'Insufficient time'):
                run_lag(self.envelope(), time.monotonic()-10, 1, 1)
            with self.assertRaisesRegex(MLRequestError, 'Insufficient time'):
                refit_lag(self.envelope(), {}, time.monotonic()-10, 1, 1)
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
