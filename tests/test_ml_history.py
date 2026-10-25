"""Past entity labels complement model predictions without future-label leakage."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from ml_backend import MLRequestError, load_problem, run
from ml_history import blend_past, entity_columns


class HistoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Building regression over time periods. Target: target. RMSE.', encoding='utf-8')
        rng = np.random.default_rng(19)
        values = rng.uniform(0, 100, size=24)
        rows = []
        for period in range(1, 5):
            for entity in range(24):
                rows.append(dict(id=len(rows), period=period, state='north',
                                 building_class='office', floor_area=100 + entity,
                                 year_built=1990 + entity % 10,
                                 target=float(values[entity] + period)))
        self.train = pd.DataFrame(rows)
        self.test = self.train.iloc[:4].drop(columns='target').copy()
        self.test['period'] = 5
        self.test['id'] = range(1000, 1004)
        self.train.to_csv(self.root / 'train.csv', index=False)
        self.test.to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': [1003, 1002, 1001, 1000], 'target': [0] * 4}).to_csv(
            self.root / 'sample_submission.csv', index=False)

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root), threads=1,
                    time_budget_seconds=35, request=request)

    def base(self, family='lightgbm', full=False):
        return run(self.envelope(action='experiment', model=family, refit=full,
                                 iterations=15, depth=3, split='time', time_column='period'))

    def test_strict_past_missing_and_ambiguous_keys_preserve_base(self):
        train = pd.DataFrame({'entity': ['a', 'a', 'b', 'b', None],
                              'period': [1, 3, 1, 1, 1]})
        query = pd.DataFrame({'entity': ['a'] * 5 + ['b', None, 'new'],
                              'period': [0, 1, 2, 3, 4, 4, 4, 4]})
        result, info = blend_past(train, [10, 30, 20, 20, 50], query,
                                  np.full(8, 100.), ['entity'], 'period')
        np.testing.assert_array_equal(result, [100, 100, 55, 55, 65, 100, 100, 100])
        self.assertEqual(info['matched_rows'], 3)
        self.assertEqual(info['ambiguous_keys'], 1)
        self.assertEqual(info['missing_training_rows'], 1)

    def test_auto_signature_requires_all_semantic_roles_and_rejects_target_id(self):
        problem = load_problem(self.envelope())
        columns, reason = entity_columns(problem, {})
        self.assertEqual(set(columns), {'state', 'building_class', 'floor_area', 'year_built'})
        self.assertIn('Heuristic', reason)
        self.assertEqual(entity_columns(dict(problem, features=['state', 'floor_area']), {})[0], [])
        for columns in (['target'], ['id'], ['state', 'state'], [], ['no_such_column']):
            with self.subTest(columns=columns), self.assertRaises(MLRequestError):
                entity_columns(problem, dict(entity_columns=columns))

    def test_history_refit_keeps_exact_score_and_uses_full_past_for_reversed_sample(self):
        base = self.base()
        history = run(self.envelope(action='history', manifest_path=base['manifest_path']))
        self.assertLess(history['score'], base['score'])
        self.assertEqual(history['history_validation']['coverage'], 1)
        self.assertEqual(history['validation_id'], base['validation_id'])
        self.assertEqual(history['fit_scope'], 'holdout_train')
        final = run(self.envelope(action='refit', manifest_path=history['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertEqual(final['training_rows'], len(self.train))
        self.assertEqual(final['score'], history['score'])
        self.assertEqual(final['validation_id'], history['validation_id'])
        data = pd.read_csv(final['candidate_path'])
        self.assertEqual(data['id'].tolist(), [1003, 1002, 1001, 1000])
        fm = json.loads(Path(final['manifest_path']).read_text())
        bm = json.loads(Path(fm['base_manifest_path']).read_text())
        from ml_portfolio import _load_model
        from ml_backend import _features
        problem = load_problem(self.envelope())
        _, tx, _ = _features(problem)
        pred = _load_model(bm, self.root).predict(tx)
        latest = self.train.loc[self.train.period == 4, 'target'].to_numpy()[:4]
        np.testing.assert_allclose(data['target'], (.5 * pred + .5 * latest)[::-1])

    def test_history_handles_native_ensemble_and_rejects_changed_base_manifest(self):
        a, b = self.base('catboost'), self.base('xgboost')
        ensemble = run(self.envelope(action='ensemble', manifest_paths=[a['manifest_path'], b['manifest_path']]))
        history = run(self.envelope(action='history', manifest_path=ensemble['manifest_path']))
        final = run(self.envelope(action='refit', manifest_path=history['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], history['score'])
        path = Path(ensemble['manifest_path'])
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(MLRequestError, 'manifest changed'):
            run(self.envelope(action='refit', manifest_path=history['manifest_path']))

    def test_invalid_history_requests_and_deadline_fail_before_model_operation(self):
        from ml_history import run_history
        base = self.base()
        for request in [dict(history_weight=float('nan')), dict(history_weight=1),
                        dict(entity_columns=['period']), dict(entity_columns=['target'])]:
            with self.subTest(request=request), self.assertRaises(MLRequestError):
                run(self.envelope(action='history', manifest_path=base['manifest_path'], **request))
        with self.assertRaisesRegex(MLRequestError, 'Insufficient time'):
            run_history(self.envelope(action='history', manifest_path=base['manifest_path']),
                        time.monotonic() - 10, 1, 1)


if __name__ == '__main__':
    unittest.main()
