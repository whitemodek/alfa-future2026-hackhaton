"""Independent adversarial checks for past-target blending and refit contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_backend import MLRequestError, run
from ml_history import blend_past, entity_columns


class HistoryLookupAdversarialTests(unittest.TestCase):
    def test_strict_past_lookup_and_query_order(self):
        train = pd.DataFrame({'entity': ['a', 'a'], 'period': [3, 1]})
        query = pd.DataFrame({'entity': ['a'] * 5, 'period': [4, 0, 3, 1, 2]})
        prediction, diagnostics = blend_past(train, [30, 10], query, [100] * 5,
                                            ['entity'], 'period')
        np.testing.assert_allclose(prediction, [65, 100, 55, 100, 55])
        self.assertEqual(diagnostics['matched_rows'], 3)

    def test_same_period_collision_discards_key_even_when_targets_agree(self):
        train = pd.DataFrame({'entity': ['a', 'a', 'a', 'b'], 'period': [1, 2, 2, 1]})
        query = pd.DataFrame({'entity': ['a', 'b'], 'period': [3, 3]})
        prediction, diagnostics = blend_past(train, [10, 20, 20, 40], query, [100, 100],
                                            ['entity'], 'period')
        np.testing.assert_allclose(prediction, [100, 70])
        self.assertEqual(diagnostics['ambiguous_keys'], 1)

    def test_nonfinite_entity_cells_fall_back_instead_of_matching(self):
        train = pd.DataFrame({'entity': [np.inf, -np.inf, np.nan, 3], 'period': [1] * 4})
        query = pd.DataFrame({'entity': [np.inf, -np.inf, np.nan, 3], 'period': [2] * 4})
        prediction, diagnostics = blend_past(train, [10] * 4, query, [100] * 4,
                                            ['entity'], 'period')
        np.testing.assert_allclose(prediction, [100, 100, 100, 55])
        self.assertEqual(diagnostics['matched_rows'], 1)

    def test_tuple_keys_and_ignored_query_target(self):
        train = pd.DataFrame({'left': ['a|b', 'a'], 'right': ['c', 'b|c'], 'period': [1, 1]})
        query = pd.DataFrame({'left': ['a', 'a|b'], 'right': ['b|c', 'c'],
                              'period': [2, 2], 'target': [99999, -99999]})
        prediction, _ = blend_past(train, [10, 30], query, [100, 100],
                                    ['left', 'right'], 'period')
        np.testing.assert_allclose(prediction, [65, 55])
        query['target'] = [0, 0]
        repeated, _ = blend_past(train, [10, 30], query, [100, 100],
                                 ['left', 'right'], 'period')
        np.testing.assert_array_equal(prediction, repeated)

    def test_string_codes_are_distinct_from_numeric_missing_keys(self):
        train = pd.DataFrame({'entity': ['inf', '__MISSING__', np.inf, None], 'period': [1] * 4})
        query = pd.DataFrame({'entity': ['inf', '__MISSING__', np.inf, None], 'period': [2] * 4})
        prediction, info = blend_past(train, [10, 20, 30, 40], query, [100] * 4,
                                      ['entity'], 'period')
        np.testing.assert_allclose(prediction, [55, 60, 100, 100])
        self.assertEqual(info['matched_rows'], 2)

    def test_invalid_shape_weight_and_period_rejected(self):
        train = pd.DataFrame({'entity': ['a', 'a'], 'period': [1, 2]})
        query = pd.DataFrame({'entity': ['a'], 'period': [3]})
        for weight in [0, 1, -1, np.nan, np.inf, True]:
            with self.subTest(weight=weight), self.assertRaises(MLRequestError):
                blend_past(train, [10, 20], query, [100], ['entity'], 'period', weight)
        with self.assertRaises(MLRequestError):
            blend_past(train, [10], query, [100], ['entity'], 'period')
        with self.assertRaises(MLRequestError):
            blend_past(train, [10, 20], query, [100, 100], ['entity'], 'period')
        query['period'] = np.nan
        with self.assertRaises(MLRequestError):
            blend_past(train, [10, 20], query, [100], ['entity'], 'period')

    def test_explicit_target_and_excluded_id_cannot_be_keys(self):
        problem = {'features': ['area', 'period'], 'target': 'target', 'id_columns': ['id']}
        for columns in [['target'], ['id'], ['area', 'area']]:
            with self.subTest(columns=columns), self.assertRaises(MLRequestError):
                entity_columns(problem, {'entity_columns': columns})


class HistoryManifestAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=scratch)
        cls.root = Path(cls.directory.name)
        cls.task = cls.root / 'task.md'
        cls.task.write_text('Temporal regression. Target: target. Metric RMSE.', encoding='utf-8')
        train = pd.DataFrame([{'id': p * 20 + e, 'entity': e, 'period': p,
                               'target': 10 if p < 3 else 20}
                              for p in [1, 2, 3] for e in range(20)])
        train.to_csv(cls.root / 'train.csv', index=False)
        pd.DataFrame({'id': range(200, 220), 'entity': range(20), 'period': [4] * 20}).to_csv(
            cls.root / 'test.csv', index=False)
        pd.DataFrame({'id': range(200, 220), 'target': [0] * 20}).to_csv(
            cls.root / 'sample_submission.csv', index=False)
        base = run(cls.envelope(action='experiment', model='lightgbm', iterations=1,
                               min_child_samples=1000, refit=False, split='time', time_column='period'))
        cls.history = run(cls.envelope(action='history', manifest_path=base['manifest_path'],
                                     entity_columns=['entity']))
        cls.original = json.loads(Path(cls.history['manifest_path']).read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @classmethod
    def envelope(cls, **request):
        return dict(task_path=str(cls.task), task_dir=str(cls.root), threads=1,
                    time_budget_seconds=30, request=request)

    def tamper(self, **changes):
        manifest = dict(self.original, **changes)
        path = self.root / '.ml-artifacts' / 'tampered-history.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path

    def test_changed_time_column_is_rejected(self):
        path = self.tamper(time_column='entity')
        with self.assertRaises(MLRequestError):
            run(self.envelope(action='refit', manifest_path=str(path)))

    def test_changed_weight_is_rejected_even_when_holdout_score_is_identical(self):
        # The base and history both predict10 on validation, so ANY weight has
        # the same RMSE10. After full refit, the base mean is13.333 but history
        # predicts20 on test. A changed weight now changes the final CSV while
        # still passing a score-only check, violating exact pipeline refit.
        self.assertEqual(self.history['score'], 10.0)
        path = self.tamper(history_weight=.25)
        with self.assertRaises(MLRequestError):
            run(self.envelope(action='refit', manifest_path=str(path)))

    def test_changed_independent_configuration_file_is_rejected(self):
        config_path = Path(self.original['history_config_path'])
        contents = config_path.read_bytes()
        try:
            config = json.loads(contents)
            config['history_weight'] = .25
            config_path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaisesRegex(MLRequestError, 'configuration changed'):
                run(self.envelope(action='refit', manifest_path=self.history['manifest_path']))
        finally:
            config_path.write_bytes(contents)


class HistoryIntegrationGuardTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Temporal regression. Target: target. Metric RMSE.', encoding='utf-8')

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root), threads=1,
                    time_budget_seconds=30, request=request)

    def case(self, entities=20, past=2, matched=None, collisions=0, numeric=True,
             metric='rmse', test_period=None, classification=False):
        rows = []
        for period in range(1, past + 2):
            for entity in range(entities):
                area = 100 + entity
                if period == past + 1 and matched is not None and entity >= matched:
                    area += 10000
                rows.append(dict(id=len(rows), period=period, state='north', building_class='office',
                                 floor_area=area if numeric else f'area_{area}',
                                 year_built=1990 + entity % 10,
                                 target=(entity % 2 if classification else (10 if period <= past else 20))))
        rows.extend(dict(row) for row in rows[:collisions])
        train = pd.DataFrame(rows)
        train['id'] = range(len(train))
        train.to_csv(self.root / 'train.csv', index=False)
        test = train.iloc[:entities].drop(columns='target').copy()
        test['id'] = range(10000, 10000 + entities)
        test['period'] = past + 2 if test_period is None else test_period
        test.to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': test['id'], 'target': 0}).to_csv(
            self.root / 'sample_submission.csv', index=False)
        if classification:
            self.task.write_text('Classification. Target: target. Metric macro F1.', encoding='utf-8')
            return self.envelope(action='portfolio', families=['extra_trees'], iterations=4,
                                 split='time', time_column='period', seed_bagging=False)
        return run(self.envelope(action='experiment', model='lightgbm', iterations=1,
                                 min_child_samples=1000, refit=False, split='time',
                                 time_column='period', metric=metric))

    def test_auto_support_gates_reject_weak_signatures_but_explicit_keys_remain_available(self):
        # The positive boundary has exactly20 entities and two past periods.
        base = self.case()
        accepted = run(self.envelope(action='history', manifest_path=base['manifest_path']))
        self.assertEqual(accepted['history_validation']['coverage'], 1.0)
        for changes in [dict(matched=2), dict(entities=40, collisions=2)]:
            with self.subTest(accepted_boundary=changes):
                base = self.case(**changes)
                accepted = run(self.envelope(action='history', manifest_path=base['manifest_path']))
                self.assertGreaterEqual(accepted['history_validation']['coverage'], .1)
        for changes in [dict(entities=19), dict(past=1), dict(matched=1),
                        dict(entities=40, collisions=3), dict(numeric=False)]:
            with self.subTest(changes=changes):
                base = self.case(**changes)
                with self.assertRaisesRegex(MLRequestError, 'sufficient unambiguous temporal support'):
                    run(self.envelope(action='history', manifest_path=base['manifest_path']))
                explicit = run(self.envelope(action='history', manifest_path=base['manifest_path'],
                                             entity_columns=['state', 'building_class', 'floor_area', 'year_built']))
                self.assertGreater(explicit['history_validation']['matched_rows'], 0)

    def test_strict_future_test_gate_and_metric_allowlist(self):
        for changes, message in [(dict(test_period=3), 'strictly later'),
                                 (dict(test_period=2), 'strictly later'),
                                 (dict(metric='mape'), 'forward time validation'),
                                 (dict(metric='neg_mape'), 'forward time validation')]:
            with self.subTest(changes=changes):
                base = self.case(**changes)
                with self.assertRaisesRegex(MLRequestError, message):
                    run(self.envelope(action='history', manifest_path=base['manifest_path']))

    def test_extra_trees_classification_portfolio_never_invokes_regression_history(self):
        envelope = self.case(entities=24, classification=True)
        with patch('ml_history.run_history', side_effect=AssertionError('Classification history invoked')) as history:
            result = run(envelope)
        self.assertEqual(result['model'], 'extra_trees')
        self.assertIsNotNone(result.get('score'))
        self.assertFalse(any(trial['model'] == 'history' for trial in result['portfolio_results']))
        history.assert_not_called()


if __name__ == '__main__':
    unittest.main()
