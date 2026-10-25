"""Different learners and ensembles preserve a single honest validation contract."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_ml_backend as fixtures
from ml_backend import MLRequestError, _score, load_problem, run


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        fixtures.MLBackendTests.setUp(self)

    envelope = fixtures.MLBackendTests.envelope
    assert_contract = fixtures.MLBackendTests.assert_contract

    def train(self, family, full=False):
        return run(self.envelope(action='experiment', model=family, iterations=70,
                                 depth=4, refit=full))

    def test_both_alternative_families_beat_dummy_and_use_same_fold(self):
        dummy = run(self.envelope(action='baseline'))
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                result = self.train(family)
                self.assertEqual(result['model'], family)
                self.assertEqual(result['fit_scope'], 'holdout_train')
                self.assertLess(result['score'], dummy['score'])
                self.assertEqual(result['validation_id'], dummy['validation_id'])
                self.assert_contract(result)

    def test_alternative_full_refit_retains_exact_validation(self):
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family):
                trained = self.train(family)
                final = run(self.envelope(action='refit', manifest_path=trained['manifest_path']))
                self.assertTrue(final['final'])
                self.assertEqual(final['fit_scope'], 'full_train')
                self.assertEqual(final['training_rows'], 160)
                self.assertEqual(final['score'], trained['score'])
                self.assertEqual(final['validation_id'], trained['validation_id'])
                self.assert_contract(final)

    def test_ensemble_scores_original_holdout_models_after_full_refits(self):
        a, b = self.train('catboost', True), self.train('lightgbm', True)
        self.assertEqual(a['fit_scope'], 'full_train')
        self.assertEqual(b['fit_scope'], 'full_train')
        manifests = [json.loads(Path(r['manifest_path']).read_text()) for r in (a, b)]
        predictions = [np.load(m['validation_path'], allow_pickle=False)['prediction'] for m in manifests]
        expected_predictions = (predictions[0] + predictions[1]) / 2
        problem = load_problem(self.envelope())
        from ml_backend import _split
        _, indices = _split(problem, {})
        expected = _score(problem, problem['y'].iloc[indices], expected_predictions, None)
        ensemble = run(self.envelope(action='ensemble', manifest_paths=[r['manifest_path'] for r in (a, b)]))
        self.assertEqual(ensemble['fit_scope'], 'full_train')
        self.assertAlmostEqual(ensemble['score'], expected, places=12)
        self.assert_contract(ensemble)

    def test_ensemble_refit_completes_every_component_and_keeps_score(self):
        a, b = self.train('catboost'), self.train('xgboost')
        ensemble = run(self.envelope(action='ensemble', manifest_paths=[a['manifest_path'], b['manifest_path']],
                                     weights=[1e308, 1e308]))
        self.assertEqual(ensemble['fit_scope'], 'holdout_train')
        self.assertGreater(ensemble['score'], 0)
        final = run(self.envelope(action='refit', manifest_path=ensemble['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertEqual(final['score'], ensemble['score'])
        self.assert_contract(final)

    def test_changed_model_or_data_cannot_be_refitted(self):
        trained = self.train('lightgbm')
        manifest = json.loads(Path(trained['manifest_path']).read_text())
        Path(manifest['model_path']).write_bytes(b'changed')
        with self.assertRaises(MLRequestError):
            run(self.envelope(action='refit', manifest_path=trained['manifest_path']))

    def test_ensemble_refuses_changed_component_manifest(self):
        a, b = self.train('catboost'), self.train('xgboost')
        ensemble = run(self.envelope(action='ensemble', manifest_paths=[a['manifest_path'], b['manifest_path']]))
        path = Path(a['manifest_path'])
        content = json.loads(path.read_text())
        content['score'] += 1
        path.write_text(json.dumps(content))
        with self.assertRaisesRegex(MLRequestError, 'manifest changed'):
            run(self.envelope(action='refit', manifest_path=ensemble['manifest_path']))

    def test_probability_ensemble_preserves_labels_and_alignment(self):
        frame = pd.read_csv(self.root / 'train.csv')
        frame['target'] = np.where(frame.x.fillna(0) > 0, 'attack', 'normal')
        frame.to_csv(self.root / 'train.csv', index=False)
        self.task.write_text('Classification. Target target. Primary Macro F1.', encoding='utf-8')
        a, b = self.train('lightgbm'), self.train('xgboost')
        result = run(self.envelope(action='ensemble', manifest_paths=[a['manifest_path'], b['manifest_path']]))
        self.assertGreater(result['score'], .8)
        self.assert_contract(result)

    def test_portfolio_still_produces_result_if_a_family_is_unavailable(self):
        from ml_models import fit as native_fit
        def unavailable(family, *args, **kwargs):
            if family == 'lightgbm':
                raise ModuleNotFoundError('optional model absent')
            return native_fit(family, *args, **kwargs)
        with patch('ml_models.fit', side_effect=unavailable):
            result = run(self.envelope(action='portfolio', iterations=40, depth=4,
                                      families=['lightgbm', 'xgboost']))
        self.assertEqual(result['model'], 'xgboost')
        self.assertEqual(result['portfolio_results'][0]['error'], 'ModuleNotFoundError')
        self.assert_contract(result)

    def test_incomparable_models_are_not_ensembled(self):
        a = self.train('catboost')
        b = run(self.envelope(action='experiment', model='lightgbm', iterations=30, split='time', refit=False))
        with self.assertRaisesRegex(MLRequestError, 'identical'):
            run(self.envelope(action='ensemble', manifest_paths=[a['manifest_path'], b['manifest_path']]))

    def test_expired_ensemble_budget_does_not_load_or_predict(self):
        from ml_portfolio import run_ensemble
        with patch('ml_portfolio.read_manifest') as read:
            with self.assertRaisesRegex(MLRequestError, 'Insufficient time'):
                run_ensemble(self.envelope(action='ensemble', manifest_paths=['a', 'b']),
                             time.monotonic() - 100, 1, 1)
        read.assert_not_called()

    def test_cheap_seed_is_used_only_in_predefined_three_model_blend(self):
        from ml_portfolio import _try_seed_bag
        a, b = self.train('catboost'), self.train('xgboost')
        bag = _try_seed_bag(self.envelope(action='portfolio'), [a, b], time.monotonic(), 120, 1)
        self.assertEqual(bag['model'], 'ensemble')
        manifest = json.loads(Path(bag['manifest_path']).read_text())
        self.assertEqual([c['weight'] for c in manifest['components']], [.5, .25, .25])
        self.assertEqual(len(manifest['components']), 3)
        self.assert_contract(bag)
