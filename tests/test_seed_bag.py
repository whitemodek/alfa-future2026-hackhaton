"""Seed averaging preserves class order, validation identity and honest refit."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_backend as fixtures
import ml_backend as b
from ml_models import ModelAdapter, SeedBagAdapter
from ml_portfolio import read_manifest, _load_model, checked_problem


class SeedBagTests(unittest.TestCase):
    setUp = fixtures.MLBackendTests.setUp
    envelope = fixtures.MLBackendTests.envelope
    assert_contract = fixtures.MLBackendTests.assert_contract

    def base(self, family='lightgbm', full=False):
        fixtures.MLBackendTests.classify(self, ('a', 'b', 'c'))
        return b.run(self.envelope(action='experiment', model=family, iterations=20, refit=full))

    def test_seed_bag_score_reproduces_and_original_model_is_not_mutated(self):
        base = self.base(full=True)
        old_path = Path(base['manifest_path'])
        before = old_path.read_bytes()
        result = b.run(self.envelope(action='seed_bag', manifest_path=str(old_path)))
        self.assert_contract(result)
        self.assertEqual(result['validation_id'], base['validation_id'])
        _, manifest = read_manifest(result['manifest_path'], self.root)
        model = _load_model(manifest, self.root, validation=True)
        self.assertIsInstance(model, SeedBagAdapter)
        self.assertEqual(len(model.members), 3)
        self.assertEqual(model.tree_count_, base['selected_iterations'] * 3)
        problem, _, valid = checked_problem(self.envelope(), manifest)
        x, _, _ = b._features(problem)
        probabilities = model.predict_proba(x.iloc[valid])
        np.testing.assert_allclose(probabilities, np.mean([m.predict_proba(x.iloc[valid]) for m in model.members], axis=0))
        reproduced = b._score(problem, problem['y'].iloc[valid], model.predict(x.iloc[valid]), probabilities)
        self.assertAlmostEqual(reproduced, result['score'])
        self.assertEqual(old_path.read_bytes(), before)
        with self.assertRaises(b.MLRequestError):
            b.run(self.envelope(action='seed_bag', manifest_path=result['manifest_path']))

    def test_bagged_ensemble_survives_full_refit_with_exact_original_score(self):
        first = self.base()
        second = b.run(self.envelope(action='experiment', model='xgboost', iterations=20, refit=False))
        source = b.run(self.envelope(action='ensemble', manifest_paths=[first['manifest_path'], second['manifest_path']]))
        result = b.run(self.envelope(action='seed_bag', manifest_path=source['manifest_path']))
        self.assertEqual(result['model'], 'ensemble')
        self.assertEqual(result['validation_id'], source['validation_id'])
        final = b.run(self.envelope(action='refit', manifest_path=result['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], result['score'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assert_contract(final)

    def test_failed_replica_does_not_change_the_source_candidate(self):
        base = self.base()
        path = Path(base['candidate_path'])
        before = path.read_bytes()
        with patch.object(ModelAdapter, '_train', side_effect=TimeoutError('replica exhausted')):
            with self.assertRaises(TimeoutError):
                b.run(self.envelope(action='seed_bag', manifest_path=base['manifest_path']))
        self.assertEqual(path.read_bytes(), before)

    def test_regression_bagging_is_rejected(self):
        base = b.run(self.envelope(action='experiment', model='lightgbm', iterations=10, refit=False))
        with self.assertRaises(b.MLRequestError):
            b.run(self.envelope(action='seed_bag', manifest_path=base['manifest_path']))


if __name__ == '__main__':
    unittest.main()
