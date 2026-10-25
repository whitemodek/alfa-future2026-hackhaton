"""Different valid representations can blend, and completed models need no fit budget."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_backend import _score, _split, load_problem, run
from ml_portfolio import refit_ensemble


class EnsembleRefinementTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.task = self.root / 'task.md'
        self.task.write_text('Regression. Target target. Metric RMSE.', encoding='utf-8')
        rng = np.random.default_rng(12)
        segment = rng.integers(0, 3, 240)
        pd.DataFrame({'id': np.arange(240), 'x': rng.normal(size=240),
                      'segment': segment, 'target': segment * 15 + rng.normal(0, .1, 240)}).to_csv(
            self.root / 'train.csv', index=False)
        pd.DataFrame({'id': ['b', 'a', 'c'], 'x': [0., 1., 2.],
                      'segment': [2, 0, 1]}).to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': ['a', 'b', 'c'], 'target': [0., 0., 0.]}).to_csv(
            self.root / 'sample_submission.csv', index=False)

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root),
                    time_budget_seconds=30, threads=1, request=request)

    def train(self, family, *, full=False, categorical=True):
        request = dict(action='experiment', model=family, iterations=100, depth=4, refit=full)
        if family == 'lightgbm':
            request.update(depth=-1, num_leaves=7, min_child_samples=5)
            if categorical:
                request['categorical_features'] = ['segment']
        result = run(self.envelope(**request))
        self.assertEqual(result['model'], family)
        return result

    def test_ensemble_preserves_each_component_numeric_or_categorical_representation(self):
        numeric = self.train('catboost')
        categorical = self.train('lightgbm')
        self.assertEqual(numeric['validation_id'], categorical['validation_id'])
        manifests = [json.loads(Path(r['manifest_path']).read_text(encoding='utf-8'))
                     for r in (numeric, categorical)]
        predictions = []
        for manifest in manifests:
            with np.load(manifest['validation_path'], allow_pickle=False) as stored:
                predictions.append(stored['prediction'].copy())
        problem = load_problem(self.envelope())
        _, valid = _split(problem, {})
        expected_score = _score(problem, problem['y'].iloc[valid],
                                .25 * predictions[0] + .75 * predictions[1], None)
        result = run(self.envelope(action='ensemble',
                     manifest_paths=[numeric['manifest_path'], categorical['manifest_path']],
                     weights=[.25, .75]))
        self.assertAlmostEqual(result['score'], expected_score, places=12)
        first = pd.read_csv(numeric['candidate_path'])
        second = pd.read_csv(categorical['candidate_path'])
        actual = pd.read_csv(result['candidate_path'])
        self.assertEqual(actual.id.tolist(), ['a', 'b', 'c'])
        np.testing.assert_allclose(actual.target, .25 * first.target + .75 * second.target,
                                   rtol=1e-12, atol=1e-12)
        final = run(self.envelope(action='refit', manifest_path=result['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['fit_scope'], 'full_train')
        self.assertEqual(final['score'], result['score'])
        self.assertEqual(pd.read_csv(final['candidate_path']).id.tolist(), ['a', 'b', 'c'])

    def test_only_pending_component_receives_remaining_fit_budget(self):
        pending = self.train('catboost')
        completed = self.train('lightgbm', full=True, categorical=False)
        self.assertEqual(completed['fit_scope'], 'full_train')
        ensemble = run(self.envelope(action='ensemble',
                       manifest_paths=[pending['manifest_path'], completed['manifest_path']]))
        manifest = json.loads(Path(ensemble['manifest_path']).read_text(encoding='utf-8'))
        completed_pending = run(self.envelope(action='refit', manifest_path=pending['manifest_path']))
        self.assertEqual(completed_pending['fit_scope'], 'full_train')
        allowances = []

        def expensive_component(envelope):
            self.assertEqual(envelope['request']['manifest_path'], pending['manifest_path'])
            allowance = envelope['time_budget_seconds']
            allowances.append(allowance)
            if allowance < 60:
                raise TimeoutError('The sole pending component needs 60 seconds')
            return completed_pending

        # One model already completed. Of 100 seconds, 90 remain after the output
        # reserve; allocating half of those to the sole pending fit is incorrect.
        with patch('ml_portfolio.time.monotonic', return_value=1000.), \
                patch('ml_backend.run', side_effect=expensive_component) as fit:
            result = refit_ensemble(self.envelope(action='refit'), manifest, 1000., 100., 1)
        self.assertEqual(fit.call_count, 1)
        self.assertGreaterEqual(allowances[0], 60)
        self.assertLessEqual(allowances[0], 90)
        self.assertTrue(result['final'])
        self.assertEqual(result['fit_scope'], 'full_train')
        self.assertEqual(result['score'], ensemble['score'])
