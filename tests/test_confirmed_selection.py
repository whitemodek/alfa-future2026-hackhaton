"""Nested selection keeps confirmation answers out of training and stopping."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_backend as fixtures
import ml_backend as b
from ml_confirmed import nested_split, select_confirmed
from qa.decision_weights import fit_decision_weights, decision_probability, confirmed_weights


class DecisionTests(unittest.TestCase):
    def test_invalid_probability_and_labels_fail_closed(self):
        for probability in ([[np.nan, 0.]], [[.2, .2]], [[1.1, -.1]]):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                fit_decision_weights(probability, np.array([0]))
        with self.assertRaises(ValueError):
            fit_decision_weights(np.array([[.5, .5]]), np.array([2]))

    def test_exact_ties_and_unsupported_rare_classes_keep_identity(self):
        p = np.array([[.9, .1], [.1, .9]] * 6)
        np.testing.assert_array_equal(fit_decision_weights(p, np.array([0, 1] * 6)), [1, 1])
        np.testing.assert_array_equal(fit_decision_weights(np.array([[.6, .4]] * 4), np.ones(4, dtype=int)), [1, 1])

    def test_weights_are_normalized_and_invalid_weights_rejected(self):
        p = np.array([[.7, .3], [.3, .7]])
        weighted = decision_probability(p, [1, 2])
        np.testing.assert_allclose(weighted.sum(axis=1), 1)
        for weights in ([0, 1], [1], [1, np.inf]):
            with self.assertRaises(ValueError):
                decision_probability(p, weights)

    def test_confirmation_rejects_bad_frozen_proposal_without_retuning(self):
        weights = np.array([1., 2.])
        frozen = weights.copy()
        accepted, report = confirmed_weights(np.array([[.6, .4], [.4, .6]]), np.array([0, 1]), weights)
        self.assertFalse(report['accepted'])
        np.testing.assert_array_equal(accepted, [1, 1])
        np.testing.assert_array_equal(weights, frozen)

    def test_both_splits_must_improve_and_direction_is_respected(self):
        base = dict(selection_score=.65, confirmation_score=.65)
        self.assertFalse(select_confirmed(base, dict(selection_score=.7, confirmation_score=.649), 'maximize', 'macro_f1')[0])
        self.assertFalse(select_confirmed(base, dict(selection_score=.651, confirmation_score=.7), 'maximize', 'macro_f1')[0])
        self.assertTrue(select_confirmed(base, dict(selection_score=.66, confirmation_score=.66), 'maximize', 'macro_f1')[0])
        self.assertTrue(select_confirmed(dict(selection_score=10, confirmation_score=10),
            dict(selection_score=9, confirmation_score=9), 'minimize', 'rmse')[0])
        with self.assertRaises(ValueError):
            select_confirmed(base, dict(selection_score=np.nan, confirmation_score=.7), 'maximize', 'macro_f1')


class ConfirmedRuntimeTests(unittest.TestCase):
    setUp = fixtures.MLBackendTests.setUp
    envelope = fixtures.MLBackendTests.envelope
    assert_contract = fixtures.MLBackendTests.assert_contract

    def test_nested_groups_never_cross_any_of_the_three_parts(self):
        frame = pd.read_csv(self.root / 'train.csv')
        frame = pd.concat([frame, frame], ignore_index=True)
        frame['id'] = np.arange(len(frame))
        frame.to_csv(self.root / 'train.csv', index=False)
        problem = b.load_problem(self.envelope(split='group'))
        outer_train, audit, train, selection = nested_split(problem, dict(split='group'))
        self.assertEqual(set(outer_train), set(train) | set(selection))
        groups = pd.util.hash_pandas_object(problem['train'][problem['features']], index=False).to_numpy()
        for a, c in ((train, selection), (train, audit), (selection, audit)):
            self.assertFalse(set(groups[a]) & set(groups[c]))

    def test_nested_time_only_trains_on_earlier_complete_periods(self):
        frame = pd.read_csv(self.root / 'train.csv')
        frame['year'] = np.repeat(np.arange(8), 20)
        frame.to_csv(self.root / 'train.csv', index=False)
        request = dict(split='time', time_column='year')
        problem = b.load_problem(self.envelope(**request))
        _, audit, train, selection = nested_split(problem, request)
        self.assertLess(frame.year.iloc[train].max(), frame.year.iloc[selection].min())
        self.assertLess(frame.year.iloc[selection].max(), frame.year.iloc[audit].min())

    def test_runtime_roundtrip_reproduces_score_and_refits_exact_selected_model(self):
        result = b.run(self.envelope(action='confirmed', refit=False))
        self.assert_contract(result)
        self.assertEqual(result['fit_scope'], 'holdout_train')
        manifest = json.loads(Path(result['manifest_path']).read_text())
        from ml_portfolio import checked_problem, _load_model
        problem, _, valid = checked_problem(self.envelope(), manifest)
        model = _load_model(manifest, self.root, validation=True)
        x, _, _ = b._features(problem)
        self.assertAlmostEqual(b._score(problem, problem['y'].iloc[valid], model.predict(x.iloc[valid]), None), result['score'])
        final = b.run(self.envelope(action='refit', manifest_path=result['manifest_path']))
        self.assertTrue(final['final'])
        self.assertEqual(final['score'], result['score'])
        self.assertEqual(final['validation_id'], result['validation_id'])

    def test_confirmation_rows_never_enter_selection_fit_or_early_stopping(self):
        import ml_models
        env = self.envelope(action='confirmed', refit=False)
        problem = b.load_problem(env)
        _, audit, _, _ = nested_split(problem, env['request'])
        forbidden = set(audit)
        real_fit, real_refit = ml_models.fit, ml_models.ModelAdapter.refit
        seen = []

        def fit(family, x, y, valid_x, valid_y, *args, **kwargs):
            self.assertFalse(forbidden & set(x.index))
            self.assertFalse(forbidden & set(valid_x.index))
            seen.append(family)
            return real_fit(family, x, y, valid_x, valid_y, *args, **kwargs)

        def refit(model, x, y, *args, **kwargs):
            self.assertFalse(forbidden & set(x.index))
            return real_refit(model, x, y, *args, **kwargs)

        with patch('ml_models.fit', fit), patch.object(ml_models.ModelAdapter, 'refit', refit):
            result = b.run(env)
        self.assertTrue(seen)
        self.assertEqual(result['status'], 'ok')

    def test_classification_confirmed_candidate_preserves_all_labels_and_refit(self):
        fixtures.MLBackendTests.classify(self, labels=('a', 'b', 'c'))
        result = b.run(self.envelope(action='confirmed', refit=True))
        self.assert_contract(result)
        self.assertTrue(set(pd.read_csv(result['candidate_path'])['target']) <= {'a', 'b', 'c'})
        self.assertEqual(result['metric'], 'macro_f1')

    def test_linear_blend_is_opt_in_and_uses_one_fixed_weight_pair(self):
        env = self.envelope(action='portfolio', linear_blend=True)
        env['time_budget_seconds'] = 90
        result = b.run(env)
        self.assert_contract(result)
        self.assertIn('linear_blend_control', result)
        self.assertTrue(any(t['model'] == 'linear_tree_ensemble' for t in result['portfolio_results']))
        manifests = [json.loads(p.read_text()) for p in (self.root / '.ml-artifacts').glob('*.json')]
        blends = [m for m in manifests if m.get('model') == 'ensemble'
                  and any(json.loads(Path(c['manifest_path']).read_text()).get('model') == 'linear'
                          for c in m.get('components', []))]
        self.assertEqual(len(blends), 1)
        self.assertEqual([c['weight'] for c in blends[0]['components']], [.75, .25])
        with self.assertRaises(b.MLRequestError):
            b.run(self.envelope(action='portfolio', linear_blend='yes'))

    def test_changed_confirmation_answers_cannot_change_selection_scores_or_tree_counts(self):
        env = self.envelope(action='confirmed', refit=False)
        problem = b.load_problem(env)
        _, audit, _, _ = nested_split(problem, env['request'])
        first = b.run(env)
        frame = pd.read_csv(self.root / 'train.csv')
        frame.loc[audit, 'target'] = frame.loc[audit, 'target'].to_numpy()[::-1]
        frame.to_csv(self.root / 'train.csv', index=False)
        second = b.run(env)
        for left, right in zip(first['confirmation']['trials'], second['confirmation']['trials']):
            self.assertEqual(left['model'], right['model'])
            self.assertEqual(left.get('selected_iterations'), right.get('selected_iterations'))
            self.assertAlmostEqual(left['selection_score'], right['selection_score'], places=10)

    def test_flat_zip_executes_confirmed_action_from_a_separate_directory(self):
        from pack_solution import build
        archive = self.root / 'experimental.zip'
        extracted = self.root / 'extracted'
        build(archive)
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn('ml_confirmed.py', bundle.namelist())
            self.assertNotIn('decision_weights.py', bundle.namelist())
            self.assertTrue(all('/' not in name for name in bundle.namelist()))
            bundle.extractall(extracted)
        request, response = self.root / 'request.json', self.root / 'response.json'
        request.write_text(json.dumps(self.envelope(action='confirmed', refit=False)), encoding='utf-8')
        environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, str(extracted / 'ml_backend.py'), str(request), str(response)],
            cwd=extracted, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=35)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        output = json.loads(response.read_text(encoding='utf-8'))
        self.assertEqual(output['status'], 'ok')
        self.assert_contract(output)


if __name__ == '__main__':
    unittest.main()
