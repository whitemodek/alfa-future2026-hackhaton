"""Probability scores must describe the values the worker actually submits."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ml_backend import MLRequestError, _split, load_problem, run


class ProbabilityContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.task = self.root / 'task.md'
        self.task.write_text('Binary classification. Target: target. Submit probability of the positive class.',
                             encoding='utf-8')
        rng = np.random.default_rng(31)
        x = rng.normal(size=220)
        self.labels = (x + rng.normal(size=220) > .1).astype(int)
        train = pd.DataFrame({'id': np.arange(len(x)), 'x': x, 'target': self.labels})
        train.to_csv(self.root / 'train.csv', index=False)
        train.drop(columns='target').iloc[:3].to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': np.arange(3), 'target': 0.}).to_csv(self.root / 'sample_submission.csv', index=False)
        problem = load_problem(self.envelope(metric='log_loss'))
        _, self.valid = _split(problem, {})
        # Test rows intentionally mirror ONLY the excluded validation features,
        # making the serialized probability score independently checkable.
        train.drop(columns='target').iloc[self.valid].to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': train.id.iloc[self.valid].to_numpy(), 'target': 0.}).to_csv(
            self.root / 'sample_submission.csv', index=False)

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root),
                    time_budget_seconds=25, threads=1,
                    request=dict(action='experiment', iterations=80, depth=4, refit=False, **request))

    def test_probability_metrics_reject_hard_label_output(self):
        # Before the guard, actual log_loss was ~14.75 for submitted 0/1 labels
        # while all three learners advertised a probability log_loss near .61.
        for metric in ('log_loss', 'roc_auc', 'average_precision'):
            for family in ('catboost', 'lightgbm', 'xgboost'):
                with self.subTest(metric=metric, family=family):
                    with self.assertRaisesRegex(MLRequestError, '(?i)probab'):
                        run(self.envelope(metric=metric, model=family, prediction_type='label'))

    def test_probability_csv_score_matches_native_holdout_for_every_family(self):
        metrics = {'log_loss': log_loss, 'roc_auc': roc_auc_score,
                   'average_precision': average_precision_score}
        for metric, evaluate in metrics.items():
            for family in ('catboost', 'lightgbm', 'xgboost'):
                with self.subTest(metric=metric, family=family):
                    result = run(self.envelope(metric=metric, model=family))
                    probabilities = pd.read_csv(result['candidate_path']).target.to_numpy()
                    self.assertTrue(np.logical_and(probabilities > 0, probabilities < 1).all())
                    expected = evaluate(self.labels[self.valid], probabilities)
                    self.assertAlmostEqual(result['score'], expected, places=10)
                    self.assertEqual(result['schema']['numeric_columns'], ['target'])

    def test_explicit_positive_class_zero_keeps_probability_meaning(self):
        # Re-encoding the stratification labels changes the deterministic fold;
        # mirror that fold, rather than accidentally checking a different subset.
        envelope = self.envelope(metric='log_loss', positive_class=0)
        problem = load_problem(envelope)
        _, valid = _split(problem, envelope['request'])
        frame = pd.read_csv(self.root / 'train.csv')
        frame.drop(columns='target').iloc[valid].to_csv(self.root / 'test.csv', index=False)
        pd.DataFrame({'id': frame.id.iloc[valid].to_numpy(), 'target': 0.}).to_csv(
            self.root / 'sample_submission.csv', index=False)
        for family in ('catboost', 'lightgbm', 'xgboost'):
            with self.subTest(family=family):
                result = run(self.envelope(metric='log_loss', model=family, positive_class=0,
                                           prediction_type='probability'))
                probabilities = pd.read_csv(result['candidate_path']).target.to_numpy()
                expected = log_loss((self.labels[valid] == 0).astype(int), probabilities)
                self.assertAlmostEqual(result['score'], expected, places=10)


if __name__ == '__main__':
    unittest.main()
