"""Task contract inference must follow the scored objective, not alias priority."""
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

import ml_backend as backend


class MetricInferenceTests(unittest.TestCase):
    def problem(self, text, **request):
        train = pd.DataFrame({'x': range(20), 'target': [0, 1] * 10})
        test = pd.DataFrame({'x': range(20, 24)})
        discovered = (Path.cwd(), text, Path('train.parquet'), Path('test.parquet'), None)
        with patch.object(backend, 'discover', return_value=discovered), \
                patch.object(backend, '_read_table', side_effect=[train, test]):
            return backend.load_problem({'request': request})

    def test_primary_accuracy_beats_additional_macro_f1(self):
        problem = self.problem('Binary classification. Main metric: Accuracy. '
                               'Additionally report Macro F1.')
        self.assertEqual(problem['metric'], 'accuracy')
        self.assertEqual(problem['prediction_type'], 'label')

    def test_russian_primary_is_not_overridden_by_other_mentions(self):
        for text in [
                'Пример: ROC AUC. Основная метрика — **Accuracy**. Дополнительно Macro F1.',
                'Дополнительно Macro F1. Главная метрика: accuracy.',
                'Целевая метрика: accuracy; вспомогательная метрика Macro F1.',
                'Метрика оценки: accuracy. Дополнительно macro F1.',
                'Основной метрикой выбрана accuracy, а не Macro F1.',
                'Primary metric: accuracy rather than Macro F1.']:
            with self.subTest(text=text):
                self.assertEqual(self.problem(text)['metric'], 'accuracy')

    def test_probability_metric_drives_type_despite_regression_example(self):
        problem = self.problem('Example of a regression metric: RMSE. '
                               'Evaluation metric: log loss. Additionally report accuracy.')
        self.assertEqual(problem['problem_type'], 'classification')
        self.assertEqual(problem['metric'], 'log_loss')
        self.assertEqual(problem['prediction_type'], 'probability')

    def test_primary_regression_metric_drives_type(self):
        problem = self.problem('Classification example: Macro F1. Primary metric: RMSE.')
        self.assertEqual(problem['problem_type'], 'regression')
        self.assertEqual(problem['metric'], 'rmse')

    def test_markdown_heading_and_postfix_declarations(self):
        for text in ['## Primary metric\n\nROC AUC\nAdditional: Macro F1',
                     '**Accuracy** is the primary metric. Additionally Macro F1.',
                     'Metric used for evaluation: accuracy. Additionally Macro F1.']:
            with self.subTest(text=text):
                expected = 'roc_auc' if 'ROC AUC' in text else 'accuracy'
                self.assertEqual(self.problem(text)['metric'], expected)

    def test_conflicting_primary_declarations_require_explicit_request(self):
        for text in ['Main metric: accuracy. Primary metric: Macro F1.',
                     'Основная метрика: accuracy и Macro F1.']:
            with self.subTest(text=text):
                with self.assertRaisesRegex(backend.MLRequestError, 'Conflicting main metric'):
                    self.problem(text)
                self.assertEqual(self.problem(text, metric='roc_auc')['metric'], 'roc_auc')

    def test_unknown_explicit_main_metric_fails_instead_of_selecting_secondary(self):
        with self.assertRaisesRegex(backend.MLRequestError, 'set metric explicitly'):
            self.problem('Main metric: custom quality. Additionally report accuracy.')

    def test_aliases_do_not_produce_false_conflicts(self):
        for metric, expected in [('Macro F1', 'macro_f1'), ('Balanced accuracy', 'balanced_accuracy'),
                                 ('root mean squared error', 'rmse'),
                                 ('mean absolute percentage error', 'mape'),
                                 ('neg_root_mean_squared_error', 'neg_rmse')]:
            with self.subTest(metric=metric):
                self.assertEqual(self.problem('Primary metric: ' + metric + '.')['metric'], expected)

    def test_legacy_unambiguous_conditions_still_work(self):
        for text, expected in [('Regression. Metric RMSE.', 'rmse'),
                               ('Classification. Metric macro F1.', 'macro_f1'),
                               ('Classification, primary macro F1.', 'macro_f1')]:
            with self.subTest(text=text):
                self.assertEqual(self.problem(text)['metric'], expected)

    def test_explicit_metric_and_type_conflict_is_rejected(self):
        with self.assertRaisesRegex(backend.MLRequestError, 'incompatible with problem_type'):
            self.problem('Main metric: accuracy.', metric='rmse', problem_type='classification')


if __name__ == '__main__':
    unittest.main()
