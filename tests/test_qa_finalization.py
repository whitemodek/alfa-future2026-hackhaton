import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from qa.check_submission import check_format, score
from qa.run_local import main, run_task


class FinalizationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / 'task'
        self.task.mkdir()
        (self.task / 'task.md').write_text('Classification, Macro F1', encoding='utf-8')
        (self.task / 'train.csv').write_text('id,x,target\n1,1,Normal\n2,2,Attack\n')
        (self.task / 'sample_submission.csv').write_text('id,target\nNA,Normal\n002,Normal\n')

    def check(self, labels):
        (self.task / 'submission.csv').write_text('id,target\nNA,' + labels[0] + '\n002,' + labels[1] + '\n')
        return check_format(self.task)[0]

    def test_labels_checked_against_public_training_data(self):
        self.assertFalse(self.check(['Normal', 'Attack']))
        self.assertTrue(self.check(['Normal', 'Typo']))

    def test_nonfinite_text_rejected_in_object_predictions(self):
        for value in ('inf', '-Infinity', '1e999', 'NaN'):
            with self.subTest(value=value):
                self.assertTrue(self.check(['Normal', value]))

    def test_known_lexical_class_names_override_numeric_placeholder(self):
        (self.task / 'train.csv').write_text('id,x,target\n1,1,inf\n2,2,NA\n3,3,001\n')
        (self.task / 'sample_submission.csv').write_text('id,target\nNA,0\n002,0\n')
        for labels in (['inf', 'NA'], ['001', 'inf']):
            with self.subTest(labels=labels):
                self.assertFalse(self.check(labels))
                problems, sub = check_format(self.task)
                self.assertEqual(sub['target'].tolist(), labels)
        self.assertTrue(self.check(['001', '1']))

    def test_known_lexical_classes_score_without_pandas_conversion(self):
        (self.task / 'train.csv').write_text('id,x,target\n1,1,001\n2,2,NA\n')
        self.assertFalse(self.check(['001', 'NA']))
        hidden = self.root / '_answers'
        hidden.mkdir()
        (hidden / 'task.csv').write_text('id,target\n002,NA\nNA,001\n')
        _, sub = check_format(self.task)
        self.assertEqual(score(self.task, sub), {'macro_f1': 1.0})

    def test_numeric_regression_still_rejects_nonfinite_text(self):
        (self.task / 'task.md').write_text('Regression RMSE', encoding='utf-8')
        (self.task / 'train.csv').write_text('id,x,target\n1,1,1.0\n2,2,2.0\n')
        (self.task / 'sample_submission.csv').write_text('id,target\nNA,0.0\n002,0.0\n')
        for value in ('inf', 'NaN', '1e999'):
            with self.subTest(value=value):
                self.assertTrue(self.check(['1.0', value]))

    def test_probability_predictions_not_treated_as_class_labels(self):
        (self.task / 'task.md').write_text('Classification, ROC-AUC probability', encoding='utf-8')
        (self.task / 'train.csv').write_text('id,x,target\n1,1,0\n2,2,1\n')
        (self.task / 'sample_submission.csv').write_text('id,target\nNA,0.5\n002,0.5\n')
        self.assertFalse(self.check(['0.25', '0.75']))

    def test_numeric_regression_not_constrained_by_train_targets(self):
        (self.task / 'task.md').write_text('Regression RMSE', encoding='utf-8')
        (self.task / 'train.csv').write_text('id,x,target\n1,1,0\n2,2,1\n')
        (self.task / 'sample_submission.csv').write_text('id,target\nNA,0.5\n002,0.5\n')
        self.assertFalse(self.check(['2.5', '-5']))

    def test_harness_exit_requires_healthy_cli_and_valid_csv(self):
        for success, exit_code, killed in ((False, 2, False), (False, -9, True), (True, 0, False)):
            result = {'success': success, 'exit_code': exit_code, 'killed_by_limit': killed,
                      'check': {'format_ok': True}}
            with self.subTest(result=result), patch('qa.run_local.run_task', return_value=result):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = main(['solution.zip', str(self.root)])
                self.assertEqual(code, 0 if success else 1)
                self.assertIn('"format_ok": true', output.getvalue())

    def test_valid_csv_survives_failed_cli_but_run_is_unsuccessful(self):
        source = ("import sys, shutil\nfrom pathlib import Path\n"
                  "root=Path(sys.argv[1]).parent\n"
                  "shutil.copyfile(root/'sample_submission.csv',root/'submission.csv')\n"
                  "raise SystemExit(3)\n")
        archive = self.root / 'failed.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('solution.py', source)
        result = run_task(archive, self.task, 15)
        self.addCleanup(shutil.rmtree, result['work_dir'], ignore_errors=True)
        self.assertEqual(result['exit_code'], 3)
        self.assertTrue(result['check']['format_ok'], result)
        self.assertFalse(result['success'])

    def test_valid_csv_survives_timeout_but_run_is_unsuccessful(self):
        source = ("import sys, shutil, time\nfrom pathlib import Path\n"
                  "root=Path(sys.argv[1]).parent\n"
                  "shutil.copyfile(root/'sample_submission.csv',root/'submission.csv')\n"
                  "time.sleep(60)\n")
        archive = self.root / 'timeout.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('solution.py', source)
        result = run_task(archive, self.task, 1.5)
        self.addCleanup(shutil.rmtree, result['work_dir'], ignore_errors=True)
        self.assertTrue(result['killed_by_limit'])
        self.assertTrue(result['check']['format_ok'], result)
        self.assertFalse(result['success'])


if __name__ == '__main__':
    unittest.main()
