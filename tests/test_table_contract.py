"""Check source CSV schemas and class identities before model fitting."""
import csv
from pathlib import Path
import tempfile
import unittest

from ml_backend import MLRequestError, _read_table, load_problem, run


class TableContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.task = self.root / 'task.md'
        self.task.write_text('Classification. Target: target. Metric macro F1.', encoding='utf-8')
        (self.root / 'test.csv').write_text('id,x\n101,1\n102,2\n', encoding='utf-8')
        (self.root / 'sample_submission.csv').write_text('id,target\n101,0\n102,0\n', encoding='utf-8')

    def train(self, labels):
        with (self.root / 'train.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'x', 'target'])
            writer.writerows((i, i, labels[i % len(labels)]) for i in range(40))

    def envelope(self, **request):
        return dict(task_path=str(self.task), task_dir=str(self.root),
                    time_budget_seconds=10, threads=1, request=request)

    def test_lexical_classes_remain_distinct_and_csv_uses_original_labels(self):
        for labels in [('001', '002'), ('001', '1'), ('1', '1.0'), ('NA', 'null')]:
            with self.subTest(labels=labels):
                self.train(labels)
                problem = load_problem(self.envelope())
                self.assertEqual(problem['classes'], sorted(labels))
                result = run(self.envelope(action='baseline'))
                self.assertLess(result['score'], 1.0)
                with Path(result['candidate_path']).open(encoding='utf-8', newline='') as stream:
                    predictions = [row['target'] for row in csv.DictReader(stream)]
                self.assertTrue(set(predictions).issubset(labels), predictions)

    def test_numeric_labels_and_positive_class_zero_remain_numeric(self):
        self.train([0, 1])
        problem = load_problem(self.envelope(positive_class=0))
        self.assertEqual(problem['classes'], [1, 0])
        self.assertTrue(all(isinstance(value, int) for value in problem['classes']))

    def test_blank_target_is_missing_and_regression_stays_numeric(self):
        self.train(['', 'class_a'])
        with self.assertRaisesRegex(MLRequestError, 'missing'):
            load_problem(self.envelope())
        self.task.write_text('Regression. Target: target. Metric RMSE.', encoding='utf-8')
        self.train(['001', '002'])
        problem = load_problem(self.envelope())
        self.assertIsNone(problem['classes'])
        self.assertEqual(set(problem['y']), {1.0, 2.0})
        self.train(['NA', '2'])
        with self.assertRaisesRegex(MLRequestError, 'missing'):
            load_problem(self.envelope())

    def test_duplicate_source_headers_rejected_before_pandas_renames_them(self):
        for separator in [',', ';', '\t', '|']:
            for names in [('x', 'x', 'target'), ('x', 'x', 'x.1', 'target')]:
                with self.subTest(separator=separator, names=names):
                    path = self.root / 'duplicate.csv'
                    with path.open('w', encoding='utf-8-sig', newline='') as stream:
                        writer = csv.writer(stream, delimiter=separator)
                        writer.writerow(names)
                        writer.writerow(range(len(names)))
                        writer.writerow(range(len(names)))
                    with self.assertRaisesRegex(MLRequestError, 'Duplicate columns'):
                        _read_table(path)


if __name__ == '__main__':
    unittest.main()
