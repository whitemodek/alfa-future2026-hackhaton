"""String class names must not be confused with floating-point sentinels."""
import csv
from pathlib import Path
import tempfile
import unittest

from artifacts import ArtifactError, ArtifactStore, SubmissionSchema, validate_submission


class LexicalSubmissionTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.test-temp'
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.candidate = self.root / 'candidate.csv'

    def write(self, labels):
        with self.candidate.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['target'])
            writer.writerows((value,) for value in labels)

    def test_explicit_lexical_classes_can_be_snapshotted_and_published(self):
        labels = ['inf', '-inf', 'NaN', 'NA', 'null']
        self.write(labels)
        schema = SubmissionSchema(columns=['target'], expected_rows=len(labels),
                                  allowed_values={'target': labels})
        store = ArtifactStore(self.root, self.root / '.agent-state', schema)
        accepted = store.consider(self.candidate)
        self.assertTrue(accepted['accepted'])
        published = store.publish()
        self.assertEqual(published.read_bytes(), self.candidate.read_bytes())

    def test_numeric_nonfinite_values_rejected_even_with_allowlist(self):
        for value in ['inf', '-inf', 'nan']:
            for declared in [False, True]:
                with self.subTest(value=value, allowed=declared):
                    self.write([value])
                    schema = SubmissionSchema(columns=['target'], numeric_columns=['target'],
                                              allowed_values={'target': [value]} if declared else {})
                    with self.assertRaises(ArtifactError):
                        validate_submission(self.candidate, self.root, schema)

    def test_unknown_labels_and_undeclared_nonfinite_spellings_rejected(self):
        for value in ['class_c', 'inf', 'NaN', 'null']:
            with self.subTest(value=value):
                self.write([value])
                schema = SubmissionSchema(columns=['target'],
                                          allowed_values={'target': ['class_a', 'class_b']})
                with self.assertRaises(ArtifactError):
                    validate_submission(self.candidate, self.root, schema)
        for value in ['inf', '-inf', 'nan']:
            self.write([value])
            with self.assertRaises(ArtifactError):
                validate_submission(self.candidate, self.root, SubmissionSchema(columns=['target']))


if __name__ == '__main__':
    unittest.main()
