import tempfile
import unittest
from pathlib import Path

from artifacts import ArtifactError, ArtifactStore, SubmissionSchema, validate_submission


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "task"
        self.task.mkdir()
        self.schema = SubmissionSchema(columns=["id", "prediction"], expected_rows=2,
                                       id_columns=["id"], numeric_columns=["prediction"])
        self.store = ArtifactStore(self.task, self.root / "state", self.schema)

    def write(self, text="id,prediction\na,1\nb,2\n", name="candidate.csv"):
        path = self.task / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_contract(self):
        result = validate_submission(self.write(), self.task, self.schema)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.columns, ("id", "prediction"))
        self.assertEqual(len(result.sha256), 64)

    def test_invalid_csv_variants(self):
        for content in ["", "id,id\na,1\nb,2\n", "id,prediction\na,1,2\nb,2\n",
                        "id,prediction\na,\nb,2\n", "id,prediction\na,NaN\nb,2\n",
                        "id,prediction\na,1e999\nb,2\n", "id,prediction\na,abc\nb,2\n",
                        "id,prediction\na,1\n", "prediction,id\n1,a\n2,b\n",
                        'id,prediction\na,"unterminated\n']:
            with self.subTest(content=content), self.assertRaises(ArtifactError):
                validate_submission(self.write(content), self.task, self.schema)

    def test_sample_order_and_count(self):
        self.write(name="sample.csv")
        schema = SubmissionSchema(sample_path="sample.csv", id_columns=["id"])
        validate_submission(self.write(), self.task, schema)
        for content in ["id,prediction\nb,2\na,1\n", "id,prediction\na,1\n",
                        "id,prediction\na,1\nb,2\nc,3\n"]:
            with self.subTest(content=content), self.assertRaises(ArtifactError):
                validate_submission(self.write(content), self.task, schema)

    def test_path_escape(self):
        outside = self.root / "outside.csv"
        outside.write_text("prediction\n1\n")
        with self.assertRaises(ArtifactError):
            validate_submission(outside, self.task)
        with self.assertRaises(ArtifactError):
            validate_submission("../outside.csv", self.task)
        try:
            (self.task / "link.csv").symlink_to(outside)
        except OSError:
            return  # Windows may not grant symlink privilege.
        with self.assertRaises(ArtifactError):
            validate_submission("link.csv", self.task)

    def test_invalid_later_candidate_preserves_best(self):
        self.store.consider(self.write())
        with self.assertRaises(ArtifactError):
            self.store.consider(self.write("id,prediction\na,nan\nb,2\n"))
        published = self.store.publish()
        self.assertEqual(published.read_text(), "id,prediction\na,1\nb,2\n")

    def test_scores_only_compared_with_matching_validation(self):
        path = self.write()
        self.assertTrue(self.store.consider(path)["accepted"])
        self.assertFalse(self.store.consider(path)["accepted"])
        self.assertTrue(self.store.consider(path, .7, "accuracy", "maximize", "split1")["accepted"])
        self.assertFalse(self.store.consider(path, .9, "accuracy", "maximize", "split2")["accepted"])
        self.assertFalse(self.store.consider(path, .9, "f1", "maximize", "split1")["accepted"])
        self.assertFalse(self.store.consider(path, .6, "accuracy", "maximize", "split1")["accepted"])
        self.assertTrue(self.store.consider(path, .8, "accuracy", "maximize", "split1")["accepted"])

    def test_minimization_and_bad_score(self):
        path = self.write()
        self.store.consider(path, 2., "loss", "minimize", "holdout")
        self.assertTrue(self.store.consider(path, 1., "loss", "minimize", "holdout")["accepted"])
        for value in [float("nan"), float("inf"), True, "0.5"]:
            with self.subTest(value=value), self.assertRaises(ArtifactError):
                self.store.consider(path, value, "loss", "minimize", "holdout")
        with self.assertRaises(ArtifactError):
            self.store.consider(path, .5)

    def test_snapshot_tampering_blocks_publication(self):
        self.store.consider(self.write())
        snapshot = self.store.state_dir / self.store.best["snapshot"]
        snapshot.write_text("id,prediction\na,3\nb,4\n")
        with self.assertRaises(ArtifactError):
            self.store.publish()
        self.assertFalse((self.task / "submission.csv").exists())

    def test_publication_revalidates_current_schema(self):
        self.store.consider(self.write())
        self.store.schema = SubmissionSchema(expected_rows=3)
        with self.assertRaises(ArtifactError):
            self.store.publish()

    def test_empty_predictions_require_explicit_zero_rows(self):
        path = self.write("prediction\n")
        with self.assertRaises(ArtifactError):
            validate_submission(path, self.task)
        self.assertEqual(validate_submission(path, self.task, SubmissionSchema(expected_rows=0)).rows, 0)

    def test_schema_rejects_malformed_config(self):
        for kwargs in [{"columns": "abc"}, {"columns": []}, {"expected_rows": True},
                       {"expected_rows": -1}, {"id_columns": ["x", "x"]},
                       {"columns": {"id": True}}, {"id_columns": None},
                       {"sample_path": 1}, {"numeric_columns": False}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ArtifactError):
                SubmissionSchema(**kwargs)

    def test_sample_mutation_cannot_break_saved_contract(self):
        sample = self.write(name="sample.csv")
        self.store.schema = SubmissionSchema(sample_path="sample.csv", id_columns=["id"])
        self.store.consider(self.write())
        sample.unlink()
        self.assertTrue(self.store.publish().is_file())

    def test_oversized_file_rejected_before_registration(self):
        candidate = self.write()
        self.store.max_bytes = 8
        with self.assertRaises(ArtifactError):
            self.store.consider(candidate)
        with self.assertRaises(ArtifactError):
            validate_submission(candidate, self.task, max_bytes=8)
        self.assertIsNone(self.store.best)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ArtifactError, "duplicate ID"):
            validate_submission(self.write("id,prediction\na,1\na,2\n"), self.task, self.schema)

    def test_allowed_values_restrict_labels(self):
        schema = SubmissionSchema(allowed_values={"prediction": ["cat", "dog"]})
        validate_submission(self.write("id,prediction\na,cat\nb,dog\n"), self.task, schema)
        with self.assertRaisesRegex(ArtifactError, "allowed_values"):
            validate_submission(self.write("id,prediction\na,cat\nb,bird\n"), self.task, schema)
        for values in [None, {"prediction": "cat"}, {"prediction": [1]}, {"prediction": []}]:
            with self.subTest(values=values), self.assertRaises(ArtifactError):
                SubmissionSchema(allowed_values=values)

    def test_final_refit_may_replace_equal_score_once(self):
        candidate = self.write()
        self.store.consider(candidate, .8, "f1", "maximize", "split1")
        self.assertFalse(self.store.consider(candidate, .8, "f1", "maximize", "split2", final=True)["accepted"])
        self.assertTrue(self.store.consider(candidate, .8, "f1", "maximize", "split1", final=True)["accepted"])
        self.assertTrue(self.store.best["final"])
        self.assertFalse(self.store.consider(candidate, .8, "f1", "maximize", "split1", final=True)["accepted"])
        self.assertFalse(self.store.consider(candidate, final=True)["accepted"])
        self.assertTrue(self.store.consider(candidate, .9, "f1", "maximize", "split1")["accepted"])
        self.assertFalse(self.store.best["final"])

    def test_unscored_final_refit_only_once(self):
        candidate = self.write()
        self.store.consider(candidate)
        self.assertTrue(self.store.consider(candidate, final=True)["accepted"])
        self.assertFalse(self.store.consider(candidate, final=True)["accepted"])
        with self.assertRaises(ArtifactError):
            self.store.consider(candidate, final="true")


if __name__ == "__main__":
    unittest.main()
