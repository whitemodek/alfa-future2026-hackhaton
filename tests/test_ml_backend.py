"""Native ML integration tests with generated data, without external services."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_backend import MLRequestError, canonical_metric, load_problem, metric_direction, run
from artifacts import ArtifactStore, SubmissionSchema


class MLBackendTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".test-temp"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / "task.md"
        self.task.write_text("Tabular regression. Metric RMSE. Target: target.", encoding="utf-8")
        rng = np.random.default_rng(25)
        feature = rng.normal(size=160)
        train = pd.DataFrame({"id": np.arange(160), "x": feature,
                              "category": np.where(feature > 0, "up", "down"),
                              "target": 12 * feature + rng.normal(size=160) * 0.05})
        train.loc[0, "category"] = None
        train.loc[1, "x"] = np.nan
        train.to_csv(self.root / "train.csv", index=False)
        pd.DataFrame({"id": [1010, 1011, 1012], "x": [-0.5, 0.1, 1.0],
                      "category": ["down", "new", None]}).to_csv(self.root / "test.csv", index=False)
        pd.DataFrame({"id": [1012, 1010, 1011], "target": [0., 0., 0.]}).to_csv(
            self.root / "sample_submission.csv", index=False)

    def envelope(self, **request):
        return {"version": 1, "task_path": str(self.task), "task_dir": str(self.root),
                "time_budget_seconds": 20, "threads": 1, "request": request}

    def assert_contract(self, response):
        store = ArtifactStore(self.root, self.root / ".agent-state", SubmissionSchema(**response["schema"]))
        accepted = store.consider(response["candidate_path"], response.get("score"),
                                  response["metric"] if "score" in response else None,
                                  response["direction"] if "score" in response else None,
                                  response.get("validation_id"))
        self.assertTrue(accepted["accepted"])
        self.assertTrue(store.publish().is_file())

    def classify(self, labels=("normal", "attack")):
        frame = pd.read_csv(self.root / "train.csv")
        frame["target"] = [labels[i % len(labels)] for i in range(len(frame))]
        frame.to_csv(self.root / "train.csv", index=False)
        self.task.write_text("Classification, target target, primary macro F1.", encoding="utf-8")

    def test_regression_training_beats_baseline_and_preserves_template_order(self):
        baseline = run(self.envelope(action="baseline"))
        experiment = run(self.envelope(action="experiment", iterations=120, depth=4))
        self.assertEqual(experiment["model"], "catboost", experiment["notes"])
        self.assertLess(experiment["score"], baseline["score"] * 0.5)
        self.assertEqual(baseline["validation_id"], experiment["validation_id"])
        self.assertNotEqual(baseline["candidate_path"], experiment["candidate_path"])
        result = pd.read_csv(experiment["candidate_path"])
        self.assertEqual(result.id.tolist(), [1012, 1010, 1011])
        self.assertTrue(np.isfinite(result.target).all())
        self.assertGreater(result.target.iloc[0], result.target.iloc[1])
        self.assertTrue(Path(experiment["manifest_path"]).is_file())

    def test_validated_model_survives_without_full_refit(self):
        baseline = run(self.envelope(action="baseline"))
        learned = run(self.envelope(action="finalize", refit=False, iterations=120, depth=4))
        self.assertEqual(learned["model"], "catboost", learned["notes"])
        self.assertLess(learned["score"], baseline["score"] * 0.5)
        self.assertEqual(learned["fit_scope"], "holdout_train")
        self.assertGreater(learned["training_rows"], 0)
        self.assertLess(learned["training_rows"], 160)
        self.assertFalse(learned.get("final", False))
        manifest = json.loads(Path(learned["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["fit_scope"], "holdout_train")
        self.assertEqual(manifest["training_rows"], learned["training_rows"])
        self.assertTrue(Path(manifest["model_path"]).is_file())
        self.assert_contract(learned)

    def test_metric_selects_unweighted_macro_f1_and_records_selected_model(self):
        from catboost import CatBoostClassifier
        train = pd.read_csv(self.root / "train.csv")
        train["target"] = np.where(train.x.fillna(0) > 0, "large", "small")
        train.to_csv(self.root / "train.csv", index=False)
        self.task.write_text("Classification, target target, primary macro F1.", encoding="utf-8")
        observed = []
        original_fit = CatBoostClassifier.fit

        def record_fit(model, *args, **kwargs):
            observed.append(model.get_params())
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostClassifier, "fit", record_fit):
            result = run(self.envelope(action="experiment", iterations=120, refit=False))
        self.assertEqual(result["model"], "catboost", result["notes"])
        self.assertEqual(observed[0]["custom_metric"], "TotalF1:average=Macro;use_weights=false")
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_iterations"], result["selected_iterations"])
        self.assertGreater(result["selected_iterations"], 0)
        self.assertEqual(manifest["training_parameters"]["iterations"], result["selected_iterations"])
        self.assert_contract(result)

    def test_no_refit_releases_training_time_reservation(self):
        from catboost import CatBoostRegressor
        deadlines = []
        original_fit = CatBoostRegressor.fit

        def record_fit(model, *args, **kwargs):
            if "eval_set" in kwargs:
                import time
                deadlines.append(kwargs["callbacks"][0].deadline - time.monotonic())
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostRegressor, "fit", record_fit):
            run(self.envelope(iterations=20, refit=True))
            run(self.envelope(iterations=20, refit=False))
        self.assertGreater(deadlines[1], deadlines[0] * 2)

    def test_failed_full_refit_keeps_validated_learned_candidate(self):
        from catboost import CatBoostRegressor
        original_fit = CatBoostRegressor.fit

        def fail_only_full_fit(model, *args, **kwargs):
            if "eval_set" not in kwargs:
                raise RuntimeError("simulated full-refit failure")
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostRegressor, "fit", fail_only_full_fit):
            result = run(self.envelope(action="finalize", iterations=120, depth=4))
        self.assertEqual(result["model"], "catboost", result["notes"])
        self.assertEqual(result["fit_scope"], "holdout_train")
        self.assertNotIn("final", result)
        self.assertIn("simulated full-refit failure", result["notes"])
        self.assert_contract(result)

    def test_partial_full_refit_keeps_validated_learned_candidate(self):
        from catboost import CatBoostRegressor
        original_fit = CatBoostRegressor.fit

        def incomplete_full_fit(model, *args, **kwargs):
            if "eval_set" not in kwargs:
                model.set_params(iterations=1)
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostRegressor, "fit", incomplete_full_fit):
            result = run(self.envelope(action="finalize", iterations=120, depth=4))
        self.assertEqual(result["model"], "catboost", result["notes"])
        self.assertEqual(result["fit_scope"], "holdout_train")
        self.assertNotIn("final", result)
        self.assertIn("before validated tree count", result["notes"])
        self.assert_contract(result)

    def test_invalid_full_refit_output_keeps_validated_learned_candidate(self):
        from catboost import CatBoostRegressor
        original_fit = CatBoostRegressor.fit

        def invalid_full_fit(model, *args, **kwargs):
            if "eval_set" not in kwargs:
                raise MLRequestError("invalid refit predictions")
            return original_fit(model, *args, **kwargs)

        with patch.object(CatBoostRegressor, "fit", invalid_full_fit):
            result = run(self.envelope(action="finalize", iterations=120, depth=4))
        self.assertEqual(result["fit_scope"], "holdout_train")
        self.assertIn("invalid refit predictions", result["notes"])
        self.assertNotIn("final", result)
        self.assert_contract(result)

    def test_string_labels_roundtrip_and_schema(self):
        self.classify()
        result = run(self.envelope(action="baseline"))
        submission = pd.read_csv(result["candidate_path"])
        self.assertTrue(set(submission.target) <= {"normal", "attack"})
        self.assertEqual(set(result["schema"]["allowed_values"]["target"]), {"normal", "attack"})
        self.assertEqual(result["direction"], "maximize")

    def test_binary_probabilities_are_finite_and_in_range(self):
        self.classify()
        result = run(self.envelope(action="experiment", metric="roc_auc", iterations=20))
        submission = pd.read_csv(result["candidate_path"])
        self.assertTrue(submission.target.between(0, 1).all())
        self.assertEqual(result["schema"]["numeric_columns"], ["target"])
        self.assertTrue(np.isfinite(result["score"]))

    def test_multiclass_probability_column_mapping(self):
        self.classify(("red", "green", "blue"))
        pd.DataFrame({"id": [1012, 1010, 1011], "red": [0, 0, 0],
                      "blue": [0, 0, 0], "green": [0, 0, 0]}).to_csv(
                          self.root / "sample_submission.csv", index=False)
        result = run(self.envelope(action="baseline", metric="log_loss"))
        submission = pd.read_csv(result["candidate_path"])
        np.testing.assert_allclose(submission[["red", "blue", "green"]].sum(axis=1), 1)
        self.assertEqual(list(submission), ["id", "red", "blue", "green"])

    def test_negative_loss_is_maximized(self):
        self.assertEqual(canonical_metric("neg_root_mean_squared_error"), "neg_rmse")
        self.assertEqual(metric_direction("neg_rmse"), "maximize")
        self.assertEqual(metric_direction("neg_r2"), "minimize")
        positive = run(self.envelope(action="baseline", metric="rmse"))
        negative = run(self.envelope(action="baseline", metric="neg_root_mean_squared_error"))
        self.assertEqual(negative["score"], -positive["score"])

    def test_numeric_class_labels_have_csv_string_schema(self):
        self.classify((0, 1))
        result = run(self.envelope(action="baseline"))
        self.assertEqual(result["schema"]["allowed_values"]["target"], ["0", "1"])
        self.assert_contract(result)

    def test_float_class_labels_have_valid_artifact_contract(self):
        self.classify((0.1, 0.2))
        result = run(self.envelope(action="baseline"))
        self.assertEqual(result["schema"]["allowed_values"]["target"], ["0.1", "0.2"])
        self.assert_contract(result)

    def test_positive_class_changes_encoding_and_validation_identity(self):
        self.classify(("attack", "normal"))
        default = run(self.envelope(action="baseline", metric="average_precision"))
        other = run(self.envelope(action="baseline", metric="average_precision", positive_class="attack"))
        self.assertNotEqual(default["validation_id"], other["validation_id"])
        problem = load_problem(self.envelope(positive_class="attack"))
        self.assertEqual(problem["classes"][1], "attack")

    def test_numeric_categorical_override_used_by_model(self):
        result = run(self.envelope(action="experiment", iterations=30, categorical_features=["x"]))
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertIn("x", manifest["categorical_features"])

    def test_repeated_run_ignores_published_submission(self):
        result = run(self.envelope(action="baseline"))
        (self.root / "submission.csv").write_bytes(Path(result["candidate_path"]).read_bytes())
        rerun = run(self.envelope(action="baseline"))
        self.assertEqual(result["schema"], rerun["schema"])

    def test_leading_zero_ids_preserved_in_candidate(self):
        (self.root / "test.csv").write_text("id,x,category\n0001,-0.5,down\n0002,0.1,new\n0003,1,up\n", encoding="utf-8")
        (self.root / "sample_submission.csv").write_text("id,target\n0003,0\n0001,0\n0002,0\n", encoding="utf-8")
        result = run(self.envelope(action="baseline"))
        candidate = pd.read_csv(result["candidate_path"], dtype={"id": str})
        self.assertEqual(candidate.id.tolist(), ["0003", "0001", "0002"])
        self.assert_contract(result)

    def test_literal_na_identifiers_preserved(self):
        (self.root / "test.csv").write_text("id,x,category\nNA,-0.5,down\nnull,0.1,new\n03,1,up\n", encoding="utf-8")
        (self.root / "sample_submission.csv").write_text("id,target\n03,0\nNA,0\nnull,0\n", encoding="utf-8")
        result = run(self.envelope(action="baseline"))
        candidate = pd.read_csv(result["candidate_path"], dtype=str, keep_default_na=False)
        self.assertEqual(candidate.id.tolist(), ["03", "NA", "null"])
        self.assert_contract(result)

    def test_non_comma_templates_are_normalized_without_modifying_inputs(self):
        for separator, extension in ((";", ".csv"), ("\t", ".tsv")):
            with self.subTest(separator=separator):
                original_path = self.root / "sample_submission.csv"
                if original_path.exists():
                    original_path.unlink()
                sample_path = self.root / ("sample_submission" + extension)
                content = separator.join(["id", "target"]) + "\n"
                content += "\n".join(separator.join([identifier, "0"]) for identifier in ["0003", "NA", "0002"]) + "\n"
                original_bytes = content.encode("utf-8")
                sample_path.write_bytes(original_bytes)
                (self.root / "test.csv").write_text("id,x,category\nNA,-0.5,down\n0002,0.1,new\n0003,1,up\n", encoding="utf-8")
                result = run(self.envelope(action="baseline", sample_path=str(sample_path)))
                normalized = Path(result["schema"]["sample_path"])
                self.assertNotEqual(normalized, sample_path)
                self.assertEqual(normalized.read_text(encoding="utf-8").splitlines()[0], "id,target")
                self.assertEqual(sample_path.read_bytes(), original_bytes)
                self.assert_contract(result)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_parquet_template_normalized_for_artifact_store(self):
        sample_path = self.root / "sample_submission.parquet"
        sample = pd.read_csv(self.root / "sample_submission.csv", dtype={"id": str})
        sample.to_parquet(sample_path, index=False)
        original_bytes = sample_path.read_bytes()
        (self.root / "sample_submission.csv").unlink()
        result = run(self.envelope(action="baseline", sample_path=str(sample_path)))
        self.assertEqual(Path(result["schema"]["sample_path"]).suffix, ".csv")
        self.assertEqual(sample_path.read_bytes(), original_bytes)
        self.assert_contract(result)

    def test_explicit_overrides_allow_nonstandard_filenames(self):
        (self.root / "train.csv").rename(self.root / "labeled.csv")
        (self.root / "test.csv").rename(self.root / "unlabeled.csv")
        result = run(self.envelope(action="inspect", train_path="labeled.csv",
                                   test_path="unlabeled.csv", target="target"))
        self.assertEqual(result["target"], "target")

    def test_missing_and_ambiguous_data_fail_clearly(self):
        with self.assertRaisesRegex(MLRequestError, "not found"):
            run(self.envelope(action="baseline", train_path="absent.csv"))
        (self.root / "train_other.csv").write_bytes((self.root / "train.csv").read_bytes())
        with self.assertRaisesRegex(MLRequestError, "unambiguously"):
            run(self.envelope(action="baseline"))

    def test_bad_submission_ids_are_rejected(self):
        pd.DataFrame({"id": [10, 11, 12], "target": [0, 0, 0]}).to_csv(
            self.root / "sample_submission.csv", index=False)
        with self.assertRaisesRegex(MLRequestError, "do not match"):
            run(self.envelope(action="baseline"))

    def test_no_template_and_tiny_data_omits_unreliable_score(self):
        (self.root / "sample_submission.csv").unlink()
        train = pd.read_csv(self.root / "train.csv").iloc[:3]
        train.to_csv(self.root / "train.csv", index=False)
        result = run(self.envelope(action="baseline"))
        self.assertNotIn("score", result)
        self.assertEqual(result["schema"]["columns"], ["id", "target"])

    def test_validation_identity_changes_when_data_or_split_changes(self):
        baseline = run(self.envelope(action="baseline"))
        alternate = run(self.envelope(action="baseline", split="time"))
        self.assertNotEqual(baseline["validation_id"], alternate["validation_id"])
        train = pd.read_csv(self.root / "train.csv")
        train.loc[2, "target"] += 1
        train.to_csv(self.root / "train.csv", index=False)
        changed = run(self.envelope(action="baseline"))
        self.assertNotEqual(baseline["validation_id"], changed["validation_id"])

    def test_cli_error_response_is_valid_json(self):
        request_path, response_path = self.root / "request.json", self.root / "response.json"
        request_path.write_text(json.dumps(self.envelope(action="unsupported")), encoding="utf-8")
        process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "ml_backend.py"),
                                  str(request_path), str(response_path)], capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 1)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response["status"], "error")
        self.assertIn("Unsupported action", response["error"])


if __name__ == "__main__":
    unittest.main()
