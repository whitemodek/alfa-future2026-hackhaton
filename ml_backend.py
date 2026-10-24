"""Tabular training worker. Run: ml_backend.py request.json response.json.

The request envelope is described in docs/ML_CONTRACT.md. Actions are inspect,
baseline (fast, no boosting), experiment/train/finalize (validated CatBoost and
full-data refit). Every successful training action writes an independent CSV.
Explicit request fields override inference: train_path, test_path, sample_path,
target, id_columns, problem_type, prediction_type, metric and direction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid


class MLRequestError(ValueError):
    pass


METRICS = {
    "rmse": "minimize", "mse": "minimize", "mae": "minimize",
    "mape": "minimize", "r2": "maximize", "macro_f1": "maximize",
    "weighted_f1": "maximize", "f1": "maximize", "accuracy": "maximize",
    "balanced_accuracy": "maximize", "roc_auc": "maximize",
    "average_precision": "maximize", "log_loss": "minimize",
}


def canonical_metric(value):
    name = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    aliases = {"root_mean_squared_error": "rmse", "mean_squared_error": "mse",
               "mean_absolute_error": "mae", "mean_absolute_percentage_error": "mape",
               "f1_macro": "macro_f1", "f1_weighted": "weighted_f1", "auc": "roc_auc",
               "roc_auc_score": "roc_auc", "cross_entropy": "log_loss",
               "logloss": "log_loss", "acc": "accuracy"}
    negative = name.startswith("neg_")
    base = aliases.get(name[4:] if negative else name, name[4:] if negative else name)
    if base not in METRICS:
        raise MLRequestError(f"Unsupported metric: {value}. Set metric explicitly to a supported name.")
    return ("neg_" if negative else "") + base


def metric_direction(name):
    base = METRICS[name.removeprefix("neg_")]
    return ("maximize" if base == "minimize" else "minimize") if name.startswith("neg_") else base


def _read_table(path, string_columns=None):
    import pandas as pd
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            head = stream.read(8192)
        try:
            separator = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
        except csv.Error:
            separator = ","
        frame = pd.read_csv(path, sep=separator, low_memory=False,
                            dtype={name: str for name in (string_columns or [])})
        if string_columns:
            # Values such as "NA" are valid identifiers, not pandas missing markers.
            raw_ids = pd.read_csv(path, sep=separator, usecols=string_columns,
                                  dtype=str, keep_default_na=False)
            for name in string_columns:
                frame[name] = raw_ids[name]
    if len(frame.columns) != len(set(frame.columns)):
        raise MLRequestError(f"Duplicate columns in {path.name}")
    frame.columns = frame.columns.map(str)
    if frame.empty:
        raise MLRequestError(f"Empty table: {path.name}")
    return frame


def _resolve_input(value, root):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise MLRequestError(f"Input file not found: {path}")
    return path.resolve()


def discover(envelope):
    """Resolve actual data files; ambiguous layouts require an explicit request."""
    request = envelope.get("request", {})
    root = Path(envelope.get("task_dir") or Path(envelope["task_path"]).parent).resolve()
    task_path = _resolve_input(envelope["task_path"], root)
    text = task_path.read_text(encoding="utf-8-sig")
    tables = sorted(p for p in root.rglob("*") if p.is_file()
                    and p.suffix.lower() in {".csv", ".tsv", ".parquet", ".pq"}
                    and not any(part.startswith((".agent", ".ml", ".venv", ".git"))
                                for part in p.relative_to(root).parts))
    if len(tables) > 500:
        raise MLRequestError("Too many data files; specify train_path and test_path.")

    def choose(role, optional=False):
        explicit = request.get(role + "_path")
        if explicit:
            return _resolve_input(explicit, root)
        patterns = {"train": r"(^|[_\W])train(ing)?([_\W]|$)",
                    "test": r"(^|[_\W])(test|predict)([_\W]|$)",
                    "sample": r"sample|submission|template"}
        choices = [p for p in tables if re.search(patterns[role], p.stem, re.I)]
        # submission.csv is produced by the orchestrator and is never an input template.
        if role == "sample":
            choices = [p for p in choices if p.stem.lower() != "submission"]
            preferred = [p for p in choices if re.search(r"sample|template", p.stem, re.I)]
            choices = preferred or choices
        if role != "sample":
            choices = [p for p in choices if not re.search(patterns["sample"], p.stem, re.I)]
        # Prefer file names actually named by the task over unrelated tables.
        mentioned = [p for p in choices if p.name.lower() in text.lower()]
        choices = mentioned or choices
        if len(choices) == 1:
            return choices[0].resolve()
        if optional and not choices:
            return None
        raise MLRequestError(f"Cannot identify {role} data unambiguously; set {role}_path.")

    train_path, test_path, sample_path = choose("train"), choose("test"), choose("sample", True)
    if train_path == test_path:
        raise MLRequestError("Training and test files must be different.")
    return root, text, train_path, test_path, sample_path


def load_problem(envelope):
    import numpy as np
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target
    request = envelope.get("request", {})
    root, text, train_path, test_path, sample_path = discover(envelope)
    train, test = _read_table(train_path), _read_table(test_path)
    sample = _read_table(sample_path) if sample_path else None
    if sample is not None and len(sample) != len(test):
        raise MLRequestError("Sample submission and test have different row counts.")
    extra = [name for name in train.columns if name not in test.columns]
    target = request.get("target") or request.get("target_column")
    if not target:
        # Exact target mentions help when auxiliary training-only columns exist.
        named = [name for name in extra if re.search(
            r"(?:target|label|целевая|целевой|predict)\s*(?:column|столбец|колонка)?\s*[:=—-]?\s*[`\"']?"
            + re.escape(name) + r"(?:[`\"']|\b)", text, re.I)]
        if len(named) == 1:
            target = named[0]
        elif len(extra) == 1:
            target = extra[0]
        elif sample is not None:
            options = [name for name in sample if name in extra]
            if len(options) == 1:
                target = options[0]
    if not isinstance(target, str) or target not in train:
        raise MLRequestError("Cannot identify one target column; set target explicitly.")
    y = train[target]
    if y.isna().any():
        raise MLRequestError("Training target contains missing values; clarify labeled rows.")
    problem_type = str(request.get("problem_type", "")).lower()
    if problem_type in {"binary", "multiclass"}:
        problem_type = "classification"
    if not problem_type:
        if re.search(r"\b(regression|rmse|mae|mse|r2)\b|регресси", text, re.I):
            problem_type = "regression"
        elif re.search(r"\b(classification|macro.?f1|accuracy|roc.?auc|log.?loss)\b|классификац", text, re.I):
            problem_type = "classification"
        else:
            kind = type_of_target(y)
            problem_type = "classification" if (kind in {"binary", "multiclass"} and
                (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20)) else "regression"
    if problem_type not in {"classification", "regression"}:
        raise MLRequestError("Only tabular regression or classification is supported.")
    if problem_type == "regression":
        y = pd.to_numeric(y, errors="raise").astype(float)
        if not np.isfinite(y.to_numpy()).all():
            raise MLRequestError("Regression target must be finite.")
        classes = None
    else:
        if pd.api.types.is_numeric_dtype(y):
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise MLRequestError("Classification labels must be finite.")
        # sklearn and CatBoost agree on a stable integer encoding; CSV restores originals.
        classes = sorted(y.unique().tolist())
        if len(classes) > 1000:
            raise MLRequestError("More than 1000 classes; check problem_type and target.")
        if "positive_class" in request:
            positive = request["positive_class"]
            if len(classes) != 2 or positive not in classes:
                raise MLRequestError("positive_class must be a label of a binary target.")
            classes = [label for label in classes if label != positive] + [positive]
        mapping = {value: index for index, value in enumerate(classes)}
        y = y.map(mapping).astype(int)
    metric = request.get("metric")
    if not metric:
        hints = [(r"neg_root_mean_squared_error", "neg_rmse"),
                 (r"neg_log_loss", "neg_log_loss"),
                 (r"macro[\s_-]*f1|f1[\s_-]*macro|macro.?averaged.?f1", "macro_f1"),
                 (r"weighted[\s_-]*f1|f1[\s_-]*weighted", "weighted_f1"),
                 (r"balanced[\s_-]*accuracy", "balanced_accuracy"),
                 (r"roc[\s_-]*auc|\bauc\b", "roc_auc"),
                 (r"average[\s_-]*precision|\bpr.?auc\b", "average_precision"),
                 (r"log[\s_-]*loss|cross[\s_-]*entropy", "log_loss"),
                 (r"\baccuracy\b", "accuracy"), (r"\brmse\b|root mean squared", "rmse"),
                 (r"\bmape\b", "mape"), (r"\bmae\b|mean absolute error", "mae"),
                 (r"\bmse\b|mean squared error", "mse"), (r"\br2\b|r²", "r2"),
                 (r"\bf1\b", "f1")]
        metric = next((name for pattern, name in hints if re.search(pattern, text, re.I)),
                      "macro_f1" if problem_type == "classification" else "rmse")
    metric = canonical_metric(metric)
    direction = metric_direction(metric)
    if request.get("direction", direction) != direction:
        raise MLRequestError(f"Metric {metric} requires direction={direction}.")
    regression_metrics = {"rmse", "mse", "mae", "mape", "r2"}
    if (metric.removeprefix("neg_") in regression_metrics) != (problem_type == "regression"):
        raise MLRequestError("Metric is incompatible with problem_type.")
    id_columns = request.get("id_columns")
    if id_columns is None:
        possible = list(sample.columns) if sample is not None else list(test.columns)
        id_columns = [name for name in possible if name in test and name != target and
                      re.search(r"(^id$|^id_|_id$|^index$|^row_?id$)", name, re.I)]
    if not isinstance(id_columns, list) or any(name not in test for name in id_columns):
        raise MLRequestError("id_columns must be a list of test column names.")
    if sample is not None and any(name not in sample for name in id_columns):
        raise MLRequestError("ID columns must also exist in sample submission.")
    if id_columns:
        # Preserve IDs byte-for-byte through CSV serialization (e.g. 00042).
        test = _read_table(test_path, string_columns=id_columns)
        sample = _read_table(sample_path, string_columns=id_columns) if sample_path else None
    features = [name for name in train if name in test and name != target and name not in id_columns]
    if not features:
        raise MLRequestError("No shared train/test features.")
    categorical_features = request.get("categorical_features", [])
    if not isinstance(categorical_features, list) or any(name not in features for name in categorical_features):
        raise MLRequestError("categorical_features must list shared feature columns.")
    output_columns = [name for name in sample if name not in id_columns] if sample is not None else [target]
    prediction_type = request.get("prediction_type")
    if prediction_type in {"proba", "predict_proba"}:
        prediction_type = "probability"
    if not prediction_type:
        if classes is not None and len(output_columns) == len(classes) and len(classes) > 1:
            prediction_type = "probability"
        elif classes is not None and (metric.removeprefix("neg_") in
                                     {"roc_auc", "log_loss", "average_precision"}):
            prediction_type = "probability"
        else:
            prediction_type = "label"
    if prediction_type not in {"label", "probability"}:
        raise MLRequestError("prediction_type must be label or probability.")
    if problem_type == "regression" and (prediction_type != "label" or len(output_columns) != 1):
        raise MLRequestError("Regression requires exactly one prediction column.")
    if classes is not None:
        if prediction_type == "label" and len(output_columns) != 1:
            raise MLRequestError("Label submission requires exactly one prediction column.")
        if prediction_type == "probability" and len(output_columns) not in {1, len(classes)}:
            raise MLRequestError("Probability columns do not match class count.")
        if prediction_type == "probability" and len(output_columns) == 1 and len(classes) != 2:
            raise MLRequestError("One probability column is supported only for binary targets.")
    return dict(root=root, text=text, train_path=train_path, test_path=test_path,
                sample_path=sample_path, train=train, test=test, sample=sample, target=target,
                y=y, classes=classes, problem_type=problem_type, metric=metric,
                direction=direction, id_columns=id_columns, features=features,
                categorical_features=categorical_features,
                output_columns=output_columns, prediction_type=prediction_type)


def _features(problem):
    """Fit conversion decisions on train only; CatBoost gets native categoricals."""
    import numpy as np
    import pandas as pd
    train = problem["train"][problem["features"]].copy()
    test = problem["test"][problem["features"]].copy()
    categorical = []
    for name in problem["features"]:
        if pd.api.types.is_numeric_dtype(train[name]) and name not in problem["categorical_features"]:
            train[name] = pd.to_numeric(train[name], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            test[name] = pd.to_numeric(test[name], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(name)
            train[name] = train[name].fillna("__MISSING__").astype(str)
            test[name] = test[name].fillna("__MISSING__").astype(str)
    return train, test, categorical


def _split(problem, request):
    import numpy as np
    from sklearn.model_selection import train_test_split
    y = problem["y"]
    if len(y) < 5:
        return None, None
    # The split never changes with model random_seed or model hyperparameters.
    indices = np.arange(len(y))
    validation_fraction = float(request.get("validation_fraction", 0.2))
    if not 0.1 <= validation_fraction <= 0.4:
        raise MLRequestError("validation_fraction must be between 0.1 and 0.4.")
    if request.get("split", "random") == "time":
        split = max(1, int(len(y) * (1 - validation_fraction)))
        return indices[:split], indices[split:]
    if request.get("split", "random") != "random":
        raise MLRequestError("split must be random or time; use time only for correctly ordered data.")
    stratify = None
    if problem["classes"] is not None:
        counts = y.value_counts()
        n_valid = math.ceil(len(y) * validation_fraction)
        if counts.min() >= 2 and n_valid >= len(counts) and len(y) - n_valid >= len(counts):
            stratify = y
    train, valid = train_test_split(indices, test_size=validation_fraction, random_state=42, stratify=stratify)
    if problem["classes"] is not None:
        # Rare labels must be seen by training, even when stratification is impossible.
        missing = set(y.unique()) - set(y.iloc[train].unique())
        move = [next(int(i) for i in valid if y.iloc[i] == label) for label in missing]
        if move:
            train = np.concatenate([train, move])
            valid = np.array([i for i in valid if i not in move], dtype=int)
    return (train, valid) if len(valid) else (None, None)


def _aligned_proba(model, x, count):
    import numpy as np
    probability = np.asarray(model.predict_proba(x), dtype=float)
    result = np.zeros((len(x), count), dtype=float)
    for column, label in enumerate(model.classes_):
        result[:, int(label)] = probability[:, column]
    return result


def _score(problem, y, prediction, probability):
    import numpy as np
    from sklearn import metrics
    name = problem["metric"].removeprefix("neg_")
    if name == "rmse":
        value = math.sqrt(metrics.mean_squared_error(y, prediction))
    elif name == "mse":
        value = metrics.mean_squared_error(y, prediction)
    elif name == "mae":
        value = metrics.mean_absolute_error(y, prediction)
    elif name == "mape":
        value = metrics.mean_absolute_percentage_error(y, prediction)
    elif name == "r2":
        value = metrics.r2_score(y, prediction)
    elif name in {"macro_f1", "weighted_f1", "f1"}:
        average = {"macro_f1": "macro", "weighted_f1": "weighted", "f1": "binary"}[name]
        if name == "f1" and len(problem["classes"]) != 2:
            raise MLRequestError("Multiclass F1 needs an explicit averaging metric such as macro_f1.")
        value = metrics.f1_score(y, prediction, average=average, zero_division=0)
    elif name == "accuracy":
        value = metrics.accuracy_score(y, prediction)
    elif name == "balanced_accuracy":
        value = metrics.balanced_accuracy_score(y, prediction)
    elif name == "log_loss":
        value = metrics.log_loss(y, probability, labels=list(range(len(problem["classes"]))))
    elif name == "average_precision":
        if len(problem["classes"]) != 2:
            raise MLRequestError("average_precision currently supports binary classification only.")
        value = metrics.average_precision_score(y, probability[:, 1])
    elif name == "roc_auc":
        if len(problem["classes"]) == 2:
            value = metrics.roc_auc_score(y, probability[:, 1])
        else:
            value = metrics.roc_auc_score(y, probability, multi_class="ovr", average="macro",
                                          labels=list(range(len(problem["classes"]))))
    else:
        raise MLRequestError("Metric is not implemented.")
    if not np.isfinite(value):
        return None
    return float(-value if problem["metric"].startswith("neg_") else value)


def _validation_id(problem, train_indices, valid_indices):
    digest = hashlib.sha256()
    # Data content rather than file names makes scores incomparable after a data change.
    with problem["train_path"].open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(json.dumps({"target": problem["target"], "type": problem["problem_type"],
                              "features": problem["features"],
                              "classes": problem["classes"],
                              "train": train_indices.tolist(), "valid": valid_indices.tolist()},
                             sort_keys=True).encode())
    return "holdout-v1-" + digest.hexdigest()[:24]


def _output(problem, prediction, probability, request):
    import numpy as np
    import pandas as pd
    sample, test, ids = problem["sample"], problem["test"], problem["id_columns"]
    result = sample.copy() if sample is not None else test[ids].copy()
    order = np.arange(len(test))
    if sample is not None and ids:
        if test.duplicated(ids).any() or sample.duplicated(ids).any():
            raise MLRequestError("Submission IDs must identify unique test rows.")
        if test[ids].isna().any().any() or sample[ids].isna().any().any():
            raise MLRequestError("Submission IDs contain missing values.")
        lookup = test[ids].copy()
        lookup["__ml_row_order__"] = np.arange(len(test))
        joined = sample[ids].merge(lookup, how="left", on=ids, sort=False, validate="one_to_one")
        if joined["__ml_row_order__"].isna().any():
            raise MLRequestError("Sample submission IDs do not match test rows.")
        order = joined["__ml_row_order__"].to_numpy(dtype=int)
    columns = problem["output_columns"]
    numeric_columns, allowed_values = [], {}
    if problem["classes"] is None:
        result[columns[0]] = np.asarray(prediction).reshape(-1)[order]
        numeric_columns = columns
    elif problem["prediction_type"] == "label":
        labels = np.array(problem["classes"])[np.asarray(prediction).reshape(-1).astype(int)]
        result[columns[0]] = labels[order]
        allowed_values[columns[0]] = [str(label) for label in problem["classes"]]
    else:
        if len(columns) == 1:
            positive = request.get("positive_class", problem["classes"][1])
            if positive not in problem["classes"]:
                raise MLRequestError("positive_class is not a training label.")
            result[columns[0]] = probability[order, problem["classes"].index(positive)]
        else:
            labels = request.get("class_order")
            if labels is None:
                # Match exact label names or common class_<label> templates.
                label_names = {str(label): label for label in problem["classes"]}
                stripped = [re.sub(r"^(?:class|prob|proba|target)[_-]", "", col, flags=re.I) for col in columns]
                if all(col in label_names for col in columns):
                    labels = [label_names[col] for col in columns]
                elif all(col in label_names for col in stripped):
                    labels = [label_names[col] for col in stripped]
                else:
                    raise MLRequestError("Cannot map probability columns to labels; set class_order.")
            if len(labels) != len(columns) or set(labels) != set(problem["classes"]):
                raise MLRequestError("class_order must contain every training label exactly once.")
            for col, label in zip(columns, labels):
                result[col] = probability[order, problem["classes"].index(label)]
        numeric_columns = columns
    if result[columns].isna().any().any():
        raise MLRequestError("Prediction contains missing values.")
    if numeric_columns and not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise MLRequestError("Prediction contains non-finite values.")
    directory = problem["root"] / ".ml-artifacts"
    directory.mkdir(exist_ok=True)
    path = directory / ("candidate-" + uuid.uuid4().hex[:12] + ".csv")
    result.to_csv(path, index=False)
    schema = dict(columns=list(result.columns), expected_rows=len(result), id_columns=ids,
                  numeric_columns=numeric_columns)
    if allowed_values:
        schema["allowed_values"] = allowed_values
    if problem["sample_path"]:
        # ArtifactStore consumes comma-separated CSV. Normalize even when the
        # provided template is TSV/semicolon CSV/parquet, retaining lexical IDs.
        template_bytes = sample.to_csv(index=False, lineterminator="\n").encode("utf-8")
        template_name = "template-" + hashlib.sha256(template_bytes).hexdigest()[:24] + ".csv"
        template_path = directory / template_name
        if not template_path.is_file() or template_path.read_bytes() != template_bytes:
            template_path.write_bytes(template_bytes)
        schema["sample_path"] = str(template_path)
    return path, schema


class _DeadlineCallback:
    def __init__(self, deadline):
        self.deadline = deadline

    def after_iteration(self, info):
        return time.monotonic() < self.deadline


def run(envelope):
    import numpy as np
    from sklearn.dummy import DummyClassifier, DummyRegressor
    started = time.monotonic()
    request = envelope.get("request", {})
    if not isinstance(request, dict):
        raise MLRequestError("request must be an object.")
    action = request.get("action", "experiment")
    if action not in {"inspect", "baseline", "experiment", "train", "finalize"}:
        raise MLRequestError(f"Unsupported action: {action}")
    budget = float(envelope.get("time_budget_seconds", 300))
    if not math.isfinite(budget) or budget < 1:
        raise MLRequestError("time_budget_seconds must be finite and at least 1.")
    threads = max(1, min(8, int(envelope.get("threads", 1))))
    problem = load_problem(envelope)
    if action == "inspect":
        return dict(version=1, status="ok", train_path=str(problem["train_path"]),
                    test_path=str(problem["test_path"]),
                    sample_path=str(problem["sample_path"]) if problem["sample_path"] else None,
                    target=problem["target"], id_columns=problem["id_columns"],
                    output_columns=problem["output_columns"],
                    problem_type=problem["problem_type"], metric=problem["metric"],
                    direction=problem["direction"], features=problem["features"][:100],
                    feature_dtypes={name: str(problem["train"][name].dtype) for name in problem["features"][:100]},
                    train_rows=len(problem["train"]), test_rows=len(problem["test"]),
                    classes=problem["classes"], prediction_type=problem["prediction_type"])
    x, test_x, categorical = _features(problem)
    y = problem["y"]
    train_i, valid_i = _split(problem, request)
    classification = problem["classes"] is not None
    dummy = DummyClassifier(strategy="prior") if classification else DummyRegressor(
        strategy="median" if problem["metric"].removeprefix("neg_") == "mae" else "mean")
    notes = []
    score, validation_id = None, None
    # Dummy estimators ignore features. Using a constant avoids categorical conversion.
    dummy_x = np.zeros((len(y), 1))
    if train_i is not None:
        dummy.fit(dummy_x[train_i], y.iloc[train_i])
        try:
            score = _score(problem, y.iloc[valid_i], dummy.predict(dummy_x[valid_i]),
                           _aligned_proba(dummy, dummy_x[valid_i], len(problem["classes"])) if classification else None)
        except ValueError as exc:
            notes.append("Validation metric unavailable on this split: " + str(exc)[:200])
        validation_id = _validation_id(problem, train_i, valid_i)
    else:
        notes.append("Too few rows for a useful holdout; score omitted.")
    # Write a complete baseline before fitting anything expensive.
    dummy.fit(dummy_x, y)
    prediction = dummy.predict(np.zeros((len(test_x), 1)))
    probability = _aligned_proba(dummy, np.zeros((len(test_x), 1)), len(problem["classes"])) if classification else None
    path, schema = _output(problem, prediction, probability, request)
    chosen, model_path = "dummy", None
    baseline_path = str(path)
    remaining = budget - (time.monotonic() - started)
    if action != "baseline" and train_i is not None and remaining >= 3 and (not classification or len(problem["classes"]) > 1):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            model_class = CatBoostClassifier if classification else CatBoostRegressor
            iterations = max(10, min(4000, int(request.get("iterations", 2000))))
            depth = max(3, min(9, int(request.get("depth", 6))))
            learning_rate = float(request.get("learning_rate", 0.06))
            if not 0.001 <= learning_rate <= 1:
                raise MLRequestError("learning_rate must be between 0.001 and 1.")
            loss = ("MultiClass" if len(problem["classes"]) > 2 else "Logloss") if classification else (
                "MAE" if problem["metric"].removeprefix("neg_") == "mae" else "RMSE")
            parameters = dict(iterations=iterations, depth=depth, learning_rate=learning_rate,
                              loss_function=loss, thread_count=threads, random_seed=int(request.get("random_seed", 42)),
                              verbose=False, allow_writing_files=False, cat_features=categorical)
            if classification:
                weighting = request.get("auto_class_weights", "SqrtBalanced" if problem["metric"] in {
                    "macro_f1", "balanced_accuracy"} else "None")
                if weighting not in {"None", "Balanced", "SqrtBalanced"}:
                    raise MLRequestError("auto_class_weights must be None, Balanced or SqrtBalanced.")
                if weighting != "None":
                    parameters["auto_class_weights"] = weighting
            model = model_class(**parameters)
            fit_started = time.monotonic()
            # Leave roughly half the remaining budget for full-data refit and output.
            validation_deadline = fit_started + max(1, (remaining - 2) * 0.42)
            model.fit(x.iloc[train_i], y.iloc[train_i], eval_set=(x.iloc[valid_i], y.iloc[valid_i]),
                      early_stopping_rounds=60, callbacks=[_DeadlineCallback(validation_deadline)])
            validation_time = time.monotonic() - fit_started
            learned_prediction = np.asarray(model.predict(x.iloc[valid_i])).reshape(-1)
            learned_probability = _aligned_proba(model, x.iloc[valid_i], len(problem["classes"])) if classification else None
            learned_score = _score(problem, y.iloc[valid_i], learned_prediction, learned_probability)
            better = learned_score is not None and (score is None or
                (learned_score >= score if problem["direction"] == "maximize" else learned_score <= score))
            if better:
                count = max(1, int(model.tree_count_))
                left = budget - (time.monotonic() - started) - 1
                estimate = validation_time * len(y) / len(train_i)
                if left < max(1, estimate * 1.1):
                    notes.append("Kept baseline: insufficient time for validated full-data refit.")
                else:
                    parameters["iterations"] = count
                    final_model = model_class(**parameters)
                    final_model.fit(x, y, callbacks=[_DeadlineCallback(time.monotonic() + left)])
                    if final_model.tree_count_ != count:
                        notes.append("Kept baseline: full refit hit deadline before validated tree count.")
                    else:
                        prediction = np.asarray(final_model.predict(test_x)).reshape(-1)
                        probability = _aligned_proba(final_model, test_x, len(problem["classes"])) if classification else None
                        path, schema = _output(problem, prediction, probability, request)
                        score, chosen = learned_score, "catboost"
                        model_path = path.with_suffix(".cbm")
                        final_model.save_model(str(model_path))
            else:
                notes.append("Dummy validation score was at least as good as CatBoost.")
        except MLRequestError:
            raise
        except Exception as exc:
            # Baseline is already a valid standalone artifact; no fabricated model score.
            notes.append(f"CatBoost failed; kept validated baseline: {type(exc).__name__}: {str(exc)[:250]}")
    elif action != "baseline":
        notes.append("Kept baseline: little time, one class, or no holdout.")
    manifest = {"model": chosen, "target": problem["target"], "features": problem["features"],
                "categorical_features": categorical, "classes": problem["classes"],
                "request": request, "validation_id": validation_id, "score": score,
                "metric": problem["metric"], "direction": problem["direction"],
                "model_path": str(model_path) if model_path else None}
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    result = dict(version=1, status="ok", candidate_path=str(path), schema=schema,
                  metric=problem["metric"], direction=problem["direction"],
                  notes="; ".join([f"{chosen}; full training data used for final predictions"] + notes),
                  model=chosen, manifest_path=str(manifest_path), baseline_path=baseline_path,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    if score is not None:
        result.update(score=score, validation_id=validation_id)
    if action == "finalize":
        result["final"] = True
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("Usage: ml_backend.py REQUEST.json RESPONSE.json", file=sys.stderr)
        return 2
    request_path, response_path = map(Path, argv)
    try:
        envelope = json.loads(request_path.read_text(encoding="utf-8-sig"))
        if envelope.get("version", 1) != 1:
            raise MLRequestError("Unsupported request version.")
        # Applied before numeric libraries are imported, including in a fresh CLI process.
        threads = max(1, min(8, int(envelope.get("threads", 1))))
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(threads)
        result = run(envelope)
    except Exception as exc:
        result = dict(version=1, status="error", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
    response_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
