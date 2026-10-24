"""Deterministic validation that respects observed periods and duplicate groups."""
from __future__ import annotations

import math
import re


def _feature_groups(frame):
    import pandas as pd
    # Hashes identify equivalent feature rows, never target values or row IDs.
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def _time_hint(name, text):
    return bool(re.search(r"(?:^|[_\W])(year|time|date|period|timestamp)(?:$|[_\W])",
                          str(name), re.I) and re.search(
        r"year|time|date|period|год|времен|период|дат", text, re.I))


def plan_validation(problem, request=None):
    """Return indices, actual strategy and an explanation, without fitting models.

    Automatic selection inspects features and test covariates only. Explicit
    ``split=random`` retains conventional stratification; ``split=time`` can
    specify a time_column and otherwise assumes existing chronological row order.
    ``split=group`` uses group_columns, or all model features by default.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    request = request or {}
    train, test, y = problem["train"], problem["test"], problem["y"]
    indices = np.arange(len(y))
    fraction = float(request.get("validation_fraction", 0.2))
    if not math.isfinite(fraction) or not 0.1 <= fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.1 and 0.4")
    strategy = request.get("split", "auto")
    automatic = strategy == "auto"
    if strategy not in {"auto", "random", "time", "group"}:
        raise ValueError("split must be auto, random, time or group")
    time_column = request.get("time_column")
    columns = request.get("group_columns")
    group_by_features = request.get("group_by_features", False)
    if not isinstance(group_by_features, bool):
        raise ValueError("group_by_features must be a boolean")
    if time_column and (columns is not None or group_by_features):
        raise ValueError("Choose time_column or grouping, not both")
    if time_column and strategy not in {"auto", "time"}:
        raise ValueError("time_column requires split=auto or time")
    if (columns is not None or group_by_features) and strategy not in {"auto", "group"}:
        raise ValueError("Grouping requires split=auto or group")
    if len(y) < 5:
        return dict(train_indices=None, valid_indices=None, strategy=strategy,
                    rationale="Too few rows for validation", diagnostics={})
    features = problem["features"]
    hashes = _feature_groups(train[features])
    feature_hashes = hashes
    duplicate_fraction = float(1 - len(np.unique(hashes)) / len(hashes))
    diagnostics = {"duplicate_feature_fraction": duplicate_fraction}
    diagnostics["test_seen_fraction"] = float(np.isin(
        _feature_groups(test[features]), hashes).mean()) if len(test) else 0.0
    rationale = "Explicit validation strategy"
    if strategy == "auto":
        if time_column:
            strategy = "time"
            rationale = "Explicit time column"
        elif columns is not None or group_by_features:
            strategy = "group"
            rationale = "Explicit grouping columns or feature grouping"
        else:
            for name in features:
                if not _time_hint(name, problem.get("text", "")):
                    continue
                a, b = train[name], test[name]
                if not pd.api.types.is_numeric_dtype(a) or a.isna().any() or b.isna().any():
                    continue
                if 2 <= a.nunique() <= 100 and len(b) and b.min() > a.max():
                    time_column, strategy = name, "time"
                    rationale = "Test periods are later than every training period"
                    break
            if strategy == "auto":
                strategy = "group" if duplicate_fraction >= 0.1 else "random"
                rationale = ("Repeated feature vectors must stay in one fold" if strategy == "group"
                             else "No temporal shift or substantial duplicate groups detected")
    if strategy == "time":
        if time_column:
            if (time_column not in train or time_column == problem.get('target')
                    or train[time_column].isna().any()):
                raise ValueError("time_column must name a nonmissing training column other than the target")
            values = train[time_column]
            periods = sorted(values.unique())
            if len(periods) < 2:
                raise ValueError("Time validation needs at least two distinct periods")
            # Hold out complete latest periods, including unsorted input rows.
            chosen = []
            for period in reversed(periods[1:]):
                chosen.append(period)
                if values.isin(chosen).sum() >= math.ceil(len(y) * fraction):
                    break
            mask = values.isin(chosen).to_numpy()
            train_i, valid_i = indices[~mask], indices[mask]
            diagnostics["time_column"] = time_column
            diagnostics["validation_periods"] = [str(x) for x in sorted(chosen)]
        else:
            cutoff = max(1, int(len(y) * (1 - fraction)))
            train_i, valid_i = indices[:cutoff], indices[cutoff:]
            rationale += "; chronological row order supplied by caller"
    elif strategy == "group":
        if columns is not None:
            if (not isinstance(columns, list) or not columns
                    or any(name not in train or name == problem.get('target') for name in columns)):
                raise ValueError("group_columns must list training columns other than the target")
            hashes = _feature_groups(train[columns])
        groups, inverse = np.unique(hashes, return_inverse=True)
        if len(groups) < 2:
            if automatic:
                return dict(train_indices=None, valid_indices=None, strategy="group",
                            rationale="No independent holdout: all rows share one feature group",
                            diagnostics=diagnostics)
            raise ValueError("Group validation needs at least two distinct groups")
        order = np.random.RandomState(42).permutation(len(groups))
        counts = np.bincount(inverse)
        grouped_rows = np.argsort(inverse, kind="stable")
        offsets = np.concatenate([[0], np.cumsum(counts)])
        training_counts = y.value_counts().to_dict() if problem.get("classes") is not None else None
        selected, size = [], 0
        for group in order:
            members = grouped_rows[offsets[group]:offsets[group + 1]]
            labels = y.iloc[members].value_counts().to_dict() if training_counts is not None else {}
            if any(training_counts[label] <= count for label, count in labels.items()):
                continue
            selected.append(group)
            size += len(members)
            for label, count in labels.items():
                training_counts[label] -= count
            if size >= math.ceil(len(y) * fraction):
                break
        mask = np.isin(inverse, selected)
        train_i, valid_i = indices[~mask], indices[mask]
        diagnostics["group_columns"] = columns or list(features)
    else:
        stratify = None
        if problem.get("classes") is not None:
            counts = y.value_counts()
            n_valid = math.ceil(len(y) * fraction)
            if counts.min() >= 2 and n_valid >= len(counts) and len(y) - n_valid >= len(counts):
                stratify = y
        train_i, valid_i = train_test_split(indices, test_size=fraction,
                                            random_state=42, stratify=stratify)
        if problem.get("classes") is not None:
            missing = set(y.unique()) - set(y.iloc[train_i].unique())
            move = [next(int(i) for i in valid_i if y.iloc[i] == label) for label in missing]
            if move:
                train_i = np.concatenate([train_i, move])
                valid_i = np.array([i for i in valid_i if i not in move], dtype=int)
    if not len(train_i) or not len(valid_i):
        if automatic:
            return dict(train_indices=None, valid_indices=None, strategy=strategy,
                        rationale="No independent holdout can preserve every training class",
                        diagnostics=diagnostics)
        raise ValueError("Cannot form nonempty independent validation folds")
    if problem.get("classes") is not None and set(y.iloc[train_i]) != set(y):
        if automatic:
            return dict(train_indices=None, valid_indices=None, strategy=strategy,
                        rationale="Future validation introduces a class absent from earlier training; score omitted",
                        diagnostics=diagnostics)
        raise ValueError("Validation would leave training without an observed class")
    diagnostics["training_rows"] = len(train_i)
    diagnostics["validation_rows"] = len(valid_i)
    diagnostics["validation_seen_fraction"] = float(np.isin(
        feature_hashes[valid_i], feature_hashes[train_i]).mean())
    return dict(train_indices=train_i, valid_indices=valid_i, strategy=strategy,
                rationale=rationale, diagnostics=diagnostics)
