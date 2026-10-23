"""Streaming submission checks and durable, validated candidate snapshots."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
from typing import Optional

MAX_CSV_BYTES = 256 * 1024 * 1024


def _bounded_copy(source, target, max_bytes):
    total = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        total += len(chunk)
        if total > max_bytes:
            raise ArtifactError(f"CSV exceeds the {max_bytes} byte size limit")
        target.write(chunk)


class ArtifactError(ValueError):
    """A candidate or its declared contract is invalid."""


@dataclass(frozen=True)
class SubmissionSchema:
    columns: Optional[tuple[str, ...]] = None
    expected_rows: Optional[int] = None
    id_columns: tuple[str, ...] = ()
    sample_path: Optional[str] = None
    numeric_columns: tuple[str, ...] = ()
    allowed_values: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("columns", "id_columns", "numeric_columns"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (list, tuple)):
                    raise ArtifactError(f"{name} must be a list of column names")
                value = tuple(value)
                if any(not isinstance(x, str) or not x.strip() for x in value):
                    raise ArtifactError(f"{name} contains an empty or invalid name")
                if len(set(value)) != len(value):
                    raise ArtifactError(f"{name} contains duplicates")
                object.__setattr__(self, name, value)
        if self.columns is not None and not self.columns:
            raise ArtifactError("columns must not be empty")
        if self.sample_path is not None and (not isinstance(self.sample_path, str) or not self.sample_path.strip()):
            raise ArtifactError("sample_path must be a nonempty path string")
        if self.id_columns is None or self.numeric_columns is None:
            raise ArtifactError("id_columns and numeric_columns must be lists")
        if not isinstance(self.allowed_values, dict):
            raise ArtifactError("allowed_values must map column names to lists of strings")
        normalized_values = {}
        for column, values in self.allowed_values.items():
            if (not isinstance(column, str) or not column.strip()
                    or not isinstance(values, (list, tuple)) or not values
                    or any(not isinstance(value, str) or not value.strip() for value in values)):
                raise ArtifactError("allowed_values must map column names to nonempty lists of strings")
            normalized_values[column] = tuple(values)
        object.__setattr__(self, "allowed_values", normalized_values)
        if self.expected_rows is not None and (
            isinstance(self.expected_rows, bool)
            or not isinstance(self.expected_rows, int) or self.expected_rows < 0
        ):
            raise ArtifactError("expected_rows must be a nonnegative integer")


@dataclass(frozen=True)
class ValidationResult:
    rows: int
    columns: tuple[str, ...]
    sha256: str


def _confined(path, task_dir):
    root = Path(task_dir).resolve(strict=True)
    candidate = Path(path)
    candidate = (root / candidate if not candidate.is_absolute() else candidate).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ArtifactError("Submission and sample files must be inside the task directory")
    return candidate


def _header(reader):
    result = next(reader, None)
    if not result or any(not x.strip() for x in result) or len(set(result)) != len(result):
        raise ArtifactError("CSV requires nonempty, unique column names")
    return tuple(result)


def _validate_csv(path, task_dir, schema, sample_override=None, max_bytes=MAX_CSV_BYTES):
    sample_file = None
    uniqueness_db = None
    try:
        if Path(path).stat().st_size > max_bytes:
            raise ArtifactError(f"CSV exceeds the {max_bytes} byte size limit")
        if schema.sample_path:
            sample = sample_override or _confined(schema.sample_path, task_dir)
            if Path(sample).stat().st_size > max_bytes:
                raise ArtifactError(f"Sample CSV exceeds the {max_bytes} byte size limit")
            sample_file = Path(sample).open("r", encoding="utf-8-sig", newline="")
            sample_reader = csv.reader(sample_file, strict=True)
            sample_header = _header(sample_reader)
        else:
            sample_reader = None
            sample_header = None
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            columns = _header(reader)
            if schema.columns is not None and columns != schema.columns:
                raise ArtifactError("CSV columns do not match the declared columns and order")
            if sample_header is not None and columns != sample_header:
                raise ArtifactError("CSV columns do not match sample submission")
            if not set(schema.id_columns + schema.numeric_columns + tuple(schema.allowed_values)).issubset(columns):
                raise ArtifactError("Declared ID or numeric column is absent")
            ids = [columns.index(x) for x in schema.id_columns]
            numeric = [columns.index(x) for x in schema.numeric_columns]
            allowed = {columns.index(key): set(values) for key, values in schema.allowed_values.items()}
            if ids:
                # A temporary SQLite index avoids growing RAM with the number of rows.
                uniqueness_db = sqlite3.connect("")
                uniqueness_db.execute("PRAGMA cache_size = -2048")
                uniqueness_db.execute("CREATE TABLE identifiers (value TEXT PRIMARY KEY) WITHOUT ROWID")
            rows = 0
            for rows, row in enumerate(reader, 1):
                if len(row) != len(columns):
                    raise ArtifactError(f"Row {rows} has an incorrect number of fields")
                if any(not value.strip() for value in row):
                    raise ArtifactError(f"Row {rows} contains a blank value")
                if uniqueness_db is not None:
                    identifier = json.dumps([row[index] for index in ids], ensure_ascii=False)
                    try:
                        uniqueness_db.execute("INSERT INTO identifiers VALUES (?)", (identifier,))
                    except sqlite3.IntegrityError as exc:
                        raise ArtifactError(f"Row {rows} has a duplicate ID") from exc
                if any(row[index] not in values for index, values in allowed.items()):
                    raise ArtifactError(f"Row {rows} contains a value outside allowed_values")
                for index in numeric:
                    try:
                        finite = math.isfinite(float(row[index]))
                    except ValueError:
                        finite = False
                    if not finite:
                        raise ArtifactError(f"Row {rows} contains a nonfinite or invalid number")
                # Even without a numeric schema, NaN/Infinity must not pass as predictions.
                for index, value in enumerate(row):
                    if index not in ids and value.strip().lower() in {
                        "nan", "+nan", "-nan", "inf", "+inf", "-inf",
                        "infinity", "+infinity", "-infinity",
                    }:
                        raise ArtifactError(f"Row {rows} contains a nonfinite prediction")
                if sample_reader is not None:
                    sample_row = next(sample_reader, None)
                    if sample_row is None or len(sample_row) != len(columns):
                        raise ArtifactError("CSV row count or sample row width differs")
                    if any(row[index] != sample_row[index] for index in ids):
                        raise ArtifactError(f"Row {rows} ID or ID order differs from sample")
            if sample_reader is not None and next(sample_reader, None) is not None:
                raise ArtifactError("CSV has fewer rows than sample submission")
            if schema.expected_rows is not None and rows != schema.expected_rows:
                raise ArtifactError("CSV row count does not match expected_rows")
            if rows == 0 and schema.expected_rows != 0:
                raise ArtifactError("Submission contains no predictions")
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return ValidationResult(rows, columns, digest.hexdigest())
    except (OSError, UnicodeError, csv.Error, sqlite3.Error) as exc:
        raise ArtifactError(f"Cannot read valid CSV: {exc}") from exc
    finally:
        if sample_file is not None:
            sample_file.close()
        if uniqueness_db is not None:
            uniqueness_db.close()


def validate_submission(path, task_dir, schema=None, max_bytes=MAX_CSV_BYTES):
    """Validate a UTF-8 CSV inside task_dir without loading its rows into memory."""
    try:
        _check_max_bytes(max_bytes)
        return _validate_csv(_confined(path, task_dir), task_dir, schema or SubmissionSchema(), max_bytes=max_bytes)
    except OSError as exc:
        raise ArtifactError(str(exc)) from exc


def _check_max_bytes(max_bytes):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ArtifactError("max_bytes must be a positive integer")


class ArtifactStore:
    """Keep best snapshots in a supplied private state directory.

    The caller should use a fresh state directory for each run.
    This is crash-safe artifact preservation, not an OS security sandbox.
    """

    def __init__(self, task_dir, state_dir, schema=None, max_bytes=MAX_CSV_BYTES):
        _check_max_bytes(max_bytes)
        self.max_bytes = max_bytes
        self.task_dir = Path(task_dir).resolve(strict=True)
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._sample_snapshot = None
        self.schema = schema or SubmissionSchema()
        self.best = None
        self._snapshot = None

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, value):
        if not isinstance(value, SubmissionSchema):
            raise ArtifactError("schema must be a SubmissionSchema")
        snapshot = None
        if value.sample_path:
            descriptor, temporary = tempfile.mkstemp(prefix="sample-", suffix=".csv", dir=self.state_dir)
            snapshot = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as target:
                    with _confined(value.sample_path, self.task_dir).open("rb") as source:
                        _bounded_copy(source, target, self.max_bytes)
            except Exception:
                snapshot.unlink(missing_ok=True)
                raise
        old_sample = self._sample_snapshot
        self._schema = value
        self._sample_snapshot = snapshot
        if old_sample is not None:
            old_sample.unlink(missing_ok=True)

    def consider(self, candidate_path, score=None, metric=None, direction="maximize", validation_id=None, final=False):
        if not isinstance(final, bool):
            raise ArtifactError("final must be a boolean")
        if direction not in ("maximize", "minimize"):
            raise ArtifactError("direction must be maximize or minimize")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ArtifactError("score must be a finite number")
            if not isinstance(metric, str) or not metric.strip() or not isinstance(validation_id, str) or not validation_id.strip():
                raise ArtifactError("Scored candidates require metric and validation_id")
        descriptor, temporary = tempfile.mkstemp(prefix="candidate-", suffix=".csv", dir=self.state_dir)
        snapshot = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as target:
                with _confined(candidate_path, self.task_dir).open("rb") as source:
                    _bounded_copy(source, target, self.max_bytes)
                target.flush()
                os.fsync(target.fileno())
            result = _validate_csv(snapshot, self.task_dir, self.schema, self._sample_snapshot, self.max_bytes)
            accepted, reason = self._prefer(score, metric, direction, validation_id, final)
            if not accepted:
                return {"accepted": False, "reason": reason, "rows": result.rows}
            metadata = {"score": score, "metric": metric, "direction": direction,
                        "validation_id": validation_id, "rows": result.rows,
                        "columns": list(result.columns), "sha256": result.sha256,
                        "snapshot": snapshot.name, "final": final}
            metadata_path = self.state_dir / "best.json"
            temporary_metadata = self.state_dir / "best.json.tmp"
            with temporary_metadata.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_metadata, metadata_path)
            previous = self._snapshot
            self._snapshot = snapshot
            self.best = metadata
            if previous is not None:
                previous.unlink(missing_ok=True)
            return {"accepted": True, "reason": reason, **metadata}
        except OSError as exc:
            raise ArtifactError(str(exc)) from exc
        finally:
            if snapshot != self._snapshot:
                snapshot.unlink(missing_ok=True)

    def _prefer(self, score, metric, direction, validation_id, final):
        if self.best is None:
            return True, "first_valid"
        if score is None:
            if self.best["score"] is None and final and not self.best["final"]:
                return True, "final_refit"
            return False, "existing_valid_candidate"
        if self.best["score"] is None:
            return True, "first_scored"
        if (metric, direction, validation_id) != (
            self.best["metric"], self.best["direction"], self.best["validation_id"]
        ):
            return False, "incomparable_validation"
        if score == self.best["score"] and final and not self.best["final"]:
            return True, "final_refit"
        improved = score > self.best["score"] if direction == "maximize" else score < self.best["score"]
        return improved, "improved_score" if improved else "score_not_improved"

    def publish(self, name="submission.csv"):
        if self._snapshot is None:
            raise ArtifactError("No validated submission is available")
        result = _validate_csv(self._snapshot, self.task_dir, self.schema, self._sample_snapshot, self.max_bytes)
        if result.sha256 != self.best["sha256"]:
            raise ArtifactError("Best snapshot content changed")
        destination = (self.task_dir / name).resolve()
        if not destination.is_relative_to(self.task_dir) or destination.suffix.lower() != ".csv":
            raise ArtifactError("Output must be a .csv file inside the task directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ArtifactError("Refusing to publish over a symlink")
        descriptor, temporary = tempfile.mkstemp(prefix=".submission-", suffix=".tmp", dir=self.task_dir)
        try:
            with os.fdopen(descriptor, "wb") as target, self._snapshot.open("rb") as source:
                _bounded_copy(source, target, self.max_bytes)
                target.flush()
                os.fsync(target.fileno())
            result = _validate_csv(temporary, self.task_dir, self.schema, self._sample_snapshot, self.max_bytes)
            if result.sha256 != self.best["sha256"]:
                raise ArtifactError("Published snapshot content changed")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination
