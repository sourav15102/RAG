"""
data_pipeline.py

Synthetic ETL / data-pipeline module: extract, transform, validate, load.
Test fixture for AST-based code chunking + RAG evaluation.
"""

import csv
import io
import json
import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
NULL_TOKENS = {"", "null", "NULL", "None", "N/A", "n/a"}
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%b-%Y")


class PipelineError(Exception):
    """Base exception for pipeline failures."""
    pass


class ValidationError(PipelineError):
    """Raised when a record fails schema validation."""

    def __init__(self, record_index: int, field_name: str, reason: str):
        self.record_index = record_index
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"Record {record_index} field '{field_name}': {reason}")


class SchemaMismatchError(PipelineError):
    """Raised when input data doesn't match the expected schema shape."""
    pass


@dataclass
class FieldSpec:
    """Declarative spec for a single field in a record schema."""
    name: str
    dtype: type
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None


@dataclass
class PipelineStats:
    """Aggregate stats collected while running a pipeline."""
    records_seen: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    errors: list[str] = field(default_factory=list)

    def success_rate(self) -> float:
        """Return the fraction of seen records that were valid."""
        if self.records_seen == 0:
            return 0.0
        return round(self.records_valid / self.records_seen, 4)

    def record_error(self, message: str) -> None:
        """Append an error message and increment the invalid counter."""
        self.errors.append(message)
        self.records_invalid += 1


def is_null_token(value: Any) -> bool:
    """Return True if a raw value represents a null/missing marker."""
    return isinstance(value, str) and value.strip() in NULL_TOKENS


def parse_flexible_date(value: str) -> Optional[datetime]:
    """Try parsing a date string against several known formats."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def coerce_type(value: Any, dtype: type) -> Any:
    """Attempt to coerce a raw value to the target dtype."""
    if value is None:
        return None
    if dtype is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y")
        return bool(value)
    if dtype is datetime:
        return parse_flexible_date(str(value))
    return dtype(value)


def read_csv_records(csv_text: str) -> list[dict]:
    """Parse CSV text into a list of dict records, treating null tokens as None."""
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for row in reader:
        cleaned = {k: (None if is_null_token(v) else v) for k, v in row.items()}
        records.append(cleaned)
    return records


def read_jsonl_records(jsonl_text: str) -> list[dict]:
    """Parse newline-delimited JSON text into a list of dict records."""
    records = []
    for i, line in enumerate(jsonl_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SchemaMismatchError(f"Invalid JSON on line {i}: {exc}") from exc
    return records


def dedupe_records(records: list[dict], key_fields: list[str]) -> list[dict]:
    """Remove duplicate records based on a composite key, keeping the first occurrence."""
    seen = set()
    unique = []
    for record in records:
        key = tuple(record.get(f) for f in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def compute_column_stats(records: list[dict], column: str) -> dict:
    """Compute basic numeric stats (min/max/mean/stdev) for a column."""
    values = [r[column] for r in records if isinstance(r.get(column), (int, float))]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stdev": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def group_by(records: list[dict], key_field: str) -> dict[Any, list[dict]]:
    """Group records into a dict keyed by the value of `key_field`."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get(key_field)].append(record)
    return dict(grouped)


def batched(iterable: Iterable, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list]:
    """Yield successive batches of at most `batch_size` items."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def legacy_flatten_nested_dict(d, parent_key="", sep=".", max_depth=10, _depth=0):
    # NOTE: legacy flattening utility carried over from the old ingestion service.
    items = {}
    if _depth > max_depth:
        items[parent_key] = json.dumps(d)
        return items
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(legacy_flatten_nested_dict(v, new_key, sep, max_depth, _depth + 1))
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.update(legacy_flatten_nested_dict(elem, f"{new_key}[{i}]", sep, max_depth, _depth + 1))
                else:
                    items[f"{new_key}[{i}]"] = elem
        else:
            items[new_key] = v
    return items


class Transform(ABC):
    """Base class for a single transform step in a pipeline."""

    @abstractmethod
    def apply(self, record: dict) -> dict:
        ...


class TrimStrings(Transform):
    """Transform that strips whitespace from all string field values."""

    def apply(self, record: dict) -> dict:
        """Return a copy of the record with all string values stripped."""
        return {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}


class RenameFields(Transform):
    """Transform that renames fields according to a provided mapping."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def apply(self, record: dict) -> dict:
        """Return a copy of the record with keys renamed per the mapping."""
        return {self.mapping.get(k, k): v for k, v in record.items()}


class DropFields(Transform):
    """Transform that removes a set of fields from each record."""

    def __init__(self, fields_to_drop: list[str]):
        self.fields_to_drop = set(fields_to_drop)

    def apply(self, record: dict) -> dict:
        """Return a copy of the record excluding the dropped fields."""
        return {k: v for k, v in record.items() if k not in self.fields_to_drop}


class SchemaValidator:
    """Validates records against a list of FieldSpecs."""

    def __init__(self, fields: list[FieldSpec]):
        self.fields = {f.name: f for f in fields}

    def validate(self, record: dict, record_index: int = 0) -> dict:
        """Validate and coerce a single record, raising ValidationError on failure."""
        result = {}
        for name, spec in self.fields.items():
            raw = record.get(name, spec.default)
            if raw is None:
                if spec.required:
                    raise ValidationError(record_index, name, "missing required field")
                result[name] = spec.default
                continue
            try:
                coerced = coerce_type(raw, spec.dtype)
            except (ValueError, TypeError) as exc:
                raise ValidationError(record_index, name, f"cannot coerce to {spec.dtype.__name__}: {exc}")
            if spec.validator and not spec.validator(coerced):
                raise ValidationError(record_index, name, "failed custom validation")
            result[name] = coerced
        return result

    def validate_all(self, records: list[dict], stats: Optional[PipelineStats] = None) -> list[dict]:
        """Validate a full batch of records, collecting stats and skipping bad ones."""
        stats = stats or PipelineStats()
        valid_records = []
        for i, record in enumerate(records):
            stats.records_seen += 1
            try:
                valid_records.append(self.validate(record, i))
                stats.records_valid += 1
            except ValidationError as exc:
                stats.record_error(str(exc))
        return valid_records


class Pipeline:
    """Orchestrates a sequence of transforms plus schema validation."""

    def __init__(self, transforms: Optional[list[Transform]] = None, validator: Optional[SchemaValidator] = None):
        self.transforms = transforms or []
        self.validator = validator
        self.stats = PipelineStats()

    def add_transform(self, transform: Transform) -> "Pipeline":
        """Append a transform step and return self for chaining."""
        self.transforms.append(transform)
        return self

    def _apply_transforms(self, record: dict) -> dict:
        for transform in self.transforms:
            record = transform.apply(record)
        return record

    def run(self, records: list[dict]) -> list[dict]:
        """Run all transforms and validation over a list of raw records."""
        transformed = [self._apply_transforms(r) for r in records]
        if self.validator is not None:
            return self.validator.validate_all(transformed, self.stats)
        self.stats.records_seen = len(transformed)
        self.stats.records_valid = len(transformed)
        return transformed

    def run_in_batches(self, records: list[dict], batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[dict]]:
        """Run the pipeline over records batch by batch, yielding each output batch."""
        for batch_records in batched(records, batch_size):
            yield self.run(batch_records)

    def summary(self) -> dict:
        """Return a summary dict of the pipeline's accumulated stats."""
        return {
            "records_seen": self.stats.records_seen,
            "records_valid": self.stats.records_valid,
            "records_invalid": self.stats.records_invalid,
            "success_rate": self.stats.success_rate(),
        }