"""Long-form ratings: one row per rating attempt, each with an explicit status."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Iterable
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab.exceptions import SchemaError


class Status(StrEnum):
    """Outcome of one rating attempt."""

    OBSERVED = "observed"
    ABSTAINED = "abstained"
    INAPPLICABLE = "inapplicable"
    PARSE_ERROR = "parse_error"
    PROVIDER_ERROR = "provider_error"


#: Statuses of failed attempts, as plain strings (the stored form).
ERROR_STATUSES: frozenset[str] = frozenset({Status.PARSE_ERROR.value, Status.PROVIDER_ERROR.value})

#: Canonical field names, in the order a RatingFrame stores them.
FIELDS: tuple[str, ...] = (
    "unit_id",
    "rater_id",
    "value",
    "status",
    "group_id",
    "stratum",
    "model_id",
    "model_family",
    "condition_id",
    "run_id",
)


def preview(labels: Iterable[Any], limit: int = 5) -> str:
    """Format the first few labels of a collection for an error message."""
    items = list(labels)
    shown = ", ".join(repr(_plain(x)) for x in items[:limit])
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return f"[{shown}]{more}"


def _plain(x: Any) -> Any:
    """Turn numpy scalars into Python scalars so they print cleanly."""
    return x.item() if isinstance(x, np.generic) else x


def canonical_text(x: Any) -> str:
    """Stable text for fingerprints: missing is empty, integral floats print as ints."""
    if x is None or (not isinstance(x, str) and bool(pd.isna(x))):
        return ""
    x = _plain(x)
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def fingerprint_rows(columns: dict[str, Iterable[Any]]) -> str:
    """sha256 over rows of named columns, independent of row order."""
    names = list(columns)
    texts = [[canonical_text(v) for v in col] for col in columns.values()]
    rows = sorted("\x1f".join(parts) for parts in zip(*texts, strict=True))
    digest = hashlib.sha256("\x1f".join(names).encode("utf-8"))
    for row in rows:
        digest.update(b"\n" + row.encode("utf-8"))
    return digest.hexdigest()


class RatingFrame:
    """Validated long-form ratings.

    Build one with :meth:`from_long`. Each row is one rating attempt by one rater
    on one unit. Its ``status`` says whether a value was observed, the rater
    abstained, the item was inapplicable, or the attempt failed (parse or provider
    error). The frame is copied on construction and never mutated afterwards.
    """

    def __init__(self, data: pd.DataFrame, declared: frozenset[str], implicit_status: bool) -> None:
        self._data = data
        self._declared = declared
        self._implicit_status = implicit_status
        self._fingerprint: str | None = None

    def fingerprint(self) -> str:
        """sha256 of the validated ratings (declared fields only).

        Rows are sorted before hashing, and values are written as canonical text,
        so the fingerprint ignores row order and dtype changes such as ``1`` vs
        ``1.0``.
        """
        if self._fingerprint is None:
            fields = [f for f in FIELDS if f in self._declared]
            self._fingerprint = fingerprint_rows({f: self._data[f] for f in fields})
        return self._fingerprint

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        unit: Hashable,
        rater: Hashable,
        value: Hashable,
        status: Hashable | None = None,
        group: Hashable | None = None,
        stratum: Hashable | None = None,
        model_id: Hashable | None = None,
        model_family: Hashable | None = None,
        condition_id: Hashable | None = None,
        run_id: Hashable | None = None,
    ) -> RatingFrame:
        """Validate a long-form table and map its columns to canonical fields.

        Each keyword names the column of ``df`` that holds that field. ``unit``,
        ``rater`` and ``value`` are required. Without ``status`` every row must
        carry a value and is recorded as observed. Pass a status column whenever
        abstentions or failed attempts exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
        mapping: dict[str, Hashable | None] = {
            "unit_id": unit,
            "rater_id": rater,
            "value": value,
            "status": status,
            "group_id": group,
            "stratum": stratum,
            "model_id": model_id,
            "model_family": model_family,
            "condition_id": condition_id,
            "run_id": run_id,
        }
        absent = [c for c in mapping.values() if c is not None and c not in df.columns]
        if absent:
            raise SchemaError(f"columns not found in the ratings table: {preview(absent)}")

        n = len(df)
        columns: dict[str, Any] = {}
        for field, column in mapping.items():
            if column is None:
                columns[field] = np.full(n, None, dtype=object)
            else:
                columns[field] = df[column].to_numpy(copy=True)
        data = pd.DataFrame(columns, index=df.index)

        for field in ("unit_id", "rater_id"):
            _require_complete(data, field, mapping[field])

        if status is None:
            missing = data.index[data["value"].isna()]
            if len(missing):
                raise SchemaError(
                    f"value is missing in {len(missing)} rows {preview(missing)}. Pass status= "
                    "with a status column so that abstentions and failed attempts are explicit."
                )
            data["status"] = Status.OBSERVED.value
        else:
            data["status"] = _normalise_status(data["status"], status)
            unobserved = data.index[(data["status"] == Status.OBSERVED) & data["value"].isna()]
            if len(unobserved):
                raise SchemaError(
                    f"{len(unobserved)} rows have status 'observed' but no value "
                    f"{preview(unobserved)}"
                )

        declared = frozenset(f for f, c in mapping.items() if c is not None)
        for field in ("group_id", "stratum"):
            if field in declared:
                _require_complete(data, field, mapping[field])
                _require_constant_within(data, field, "unit_id")

        is_llm = data["model_id"].notna()
        if is_llm.any():
            if "condition_id" not in declared:
                raise SchemaError(
                    "LLM rows (non-null model_id) need condition_id: the frozen prompt, rubric, "
                    "parser, candidate order and generation settings of the judge."
                )
            bad = data.index[is_llm & data["condition_id"].isna()]
            if len(bad):
                raise SchemaError(f"{len(bad)} LLM rows have no condition_id {preview(bad)}")
            if "model_family" in declared:
                _require_constant_within(data[is_llm], "model_family", "model_id")

        return cls(data, declared | {"status"}, implicit_status=status is None)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        counts = self._data["status"].value_counts()
        status = ", ".join(f"{k}={int(v)}" for k, v in counts.items())
        return (
            f"RatingFrame(rows={len(self)}, units={self.n_units}, "
            f"raters={self._data['rater_id'].nunique()}, status: {status})"
        )

    @property
    def declared(self) -> frozenset[str]:
        """Canonical fields that were mapped to a column (``status`` always counts)."""
        return self._declared

    @property
    def implicit_status(self) -> bool:
        """True when no status column was given and every row was taken as observed."""
        return self._implicit_status

    @property
    def n_units(self) -> int:
        return int(self._data["unit_id"].nunique())

    @property
    def is_llm(self) -> npt.NDArray[np.bool_]:
        """Mask of rows produced by an LLM judge (rows with a model_id)."""
        return self._data["model_id"].notna().to_numpy()

    def to_frame(self) -> pd.DataFrame:
        """A copy of the validated table, with canonical column names."""
        return self._data.copy()

    def _subset(self, mask: npt.NDArray[np.bool_]) -> RatingFrame:
        """The rows selected by a boolean mask. Every row and per-unit check that
        held for the whole frame still holds for a subset, so nothing is re-run."""
        return RatingFrame(self._data[mask].copy(), self._declared, self._implicit_status)

    def observed_values(self) -> list[Any]:
        """Distinct observed values, sorted when they are mutually comparable.

        Useful for declaring a value domain explicitly:
        ``values=ratings.observed_values()``.
        """
        observed = self._data.loc[self._data["status"] == Status.OBSERVED, "value"]
        unique = [_plain(v) for v in pd.unique(observed)]
        try:
            return sorted(unique)
        except TypeError:
            return unique


def _require_complete(data: pd.DataFrame, field: str, column: Hashable | None) -> None:
    missing = data.index[data[field].isna()]
    if len(missing):
        raise SchemaError(
            f"{field} (column {column!r}) is missing in {len(missing)} rows {preview(missing)}"
        )


def _require_constant_within(data: pd.DataFrame, field: str, key: str) -> None:
    per_key = data.groupby(key, sort=False)[field].nunique()
    bad = per_key.index[per_key > 1]
    if len(bad):
        raise SchemaError(
            f"each {key} must have exactly one {field}; {len(bad)} have several {preview(bad)}"
        )


def _normalise_status(raw: pd.Series[Any], column: Hashable) -> pd.Series[Any]:
    missing = raw.index[raw.isna()]
    if len(missing):
        raise SchemaError(
            f"status (column {column!r}) is missing in {len(missing)} rows {preview(missing)}"
        )
    labels = raw.map(str)
    legal = {s.value for s in Status}
    unknown = sorted(set(labels[~labels.isin(legal)]))
    if unknown:
        raise SchemaError(
            f"unknown status labels {preview(unknown)}; expected one of {sorted(legal)}"
        )
    return labels.astype(object)
