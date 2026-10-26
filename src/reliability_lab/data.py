"""Long-form ratings: one row per rating attempt, each with an explicit status."""

from __future__ import annotations

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
