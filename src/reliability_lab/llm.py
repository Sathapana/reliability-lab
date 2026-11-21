"""LLM judges: the trace sidecar, design-specific trace checks and diagnostics.

A ``condition_id`` in the ratings names a frozen judge configuration: prompt,
rubric, parser, candidate order and generation settings. The trace sidecar
records what each configuration was, so a study can check that its coders differ
only in what the design compares. For example, prompt robustness should vary the
prompt and nothing else. Whatever the trace does not record is reported as a
provenance gap.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from reliability_lab.alpha import Distance, compute_alpha
from reliability_lab.data import (
    TELEMETRY_FIELDS,
    ArrowStreamExportable,
    RatingFrame,
    _as_pandas,
    _clean_column,
    _is_missing,
    _plain,
    _validate_nonneg_numeric,
    canonical_text,
    fingerprint_rows,
    preview,
)
from reliability_lab.design import Design
from reliability_lab.exceptions import DesignError, SchemaError, UndefinedAlphaError
from reliability_lab.report import Diagnostic

#: Fields a trace sidecar can record, in order.
TRACE_FIELDS: tuple[str, ...] = (
    "provider",
    "prompt_hash",
    "rubric_version",
    "temperature",
    "seed",
    "deterministic",
    "parser_version",
    "candidate_order",
    "request_time",
    "retry_policy",
)
#: Fields that define how a judge generates and parses a rating.
CONFIG_FIELDS: tuple[str, ...] = (
    "provider",
    "prompt_hash",
    "rubric_version",
    "temperature",
    "seed",
    "deterministic",
    "parser_version",
    "candidate_order",
)
#: Fields a unit covariate sidecar can record, in order.
UNIT_FIELDS: tuple[str, ...] = (
    "length",
    "length_a",
    "length_b",
    "output_family",
    "output_family_a",
    "output_family_b",
)
#: z for 95% Wilson intervals.
Z95 = 1.959963984540054

IntArray = npt.NDArray[np.intp]


class TraceSidecar:
    """One row per judge configuration (``condition_id``) describing how it ran.

    Build one with :meth:`from_frame`. ``candidate_order`` lists the candidate
    labels in presentation order, for example ``"A,B"`` or ``["B", "A"]``. It is
    needed for order-robustness studies of pairwise judgments.
    """

    def __init__(self, data: pd.DataFrame, fields: tuple[str, ...]) -> None:
        self._data = data
        self._fields = fields

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame | ArrowStreamExportable,
        *,
        condition: Hashable,
        provider: Hashable | None = None,
        prompt_hash: Hashable | None = None,
        rubric_version: Hashable | None = None,
        temperature: Hashable | None = None,
        seed: Hashable | None = None,
        deterministic: Hashable | None = None,
        parser_version: Hashable | None = None,
        candidate_order: Hashable | None = None,
        request_time: Hashable | None = None,
        retry_policy: Hashable | None = None,
    ) -> TraceSidecar:
        """Validate a table of configurations and map its columns to trace fields.

        ``df`` is a pandas DataFrame or a table that exports an Arrow stream, as in
        :meth:`RatingFrame.from_long`.
        """
        df = _as_pandas(df)
        mapping = {
            "provider": provider,
            "prompt_hash": prompt_hash,
            "rubric_version": rubric_version,
            "temperature": temperature,
            "seed": seed,
            "deterministic": deterministic,
            "parser_version": parser_version,
            "candidate_order": candidate_order,
            "request_time": request_time,
            "retry_policy": retry_policy,
        }
        absent = [c for c in (condition, *mapping.values()) if c is not None and c not in df]
        if absent:
            raise SchemaError(f"columns not found in the trace table: {preview(absent)}")
        conditions = df[condition]
        if conditions.isna().any():
            raise SchemaError("condition_id is missing in some trace rows")
        repeated = conditions[conditions.duplicated()].unique()
        if len(repeated):
            raise SchemaError(
                f"the trace has several rows for conditions {preview(repeated)}; it needs one "
                "row per condition_id"
            )
        fields = tuple(f for f in TRACE_FIELDS if mapping[f] is not None)
        data = pd.DataFrame(
            {f: _clean_column(df[mapping[f]]) for f in fields},
            index=pd.Index([_plain(c) for c in conditions], dtype=object, name="condition_id"),
        )
        if "temperature" in fields:
            numeric = pd.to_numeric(data["temperature"], errors="coerce")
            bad = data.index[numeric.isna() & data["temperature"].notna()]
            if len(bad):
                raise SchemaError(f"temperature must be numeric; conditions {preview(bad)} are not")
            data["temperature"] = numeric.astype(object).where(numeric.notna(), None)
        if "deterministic" in fields:
            data["deterministic"] = [_as_bool(v) for v in data["deterministic"]]
        if "candidate_order" in fields:
            data["candidate_order"] = pd.Series(
                [_as_order(v) for v in data["candidate_order"]], index=data.index, dtype=object
            )
        return cls(data, fields)

    @property
    def fields(self) -> tuple[str, ...]:
        """The trace fields this sidecar records."""
        return self._fields

    @property
    def conditions(self) -> list[Any]:
        return list(self._data.index)

    def __contains__(self, condition: object) -> bool:
        return condition in self._data.index

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"TraceSidecar(conditions={len(self)}, fields={list(self._fields)})"

    def value(self, condition: Any, field: str) -> Any:
        """A field of one condition, or None when it is not recorded."""
        if field not in self._fields or condition not in self._data.index:
            return None
        value = _plain(self._data.at[condition, field])
        return None if _is_missing(value) else value

    def rows(self, conditions: Iterable[Any] | None = None) -> pd.DataFrame:
        """The trace rows of ``conditions`` (all when None) that the sidecar holds."""
        if conditions is None:
            return self._data.copy()
        wanted = [c for c in dict.fromkeys(conditions) if c in self._data.index]
        return self._data.loc[wanted].copy()

    def to_frame(self) -> pd.DataFrame:
        return self._data.reset_index()

    def fingerprint(self) -> str:
        frame = self.to_frame()
        return fingerprint_rows({c: [_text(v) for v in frame[c]] for c in frame.columns})

    def to_dict(self, conditions: Iterable[Any] | None = None) -> dict[str, Any]:
        rows = self.rows(conditions).reset_index()
        return {
            "fields": list(self._fields),
            "fingerprint": self.fingerprint(),
            "rows": [
                {k: (list(v) if isinstance(v, tuple) else _plain(v)) for k, v in record.items()}
                for record in rows.to_dict(orient="records")
            ],
        }


def _as_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise SchemaError(f"deterministic must be true or false; got {value!r}")


def _as_order(value: Any) -> tuple[str, ...] | None:
    if _is_missing(value):
        return None
    labels = (
        [part.strip() for part in value.split(",")]
        if isinstance(value, str)
        else [str(v) for v in value]
    )
    if len(labels) < 2 or len(set(labels)) != len(labels) or "" in labels:
        raise SchemaError(
            f"candidate_order must list two or more distinct candidates; got {value!r}"
        )
    return tuple(labels)


def _text(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(value)
    return canonical_text(value)


# -- trace checks ---------------------------------------------------------------------


@dataclass(frozen=True)
class TraceCheck:
    """What a trace says about a design: confounds (warned), notes and gaps."""

    confounds: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()


def merge_trace_checks(checks: Iterable[TraceCheck]) -> TraceCheck:
    """Combine the checks of several configurations, dropping repeated messages."""
    collected = list(checks)
    return TraceCheck(
        *(
            tuple(dict.fromkeys(message for c in collected for message in getattr(c, part)))
            for part in ("confounds", "notes", "gaps")
        )
    )


def check_trace(
    design: Design,
    trace: TraceSidecar | None,
    data: pd.DataFrame,
    values: Sequence[Any],
) -> TraceCheck:
    """Check that coders differ only in what ``design`` compares, as far as the trace says."""
    is_llm = data["model_id"].notna()
    if trace is None:
        if design is Design.ORDER_ROBUSTNESS:
            raise DesignError(
                "order_robustness needs a trace sidecar recording candidate_order for every "
                "condition"
            )
        if is_llm.any():
            return TraceCheck(
                gaps=(
                    "No trace sidecar: provider, prompt hash, temperature and parser version "
                    "are not recorded, so changes between configurations cannot be checked.",
                )
            )
        return TraceCheck()
    if not is_llm.any():
        raise DesignError(
            "a trace sidecar describes LLM judge configurations, but these ratings have no LLM rows"
        )

    conditions = list(pd.unique(data.loc[is_llm, "condition_id"]))
    missing = [c for c in conditions if c not in trace]
    gaps = []
    if missing:
        gaps.append(f"{len(missing)} conditions have no trace row {preview(missing)}.")
    unrecorded = [f for f in CONFIG_FIELDS if f not in trace.fields]
    if unrecorded:
        gaps.append(f"The trace does not record {', '.join(unrecorded)}.")
    present = [c for c in conditions if c in trace]

    confounds: list[str] = []
    notes: list[str] = []
    if design is Design.REPEATABILITY and present:
        condition = present[0]
        temperature = trace.value(condition, "temperature")
        if temperature is not None and float(temperature) > 0:
            notes.append(
                f"Temperature is {temperature} for {condition!r}: disagreement across runs "
                "includes sampling variability, by design."
            )
        if (
            trace.value(condition, "seed") is None
            and trace.value(condition, "deterministic") is None
        ):
            notes.append(f"No seed or determinism setting is recorded for {condition!r}.")
    elif design is Design.PROMPT_ROBUSTNESS:
        if "prompt_hash" in trace.fields:
            hashes = _by_condition(trace, present, "prompt_hash")
            if len(set(hashes.values())) < len(hashes):
                confounds.append(
                    f"Conditions share a prompt_hash {hashes}; prompt robustness compares "
                    "different prompts."
                )
        confounds.extend(
            f"Conditions differ in {field} as well as the prompt "
            f"{_by_condition(trace, present, field)}; differences are not attributable to the "
            "prompt alone."
            for field in _differing(trace, present, exclude=("prompt_hash",))
        )
    elif design is Design.INTER_MODEL:
        confounds.extend(
            f"The models ran with different {field} {_by_condition(trace, present, field)}; "
            "differences mix model and configuration."
            for field in _differing(trace, present, exclude=("provider",))
        )
    elif design is Design.ORDER_ROBUSTNESS:
        _check_orders(trace, conditions, values)
        confounds.extend(
            f"Conditions differ in {field} as well as candidate order "
            f"{_by_condition(trace, conditions, field)}; order effects are confounded."
            for field in _differing(trace, conditions, exclude=("candidate_order",))
        )
    return TraceCheck(tuple(confounds), tuple(notes), tuple(gaps))


def _check_orders(trace: TraceSidecar, conditions: Sequence[Any], values: Sequence[Any]) -> None:
    if "candidate_order" not in trace.fields:
        raise DesignError("order_robustness needs candidate_order in the trace sidecar")
    orders = {c: trace.value(c, "candidate_order") for c in conditions}
    unordered = [c for c, o in orders.items() if o is None]
    if unordered:
        raise DesignError(f"conditions {preview(unordered)} have no candidate_order in the trace")
    candidate_sets = {frozenset(o) for o in orders.values()}
    if len(candidate_sets) > 1:
        raise DesignError(
            f"the conditions order different candidates {orders}; an order swap presents the "
            "same candidates in a different order"
        )
    if len(set(orders.values())) < 2:
        raise DesignError(
            f"every condition presents the candidates in the same order {orders}; an order "
            "swap needs at least two orders"
        )
    labels = {str(v) for v in values}
    unknown = sorted(next(iter(candidate_sets)) - labels)
    if unknown:
        raise DesignError(
            f"candidates {unknown} in candidate_order are not declared values {list(values)}; "
            "labels must name the winning candidate"
        )


def _by_condition(trace: TraceSidecar, conditions: Sequence[Any], field: str) -> dict[Any, Any]:
    return {c: _plain_value(trace.value(c, field)) for c in conditions}


def _plain_value(value: Any) -> Any:
    return ",".join(value) if isinstance(value, tuple) else _plain(value)


def _differing(
    trace: TraceSidecar, conditions: Sequence[Any], exclude: tuple[str, ...]
) -> list[str]:
    """Recorded configuration fields whose values differ across ``conditions``."""
    if len(conditions) < 2:
        return []
    return [
        field
        for field in CONFIG_FIELDS
        if field in trace.fields
        and field not in exclude
        and len({_text(trace.value(c, field)) for c in conditions}) > 1
    ]


def configuration_diff(
    trace: TraceSidecar, conditions_a: Iterable[Any], conditions_b: Iterable[Any]
) -> dict[str, dict[str, list[Any]]]:
    """Trace fields whose values differ between two sets of conditions."""
    a, b = list(dict.fromkeys(conditions_a)), list(dict.fromkeys(conditions_b))
    diff = {}
    for field in trace.fields:
        values_a = sorted({_text(trace.value(c, field)) for c in a if c in trace})
        values_b = sorted({_text(trace.value(c, field)) for c in b if c in trace})
        if values_a != values_b:
            diff[field] = {"a": values_a, "b": values_b}
    return diff


# -- diagnostics ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CodedRatings:
    """One value code per unit and coder (-1 when missing), as a units x coders matrix."""

    matrix: IntArray
    unit_labels: npt.NDArray[np.object_]
    coder_names: npt.NDArray[np.object_]
    values: tuple[Any, ...]

    @classmethod
    def build(
        cls,
        unit_idx: IntArray,
        coder_idx: IntArray,
        value_idx: IntArray,
        unit_labels: npt.NDArray[np.object_],
        coder_names: npt.NDArray[np.object_],
        values: tuple[Any, ...],
    ) -> CodedRatings:
        matrix = np.full((len(unit_labels), len(coder_names)), -1, dtype=np.intp)
        matrix[unit_idx, coder_idx] = value_idx
        return cls(matrix, unit_labels, coder_names, values)


def coder_agreement(coded: CodedRatings) -> Diagnostic:
    """Exact agreement between every pair of coders on the units both rated."""
    rows = []
    m = coded.matrix
    for i, j in combinations(range(len(coded.coder_names)), 2):
        shared = (m[:, i] >= 0) & (m[:, j] >= 0)
        n = int(shared.sum())
        agree = float((m[shared, i] == m[shared, j]).mean()) if n else math.nan
        rows.append(
            {
                "coder_a": coded.coder_names[i],
                "coder_b": coded.coder_names[j],
                "shared_units": n,
                "exact_agreement": agree,
            }
        )
    table = pd.DataFrame(rows, columns=["coder_a", "coder_b", "shared_units", "exact_agreement"])
    defined = table["exact_agreement"].dropna()
    return Diagnostic(
        "coder_agreement",
        {
            "pairs": len(table),
            "mean_exact_agreement": float(defined.mean()) if len(defined) else math.nan,
            "min_exact_agreement": float(defined.min()) if len(defined) else math.nan,
        },
        table,
        ("Exact agreement is the share of shared units where two coders give the same value.",),
    )


def run_agreement(coded: CodedRatings, deterministic: bool | None) -> Diagnostic:
    """For repeated runs: how many units get the same value in every run."""
    m = coded.matrix
    runs = (m >= 0).sum(axis=1)
    distinct = np.array([len(set(row[row >= 0])) for row in m])
    eligible = runs >= 2
    unanimous = eligible & (distinct == 1)
    n_eligible = int(eligible.sum())
    disagreeing = int((eligible & (distinct > 1)).sum())
    table = pd.DataFrame(
        {
            "unit_id": coded.unit_labels,
            "runs": runs,
            "distinct_values": distinct,
            "unanimous": unanimous,
        }
    )
    notes = ["A unit is unanimous when every run gave it the same value."]
    if deterministic and disagreeing:
        notes.append(
            f"Runs disagree on {disagreeing} units although the configuration is declared "
            "deterministic."
        )
    return Diagnostic(
        "run_agreement",
        {
            "units_with_two_or_more_runs": n_eligible,
            "unanimous_share": int(unanimous.sum()) / n_eligible if n_eligible else math.nan,
            "units_with_disagreement": disagreeing,
        },
        table[eligible].reset_index(drop=True),
        tuple(notes),
    )


def order_swap(coded: CodedRatings, orders: Mapping[Any, tuple[str, ...]]) -> Diagnostic:
    """Order robustness for pairwise judgments: flips under a swap, and position bias.

    ``orders`` maps each coder to its candidate order. Values are labels naming the
    winning candidate (or a non-candidate label such as a tie). The reversal rate
    counts units where one order picks one candidate and the other order picks
    another. Position bias is the share of decisive judgments won by the candidate
    shown first; 0.5 means no bias.
    """
    m = coded.matrix
    labels = np.array([str(v) for v in coded.values], dtype=object)
    names = list(coded.coder_names)
    candidates = set(next(iter(orders.values())))
    decisive_code = np.array([label in candidates for label in labels])

    compared = differ = reversed_ = 0
    for i, j in combinations(range(len(names)), 2):
        if orders[names[i]] == orders[names[j]]:
            continue
        shared = (m[:, i] >= 0) & (m[:, j] >= 0)
        a, b = m[shared, i], m[shared, j]
        compared += int(shared.sum())
        differ += int((a != b).sum())
        reversed_ += int(((a != b) & decisive_code[a] & decisive_code[b]).sum())

    rows = []
    for k, name in enumerate(names):
        column = m[:, k]
        observed = column[column >= 0]
        decisive = observed[decisive_code[observed]]
        first = orders[name][0]
        wins = int((labels[decisive] == first).sum())
        rows.append(_bias_row(name, ",".join(orders[name]), len(decisive), wins))
    pooled = _bias_row(
        "pooled", "", sum(r["decisive"] for r in rows), sum(r["first_wins"] for r in rows)
    )
    table = pd.DataFrame([*rows, pooled])
    return Diagnostic(
        "order_swap",
        {
            "units_compared": compared,
            "disagreement_rate": differ / compared if compared else math.nan,
            "reversal_rate": reversed_ / compared if compared else math.nan,
            "p_first_shown_wins": pooled["share"],
            "p_first_shown_wins_low": pooled["wilson_low"],
            "p_first_shown_wins_high": pooled["wilson_high"],
            "decisive_judgments": pooled["decisive"],
        },
        table,
        (
            "Units are compared across conditions with different candidate orders. A reversal "
            "is a different winning candidate; a disagreement also counts changes to or from "
            "a non-candidate label such as a tie.",
            "P(first shown wins) is over decisive judgments, with a 95% Wilson interval; 0.5 "
            "means no position bias. Labels are taken to name the winning candidate, not its "
            "position.",
        ),
    )


def _bias_row(coder: Any, order: str, n: int, wins: int) -> dict[str, Any]:
    share, low, high = wilson(wins, n)
    return {
        "coder": coder,
        "candidate_order": order,
        "decisive": n,
        "first_wins": wins,
        "share": share,
        "wilson_low": low,
        "wilson_high": high,
    }


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Proportion with its Wilson score interval (NaN when n = 0)."""
    if n == 0:
        return math.nan, math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


# -- unit covariates & v0.4.0 bias / telemetry evidence -------------------------------


class UnitSidecar:
    """One row per rated unit (``unit_id``) recording item covariates for bias diagnostics.

    Build one with :meth:`from_frame`. Pointwise studies pass ``length`` and/or
    ``output_family``; pairwise comparison studies pass ``length_a`` + ``length_b``
    and/or ``output_family_a`` + ``output_family_b``.
    """

    def __init__(self, data: pd.DataFrame, fields: tuple[str, ...]) -> None:
        self._data = data
        self._fields = fields

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame | ArrowStreamExportable,
        *,
        unit: Hashable,
        length: Hashable | None = None,
        length_a: Hashable | None = None,
        length_b: Hashable | None = None,
        output_family: Hashable | None = None,
        output_family_a: Hashable | None = None,
        output_family_b: Hashable | None = None,
    ) -> UnitSidecar:
        """Validate a unit covariate table and map its columns to canonical unit fields."""
        df = _as_pandas(df)
        mapping: dict[str, Hashable | None] = {
            "length": length,
            "length_a": length_a,
            "length_b": length_b,
            "output_family": output_family,
            "output_family_a": output_family_a,
            "output_family_b": output_family_b,
        }
        fields = tuple(f for f in UNIT_FIELDS if mapping[f] is not None)
        if not fields:
            raise SchemaError("unit sidecar must map at least one covariate field")

        if length is not None and (length_a is not None or length_b is not None):
            raise SchemaError("cannot combine pointwise 'length' with 'length_a' / 'length_b'")
        if (length_a is None) != (length_b is None):
            raise SchemaError("'length_a' and 'length_b' must be declared together")
        if output_family is not None and (
            output_family_a is not None or output_family_b is not None
        ):
            raise SchemaError(
                "cannot combine pointwise 'output_family' with 'output_family_a' / "
                "'output_family_b'"
            )
        if (output_family_a is None) != (output_family_b is None):
            raise SchemaError("'output_family_a' and 'output_family_b' must be declared together")

        absent = [c for c in (unit, *mapping.values()) if c is not None and c not in df.columns]
        if absent:
            raise SchemaError(f"columns not found in the unit table: {preview(absent)}")

        units = df[unit]
        if units.isna().any():
            raise SchemaError("unit_id is missing in some unit sidecar rows")
        plain_units = [_plain(u) for u in units]
        index = pd.Index(plain_units, dtype=object, name="unit_id")
        if index.has_duplicates:
            dup = pd.unique(index[index.duplicated()])
            raise SchemaError(
                f"the unit sidecar has several rows for units {preview(dup)}; it needs one row "
                "per unit_id"
            )

        data = pd.DataFrame(
            {f: _clean_column(df[mapping[f]]) for f in fields},
            index=index,
        )
        for length_field in ("length", "length_a", "length_b"):
            if length_field in fields:
                data[length_field] = _validate_nonneg_numeric(
                    data[length_field],
                    field=length_field,
                    column=mapping[length_field],
                    index=data.index,
                    whole=False,
                )
        return cls(data, fields)

    @property
    def fields(self) -> tuple[str, ...]:
        """The unit covariate fields this sidecar records."""
        return self._fields

    @property
    def units(self) -> list[Any]:
        return list(self._data.index)

    def __contains__(self, unit_id: object) -> bool:
        return unit_id in self._data.index

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"UnitSidecar(units={len(self)}, fields={list(self._fields)})"

    def value(self, unit_id: Any, field: str) -> Any:
        """A covariate field of one unit, or None when missing."""
        if field not in self._fields or unit_id not in self._data.index:
            return None
        value = _plain(self._data.at[unit_id, field])
        return None if _is_missing(value) else value

    def rows(self, units: Iterable[Any] | None = None) -> pd.DataFrame:
        """Covariate rows of ``units`` (all when None) that the sidecar holds."""
        if units is None:
            return self._data.copy()
        wanted = [u for u in dict.fromkeys(_plain(x) for x in units) if u in self._data.index]
        return self._data.loc[wanted].copy()

    def to_frame(self) -> pd.DataFrame:
        return self._data.reset_index()

    def fingerprint(self) -> str:
        frame = self.to_frame()
        return fingerprint_rows({c: [canonical_text(v) for v in frame[c]] for c in frame.columns})

    def to_dict(self, units: Iterable[Any] | None = None) -> dict[str, Any]:
        rows = self.rows(units).reset_index()
        return {
            "fields": list(self._fields),
            "fingerprint": self.fingerprint(),
            "rows": [
                {k: _plain(v) for k, v in record.items()} for record in rows.to_dict("records")
            ],
        }


def check_units(units: UnitSidecar, unit_labels: Iterable[Any]) -> tuple[str, ...]:
    """Identify missing units or incomplete covariate columns for provenance gaps."""
    gaps: list[str] = []
    plain_labels = [_plain(u) for u in unit_labels]
    missing = [u for u in plain_labels if u not in units]
    if missing:
        gaps.append(
            f"{len(missing)} units have no covariate row in the unit sidecar {preview(missing)}."
        )
    present = units.rows(plain_labels)
    if not present.empty:
        incomplete = [f for f in units.fields if present[f].isna().any()]
        if incomplete:
            gaps.append(
                f"Unit sidecar has missing values for {', '.join(incomplete)} on some rated units."
            )
    return tuple(gaps)


def family_alpha(
    coded: CodedRatings,
    families: Mapping[Any, Any],
    distance: Distance,
) -> Diagnostic | None:
    """Within- and between-family pairwise alpha for ``inter_model`` designs."""
    m = coded.matrix
    names = list(coded.coder_names)
    pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(len(names)), 2):
        fam_a = families.get(names[i])
        fam_b = families.get(names[j])
        if _is_missing(fam_a) or _is_missing(fam_b):
            continue
        col_i = m[:, i]
        col_j = m[:, j]
        shared = (col_i >= 0) & (col_j >= 0)
        n_shared = int(shared.sum())
        agree = float((col_i[shared] == col_j[shared]).mean()) if n_shared else math.nan
        valid_i = col_i >= 0
        valid_j = col_j >= 0
        unit_idx = np.concatenate([np.flatnonzero(valid_i), np.flatnonzero(valid_j)]).astype(
            np.intp
        )
        val_idx = np.concatenate([col_i[valid_i], col_j[valid_j]]).astype(np.intp)
        try:
            comp = compute_alpha(unit_idx, val_idx, len(coded.unit_labels), coded.values, distance)
            alpha_ij = float(comp.components.alpha)
        except UndefinedAlphaError:
            alpha_ij = math.nan
        relation = "within" if fam_a == fam_b else "between"
        pairs.append(
            {
                "coder_a": names[i],
                "coder_b": names[j],
                "family_a": _plain(fam_a),
                "family_b": _plain(fam_b),
                "relation": relation,
                "shared_units": n_shared,
                "exact_agreement": agree,
                "alpha": alpha_ij,
            }
        )
    if not pairs:
        return None
    table = pd.DataFrame(pairs)
    within_vals = table.loc[table["relation"] == "within", "alpha"].dropna()
    between_vals = table.loc[table["relation"] == "between", "alpha"].dropna()
    within_mean = float(within_vals.mean()) if len(within_vals) else math.nan
    between_mean = float(between_vals.mean()) if len(between_vals) else math.nan
    gap = (
        within_mean - between_mean
        if math.isfinite(within_mean) and math.isfinite(between_mean)
        else math.nan
    )
    distinct_families = {fam for c in names if not _is_missing(fam := families.get(c))}
    return Diagnostic(
        name="family_alpha",
        values={
            "n_families": len(distinct_families),
            "within_pairs": len(within_vals),
            "between_pairs": len(between_vals),
            "within_family_alpha": within_mean,
            "between_family_alpha": between_mean,
            "family_gap": gap,
        },
        table=table,
        notes=(
            "Pairwise two-coder alphas grouped by whether the two models share model_family.",
            "Higher within-family alpha than between-family alpha indicates correlated "
            "same-family errors.",
        ),
    )


def _spearman_safe(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Spearman rank correlation and p-value, or (nan, nan) for constant/short inputs."""
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan, math.nan
    res = stats.spearmanr(x, y)
    r = float(res.statistic)
    p = float(res.pvalue)
    return (r, p) if math.isfinite(r) and math.isfinite(p) else (math.nan, math.nan)


def _ordered_scores(values: tuple[Any, ...], distance: Distance) -> npt.NDArray[np.float64] | None:
    """Numeric rank/value score per domain index for ordered/numeric measurements."""
    if distance.name == "nominal" or distance.name.startswith("circular"):
        return None
    is_num = all(
        isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool | np.bool_)
        for v in values
    )
    if is_num and distance.name != "ordinal":
        return np.array([float(v) for v in values], dtype=float)
    return np.arange(len(values), dtype=float)


def _pairwise_candidates(
    values: tuple[Any, ...],
    orders: Mapping[Any, tuple[str, ...]] | None,
) -> tuple[str, str] | None:
    """Identify the two candidate labels (label_a, label_b) for a pairwise study."""
    str_values = [str(v) for v in values]
    value_set = set(str_values)
    if "A" in value_set and "B" in value_set:
        return ("A", "B")
    if orders:
        first_order = next(iter(orders.values()))
        if len(first_order) == 2 and set(first_order) <= value_set:
            return (first_order[0], first_order[1])
    if len(values) >= 2:
        return (str_values[0], str_values[-1])
    return None


def length_bias(
    coded: CodedRatings,
    units: UnitSidecar,
    distance: Distance,
    orders: Mapping[Any, tuple[str, ...]] | None = None,
) -> Diagnostic | None:
    """Length-bias diagnostic from unit length covariates (pointwise or pairwise)."""
    m = coded.matrix
    unit_labels = [_plain(u) for u in coded.unit_labels]
    names = list(coded.coder_names)
    caveat = (
        "A correlation with length describes the data and is not proof of bias on its own, "
        "because longer responses can differ in quality; a causal claim needs pairs matched "
        "on length."
    )

    if "length" in units.fields:
        score_map = _ordered_scores(coded.values, distance)
        if score_map is None:
            return None
        lengths = np.array(
            [
                float(v) if (v := units.value(u, "length")) is not None else math.nan
                for u in unit_labels
            ],
            dtype=float,
        )
        rows: list[dict[str, Any]] = []
        for k, name in enumerate(names):
            obs = (m[:, k] >= 0) & np.isfinite(lengths)
            r, p = _spearman_safe(lengths[obs], score_map[m[obs, k]])
            rows.append({"coder": name, "n_units": int(obs.sum()), "spearman_r": r, "p_value": p})
        # Pooled: correlate unit mean score across observed coders with unit length.
        rated_mask = m >= 0
        counts = rated_mask.sum(axis=1)
        valid_u = (counts > 0) & np.isfinite(lengths)
        unit_sums = np.where(rated_mask, score_map[np.maximum(m, 0)], 0.0).sum(axis=1)
        mean_scores = np.where(counts > 0, unit_sums / np.maximum(counts, 1), math.nan)
        pooled_r, pooled_p = _spearman_safe(lengths[valid_u], mean_scores[valid_u])
        coder_rs = [row["spearman_r"] for row in rows if math.isfinite(row["spearman_r"])]
        mean_coder_r = float(np.mean(coder_rs)) if coder_rs else math.nan
        table = pd.DataFrame(rows)
        return Diagnostic(
            name="length_bias",
            values={
                "mode": "pointwise",
                "n_units": int(valid_u.sum()),
                "spearman_r": pooled_r,
                "p_value": pooled_p,
                "mean_coder_spearman_r": mean_coder_r,
            },
            table=table,
            notes=(
                "Spearman rank correlation between response length and judgment score (pooled "
                "uses per-unit mean rating).",
                caveat,
            ),
        )

    if "length_a" in units.fields and "length_b" in units.fields:
        candidates = _pairwise_candidates(coded.values, orders)
        if candidates is None:
            return None
        label_a, label_b = candidates
        str_values = [str(v) for v in coded.values]
        pref = np.zeros(len(coded.values), dtype=float)
        idx_a = [i for i, lbl in enumerate(str_values) if lbl == label_a]
        idx_b = [i for i, lbl in enumerate(str_values) if lbl == label_b]
        for idx in idx_a:
            pref[idx] = 1.0
        for idx in idx_b:
            pref[idx] = -1.0
        decisive_mask_code = pref != 0.0

        len_a = np.array(
            [
                float(v) if (v := units.value(u, "length_a")) is not None else math.nan
                for u in unit_labels
            ],
            dtype=float,
        )
        len_b = np.array(
            [
                float(v) if (v := units.value(u, "length_b")) is not None else math.nan
                for u in unit_labels
            ],
            dtype=float,
        )
        diff_len = len_a - len_b
        has_len = np.isfinite(diff_len)

        rows = []
        total_decisive_diff = 0
        total_longer_wins = 0
        for k, name in enumerate(names):
            obs = (m[:, k] >= 0) & has_len
            r, p = _spearman_safe(diff_len[obs], pref[m[obs, k]])
            dec = obs & decisive_mask_code[np.maximum(m[:, k], 0)] & (diff_len != 0.0)
            n_dec = int(dec.sum())
            longer_is_a = diff_len[dec] > 0.0
            picked_a = pref[m[dec, k]] > 0.0
            wins = int((longer_is_a == picked_a).sum())
            share, low, high = wilson(wins, n_dec)
            total_decisive_diff += n_dec
            total_longer_wins += wins
            rows.append(
                {
                    "coder": name,
                    "n_units": int(obs.sum()),
                    "spearman_r": r,
                    "p_value": p,
                    "decisive_with_length_diff": n_dec,
                    "longer_wins": wins,
                    "longer_wins_share": share,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )

        rated_mask = m >= 0
        counts = rated_mask.sum(axis=1)
        valid_u = (counts > 0) & has_len
        unit_sums = np.where(rated_mask, pref[np.maximum(m, 0)], 0.0).sum(axis=1)
        mean_pref = np.where(counts > 0, unit_sums / np.maximum(counts, 1), math.nan)
        pooled_r, pooled_p = _spearman_safe(diff_len[valid_u], mean_pref[valid_u])
        pooled_share, pooled_low, pooled_high = wilson(total_longer_wins, total_decisive_diff)

        diag_values: dict[str, Any] = {
            "mode": "pairwise",
            "candidate_a": label_a,
            "candidate_b": label_b,
            "n_units": int(valid_u.sum()),
            "spearman_r": pooled_r,
            "p_value": pooled_p,
            "decisive_with_length_diff": total_decisive_diff,
            "longer_wins_share": pooled_share,
            "wilson_low": pooled_low,
            "wilson_high": pooled_high,
        }
        if orders:
            consistent_n = 0
            consistent_longer = 0
            for u_idx in range(len(unit_labels)):
                if not has_len[u_idx] or diff_len[u_idx] == 0.0:
                    continue
                order_picks: dict[tuple[str, ...], int] = {}
                consistent = True
                for k, cname in enumerate(names):
                    code = int(m[u_idx, k])
                    if code < 0 or not decisive_mask_code[code]:
                        continue
                    order_picks.setdefault(orders[cname], code)
                    if order_picks[orders[cname]] != code:
                        consistent = False
                if not consistent or len(order_picks) < 2:
                    continue
                unique_codes = set(order_picks.values())
                if len(unique_codes) == 1:
                    consistent_n += 1
                    picked_code = next(iter(unique_codes))
                    if (diff_len[u_idx] > 0.0) == (pref[picked_code] > 0.0):
                        consistent_longer += 1
            c_share, c_low, c_high = wilson(consistent_longer, consistent_n)
            diag_values.update(
                {
                    "consistent_units_with_length_diff": consistent_n,
                    "consistent_longer_wins": consistent_longer,
                    "consistent_longer_wins_share": c_share,
                    "consistent_wilson_low": c_low,
                    "consistent_wilson_high": c_high,
                }
            )

        return Diagnostic(
            name="length_bias",
            values=diag_values,
            table=pd.DataFrame(rows),
            notes=(
                "Spearman rank correlation between (length_a - length_b) and preference (+1 for "
                f"{label_a!r}, -1 for {label_b!r}), plus the longer-candidate win share (0.5 "
                "means no length preference).",
                caveat,
            ),
        )
    return None


def self_enhancement(
    coded: CodedRatings,
    units: UnitSidecar,
    families: Mapping[Any, Any],
    distance: Distance,
    orders: Mapping[Any, tuple[str, ...]] | None = None,
) -> Diagnostic | None:
    """Self-enhancement diagnostic: judge preference for outputs from its own model_family."""
    m = coded.matrix
    unit_labels = [_plain(u) for u in coded.unit_labels]
    names = list(coded.coder_names)

    if "output_family" in units.fields:
        score_map = _ordered_scores(coded.values, distance)
        if score_map is None:
            return None
        out_fams = [units.value(u, "output_family") for u in unit_labels]
        rows: list[dict[str, Any]] = []
        all_own: list[float] = []
        all_other: list[float] = []
        for k, name in enumerate(names):
            judge_fam = families.get(name)
            if _is_missing(judge_fam):
                continue
            own_scores: list[float] = []
            other_scores: list[float] = []
            for u_idx, out_fam in enumerate(out_fams):
                code = int(m[u_idx, k])
                if code < 0 or _is_missing(out_fam):
                    continue
                score = float(score_map[code])
                if out_fam == judge_fam:
                    own_scores.append(score)
                else:
                    other_scores.append(score)
            if not own_scores and not other_scores:
                continue
            own_mean = float(np.mean(own_scores)) if own_scores else math.nan
            other_mean = float(np.mean(other_scores)) if other_scores else math.nan
            shift = (
                own_mean - other_mean
                if math.isfinite(own_mean) and math.isfinite(other_mean)
                else math.nan
            )
            all_own.extend(own_scores)
            all_other.extend(other_scores)
            rows.append(
                {
                    "coder": name,
                    "judge_family": _plain(judge_fam),
                    "own_ratings": len(own_scores),
                    "other_ratings": len(other_scores),
                    "own_mean": own_mean,
                    "other_mean": other_mean,
                    "mean_shift": shift,
                }
            )
        if not rows:
            return None
        own_p = float(np.mean(all_own)) if all_own else math.nan
        other_p = float(np.mean(all_other)) if all_other else math.nan
        shift_p = own_p - other_p if math.isfinite(own_p) and math.isfinite(other_p) else math.nan
        return Diagnostic(
            name="self_enhancement",
            values={
                "mode": "pointwise",
                "own_ratings": len(all_own),
                "other_ratings": len(all_other),
                "own_mean": own_p,
                "other_mean": other_p,
                "mean_shift": shift_p,
            },
            table=pd.DataFrame(rows),
            notes=(
                "Difference in mean score (own_family - other_family) comparing outputs generated "
                "by the judge's model_family against outputs from other families.",
            ),
        )

    if "output_family_a" in units.fields and "output_family_b" in units.fields:
        candidates = _pairwise_candidates(coded.values, orders)
        if candidates is None:
            return None
        label_a, label_b = candidates
        str_values = [str(v) for v in coded.values]
        codes_a = {i for i, lbl in enumerate(str_values) if lbl == label_a}
        codes_b = {i for i, lbl in enumerate(str_values) if lbl == label_b}

        fam_a = [units.value(u, "output_family_a") for u in unit_labels]
        fam_b = [units.value(u, "output_family_b") for u in unit_labels]
        rows = []
        total_dec = 0
        total_own = 0
        for k, name in enumerate(names):
            judge_fam = families.get(name)
            if _is_missing(judge_fam):
                continue
            dec = 0
            own = 0
            for u_idx in range(len(unit_labels)):
                fa, fb = fam_a[u_idx], fam_b[u_idx]
                if _is_missing(fa) or _is_missing(fb):
                    continue
                if (fa == judge_fam) == (fb == judge_fam):
                    continue  # neither or both belong to the judge's family
                code = int(m[u_idx, k])
                if code in codes_a:
                    dec += 1
                    if fa == judge_fam:
                        own += 1
                elif code in codes_b:
                    dec += 1
                    if fb == judge_fam:
                        own += 1
            share, low, high = wilson(own, dec)
            total_dec += dec
            total_own += own
            rows.append(
                {
                    "coder": name,
                    "judge_family": _plain(judge_fam),
                    "decisive": dec,
                    "own_family_wins": own,
                    "own_family_win_share": share,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
        if not rows:
            return None
        p_share, p_low, p_high = wilson(total_own, total_dec)
        diag_values: dict[str, Any] = {
            "mode": "pairwise",
            "decisive": total_dec,
            "own_family_wins": total_own,
            "own_family_win_share": p_share,
            "wilson_low": p_low,
            "wilson_high": p_high,
        }
        if orders:
            # Order-consistent own-family wins when all orders of the same judge family agree.
            c_dec = 0
            c_own = 0
            for u_idx in range(len(unit_labels)):
                fa, fb = fam_a[u_idx], fam_b[u_idx]
                if _is_missing(fa) or _is_missing(fb):
                    continue
                judges_present = {
                    families.get(cname) for cname in names if not _is_missing(families.get(cname))
                }
                if len(judges_present) != 1:
                    continue
                jfam = next(iter(judges_present))
                if (fa == jfam) == (fb == jfam):
                    continue
                picked: set[int] = set()
                for k in range(len(names)):
                    code = int(m[u_idx, k])
                    if code in codes_a or code in codes_b:
                        picked.add(code)
                if len(picked) == 1:
                    code = next(iter(picked))
                    c_dec += 1
                    if (code in codes_a and fa == jfam) or (code in codes_b and fb == jfam):
                        c_own += 1
            cs, cl, ch = wilson(c_own, c_dec)
            diag_values.update(
                {
                    "consistent_decisive": c_dec,
                    "consistent_own_family_wins": c_own,
                    "consistent_own_family_win_share": cs,
                    "consistent_wilson_low": cl,
                    "consistent_wilson_high": ch,
                }
            )
        return Diagnostic(
            name="self_enhancement",
            values=diag_values,
            table=pd.DataFrame(rows),
            notes=(
                "Share of decisive head-to-head judgments where one candidate comes from the "
                "judge's model_family and wins (0.5 means no self-family preference).",
            ),
        )
    return None


def summarize_telemetry(
    ratings: RatingFrame,
    kept: npt.NDArray[np.bool_] | None,
    attempt_coders: npt.NDArray[np.object_],
) -> pd.DataFrame | None:
    """Summarize declared per-attempt call telemetry fields per coder."""
    fields = [f for f in TELEMETRY_FIELDS if f in ratings.declared]
    if not fields:
        return None
    data = ratings._data
    counted_mask = np.ones(len(data), dtype=bool) if kept is None else kept
    coders = pd.unique(attempt_coders)

    rows: list[dict[str, Any]] = []
    for coder in coders:
        att_m = attempt_coders == coder
        cnt_m = att_m & counted_mask
        n_att = int(att_m.sum())
        n_cnt = int(cnt_m.sum())
        row: dict[str, Any] = {
            "coder": _plain(coder),
            "attempts": n_att,
            "counted_ratings": n_cnt,
        }
        for f in fields:
            att_vals = pd.to_numeric(data.loc[att_m, f], errors="coerce")
            cnt_vals = pd.to_numeric(data.loc[cnt_m, f], errors="coerce")
            att_sum = float(att_vals.sum(min_count=1)) if n_att else math.nan
            cnt_sum = float(cnt_vals.sum(min_count=1)) if n_cnt else math.nan
            row[f"total_{f}"] = att_sum
            row[f"counted_{f}"] = cnt_sum
            row[f"mean_{f}_per_rating"] = (
                att_sum / n_cnt if n_cnt > 0 and math.isfinite(att_sum) else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
