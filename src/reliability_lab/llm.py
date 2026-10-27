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

from reliability_lab.data import _plain, canonical_text, fingerprint_rows, preview
from reliability_lab.design import Design
from reliability_lab.exceptions import DesignError, SchemaError
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
        df: pd.DataFrame,
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
        """Validate a table of configurations and map its columns to trace fields."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
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
            {f: df[mapping[f]].to_numpy(dtype=object, copy=True) for f in fields},
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
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else value

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
    if value is None or (isinstance(value, float) and math.isnan(value)):
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
    if value is None or (isinstance(value, float) and math.isnan(value)):
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
