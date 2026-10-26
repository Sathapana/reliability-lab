"""Typed results: alpha estimates, intervals, comparisons and their serialisation.

File output (JSON, Parquet) and the provenance manifest arrive in phase 3. For now
results convert to JSON-safe dicts and pandas tables.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab._version import __version__
from reliability_lab.exceptions import ReliabilityLabError, ReliabilityWarning

if TYPE_CHECKING:
    from reliability_lab.inference import StudySample

Scheme = Literal["units", "kalpha"]
Level = Literal["unit", "group"]
Method = Literal["percentile", "bca"]


@dataclass(frozen=True)
class RecordedWarning:
    """A warning raised while building a result, kept with the result."""

    category: str
    message: str


def emit(category: type[ReliabilityWarning], message: str, stacklevel: int) -> RecordedWarning:
    """Raise a warning now and return a record of it for the result.

    ``stacklevel`` counts from the caller of ``emit``, as in ``warnings.warn``.
    """
    warnings.warn(message, category, stacklevel=stacklevel + 1)
    return RecordedWarning(category.__name__, message)


@dataclass(frozen=True, eq=False)
class AlphaEstimate:
    """Alpha under one missing-data policy, with its white-box decomposition.

    ``per_unit`` has one row per unit: ``pairable_values`` (m_u),
    ``disagreement`` (mean squared distance over the unit's ordered pairs of
    values, NaN when the unit is unpairable) and ``d_o_contribution`` (the unit's
    share of D_o; the column sums to ``d_o``).
    """

    name: str
    policy: str
    alpha: float
    d_o: float
    d_e: float
    observed_agreement: float
    n_pairable_values: int
    n_pairable_units: int
    n_unpairable_units: int
    coincidence: pd.DataFrame
    expected: pd.DataFrame
    category_distribution: pd.Series[float]
    per_unit: pd.DataFrame
    undefined_reason: str | None = None
    n_imputed: int = 0
    n_unanchored: int = 0

    @property
    def defined(self) -> bool:
        return self.undefined_reason is None

    def summary_row(self) -> dict[str, Any]:
        return {
            "estimate": self.name,
            "policy": self.policy,
            "alpha": self.alpha,
            "d_o": self.d_o,
            "d_e": self.d_e,
            "observed_agreement": self.observed_agreement,
            "n_pairable_values": self.n_pairable_values,
            "n_pairable_units": self.n_pairable_units,
            "n_unpairable_units": self.n_unpairable_units,
            "n_imputed": self.n_imputed,
            "n_unanchored": self.n_unanchored,
            "undefined_reason": self.undefined_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.summary_row()
        out["coincidence"] = _matrix_dict(self.coincidence)
        out["expected"] = _matrix_dict(self.expected)
        out["category_distribution"] = [
            {"value": k, "share": v} for k, v in self.category_distribution.items()
        ]
        out["per_unit"] = self.per_unit.to_dict(orient="records")
        return _jsonable(out)  # type: ignore[no-any-return]


@dataclass(frozen=True, eq=False)
class Coverage:
    """How completely the coders rated the units, and how attempts ended.

    ``per_coder`` has one row per coder: counts per status, ``coverage`` (share of
    units with an observed value), ``abstention_rate`` and ``error_rate`` (parse and
    provider errors, as a share of the coder's attempts).
    """

    n_units: int
    n_ratings: int
    n_coders: int
    n_observed: int
    coding_density: float
    status_counts: Mapping[str, int]
    per_coder: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "n_units": self.n_units,
                "n_ratings": self.n_ratings,
                "n_coders": self.n_coders,
                "n_observed": self.n_observed,
                "coding_density": self.coding_density,
                "status_counts": dict(self.status_counts),
                "per_coder": self.per_coder.reset_index().to_dict(orient="records"),
            }
        )


@dataclass(frozen=True, eq=False)
class AlphaResult:
    """Alpha for one declared study, under every declared missing-data policy.

    ``estimates`` maps an estimate name to an :class:`AlphaEstimate`. The names are
    ``as_missing`` (always present, the classic treatment), ``as_category``, and
    ``sensitivity_pessimistic`` / ``sensitivity_optimistic``.
    """

    design: str
    measurement: str
    values: tuple[Any, ...]
    coders: tuple[str, ...]
    missing_policies: tuple[str, ...]
    abstain_as: Any
    inapplicable_as: Any
    estimates: Mapping[str, AlphaEstimate]
    coverage: Coverage
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]
    _sample: StudySample | None = field(default=None, repr=False)

    @property
    def alpha(self) -> float:
        """Alpha with failed attempts treated as missing (the reference policy)."""
        return self.estimates["as_missing"].alpha

    def __getitem__(self, name: str) -> AlphaEstimate:
        return self.estimates[name]

    def bootstrap(
        self,
        *,
        scheme: Scheme,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        level: Level | None = None,
        strata: bool = False,
        method: Method = "percentile",
        alpha_min: float | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> BootstrapResult:
        """Confidence intervals for every estimate, by a named resampling scheme.

        ``scheme`` is required, so no interval is reported without its method:

        - ``"units"`` resamples whole units, or whole groups (``level="group"``, the
          default when groups are declared), optionally within declared strata.
          D_e is recomputed in every replicate.
        - ``"kalpha"`` follows Krippendorff's KALPHA algorithm: it resamples pairs
          of values and holds D_e at its observed value.

        ``method="bca"`` (units scheme only) is experimental until coverage
        simulations validate it for alpha. ``alpha_min`` adds the share of
        replicates below that minimum. All estimates use the same resamples, so
        their intervals are paired.
        """
        from reliability_lab.inference import bootstrap

        if self._sample is None:
            raise ReliabilityLabError(
                "this result has no data attached; bootstrap a result returned by "
                "ReliabilityStudy.alpha()"
            )
        return bootstrap(
            self,
            self._sample,
            scheme=scheme,
            n_resamples=n_resamples,
            confidence=confidence,
            level=level,
            strata=strata,
            method=method,
            alpha_min=alpha_min,
            random_state=random_state,
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate."""
        return pd.DataFrame([e.summary_row() for e in self.estimates.values()])

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "kind": "alpha_result",
                "reliability_lab_version": __version__,
                "design": self.design,
                "measurement": self.measurement,
                "values": list(self.values),
                "coders": list(self.coders),
                "missing_policies": list(self.missing_policies),
                "abstain_as": self.abstain_as,
                "inapplicable_as": self.inapplicable_as,
                "estimates": {k: e.to_dict() for k, e in self.estimates.items()},
                "coverage": self.coverage.to_dict(),
                "warnings": _warning_dicts(self.warnings),
                "notes": list(self.notes),
            }
        )

    def summary(self) -> str:
        cov = self.coverage
        status = ", ".join(f"{k} {v}" for k, v in cov.status_counts.items() if v)
        lines = [
            f"Krippendorff's alpha | design: {self.design} | measurement: {self.measurement}",
            f"values: {', '.join(map(str, self.values))} | coders by: {', '.join(self.coders)}",
            f"units {cov.n_units} | ratings {cov.n_ratings} | coders {cov.n_coders} | "
            f"coding density {cov.coding_density:.3f}",
            f"status: {status}",
            "",
            f"{'estimate':<26}{'alpha':>8}{'D_o':>9}{'D_e':>9}{'agreement':>11}{'pairable':>10}",
        ]
        for e in self.estimates.values():
            lines.append(
                f"{e.name:<26}{_fmt(e.alpha):>8}{_fmt(e.d_o):>9}{_fmt(e.d_e):>9}"
                f"{_fmt(e.observed_agreement):>11}{e.n_pairable_values:>10}"
            )
        primary = self.estimates["as_missing"]
        if primary.n_unpairable_units:
            lines.append(
                f"\n{primary.n_unpairable_units} of {cov.n_units} units have fewer than two "
                "pairable values (as_missing) and contribute nothing."
            )
        for e in self.estimates.values():
            if e.undefined_reason:
                lines.append(f"{e.name} is undefined: {e.undefined_reason}.")
            if e.n_unanchored:
                lines.append(
                    f"{e.name}: {e.n_unanchored} failed attempts sit in units with no other "
                    "value and stay missing."
                )
        _append_warnings_and_notes(lines, self.warnings, self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class Interval:
    """A bootstrap interval for one estimate, or for a difference of estimates.

    ``n_failed`` counts replicates where alpha was undefined; the interval uses the
    others. ``p_below_min`` is the share of defined replicates below ``alpha_min``,
    when one was given. ``distribution`` holds every replicate (NaN for failures).
    """

    estimate: str
    point: float
    low: float
    high: float
    n_resamples: int
    n_failed: int
    p_below_min: float | None = None
    undefined_reason: str | None = None
    distribution: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def defined(self) -> bool:
        return self.undefined_reason is None

    def summary_row(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "n_resamples": self.n_resamples,
            "n_failed": self.n_failed,
            "p_below_min": self.p_below_min,
            "undefined_reason": self.undefined_reason,
        }

    def to_dict(self, include_distribution: bool = False) -> dict[str, Any]:
        out = self.summary_row()
        if include_distribution:
            out["distribution"] = self.distribution.tolist()
        return _jsonable(out)  # type: ignore[no-any-return]


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Bootstrap intervals for every estimate of one study.

    ``level`` is None for the KALPHA scheme, which resamples pairs of values rather
    than units. ``seed`` reproduces the resamples when passed back as
    ``random_state``.
    """

    scheme: str
    level: str | None
    strata: bool
    method: str
    confidence: float
    n_resamples: int
    seed: int
    alpha_min: float | None
    experimental: bool
    n_clusters: int | None
    design: str
    measurement: str
    intervals: Mapping[str, Interval]
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]

    def __getitem__(self, name: str) -> Interval:
        return self.intervals[name]

    def describe(self) -> str:
        """One line naming the scheme, method, level and seed."""
        if self.scheme == "kalpha":
            how = "scheme kalpha (pairs of values, D_e fixed)"
        else:
            within = " within strata" if self.strata else ""
            how = f"scheme units (resampling {self.n_clusters} {self.level}s{within})"
        flag = " [experimental]" if self.experimental else ""
        return (
            f"{how} | method {self.method}{flag} | {self.confidence:.0%} | "
            f"{self.n_resamples} resamples | seed {self.seed}"
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate."""
        return pd.DataFrame([i.summary_row() for i in self.intervals.values()])

    def metadata(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "level": self.level,
            "strata": self.strata,
            "method": self.method,
            "experimental": self.experimental,
            "confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "alpha_min": self.alpha_min,
            "n_clusters": self.n_clusters,
        }

    def to_dict(self, include_distribution: bool = False) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "kind": "bootstrap_result",
                "reliability_lab_version": __version__,
                "design": self.design,
                "measurement": self.measurement,
                **self.metadata(),
                "intervals": {
                    k: i.to_dict(include_distribution) for k, i in self.intervals.items()
                },
                "warnings": _warning_dicts(self.warnings),
                "notes": list(self.notes),
            }
        )

    def summary(self) -> str:
        with_min = self.alpha_min is not None
        lines = [
            f"Bootstrap intervals | design: {self.design} | measurement: {self.measurement}",
            self.describe(),
            "",
            f"{'estimate':<26}{'alpha':>8}{'low':>9}{'high':>9}{'failed':>8}"
            + (f"{'P(<min)':>9}" if with_min else ""),
        ]
        for i in self.intervals.values():
            p = i.p_below_min if i.p_below_min is not None else math.nan
            lines.append(
                f"{i.estimate:<26}{_fmt(i.point):>8}{_fmt(i.low):>9}{_fmt(i.high):>9}"
                f"{i.n_failed:>8}" + (f"{_fmt(p):>9}" if with_min else "")
            )
        lines.extend(_interval_footnotes(self.intervals.values()))
        _append_warnings_and_notes(lines, self.warnings, self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """A paired comparison of two configurations on the same units.

    ``differences`` holds intervals on alpha_b - alpha_a, computed from the same
    resampled units or groups in every replicate. ``arm_a`` and ``arm_b`` hold each
    configuration's own intervals from those same resamples, and ``result_a`` and
    ``result_b`` the arms' full alpha results.
    """

    claim: str
    a: Mapping[str, Any]
    b: Mapping[str, Any]
    design: str
    measurement: str
    result_a: AlphaResult
    result_b: AlphaResult
    arm_a: BootstrapResult
    arm_b: BootstrapResult
    differences: Mapping[str, Interval]
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]

    def __getitem__(self, name: str) -> Interval:
        return self.differences[name]

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate: each arm's alpha and the difference interval."""
        rows = []
        for name, diff in self.differences.items():
            row: dict[str, Any] = {
                "estimate": name,
                "alpha_a": self.arm_a[name].point,
                "alpha_b": self.arm_b[name].point,
            }
            row.update({f"diff_{k}": v for k, v in diff.summary_row().items() if k != "estimate"})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self, include_distribution: bool = False) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "kind": "comparison_result",
                "reliability_lab_version": __version__,
                "claim": self.claim,
                "a": dict(self.a),
                "b": dict(self.b),
                "design": self.design,
                "measurement": self.measurement,
                **self.arm_a.metadata(),
                "differences": {
                    k: i.to_dict(include_distribution) for k, i in self.differences.items()
                },
                "arm_a": self.arm_a.to_dict(include_distribution),
                "arm_b": self.arm_b.to_dict(include_distribution),
                "warnings": _warning_dicts(self.warnings),
                "notes": list(self.notes),
            }
        )

    def summary(self) -> str:
        lines = [
            f"Paired comparison (claim: {self.claim}) | design: {self.design} | "
            f"measurement: {self.measurement}",
            f"a: {dict(self.a)} | b: {dict(self.b)} | difference = b - a",
            self.arm_a.describe(),
            "",
            f"{'estimate':<26}{'alpha_a':>9}{'alpha_b':>9}{'diff':>9}{'low':>9}{'high':>9}"
            f"{'failed':>8}",
        ]
        for name, d in self.differences.items():
            lines.append(
                f"{name:<26}{_fmt(self.arm_a[name].point):>9}{_fmt(self.arm_b[name].point):>9}"
                f"{_fmt(d.point):>9}{_fmt(d.low):>9}{_fmt(d.high):>9}{d.n_failed:>8}"
            )
        lines.extend(_interval_footnotes(self.differences.values()))
        _append_warnings_and_notes(lines, self.warnings, self.notes)
        return "\n".join(lines)


def _interval_footnotes(intervals: Iterable[Interval]) -> list[str]:
    return [
        f"{i.estimate}: no interval, {i.undefined_reason}." for i in intervals if i.undefined_reason
    ]


def _append_warnings_and_notes(
    lines: list[str], warns: tuple[RecordedWarning, ...], notes: tuple[str, ...]
) -> None:
    if warns:
        lines.append("\nWarnings")
        lines.extend(f"  - {w.category}: {w.message}" for w in warns)
    if notes:
        lines.append("\nNotes")
        lines.extend(f"  - {n}" for n in notes)


def _warning_dicts(warns: tuple[RecordedWarning, ...]) -> list[dict[str, str]]:
    return [{"category": w.category, "message": w.message} for w in warns]


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"


def _matrix_dict(frame: pd.DataFrame) -> dict[str, Any]:
    return {"values": list(frame.index), "matrix": frame.to_numpy().tolist()}


def _jsonable(x: Any) -> Any:
    """Recursively convert numpy and pandas scalars to JSON-safe Python values."""
    if isinstance(x, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is pd.NA or x is pd.NaT:
        return None
    return x
