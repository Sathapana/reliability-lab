"""Typed results: alpha estimates, intervals, comparisons and their serialisation.

Every result converts to a versioned, JSON-safe dict with its provenance
(``to_dict`` / ``to_json``) and to tidy pandas tables (``tables`` /
``to_parquet``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


#: Version of the JSON layout of every result type.
SCHEMA_VERSION = 1


@dataclass(frozen=True, eq=False)
class Provenance:
    """Where a result came from: software, data, declarations, split and known gaps.

    ``ratings_fingerprint`` is the sha256 of the validated ratings
    (:meth:`RatingFrame.fingerprint`). ``split`` describes the partition when the
    study came from a split. ``trace`` holds the trace sidecar's fields,
    fingerprint and the rows of the conditions in use. ``gaps`` lists what the data
    cannot vouch for, such as a missing status column, model family or trace field.
    """

    versions: Mapping[str, str]
    platform: str
    created_at: str
    ratings_fingerprint: str
    n_rows: int
    declarations: Mapping[str, Any]
    split: Mapping[str, Any] | None
    gaps: tuple[str, ...]
    trace: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "versions": dict(self.versions),
                "platform": self.platform,
                "created_at": self.created_at,
                "ratings_fingerprint": self.ratings_fingerprint,
                "n_rows": self.n_rows,
                "declarations": dict(self.declarations),
                "split": None if self.split is None else dict(self.split),
                "trace": None if self.trace is None else dict(self.trace),
                "gaps": list(self.gaps),
            }
        )


@dataclass(frozen=True, eq=False)
class Diagnostic:
    """Design-specific evidence beside alpha, such as run unanimity or order flips.

    ``values`` holds the headline numbers, ``table`` the detail (one row per coder
    pair, unit or condition) and ``notes`` how to read them.
    """

    name: str
    values: Mapping[str, Any]
    table: pd.DataFrame | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "name": self.name,
                "values": dict(self.values),
                "table": None if self.table is None else self.table.to_dict(orient="records"),
                "notes": list(self.notes),
            }
        )


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
    provenance: Provenance | None = None
    diagnostics: Mapping[str, Diagnostic] = field(default_factory=dict)
    trace_findings: tuple[str, ...] = ()
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

        ``method="bca"`` (units scheme only) gave the most accurate coverage in the
        simulations of ``docs/guidance/bootstrap-coverage.md``. On small samples
        dominated by one category it can fail to produce an interval; that is
        warned about, and ``method="percentile"`` then still gives one.
        ``alpha_min`` adds the share of replicates below that minimum. All
        estimates use the same resamples, so their intervals are paired.
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

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``estimates``, ``per_unit``, ``coincidence`` (long), ``coverage``,
        and ``diagnostic_<name>`` for each diagnostic with a table."""
        per_unit = pd.concat(
            [e.per_unit.assign(estimate=name) for name, e in self.estimates.items()],
            ignore_index=True,
        )
        coincidence = pd.concat(
            [
                pd.DataFrame(
                    {
                        "estimate": name,
                        "value_a": np.repeat(e.coincidence.index.to_numpy(), len(e.coincidence)),
                        "value_b": np.tile(e.coincidence.columns.to_numpy(), len(e.coincidence)),
                        "coincidence": e.coincidence.to_numpy().ravel(),
                        "expected": e.expected.to_numpy().ravel(),
                    }
                )
                for name, e in self.estimates.items()
            ],
            ignore_index=True,
        )
        tables = {
            "estimates": self.to_frame(),
            "per_unit": per_unit[["estimate", *self.estimates["as_missing"].per_unit.columns]],
            "coincidence": coincidence,
            "coverage": self.coverage.per_coder.reset_index(),
        }
        for name, diagnostic in self.diagnostics.items():
            if diagnostic.table is not None:
                tables[f"diagnostic_{name}"] = diagnostic.table
        return tables

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """Strict JSON of :meth:`to_dict`, written as UTF-8 when ``path`` is given."""
        return _write_json(self.to_dict(), path)

    def to_parquet(self, directory: str | os.PathLike[str]) -> Path:
        """Write each of :meth:`tables` as Parquet, plus ``result.json`` (needs pyarrow)."""
        return _write_parquet(self.tables(), self.to_dict(), directory)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                **_header("alpha-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
                "design": self.design,
                "measurement": self.measurement,
                "values": list(self.values),
                "coders": list(self.coders),
                "missing_policies": list(self.missing_policies),
                "abstain_as": self.abstain_as,
                "inapplicable_as": self.inapplicable_as,
                "estimates": {k: e.to_dict() for k, e in self.estimates.items()},
                "coverage": self.coverage.to_dict(),
                "diagnostics": {k: d.to_dict() for k, d in self.diagnostics.items()},
                "trace_findings": list(self.trace_findings),
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
        if self.diagnostics:
            lines.append("\nDiagnostics")
            for d in self.diagnostics.values():
                shown = ", ".join(f"{k} {_fmt_any(v)}" for k, v in d.values.items())
                lines.append(f"  {d.name}: {shown}")
        if self.trace_findings:
            lines.append("\nTrace findings")
            lines.extend(f"  - {f}" for f in self.trace_findings)
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
    ``random_state``. ``experimental`` marks a method not yet validated by the
    coverage simulations; no current method is.
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
    provenance: Provenance | None = None

    def __getitem__(self, name: str) -> Interval:
        return self.intervals[name]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``intervals`` and ``replicates`` (estimate, replicate, alpha)."""
        return {
            "intervals": self.to_frame(),
            "replicates": _replicates({n: {"alpha": i} for n, i in self.intervals.items()}),
        }

    def to_json(
        self, path: str | os.PathLike[str] | None = None, *, include_distribution: bool = False
    ) -> str:
        """Strict JSON of :meth:`to_dict`, written as UTF-8 when ``path`` is given."""
        return _write_json(self.to_dict(include_distribution), path)

    def to_parquet(self, directory: str | os.PathLike[str]) -> Path:
        """Write each of :meth:`tables` as Parquet, plus ``result.json`` (needs pyarrow)."""
        return _write_parquet(self.tables(), self.to_dict(), directory)

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
                **_header("bootstrap-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
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
    ``result_b`` the arms' full alpha results. ``configuration_diff`` lists the trace
    fields whose values differ between the arms (None without a trace sidecar).
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
    configuration_diff: Mapping[str, Mapping[str, list[Any]]] | None = None

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

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``differences`` and ``replicates`` (alpha_a, alpha_b, difference)."""
        return {
            "differences": self.to_frame(),
            "replicates": _replicates(
                {
                    name: {
                        "alpha_a": self.arm_a[name],
                        "alpha_b": self.arm_b[name],
                        "difference": diff,
                    }
                    for name, diff in self.differences.items()
                }
            ),
        }

    def to_json(
        self, path: str | os.PathLike[str] | None = None, *, include_distribution: bool = False
    ) -> str:
        """Strict JSON of :meth:`to_dict`, written as UTF-8 when ``path`` is given."""
        return _write_json(self.to_dict(include_distribution), path)

    def to_parquet(self, directory: str | os.PathLike[str]) -> Path:
        """Write each of :meth:`tables` as Parquet, plus ``result.json`` (needs pyarrow)."""
        return _write_parquet(self.tables(), self.to_dict(), directory)

    def to_dict(self, include_distribution: bool = False) -> dict[str, Any]:
        provenance = {
            arm: None if res.provenance is None else res.provenance.to_dict()
            for arm, res in (("a", self.result_a), ("b", self.result_b))
        }
        return _jsonable(  # type: ignore[no-any-return]
            {
                **_header("comparison-result"),
                "provenance": provenance,
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
                "configuration_diff": None
                if self.configuration_diff is None
                else {k: dict(v) for k, v in self.configuration_diff.items()},
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
        ]
        if self.configuration_diff:
            changed = "; ".join(
                f"{k}: {v['a']} -> {v['b']}" for k, v in self.configuration_diff.items()
            )
            lines.append(f"configuration changes: {changed}")
        lines += [
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


def _header(kind: str) -> dict[str, Any]:
    return {
        "schema": f"reliability-lab/{kind}",
        "schema_version": SCHEMA_VERSION,
        "reliability_lab_version": __version__,
    }


def _replicates(columns: Mapping[str, Mapping[str, Interval]]) -> pd.DataFrame:
    """Long table of replicates: one row per estimate and replicate."""
    frames = []
    for name, intervals in columns.items():
        first = next(iter(intervals.values()))
        frame = pd.DataFrame({"estimate": name, "replicate": np.arange(len(first.distribution))})
        for column, interval in intervals.items():
            frame[column] = interval.distribution
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_json(payload: dict[str, Any], path: str | os.PathLike[str] | None) -> str:
    text = json.dumps(payload, indent=2, allow_nan=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _has_pyarrow() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _write_parquet(
    tables: Mapping[str, pd.DataFrame], payload: dict[str, Any], directory: str | os.PathLike[str]
) -> Path:
    if not _has_pyarrow():
        raise ImportError(
            "to_parquet needs pyarrow; install it with: pip install 'reliability-lab[parquet]'"
        )
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        _arrow_safe(frame).to_parquet(out / f"{name}.parquet", index=False)
    _write_json(payload, out / "result.json")
    return out


def _arrow_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns that mix types (say ints and ``"<error>"``) to text."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        kinds = {type(_jsonable(v)) for v in out[column] if v is not None and not pd.isna(v)}
        if len(kinds) > 1:
            out[column] = [None if v is None or pd.isna(v) else str(v) for v in out[column]]
    return out


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"


def _fmt_any(x: Any) -> str:
    if isinstance(x, float | np.floating):
        return _fmt(float(x))
    return str(x)


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
