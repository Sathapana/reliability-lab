"""Typed results: alpha estimates, coverage diagnostics and their serialisation.

File output (JSON, Parquet) and the provenance manifest arrive in phase 3. For now
results convert to JSON-safe dicts and pandas tables.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from reliability_lab._version import __version__


@dataclass(frozen=True)
class RecordedWarning:
    """A warning raised while building a result, kept with the result."""

    category: str
    message: str


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

    @property
    def alpha(self) -> float:
        """Alpha with failed attempts treated as missing (the reference policy)."""
        return self.estimates["as_missing"].alpha

    def __getitem__(self, name: str) -> AlphaEstimate:
        return self.estimates[name]

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
                "warnings": [{"category": w.category, "message": w.message} for w in self.warnings],
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
        if self.warnings:
            lines.append("\nWarnings")
            lines.extend(f"  - {w.category}: {w.message}" for w in self.warnings)
        if self.notes:
            lines.append("\nNotes")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


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
