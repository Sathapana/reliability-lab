"""Gwet's AC1 (nominal) and AC2 (weighted) inter-rater agreement coefficients.

Reference:
    Gwet, K. L. (2008). Computing inter-rater reliability and its variance in the
    presence of high agreement. British Journal of Mathematical and Statistical
    Psychology, 61(1), 29-48.

Gwet's AC1 and AC2 are reported as standalone coefficients with their own report
types (:class:`GwetEstimate` and :class:`GwetResult`), never as diagnostics of
Krippendorff's alpha. They share the study's declarations (value domain, coders,
design, and missing-data policies).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse, stats

from reliability_lab.alpha import Distance
from reliability_lab.exceptions import BootstrapWarning, DesignError
from reliability_lab.inference import (
    METHODS,
    ON_UNDEFINED,
    UNITS_NOTE,
    StudySample,
    _interval_warnings,
    _resolve_level,
    bca_interval,
    bca_weights,
    cluster_plan,
    draw_weights,
    percentile_interval,
    with_fallback,
)
from reliability_lab.report import (
    BootstrapResult,
    Coverage,
    Interval,
    Level,
    Method,
    OnUndefined,
    Provenance,
    RecordedWarning,
    _append_warnings_and_notes,
    _arrow_safe,
    _fmt,
    _header,
    _jsonable,
    _matrix_dict,
    _warning_dicts,
    emit,
)

if TYPE_CHECKING:
    from reliability_lab.study import ReliabilityStudy

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

GWET_WEIGHT_NAMES: tuple[str, ...] = (
    "identity",
    "quadratic",
    "ordinal",
    "linear",
    "radical",
    "ratio",
    "circular",
    "bipolar",
)


def _categ_vec(values: Sequence[Any]) -> FloatArray:
    """Category coordinates matching Gwet (2008) / irrCAC.weights.Weights."""
    if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values):
        return np.asarray(values, dtype=float)
    return np.arange(1, len(values) + 1, dtype=float)


def gwet_weight_matrix(name: str, values: Sequence[Any]) -> FloatArray:
    """Build a q x q Gwet agreement weight matrix W in [0, 1] with unit diagonal."""
    if name not in GWET_WEIGHT_NAMES:
        raise DesignError(
            f"unknown Gwet weights {name!r}; choose from {list(GWET_WEIGHT_NAMES)} or pass a "
            "q x q matrix"
        )
    q = len(values)
    if q <= 1:
        return np.ones((q, q), dtype=float)
    if name == "identity":
        return np.eye(q, dtype=float)

    if name == "ordinal":
        idx = np.arange(q, dtype=float)
        nkl = np.abs(idx[:, None] - idx[None, :]) + 1.0
        raw = nkl * (nkl - 1.0) / 2.0
        np.fill_diagonal(raw, 0.0)
        max_val = float(np.max(raw))
        return np.eye(q, dtype=float) if max_val == 0.0 else 1.0 - raw / max_val

    c = _categ_vec(values)
    xmin, xmax = float(np.min(c)), float(np.max(c))
    span = abs(xmax - xmin)
    if span == 0.0:
        return np.eye(q, dtype=float)

    ck = c[:, None]
    cl = c[None, :]
    diff = ck - cl

    if name == "quadratic":
        return np.asarray(1.0 - (diff / span) ** 2, dtype=float)
    if name == "linear":
        return np.asarray(1.0 - np.abs(diff) / span, dtype=float)
    if name == "radical":
        return np.asarray(1.0 - np.sqrt(np.abs(diff)) / math.sqrt(span), dtype=float)
    if name == "ratio":
        if np.any(c == 0) or (xmax + xmin) == 0:
            raise DesignError("ratio Gwet weights need non-zero categories")
        num = (diff / (ck + cl)) ** 2
        den = ((xmax - xmin) / (xmax + xmin)) ** 2
        return np.asarray(1.0 - num / den, dtype=float)
    if name == "circular":
        u_period = xmax - xmin + 1.0
        raw = np.sin(math.pi * diff / u_period) ** 2
        max_val = float(np.max(raw))
        return np.eye(q, dtype=float) if max_val == 0.0 else 1.0 - raw / max_val
    if name == "bipolar":
        denom = (ck + cl - 2.0 * xmin) * (2.0 * xmax - ck - cl)
        safe = np.where((diff == 0) | (denom == 0), 1.0, denom)
        raw = np.where(diff == 0, 0.0, (diff**2) / safe)
        max_val = float(np.max(raw))
        return np.eye(q, dtype=float) if max_val == 0.0 else 1.0 - raw / max_val
    raise AssertionError(f"unhandled weight name {name!r}")


def resolve_gwet_weights(
    weights: str | npt.ArrayLike | None,
    values: Sequence[Any],
    distance: Distance,
    marginals: FloatArray,
) -> tuple[str, FloatArray]:
    """Resolve Gwet agreement weights (name, q x q matrix) from a spec or study Distance."""
    q = len(values)
    if weights is None:
        if distance.name == "nominal":
            return "identity", np.eye(q, dtype=float)
        if distance.name == "interval":
            return "quadratic", gwet_weight_matrix("quadratic", values)
        d2 = distance.matrix(values, marginals)
        max_d2 = float(np.max(d2)) if d2.size else 0.0
        w = np.eye(q, dtype=float) if max_d2 == 0.0 else 1.0 - d2 / max_d2
        return distance.name, np.asarray(w, dtype=float)
    if isinstance(weights, str):
        return weights, gwet_weight_matrix(weights, values)

    custom = np.asarray(weights, dtype=float)
    if custom.shape != (q, q):
        raise DesignError(f"Gwet weights must have shape ({q}, {q}), got {custom.shape}")
    if not np.isfinite(custom).all():
        raise DesignError("Gwet weights have non-finite entries")
    if (custom < -1e-12).any() or (custom > 1.0 + 1e-12).any():
        raise DesignError("Gwet weights must lie in [0, 1]")
    if not np.allclose(custom, custom.T, rtol=1e-12, atol=1e-12):
        raise DesignError("Gwet weights must be symmetric")
    if not np.allclose(np.diag(custom), 1.0, rtol=1e-12, atol=1e-12):
        raise DesignError("Gwet weights must have 1 on the diagonal")
    return "custom", custom


def _coefficient_label(weights_matrix: FloatArray) -> str:
    q = weights_matrix.shape[0]
    if np.allclose(weights_matrix, np.eye(q, dtype=float), rtol=1e-12, atol=1e-12):
        return "AC1"
    return "AC2"


@dataclass(frozen=True, eq=False)
class GwetComputation:
    """Internal result of :func:`compute_gwet`."""

    coefficient_name: str
    weights_name: str
    ac: float
    pa: float
    pe: float
    se: float
    low: float
    high: float
    z: float
    p_value: float
    confidence: float
    n_units: int
    n_pairable_units: int
    n_ratings: int
    values: tuple[Any, ...]
    weights_matrix: FloatArray
    undefined_reason: str | None = None


def compute_gwet(
    unit_codes: IntArray,
    value_codes: IntArray,
    values: Sequence[Any],
    weights_matrix: FloatArray,
    *,
    weights_name: str = "identity",
    confidence: float = 0.95,
) -> GwetComputation:
    """Compute Gwet's AC1/AC2 and its subject-level linearized variance from coded ratings.

    Follows Gwet (2008) and matches ``irrCAC.raw.CAC.gwet()``: subjects with at least
    one observed rating (``r_i >= 1``) enter the category propensity ``pi_vec`` and the
    subject count ``n``, while subjects with ``r_i >= 2`` enter ``pa`` (``n2more``).
    """
    if not (0.0 < confidence < 1.0) or not math.isfinite(confidence):
        raise DesignError(f"confidence must be in (0, 1), got {confidence}")
    vals = tuple(values)
    q = len(vals)
    w = np.asarray(weights_matrix, dtype=float)
    coeff_name = _coefficient_label(w)
    n_ratings = len(value_codes)

    if n_ratings == 0:
        return GwetComputation(
            coefficient_name=coeff_name,
            weights_name=weights_name,
            ac=np.nan,
            pa=np.nan,
            pe=np.nan,
            se=np.nan,
            low=np.nan,
            high=np.nan,
            z=np.nan,
            p_value=np.nan,
            confidence=confidence,
            n_units=0,
            n_pairable_units=0,
            n_ratings=0,
            values=vals,
            weights_matrix=w,
            undefined_reason="no ratings",
        )

    n_total_units = int(np.max(unit_codes)) + 1
    agree_all = sparse.csr_array(
        (np.ones(n_ratings, dtype=float), (unit_codes, value_codes)),
        shape=(n_total_units, q),
    ).toarray()
    ri_all = agree_all.sum(axis=1)
    active = ri_all >= 1
    agree_mat = agree_all[active]
    ri_vec = ri_all[active]
    n = len(ri_vec)
    pairable_mask = ri_vec >= 2
    n2more = int(pairable_mask.sum())

    if n2more == 0:
        return GwetComputation(
            coefficient_name=coeff_name,
            weights_name=weights_name,
            ac=np.nan,
            pa=np.nan,
            pe=np.nan,
            se=np.nan,
            low=np.nan,
            high=np.nan,
            z=np.nan,
            p_value=np.nan,
            confidence=confidence,
            n_units=n,
            n_pairable_units=0,
            n_ratings=n_ratings,
            values=vals,
            weights_matrix=w,
            undefined_reason="fewer than two ratings in every unit",
        )
    if q < 2:
        return GwetComputation(
            coefficient_name=coeff_name,
            weights_name=weights_name,
            ac=np.nan,
            pa=np.nan,
            pe=np.nan,
            se=np.nan,
            low=np.nan,
            high=np.nan,
            z=np.nan,
            p_value=np.nan,
            confidence=confidence,
            n_units=n,
            n_pairable_units=n2more,
            n_ratings=n_ratings,
            values=vals,
            weights_matrix=w,
            undefined_reason="the value domain has fewer than two values",
        )

    agree_mat_w = agree_mat @ w.T
    sum_q = (agree_mat * (agree_mat_w - 1.0)).sum(axis=1)
    den_ivec = ri_vec * (ri_vec - 1.0)
    safe_den = np.where(den_ivec == 0.0, -1.0, den_ivec)
    pa_ivec = sum_q / safe_den
    pa = float(pa_ivec[pairable_mask].sum() / n2more)

    pi_vec = (agree_mat / ri_vec[:, None]).sum(axis=0) / n
    w_sum = float(w.sum())
    pe = float(w_sum * np.sum(pi_vec * (1.0 - pi_vec)) / (q * (q - 1.0)))

    if math.isclose(1.0 - pe, 0.0, abs_tol=1e-15):
        return GwetComputation(
            coefficient_name=coeff_name,
            weights_name=weights_name,
            ac=np.nan,
            pa=pa,
            pe=pe,
            se=np.nan,
            low=np.nan,
            high=np.nan,
            z=np.nan,
            p_value=np.nan,
            confidence=confidence,
            n_units=n,
            n_pairable_units=n2more,
            n_ratings=n_ratings,
            values=vals,
            weights_matrix=w,
            undefined_reason="expected agreement is 1",
        )

    ac = float((pa - pe) / (1.0 - pe))

    if n < 2:
        return GwetComputation(
            coefficient_name=coeff_name,
            weights_name=weights_name,
            ac=ac,
            pa=pa,
            pe=pe,
            se=np.nan,
            low=np.nan,
            high=np.nan,
            z=np.nan,
            p_value=np.nan,
            confidence=confidence,
            n_units=n,
            n_pairable_units=n2more,
            n_ratings=n_ratings,
            values=vals,
            weights_matrix=w,
        )

    pe_r2 = pe * pairable_mask.astype(float)
    ac_ivec = (n / n2more) * (pa_ivec - pe_r2) / (1.0 - pe)
    pe_ivec = (w_sum / (q * (q - 1.0))) * (agree_mat @ (1.0 - pi_vec)) / ri_vec
    ac_ivec_x = ac_ivec - 2.0 * (1.0 - ac) * (pe_ivec - pe) / (1.0 - pe)
    var_ac = float(np.sum((ac_ivec_x - ac) ** 2) / (n * (n - 1.0)))
    se = math.sqrt(max(0.0, var_ac))
    se_eff = 1e-15 if se == 0.0 else se
    z = float(ac / se_eff)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(z), df=n - 1)))
    lcb, ucb = stats.t.interval(confidence, df=n - 1, loc=ac, scale=se_eff)
    low = float(lcb)
    high = float(min(1.0, float(ucb)))

    return GwetComputation(
        coefficient_name=coeff_name,
        weights_name=weights_name,
        ac=ac,
        pa=pa,
        pe=pe,
        se=se,
        low=low,
        high=high,
        z=z,
        p_value=p_value,
        confidence=confidence,
        n_units=n,
        n_pairable_units=n2more,
        n_ratings=n_ratings,
        values=vals,
        weights_matrix=w,
    )


def _replicate_gwet(
    agree_all: FloatArray,
    ri_all: FloatArray,
    cluster_of_unit: IntArray,
    n_clusters: int,
    weights: npt.ArrayLike,
    w_mat: FloatArray,
    distance: Distance | None,
    values: Sequence[Any],
    dynamic_ordinal: bool,
) -> FloatArray:
    """Vectorized Gwet AC1/AC2 across bootstrap cluster weight vectors."""
    w_clusters = np.atleast_2d(np.asarray(weights, dtype=float))
    unit_w = w_clusters[:, cluster_of_unit]  # (B, U)
    q = agree_all.shape[1]
    if q < 2:
        return np.full(len(w_clusters), np.nan)

    active = (ri_all >= 1).astype(float)
    pairable = (ri_all >= 2).astype(float)

    n_active = unit_w @ active  # (B,)
    n2more = unit_w @ pairable  # (B,)

    prop_unit = np.zeros_like(agree_all)
    mask1 = ri_all >= 1
    prop_unit[mask1] = agree_all[mask1] / ri_all[mask1, None]
    sum_prop = unit_w @ prop_unit  # (B, q)

    with np.errstate(divide="ignore", invalid="ignore"):
        pi_mat = np.where(n_active[:, None] > 0, sum_prop / n_active[:, None], np.nan)

    if not dynamic_ordinal:
        agree_w = agree_all @ w_mat.T
        sum_q = (agree_all * (agree_w - 1.0)).sum(axis=1)
        pa_unit = np.zeros(len(ri_all), dtype=float)
        mask2 = ri_all >= 2
        pa_unit[mask2] = sum_q[mask2] / (ri_all[mask2] * (ri_all[mask2] - 1.0))
        sum_pa = unit_w @ pa_unit
        w_sum = float(w_mat.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            pa = np.where(n2more > 0, sum_pa / n2more, np.nan)
            pe = w_sum * np.sum(pi_mat * (1.0 - pi_mat), axis=1) / (q * (q - 1.0))
            denom = 1.0 - pe
            out = np.where((n2more > 0) & (np.abs(denom) > 1e-15), (pa - pe) / denom, np.nan)
        return np.asarray(out, dtype=float)

    # Dynamic marginal-dependent distance (e.g. study ordinal distance with weights=None)
    assert distance is not None
    pairable_agree = np.where(pairable[:, None] > 0, agree_all, 0.0)
    marginals_b = unit_w @ pairable_agree  # (B, q)
    out = np.full(len(w_clusters), np.nan, dtype=float)
    mask2 = ri_all >= 2
    denom_u = np.where(mask2, ri_all * (ri_all - 1.0), 1.0)
    for b in range(len(w_clusters)):
        if n2more[b] <= 0 or n_active[b] <= 0:
            continue
        d2 = distance.matrix(values, marginals_b[b])
        max_d2 = float(np.max(d2)) if d2.size else 0.0
        wb_mat = np.eye(q, dtype=float) if max_d2 == 0.0 else 1.0 - d2 / max_d2
        agree_w = agree_all @ wb_mat.T
        sum_q = (agree_all * (agree_w - 1.0)).sum(axis=1)
        pa_u = np.where(mask2, sum_q / denom_u, 0.0)
        pa_val = float((unit_w[b] @ pa_u) / n2more[b])
        w_sum = float(wb_mat.sum())
        pe_val = float(w_sum * np.sum(pi_mat[b] * (1.0 - pi_mat[b])) / (q * (q - 1.0)))
        if abs(1.0 - pe_val) > 1e-15:
            out[b] = (pa_val - pe_val) / (1.0 - pe_val)
    return out


@dataclass(frozen=True, eq=False)
class GwetEstimate:
    """One Gwet's AC1 or AC2 estimate under a single missing-data policy."""

    name: str
    policy: str
    coefficient_name: str
    weights_name: str
    ac: float
    pa: float
    pe: float
    se: float
    low: float
    high: float
    z: float
    p_value: float
    confidence: float
    n_units: int
    n_pairable_units: int
    n_ratings: int
    values: tuple[Any, ...]
    weights_matrix: pd.DataFrame
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
            "coefficient": self.coefficient_name,
            "weights": self.weights_name,
            "ac": self.ac,
            "pa": self.pa,
            "pe": self.pe,
            "se": self.se,
            "low": self.low,
            "high": self.high,
            "z": self.z,
            "p_value": self.p_value,
            "confidence": self.confidence,
            "n_units": self.n_units,
            "n_pairable_units": self.n_pairable_units,
            "n_ratings": self.n_ratings,
            "n_imputed": self.n_imputed,
            "n_unanchored": self.n_unanchored,
            "undefined_reason": self.undefined_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                **self.summary_row(),
                "values": list(self.values),
                "weights_matrix": _matrix_dict(self.weights_matrix),
            }
        )


@dataclass(frozen=True, eq=False)
class GwetResult:
    """Gwet's AC1 or AC2 report for one study across all declared missing-data policies."""

    design: str
    measurement: str
    values: tuple[Any, ...]
    coders: tuple[str, ...]
    missing_policies: tuple[str, ...]
    abstain_as: str
    inapplicable_as: str
    estimates: Mapping[str, GwetEstimate]
    coverage: Coverage
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]
    provenance: Provenance | None = None
    _sample: StudySample | None = field(default=None, repr=False)
    _weights_spec: str | npt.ArrayLike | None = field(default=None, repr=False)

    @property
    def ac(self) -> float:
        """The reference ``as_missing`` Gwet AC1/AC2 estimate."""
        return self.estimates["as_missing"].ac

    @property
    def coefficient_name(self) -> str:
        """``"AC1"`` (unweighted / nominal) or ``"AC2"`` (weighted)."""
        return self.estimates["as_missing"].coefficient_name

    @property
    def weights_name(self) -> str:
        """Name of the agreement weight scheme used."""
        return self.estimates["as_missing"].weights_name

    def __getitem__(self, name: str) -> GwetEstimate:
        return self.estimates[name]

    def bootstrap(
        self,
        *,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        level: Level | None = None,
        strata: bool = False,
        method: Method = "percentile",
        random_state: int | np.random.Generator | None = None,
        alpha_min: float | None = None,
        on_undefined: OnUndefined = "nan",
    ) -> BootstrapResult:
        """Unit- or group-cluster bootstrap intervals for every Gwet AC1/AC2 estimate."""
        if self._sample is None:
            raise DesignError("this GwetResult was not built by ReliabilityStudy.gwet()")
        if method not in METHODS:
            raise DesignError(f"unknown bootstrap method {method!r}; choose from {list(METHODS)}")
        if on_undefined not in ON_UNDEFINED:
            raise DesignError(
                f"unknown on_undefined {on_undefined!r}; choose from {list(ON_UNDEFINED)}"
            )
        if n_resamples < 1 or isinstance(n_resamples, bool):
            raise DesignError(f"n_resamples must be at least 1, got {n_resamples}")
        if not (0.0 < confidence < 1.0) or not math.isfinite(confidence):
            raise DesignError(f"confidence must be in (0, 1), got {confidence}")
        if alpha_min is not None and not math.isfinite(alpha_min):
            raise DesignError(f"alpha_min must be finite, got {alpha_min}")

        sample = self._sample
        seed = (
            int(random_state.integers(0, 2**31 - 1))
            if isinstance(random_state, np.random.Generator)
            else int(np.random.SeedSequence().generate_state(1)[0]) % (2**31 - 1)
            if random_state is None
            else int(random_state)
        )
        recorded: list[RecordedWarning] = []
        notes: list[str] = []

        level_used, level_warning = _resolve_level(sample, level)
        if level_warning:
            recorded.append(emit(BootstrapWarning, level_warning, stacklevel=2))
        if method == "bca" and strata:
            raise DesignError(
                "BCa intervals cannot resample within strata; use method='percentile'"
            )
        plan = cluster_plan(sample, level_used, strata)
        n_clusters = plan.n_clusters
        weights = (
            draw_weights(np.random.default_rng(seed), plan, n_resamples)
            if method == "percentile"
            else bca_weights(seed, n_clusters, n_resamples)
        )

        intervals: dict[str, Interval] = {}
        n_total_units = sample.n_units
        for name, est in self.estimates.items():
            inp = sample.inputs[name]
            q = len(inp.values)
            agree_all = sparse.csr_array(
                (np.ones(len(inp.value_idx), dtype=float), (inp.unit_idx, inp.value_idx)),
                shape=(n_total_units, q),
            ).toarray()
            ri_all = agree_all.sum(axis=1)
            w_mat = est.weights_matrix.to_numpy(dtype=float)
            dyn_ord = self._weights_spec is None and sample.distance.depends_on_marginals

            distribution = _replicate_gwet(
                agree_all=agree_all,
                ri_all=ri_all,
                cluster_of_unit=plan.cluster_of_unit,
                n_clusters=n_clusters,
                weights=weights,
                w_mat=w_mat,
                distance=sample.distance,
                values=inp.values,
                dynamic_ordinal=dyn_ord,
            )
            if method == "percentile":
                intervals[name] = percentile_interval(
                    name, est.ac, distribution, confidence, alpha_min
                )
            else:
                ones_w = np.ones((1, n_clusters), dtype=float)
                theta_hat = float(
                    _replicate_gwet(
                        agree_all=agree_all,
                        ri_all=ri_all,
                        cluster_of_unit=plan.cluster_of_unit,
                        n_clusters=n_clusters,
                        weights=ones_w,
                        w_mat=w_mat,
                        distance=sample.distance,
                        values=inp.values,
                        dynamic_ordinal=dyn_ord,
                    )[0]
                )
                jack_weights = np.ones((n_clusters, n_clusters), dtype=float) - np.eye(
                    n_clusters, dtype=float
                )
                jackknife = _replicate_gwet(
                    agree_all=agree_all,
                    ri_all=ri_all,
                    cluster_of_unit=plan.cluster_of_unit,
                    n_clusters=n_clusters,
                    weights=jack_weights,
                    w_mat=w_mat,
                    distance=sample.distance,
                    values=inp.values,
                    dynamic_ordinal=dyn_ord,
                )
                interval = bca_interval(
                    name, est.ac, distribution, theta_hat, jackknife, confidence, alpha_min
                )
                intervals[name] = with_fallback(interval, confidence, on_undefined, alpha_min)

        notes.append(UNITS_NOTE.format(level=level_used, within=" within strata" if strata else ""))
        if len(self.estimates) > 1:
            notes.append("All estimates use the same resamples, so their intervals are paired.")
        recorded.extend(_interval_warnings(intervals, n_resamples))

        return BootstrapResult(
            scheme="units",
            level=level_used,
            strata=strata,
            method=method,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            alpha_min=alpha_min,
            experimental=False,
            n_clusters=n_clusters,
            design=self.design,
            measurement=f"{self.measurement} ({self.coefficient_name})",
            intervals=intervals,
            warnings=tuple(recorded),
            notes=tuple(notes),
            provenance=self.provenance,
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate."""
        return pd.DataFrame([e.summary_row() for e in self.estimates.values()])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``estimates``, ``weights``, and ``coverage``."""
        primary = self.estimates["as_missing"]
        return {
            "estimates": self.to_frame(),
            "weights": primary.weights_matrix.rename_axis("value").reset_index(),
            "coverage": self.coverage.per_coder.reset_index(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                **_header("gwet-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
                "design": self.design,
                "measurement": self.measurement,
                "coefficient": self.coefficient_name,
                "weights": self.weights_name,
                "values": list(self.values),
                "coders": list(self.coders),
                "missing_policies": list(self.missing_policies),
                "abstain_as": self.abstain_as,
                "inapplicable_as": self.inapplicable_as,
                "coverage": self.coverage.to_dict(),
                "estimates": {k: e.to_dict() for k, e in self.estimates.items()},
                "warnings": _warning_dicts(self.warnings),
                "notes": list(self.notes),
            }
        )

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """Strict JSON of :meth:`to_dict`, written as UTF-8 when ``path`` is given."""
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_parquet(self, directory: str | os.PathLike[str]) -> Path:
        """Write each of :meth:`tables` as Parquet, plus ``result.json`` (needs pyarrow)."""
        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError(
                "to_parquet needs pyarrow; install it with: pip install 'reliability-lab[parquet]'"
            )
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables().items():
            _arrow_safe(frame).to_parquet(out / f"{name}.parquet", index=False)
        self.to_json(out / "result.json")
        return out

    def summary(self) -> str:
        """Plain-text report of Gwet's AC1/AC2 across all declared missing-data policies."""
        c = self.coverage
        primary = self.estimates["as_missing"]
        lines = [
            f"Gwet's {self.coefficient_name} (weights: {self.weights_name}) | "
            f"design: {self.design} | measurement: {self.measurement}",
            f"Units: {c.n_units} ({primary.n_pairable_units} with >=2 observed ratings) | "
            f"Coders: {c.n_coders} | Ratings: {c.n_ratings} (density {c.coding_density:.1%})",
            "",
            f"{'estimate':<26}{self.coefficient_name:>8}{'p_a':>9}{'p_e':>9}{'SE':>9}"
            f"{'low':>9}{'high':>9}{'p':>9}",
        ]
        for e in self.estimates.values():
            lines.append(
                f"{e.name:<26}{_fmt(e.ac):>8}{_fmt(e.pa):>9}{_fmt(e.pe):>9}{_fmt(e.se):>9}"
                f"{_fmt(e.low):>9}{_fmt(e.high):>9}{_fmt(e.p_value):>9}"
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


def gwet(
    study: ReliabilityStudy,
    *,
    weights: str | npt.ArrayLike | None = None,
    confidence: float = 0.95,
) -> GwetResult:
    """Compute Gwet's AC1 (nominal) or AC2 (weighted) for a :class:`ReliabilityStudy`."""
    return study.gwet(weights=weights, confidence=confidence)
