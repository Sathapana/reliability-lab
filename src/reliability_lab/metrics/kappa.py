"""Fleiss's, Conger's, and Cohen's kappa coefficients (unweighted and weighted).

References:
    Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and
    Psychological Measurement, 20(1), 37-46.
    Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for
    scaled disagreement or partial credit. Psychological Bulletin, 70(4), 213-220.
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
    Psychological Bulletin, 76(5), 378-382.
    Conger, A. J. (1980). Integration and generalization of kappas for multiple
    raters. Psychological Bulletin, 88(2), 322-328.
    Gwet, K. L. (2021). Handbook of Inter-Rater Reliability (5th ed.). AgreeStat
    Analytics.

Kappas are reported as standalone coefficients (:class:`KappaEstimate` and
:class:`KappaResult`), never as diagnostics of Krippendorff's alpha. They share the
study's declarations (value domain, coders, design, and missing-data policies).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

KappaVariant = Literal["conger", "fleiss", "cohen"]
KAPPA_VARIANTS: tuple[str, ...] = ("conger", "fleiss", "cohen")

_VARIANT_LABELS: dict[str, str] = {
    "conger": "Conger's kappa",
    "fleiss": "Fleiss' kappa",
    "cohen": "Cohen's kappa",
}


@dataclass(frozen=True, eq=False)
class KappaComputation:
    """Internal result of :func:`compute_kappa`."""

    variant: str
    coefficient_name: str
    weights_name: str
    kappa: float
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
    n_coders: int
    n_ratings: int
    values: tuple[Any, ...]
    weights_matrix: FloatArray
    undefined_reason: str | None = None


def compute_kappa(
    unit_codes: IntArray,
    value_codes: IntArray,
    coder_codes: IntArray,
    n_coders: int,
    values: Sequence[Any],
    weights_matrix: FloatArray,
    *,
    variant: KappaVariant = "conger",
    weights_name: str = "identity",
    confidence: float = 0.95,
) -> KappaComputation:
    """Compute Conger's, Fleiss's, or Cohen's kappa and its linearized variance.

    Follows Gwet (2021) and matches ``irrCAC.raw.CAC.fleiss()`` and
    ``irrCAC.raw.CAC.conger()``. For ``variant="cohen"``, requires ``n_coders == 2``
    and evaluates the two-coder Conger/Cohen formulation.
    """
    if variant not in KAPPA_VARIANTS:
        raise DesignError(f"unknown kappa variant {variant!r}; choose from {list(KAPPA_VARIANTS)}")
    if variant == "cohen" and n_coders != 2:
        raise DesignError(
            f"variant='cohen' requires exactly 2 coders, but this study has {n_coders}; "
            "use variant='conger' for 3 or more coders"
        )
    if not (0.0 < confidence < 1.0) or not math.isfinite(confidence):
        raise DesignError(f"confidence must be in (0, 1), got {confidence}")

    vals = tuple(values)
    q = len(vals)
    w = np.asarray(weights_matrix, dtype=float)
    coeff_name = _VARIANT_LABELS[variant]
    n_ratings = len(value_codes)

    if n_ratings == 0:
        return KappaComputation(
            variant=variant,
            coefficient_name=coeff_name,
            weights_name=weights_name,
            kappa=np.nan,
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
            n_coders=n_coders,
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
        return KappaComputation(
            variant=variant,
            coefficient_name=coeff_name,
            weights_name=weights_name,
            kappa=np.nan,
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
            n_coders=n_coders,
            n_ratings=n_ratings,
            values=vals,
            weights_matrix=w,
            undefined_reason="fewer than two ratings in every unit",
        )
    if q < 2:
        return KappaComputation(
            variant=variant,
            coefficient_name=coeff_name,
            weights_name=weights_name,
            kappa=np.nan,
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
            n_coders=n_coders,
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

    if variant == "fleiss":
        pi_vec = (agree_mat / ri_vec[:, None]).sum(axis=0) / n
        pe = float(np.sum(w * np.outer(pi_vec, pi_vec)))
        if math.isclose(1.0 - pe, 0.0, abs_tol=1e-15):
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=np.nan,
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
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
                undefined_reason="expected agreement is 1",
            )
        kappa_val = float((pa - pe) / (1.0 - pe))
        if n < 2:
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=kappa_val,
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
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
            )
        pe_r2 = pe * pairable_mask.astype(float)
        kappa_ivec = (n / n2more) * (pa_ivec - pe_r2) / (1.0 - pe)
        pi_vec_w = 0.5 * ((w @ pi_vec) + (w.T @ pi_vec))
        pe_ivec = (agree_mat @ pi_vec_w) / ri_vec
        kappa_ivec_x = kappa_ivec - 2.0 * (1.0 - kappa_val) * (pe_ivec - pe) / (1.0 - pe)
    else:
        # Conger / Cohen: build active-unit x coder classification matrix
        r = n_coders
        if r < 2:
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=np.nan,
                pa=pa,
                pe=np.nan,
                se=np.nan,
                low=np.nan,
                high=np.nan,
                z=np.nan,
                p_value=np.nan,
                confidence=confidence,
                n_units=n,
                n_pairable_units=n2more,
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
                undefined_reason="fewer than two coders",
            )
        active_map = np.full(n_total_units, -1, dtype=np.intp)
        active_map[active] = np.arange(n, dtype=np.intp)
        act_u = active_map[unit_codes]
        keep = act_u >= 0
        act_u = act_u[keep]
        act_c = coder_codes[keep]
        act_v = value_codes[keep]

        classif_mat = np.zeros((r, q), dtype=float)
        np.add.at(classif_mat, (act_c, act_v), 1.0)
        ng_vec = classif_mat.sum(axis=1, keepdims=True)  # (r, 1)
        if np.any(ng_vec == 0):
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=np.nan,
                pa=pa,
                pe=np.nan,
                se=np.nan,
                low=np.nan,
                high=np.nan,
                z=np.nan,
                p_value=np.nan,
                confidence=confidence,
                n_units=n,
                n_pairable_units=n2more,
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
                undefined_reason="at least one coder rated zero units",
            )
        pgk_mat = classif_mat / ng_vec  # (r, q)
        p_mean_k = (pgk_mat.sum(axis=0) / r).reshape(-1, 1)  # (q, 1)
        s2kl_mat = (pgk_mat.T @ pgk_mat - r * (p_mean_k @ p_mean_k.T)) / (r - 1.0)
        pe = float(np.sum(w * (p_mean_k @ p_mean_k.T - s2kl_mat / r)))
        if math.isclose(1.0 - pe, 0.0, abs_tol=1e-15):
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=np.nan,
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
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
                undefined_reason="expected agreement is 1",
            )
        kappa_val = float((pa - pe) / (1.0 - pe))
        if n < 2:
            return KappaComputation(
                variant=variant,
                coefficient_name=coeff_name,
                weights_name=weights_name,
                kappa=kappa_val,
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
                n_coders=n_coders,
                n_ratings=n_ratings,
                values=vals,
                weights_matrix=w,
            )

        # Vectorized Conger subject-level linearization matching irrCAC.raw.CAC.conger
        # M[g, k] = (n / n_g) * (r * p_mean_k[k] - pgk_mat[g, k])
        m_gk = (n / ng_vec) * (r * p_mean_k.T - pgk_mat)  # (r, q)
        h_gl = m_gk @ w  # (r, q)
        c_g = np.sum(h_gl * pgk_mat, axis=1)  # (r,)

        epsi_ig = np.zeros((n, r), dtype=float)
        delta_term = np.zeros((n, r), dtype=float)
        epsi_ig[act_u, act_c] = 1.0
        delta_term[act_u, act_c] = h_gl[act_c, act_v]

        lambda_ig_mat = delta_term - (epsi_ig - (ng_vec.T / n)) * c_g[None, :]
        pe_ivec = lambda_ig_mat.sum(axis=1) / (r * (r - 1.0))

        pe_r2 = pe * pairable_mask.astype(float)
        conger_ivec = (n / n2more) * (pa_ivec - pe_r2) / (1.0 - pe)
        kappa_ivec_x = conger_ivec - 2.0 * (1.0 - kappa_val) * (pe_ivec - pe) / (1.0 - pe)

    var_k = float(np.sum((kappa_ivec_x - kappa_val) ** 2) / (n * (n - 1.0)))
    se = math.sqrt(max(0.0, var_k))
    se_eff = 1e-15 if se == 0.0 else se
    z = float(kappa_val / se_eff)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(z), df=n - 1)))
    lcb, ucb = stats.t.interval(confidence, df=n - 1, loc=kappa_val, scale=se_eff)
    low = float(lcb)
    high = float(min(1.0, float(ucb)))

    return KappaComputation(
        variant=variant,
        coefficient_name=coeff_name,
        weights_name=weights_name,
        kappa=kappa_val,
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
        n_coders=n_coders,
        n_ratings=n_ratings,
        values=vals,
        weights_matrix=w,
    )


def _replicate_kappa(
    agree_all: FloatArray,
    ri_all: FloatArray,
    coder_unit_cat: FloatArray | None,
    cluster_of_unit: IntArray,
    weights: npt.ArrayLike,
    w_mat: FloatArray,
    variant: KappaVariant,
    distance: Distance | None,
    values: Sequence[Any],
    dynamic_ordinal: bool,
) -> FloatArray:
    """Vectorized Fleiss or Conger/Cohen kappa across bootstrap cluster weight vectors."""
    w_clusters = np.atleast_2d(np.asarray(weights, dtype=float))
    unit_w = w_clusters[:, cluster_of_unit]  # (B, U)
    b_count, u_count = unit_w.shape
    q = agree_all.shape[1]
    if q < 2:
        return np.full(b_count, np.nan)

    active = (ri_all >= 1).astype(float)
    pairable = (ri_all >= 2).astype(float)
    n_active = unit_w @ active  # (B,)
    n2more = unit_w @ pairable  # (B,)

    prop_unit = np.zeros_like(agree_all)
    mask1 = ri_all >= 1
    prop_unit[mask1] = agree_all[mask1] / ri_all[mask1, None]

    mask2 = ri_all >= 2
    denom_u = np.where(mask2, ri_all * (ri_all - 1.0), 1.0)

    if not dynamic_ordinal:
        agree_w = agree_all @ w_mat.T
        sum_q = (agree_all * (agree_w - 1.0)).sum(axis=1)
        pa_unit = np.where(mask2, sum_q / denom_u, 0.0)
        sum_pa = unit_w @ pa_unit
        with np.errstate(divide="ignore", invalid="ignore"):
            pa = np.where(n2more > 0, sum_pa / n2more, np.nan)

        if variant == "fleiss":
            sum_prop = unit_w @ prop_unit  # (B, q)
            with np.errstate(divide="ignore", invalid="ignore"):
                pi_mat = np.where(n_active[:, None] > 0, sum_prop / n_active[:, None], np.nan)
                pe = np.einsum("bk,kl,bl->b", pi_mat, w_mat, pi_mat)
                denom = 1.0 - pe
                out = np.where((n2more > 0) & (np.abs(denom) > 1e-15), (pa - pe) / denom, np.nan)
            return np.asarray(out, dtype=float)

        assert coder_unit_cat is not None  # shape (r, U, q)
        r = coder_unit_cat.shape[0]
        if r < 2:
            return np.full(b_count, np.nan)
        # classif_b: (B, r, q) = einsum("bu,ruk->brk", unit_w, coder_unit_cat)
        c_flat = coder_unit_cat.transpose(1, 0, 2).reshape(u_count, r * q)
        classif_b = (unit_w @ c_flat).reshape(b_count, r, q)
        ng_b = classif_b.sum(axis=2, keepdims=True)  # (B, r, 1)
        valid_b = (n2more > 0) & np.all(ng_b[:, :, 0] > 0, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pgk_b = np.where(ng_b > 0, classif_b / ng_b, 0.0)  # (B, r, q)
            sum_p = pgk_b.sum(axis=1)  # (B, q)
            total_quad = np.einsum("bk,kl,bl->b", sum_p, w_mat, sum_p)
            self_quad = np.einsum("brk,kl,brl->b", pgk_b, w_mat, pgk_b)
            pe = (total_quad - self_quad) / (r * (r - 1.0))
            denom = 1.0 - pe
            out = np.where(valid_b & (np.abs(denom) > 1e-15), (pa - pe) / denom, np.nan)
        return np.asarray(out, dtype=float)

    # Dynamic ordinal distance (weights=None with study ordinal distance)
    assert distance is not None
    pairable_agree = np.where(pairable[:, None] > 0, agree_all, 0.0)
    marginals_b = unit_w @ pairable_agree  # (B, q)
    out = np.full(b_count, np.nan, dtype=float)
    sum_prop = unit_w @ prop_unit

    if variant != "fleiss":
        assert coder_unit_cat is not None
        r = coder_unit_cat.shape[0]
        c_flat = coder_unit_cat.transpose(1, 0, 2).reshape(u_count, r * q)
        classif_all = (unit_w @ c_flat).reshape(b_count, r, q)
    else:
        r = 0
        classif_all = np.empty((0, 0, 0))

    for b in range(b_count):
        if n2more[b] <= 0 or n_active[b] <= 0:
            continue
        d2 = distance.matrix(values, marginals_b[b])
        max_d2 = float(np.max(d2)) if d2.size else 0.0
        wb_mat = np.eye(q, dtype=float) if max_d2 == 0.0 else 1.0 - d2 / max_d2
        agree_w = agree_all @ wb_mat.T
        sum_q = (agree_all * (agree_w - 1.0)).sum(axis=1)
        pa_u = np.where(mask2, sum_q / denom_u, 0.0)
        pa_val = float((unit_w[b] @ pa_u) / n2more[b])
        if variant == "fleiss":
            pi_b = sum_prop[b] / n_active[b]
            pe_val = float(pi_b @ wb_mat @ pi_b)
        else:
            if r < 2:
                continue
            cb = classif_all[b]  # (r, q)
            ng = cb.sum(axis=1, keepdims=True)
            if np.any(ng == 0):
                continue
            pgk = cb / ng
            sp = pgk.sum(axis=0)
            pe_val = float((sp @ wb_mat @ sp - np.sum((pgk @ wb_mat) * pgk)) / (r * (r - 1.0)))
        if abs(1.0 - pe_val) > 1e-15:
            out[b] = (pa_val - pe_val) / (1.0 - pe_val)
    return out


@dataclass(frozen=True, eq=False)
class KappaEstimate:
    """One Fleiss's, Conger's, or Cohen's kappa estimate under a single missing-data policy."""

    name: str
    policy: str
    variant: str
    coefficient_name: str
    weights_name: str
    kappa: float
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
    n_coders: int
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
            "variant": self.variant,
            "coefficient": self.coefficient_name,
            "weights": self.weights_name,
            "kappa": self.kappa,
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
            "n_coders": self.n_coders,
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
class KappaResult:
    """Kappa report (Conger's, Fleiss's, or Cohen's) across all declared missing-data policies."""

    design: str
    measurement: str
    variant: KappaVariant
    values: tuple[Any, ...]
    coders: tuple[str, ...]
    missing_policies: tuple[str, ...]
    abstain_as: str
    inapplicable_as: str
    estimates: Mapping[str, KappaEstimate]
    coverage: Coverage
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]
    provenance: Provenance | None = None
    _sample: StudySample | None = field(default=None, repr=False)
    _weights_spec: str | npt.ArrayLike | None = field(default=None, repr=False)

    @property
    def kappa(self) -> float:
        """The reference ``as_missing`` kappa estimate."""
        return self.estimates["as_missing"].kappa

    @property
    def coefficient_name(self) -> str:
        """``"Conger's kappa"``, ``"Fleiss' kappa"``, or ``"Cohen's kappa"``."""
        return self.estimates["as_missing"].coefficient_name

    @property
    def weights_name(self) -> str:
        """Name of the agreement weight scheme used."""
        return self.estimates["as_missing"].weights_name

    def __getitem__(self, name: str) -> KappaEstimate:
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
        """Unit- or group-cluster bootstrap intervals for every kappa estimate."""
        if self._sample is None:
            raise DesignError("this KappaResult was not built by ReliabilityStudy.kappa()")
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
        r = sample.n_coders
        for name, est in self.estimates.items():
            inp = sample.inputs[name]
            q = len(inp.values)
            agree_all = sparse.csr_array(
                (np.ones(len(inp.value_idx), dtype=float), (inp.unit_idx, inp.value_idx)),
                shape=(n_total_units, q),
            ).toarray()
            ri_all = agree_all.sum(axis=1)
            coder_unit_cat: FloatArray | None = None
            if self.variant != "fleiss":
                coder_unit_cat = np.zeros((r, n_total_units, q), dtype=float)
                np.add.at(coder_unit_cat, (inp.coder_idx, inp.unit_idx, inp.value_idx), 1.0)
            w_mat = est.weights_matrix.to_numpy(dtype=float)
            dyn_ord = self._weights_spec is None and sample.distance.depends_on_marginals

            distribution = _replicate_kappa(
                agree_all=agree_all,
                ri_all=ri_all,
                coder_unit_cat=coder_unit_cat,
                cluster_of_unit=plan.cluster_of_unit,
                weights=weights,
                w_mat=w_mat,
                variant=self.variant,
                distance=sample.distance,
                values=inp.values,
                dynamic_ordinal=dyn_ord,
            )
            if method == "percentile":
                intervals[name] = percentile_interval(
                    name, est.kappa, distribution, confidence, alpha_min
                )
            else:
                ones_w = np.ones((1, n_clusters), dtype=float)
                theta_hat = float(
                    _replicate_kappa(
                        agree_all=agree_all,
                        ri_all=ri_all,
                        coder_unit_cat=coder_unit_cat,
                        cluster_of_unit=plan.cluster_of_unit,
                        weights=ones_w,
                        w_mat=w_mat,
                        variant=self.variant,
                        distance=sample.distance,
                        values=inp.values,
                        dynamic_ordinal=dyn_ord,
                    )[0]
                )
                jack_weights = np.ones((n_clusters, n_clusters), dtype=float) - np.eye(
                    n_clusters, dtype=float
                )
                jackknife = _replicate_kappa(
                    agree_all=agree_all,
                    ri_all=ri_all,
                    coder_unit_cat=coder_unit_cat,
                    cluster_of_unit=plan.cluster_of_unit,
                    weights=jack_weights,
                    w_mat=w_mat,
                    variant=self.variant,
                    distance=sample.distance,
                    values=inp.values,
                    dynamic_ordinal=dyn_ord,
                )
                interval = bca_interval(
                    name, est.kappa, distribution, theta_hat, jackknife, confidence, alpha_min
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
                **_header("kappa-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
                "design": self.design,
                "measurement": self.measurement,
                "variant": self.variant,
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
        """Plain-text report of kappa across all declared missing-data policies."""
        c = self.coverage
        primary = self.estimates["as_missing"]
        lines = [
            f"{self.coefficient_name} (weights: {self.weights_name}) | "
            f"design: {self.design} | measurement: {self.measurement}",
            f"Units: {c.n_units} ({primary.n_pairable_units} with >=2 observed ratings) | "
            f"Coders: {c.n_coders} | Ratings: {c.n_ratings} (density {c.coding_density:.1%})",
            "",
            f"{'estimate':<26}{'kappa':>8}{'p_a':>9}{'p_e':>9}{'SE':>9}"
            f"{'low':>9}{'high':>9}{'p':>9}",
        ]
        for e in self.estimates.values():
            lines.append(
                f"{e.name:<26}{_fmt(e.kappa):>8}{_fmt(e.pa):>9}{_fmt(e.pe):>9}{_fmt(e.se):>9}"
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


def kappa(
    study: ReliabilityStudy,
    *,
    variant: KappaVariant = "conger",
    weights: str | npt.ArrayLike | None = None,
    confidence: float = 0.95,
) -> KappaResult:
    """Compute Conger's, Fleiss's, or Cohen's kappa for a :class:`ReliabilityStudy`."""
    return study.kappa(variant=variant, weights=weights, confidence=confidence)
