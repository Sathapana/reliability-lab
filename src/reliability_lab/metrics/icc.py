"""Intraclass Correlation Coefficients (ICC) and Generalizability Theory (G-theory).

References:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in
    assessing rater reliability. Psychological Bulletin, 86(2), 420-428.
    McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass
    correlation coefficients. Psychological Methods, 1(1), 30-46.
    Brennan, R. L. (2001). Generalizability Theory. Springer.

ICCs and G-theory variance components are reported as standalone coefficients
(:class:`ICCEstimate`, :class:`ICCResult`, :class:`GTheoryEstimate`, and
:class:`GTheoryResult`), never as diagnostics of Krippendorff's alpha. They use the
study's declarations (numeric value domain, coders, design, and missing-data
policies).
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
    _warning_dicts,
    emit,
)

if TYPE_CHECKING:
    from reliability_lab.study import ReliabilityStudy

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

ICCModel = Literal["one_way", "two_way_random", "two_way_mixed"]
ICCAgreement = Literal["agreement", "consistency"]
ICCUnit = Literal["single", "average"]
GTheoryCoefficient = Literal["e_rho2", "phi"]

ICC_MODELS: tuple[str, ...] = ("one_way", "two_way_random", "two_way_mixed")
ICC_AGREEMENTS: tuple[str, ...] = ("agreement", "consistency")
ICC_UNITS: tuple[str, ...] = ("single", "average")
GTHEORY_COEFFICIENTS: tuple[str, ...] = ("e_rho2", "phi")


def _validate_numeric_domain(values: Sequence[Any]) -> FloatArray:
    """Ensure every declared domain value is a finite number (int/float, not bool)."""
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            raise DesignError(
                "ICC and G-theory require numeric values in the declared value domain; "
                f"got non-numeric value {v!r}"
            )
        fv = float(v)
        if not math.isfinite(fv):
            raise DesignError(f"ICC and G-theory require finite numeric values; got {v!r}")
        out.append(fv)
    return np.asarray(out, dtype=float)


def _icc_code(model: ICCModel, agreement: ICCAgreement, unit: ICCUnit) -> str:
    suffix = "1" if unit == "single" else "k"
    if model == "one_way":
        return f"ICC(1,{suffix})"
    if agreement == "agreement":
        return f"ICC(2,{suffix})"
    return f"ICC(3,{suffix})"


@dataclass(frozen=True, eq=False)
class VarianceComponents:
    """ANOVA / G-study variance decomposition (unit, coder, residual).

    ``sigma2_unit``, ``sigma2_coder``, and ``sigma2_residual`` are the unbiased
    method-of-moments (ANOVA / Henderson's Method 1) estimates, which can be slightly
    negative when true variance is near zero. ``share_unit``, ``share_coder``, and
    ``share_residual`` are the non-negative variance shares (clamping negative
    estimates at zero, following Brennan, 2001). ``e_rho2`` and ``phi`` are the
    generalizability and dependability coefficients evaluated at ``k_effective``
    raters.
    """

    sigma2_unit: float
    sigma2_coder: float
    sigma2_residual: float
    share_unit: float
    share_coder: float
    share_residual: float
    ms_unit: float
    ms_coder: float
    ms_residual: float
    df_unit: float
    df_coder: float
    df_residual: float
    k_effective: float
    n_Effective: float
    e_rho2: float
    phi: float
    balanced: bool

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                "sigma2_unit": self.sigma2_unit,
                "sigma2_coder": self.sigma2_coder,
                "sigma2_residual": self.sigma2_residual,
                "share_unit": self.share_unit,
                "share_coder": self.share_coder,
                "share_residual": self.share_residual,
                "ms_unit": self.ms_unit,
                "ms_coder": self.ms_coder,
                "ms_residual": self.ms_residual,
                "df_unit": self.df_unit,
                "df_coder": self.df_coder,
                "df_residual": self.df_residual,
                "k_effective": self.k_effective,
                "e_rho2": self.e_rho2,
                "phi": self.phi,
                "balanced": self.balanced,
            }
        )


def compute_variance_components(
    unit_codes: IntArray,
    coder_codes: IntArray,
    y_values: FloatArray,
    n_coders: int,
    *,
    one_way: bool = False,
    complete_case: bool = False,
) -> tuple[VarianceComponents | None, int, int, int, str | None]:
    """Compute ANOVA / Henderson's Method 1 variance components from coded numeric ratings.

    Returns ``(components, n_units, n_pairable_units, n_ratings, undefined_reason)``.
    When data are balanced (every active unit rated by all ``n_coders``), Henderson's
    Method 1 reduces identically to standard balanced two-way ANOVA.
    """
    n_ratings_in = len(y_values)
    if n_ratings_in == 0:
        return None, 0, 0, 0, "no ratings"

    n_total_units = int(np.max(unit_codes)) + 1
    z_all = sparse.csr_array(
        (np.ones(n_ratings_in, dtype=float), (unit_codes, coder_codes)),
        shape=(n_total_units, n_coders),
    ).toarray()
    y_all = sparse.csr_array(
        (y_values, (unit_codes, coder_codes)),
        shape=(n_total_units, n_coders),
    ).toarray()
    y2_all = sparse.csr_array(
        (y_values * y_values, (unit_codes, coder_codes)),
        shape=(n_total_units, n_coders),
    ).toarray()

    ri_all = z_all.sum(axis=1)
    n_active_units = int((ri_all >= 1).sum())
    keep_u = ri_all == n_coders if complete_case else ri_all >= 2

    z_mat = z_all[keep_u]
    y_mat = y_all[keep_u]
    y2_mat = y2_all[keep_u]
    ri_vec = ri_all[keep_u]
    n_pairable = len(ri_vec)
    n_ratings = int(ri_vec.sum())

    if n_pairable < 2:
        reason = (
            "fewer than two complete units rated by all coders"
            if complete_case
            else "fewer than two units with two or more ratings"
        )
        return None, n_active_units, n_pairable, n_ratings, reason

    # Filter to coders that rated at least one kept unit
    nj_full = z_mat.sum(axis=0)
    active_c = nj_full >= 1
    z_mat = z_mat[:, active_c]
    y_mat = y_mat[:, active_c]
    y2_mat = y2_mat[:, active_c]
    nj_vec = nj_full[active_c]
    k = len(nj_vec)
    n = n_pairable
    total_n = float(n_ratings)

    if k < 2:
        return None, n_active_units, n_pairable, n_ratings, "fewer than two active coders"

    is_balanced = bool(np.all(ri_vec == k) and np.all(nj_vec == n))

    yi_dot = y_mat.sum(axis=1)
    yj_dot = y_mat.sum(axis=0)
    y_dotdot = float(yi_dot.sum())

    t_o = float(y2_mat.sum())
    t_u = float(np.sum((yi_dot * yi_dot) / ri_vec))
    t_c = float(np.sum((yj_dot * yj_dot) / nj_vec))
    t_mu = (y_dotdot * y_dotdot) / total_n

    k1 = float(np.sum(ri_vec * ri_vec) / total_n)
    k2 = float(np.sum(nj_vec * nj_vec) / total_n)
    df_u = float(n - 1)
    k0 = (total_n - k1) / df_u

    if one_way:
        ss_u = max(0.0, t_u - t_mu)
        ss_w = max(0.0, t_o - t_u)
        df_w = float(total_n - n)
        if df_w <= 0:
            return (
                None,
                n_active_units,
                n_pairable,
                n_ratings,
                "zero within-unit degrees of freedom",
            )
        ms_u = ss_u / df_u
        ms_w = ss_w / df_w
        sigma2_e = ms_w
        sigma2_u = (ms_u - ms_w) / k0 if k0 > 0 else np.nan
        sigma2_c = 0.0
        ms_c = np.nan
        df_c = 0.0
        df_e = df_w
        ms_e = ms_w
        n0 = float(n)
    else:
        df_c = float(k - 1)
        df_e = float(total_n - n - k + 1)
        n0 = (total_n - k2) / df_c if df_c > 0 else float(n)
        if df_e <= 0:
            return None, n_active_units, n_pairable, n_ratings, "zero residual degrees of freedom"
        if is_balanced:
            ss_u = max(0.0, t_u - t_mu)
            ss_c = max(0.0, t_c - t_mu)
            ss_e = max(0.0, t_o - t_u - t_c + t_mu)
            ms_u = ss_u / df_u
            ms_c = ss_c / df_c
            ms_e = ss_e / df_e
            sigma2_e = ms_e
            sigma2_u = (ms_u - ms_e) / k
            sigma2_c = (ms_c - ms_e) / n
        else:
            # Henderson's Method 1 linear system for (sigma2_u, sigma2_c, sigma2_e)
            lhs = np.array(
                [
                    [total_n - k1, n - k2, n - 1.0],
                    [k - k1, total_n - k2, k - 1.0],
                    [total_n - k1, total_n - k2, total_n - 1.0],
                ],
                dtype=float,
            )
            rhs = np.array([t_u - t_mu, t_c - t_mu, t_o - t_mu], dtype=float)
            try:
                sol = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                return (
                    None,
                    n_active_units,
                    n_pairable,
                    n_ratings,
                    "singular design matrix for unbalanced variance components",
                )
            sigma2_u = float(sol[0])
            sigma2_c = float(sol[1])
            sigma2_e = max(0.0, float(sol[2]))
            ms_e = sigma2_e
            ms_u = max(0.0, sigma2_e + k0 * sigma2_u)
            ms_c = max(0.0, sigma2_e + n0 * sigma2_c)

    s_u = max(0.0, sigma2_u)
    s_c = 0.0 if one_way else max(0.0, sigma2_c)
    s_e = max(0.0, sigma2_e)
    s_tot = s_u + s_c + s_e
    if s_tot > 0:
        share_u = s_u / s_tot
        share_c = s_c / s_tot
        share_e = s_e / s_tot
    else:
        share_u = share_c = share_e = np.nan

    rel_denom = s_u + (s_e / k0) if k0 > 0 else 0.0
    abs_denom = s_u + ((s_c + s_e) / k0) if k0 > 0 else 0.0
    e_rho2 = s_u / rel_denom if rel_denom > 0 else np.nan
    phi = s_u / abs_denom if abs_denom > 0 else np.nan

    return (
        VarianceComponents(
            sigma2_unit=float(sigma2_u),
            sigma2_coder=float(sigma2_c),
            sigma2_residual=float(sigma2_e),
            share_unit=float(share_u),
            share_coder=float(share_c),
            share_residual=float(share_e),
            ms_unit=float(ms_u),
            ms_coder=float(ms_c),
            ms_residual=float(ms_e),
            df_unit=float(df_u),
            df_coder=float(df_c),
            df_residual=float(df_e),
            k_effective=float(k0),
            n_Effective=float(n0),
            e_rho2=float(e_rho2),
            phi=float(phi),
            balanced=is_balanced,
        ),
        n_active_units,
        n_pairable,
        n_ratings,
        None,
    )


@dataclass(frozen=True, eq=False)
class ICCComputation:
    """Internal result of :func:`compute_icc`."""

    code: str
    model: ICCModel
    agreement: ICCAgreement
    unit: ICCUnit
    icc: float
    f_value: float
    df1: float
    df2: float
    p_value: float
    low: float
    high: float
    confidence: float
    n_units: int
    n_pairable_units: int
    n_coders: int
    n_ratings: int
    variance_components: VarianceComponents | None
    undefined_reason: str | None = None


def compute_icc(
    unit_codes: IntArray,
    coder_codes: IntArray,
    y_values: FloatArray,
    n_coders: int,
    *,
    model: ICCModel = "two_way_random",
    agreement: ICCAgreement = "agreement",
    unit: ICCUnit = "single",
    complete_case: bool = False,
    confidence: float = 0.95,
) -> ICCComputation:
    """Compute Shrout & Fleiss (1979) / McGraw & Wong (1996) ICC and G-theory components."""
    if model not in ICC_MODELS:
        raise DesignError(f"unknown ICC model {model!r}; choose from {list(ICC_MODELS)}")
    if agreement not in ICC_AGREEMENTS:
        raise DesignError(
            f"unknown ICC agreement {agreement!r}; choose from {list(ICC_AGREEMENTS)}"
        )
    if unit not in ICC_UNITS:
        raise DesignError(f"unknown ICC unit {unit!r}; choose from {list(ICC_UNITS)}")
    if model == "one_way" and agreement == "consistency":
        raise DesignError(
            "model='one_way' does not separate coder main effects and only supports "
            "agreement='agreement'; use model='two_way_mixed' or 'two_way_random' for "
            "consistency"
        )
    if not (0.0 < confidence < 1.0) or not math.isfinite(confidence):
        raise DesignError(f"confidence must be in (0, 1), got {confidence}")

    code = _icc_code(model, agreement, unit)
    vc, n_units, n_pairable, n_ratings, reason = compute_variance_components(
        unit_codes,
        coder_codes,
        y_values,
        n_coders,
        one_way=(model == "one_way"),
        complete_case=complete_case,
    )
    if vc is None or reason is not None:
        return ICCComputation(
            code=code,
            model=model,
            agreement=agreement,
            unit=unit,
            icc=np.nan,
            f_value=np.nan,
            df1=np.nan,
            df2=np.nan,
            p_value=np.nan,
            low=np.nan,
            high=np.nan,
            confidence=confidence,
            n_units=n_units,
            n_pairable_units=n_pairable,
            n_coders=n_coders,
            n_ratings=n_ratings,
            variance_components=None,
            undefined_reason=reason or "variance components are undefined",
        )

    ms_u = vc.ms_unit
    ms_e = vc.ms_residual
    ms_c = vc.ms_coder
    df_u = vc.df_unit
    df_c = vc.df_coder
    df_e = vc.df_residual
    k0 = vc.k_effective
    n0 = vc.n_Effective

    if ms_u == 0.0 and ms_e == 0.0:
        return ICCComputation(
            code=code,
            model=model,
            agreement=agreement,
            unit=unit,
            icc=np.nan,
            f_value=np.nan,
            df1=df_u,
            df2=df_e,
            p_value=np.nan,
            low=np.nan,
            high=np.nan,
            confidence=confidence,
            n_units=n_units,
            n_pairable_units=n_pairable,
            n_coders=n_coders,
            n_ratings=n_ratings,
            variance_components=vc,
            undefined_reason="total variance is zero (all ratings are identical)",
        )

    alpha_tail = 0.5 * (1.0 - confidence)
    f_val = float(ms_u / ms_e) if ms_e > 0 else np.inf
    p_val = float(1.0 - stats.f.cdf(f_val, df_u, df_e)) if math.isfinite(f_val) else 0.0

    if model == "one_way" or agreement == "consistency":
        # ICC(1,1), ICC(1,k), ICC(3,1), ICC(3,k)
        if unit == "single":
            denom = ms_u + (k0 - 1.0) * ms_e
            icc_val = (ms_u - ms_e) / denom if denom != 0 else np.nan
        else:
            icc_val = (ms_u - ms_e) / ms_u if ms_u != 0 else np.nan

        df1_out, df2_out = df_u, df_e
        if math.isfinite(f_val) and f_val > 0:
            f_crit_low = float(stats.f.ppf(1.0 - alpha_tail, df_u, df_e))
            f_crit_high = float(stats.f.ppf(1.0 - alpha_tail, df_e, df_u))
            f_l = f_val / f_crit_low
            f_u = f_val * f_crit_high
            if unit == "single":
                low = (f_l - 1.0) / (f_l + (k0 - 1.0))
                high = (f_u - 1.0) / (f_u + (k0 - 1.0))
            else:
                low = 1.0 - 1.0 / f_l
                high = 1.0 - 1.0 / f_u
        elif math.isinf(f_val):
            low, high = 1.0, 1.0
        else:
            low, high = -1.0, 1.0
    else:
        # Two-way agreement: ICC(2,1) and ICC(2,k)
        denom_single = ms_u + (k0 - 1.0) * ms_e + (k0 / n0) * (ms_c - ms_e)
        icc_single = (ms_u - ms_e) / denom_single if denom_single != 0 else np.nan
        if unit == "single":
            icc_val = icc_single
        else:
            denom_avg = ms_u + (ms_c - ms_e) / n0
            icc_val = (ms_u - ms_e) / denom_avg if denom_avg != 0 else np.nan

        if math.isfinite(icc_single) and ms_e > 0 and abs(1.0 - icc_single) > 1e-12:
            f_j = ms_c / ms_e
            a_coef = (k0 * icc_single) / (n0 * (1.0 - icc_single))
            b_coef = 1.0 + (k0 * icc_single * (n0 - 1.0)) / (n0 * (1.0 - icc_single))
            v_num = (a_coef * f_j + b_coef) ** 2
            v_den = ((a_coef * f_j) ** 2) / df_c + (b_coef**2) / df_e
            v_df = v_num / v_den if v_den > 0 else df_e
            v_df = max(1.0, float(v_df))
            df1_out, df2_out = df_u, v_df

            f_star = float(stats.f.ppf(1.0 - alpha_tail, df_u, v_df))
            f_dstar = float(stats.f.ppf(1.0 - alpha_tail, v_df, df_u))
            c_term = k0 * ms_c + (k0 * n0 - k0 - n0) * ms_e
            low_1 = (n0 * (ms_u - f_star * ms_e)) / (f_star * c_term + n0 * ms_u)
            high_1 = (n0 * (f_dstar * ms_u - ms_e)) / (c_term + n0 * f_dstar * ms_u)
            if unit == "single":
                low, high = float(low_1), float(high_1)
            else:
                low = float((k0 * low_1) / (1.0 + (k0 - 1.0) * low_1))
                high = float((k0 * high_1) / (1.0 + (k0 - 1.0) * high_1))
        elif ms_e == 0.0 and ms_u > 0:
            df1_out, df2_out = df_u, df_e
            low, high = float(icc_val), float(icc_val)
        else:
            df1_out, df2_out = df_u, df_e
            low, high = np.nan, np.nan

    return ICCComputation(
        code=code,
        model=model,
        agreement=agreement,
        unit=unit,
        icc=float(icc_val),
        f_value=float(f_val),
        df1=float(df1_out),
        df2=float(df2_out),
        p_value=float(p_val),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_units=n_units,
        n_pairable_units=n_pairable,
        n_coders=n_coders,
        n_ratings=n_ratings,
        variance_components=vc,
    )


def _replicate_icc_or_gtheory(
    z_all: FloatArray,
    y_all: FloatArray,
    y2_all: FloatArray,
    cluster_of_unit: IntArray,
    weights: npt.ArrayLike,
    *,
    mode: Literal["icc", "e_rho2", "phi"],
    model: ICCModel = "two_way_random",
    agreement: ICCAgreement = "agreement",
    unit: ICCUnit = "single",
    complete_case: bool = False,
    target_k: float | None = None,
) -> FloatArray:
    """Vectorized ICC or G-theory (E_rho2 / Phi) across bootstrap cluster weight vectors."""
    w_clusters = np.atleast_2d(np.asarray(weights, dtype=float))
    unit_w = w_clusters[:, cluster_of_unit]  # (B, U)
    b_count = len(unit_w)
    n_coders = z_all.shape[1]

    ri_all = z_all.sum(axis=1)
    keep_u = ri_all == n_coders if complete_case else ri_all >= 2

    if int(keep_u.sum()) < 2:
        return np.full(b_count, np.nan)

    uw = unit_w[:, keep_u]  # (B, U_kept)
    z_m = z_all[keep_u]  # (U_kept, R)
    y_m = y_all[keep_u]  # (U_kept, R)
    y2_m = y2_all[keep_u]  # (U_kept, R)
    ri = ri_all[keep_u]  # (U_kept,)

    yi_dot = y_m.sum(axis=1)  # (U_kept,)
    t_u_unit = (yi_dot * yi_dot) / ri
    t_o_unit = y2_m.sum(axis=1)
    ri_sq = ri * ri

    n_b = uw.sum(axis=1)  # (B,)
    total_n_b = uw @ ri  # (B,)
    t_u_b = uw @ t_u_unit  # (B,)
    t_o_b = uw @ t_o_unit  # (B,)
    y_dotdot_b = uw @ yi_dot  # (B,)
    sum_ri_sq_b = uw @ ri_sq  # (B,)

    nj_b = uw @ z_m  # (B, R)
    yj_b = uw @ y_m  # (B, R)

    out = np.full(b_count, np.nan, dtype=float)
    one_way = model == "one_way" and mode == "icc"

    for b in range(b_count):
        nb = float(n_b[b])
        tot_n = float(total_n_b[b])
        if nb < 2.0 or tot_n <= nb:
            continue
        njb = nj_b[b]
        active_c = njb > 0
        kb = int(active_c.sum())
        if kb < 2:
            continue
        njb_act = njb[active_c]
        yjb_act = yj_b[b, active_c]

        t_o = float(t_o_b[b])
        t_u = float(t_u_b[b])
        t_c = float(np.sum((yjb_act * yjb_act) / njb_act))
        t_mu = float((y_dotdot_b[b] ** 2) / tot_n)

        k1 = float(sum_ri_sq_b[b] / tot_n)
        k2 = float(np.sum(njb_act * njb_act) / tot_n)
        df_u = nb - 1.0
        k0 = (tot_n - k1) / df_u
        if k0 <= 0:
            continue

        if one_way:
            df_w = tot_n - nb
            if df_w <= 0:
                continue
            ms_u = max(0.0, t_u - t_mu) / df_u
            ms_w = max(0.0, t_o - t_u) / df_w
            if ms_u == 0.0 and ms_w == 0.0:
                continue
            if unit == "single":
                denom = ms_u + (k0 - 1.0) * ms_w
                out[b] = (ms_u - ms_w) / denom if denom != 0 else np.nan
            else:
                out[b] = (ms_u - ms_w) / ms_u if ms_u != 0 else np.nan
            continue

        df_c = float(kb - 1)
        df_e = tot_n - nb - kb + 1.0
        if df_e <= 0:
            continue
        n0 = (tot_n - k2) / df_c if df_c > 0 else nb

        if complete_case:
            ms_u = max(0.0, t_u - t_mu) / df_u
            ms_c = max(0.0, t_c - t_mu) / df_c
            ms_e = max(0.0, t_o - t_u - t_c + t_mu) / df_e
            sig_e = ms_e
            sig_u = (ms_u - ms_e) / kb
            sig_c = (ms_c - ms_e) / nb
        else:
            lhs = np.array(
                [
                    [tot_n - k1, nb - k2, nb - 1.0],
                    [kb - k1, tot_n - k2, kb - 1.0],
                    [tot_n - k1, tot_n - k2, tot_n - 1.0],
                ],
                dtype=float,
            )
            rhs = np.array([t_u - t_mu, t_c - t_mu, t_o - t_mu], dtype=float)
            try:
                sol = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                continue
            sig_u = float(sol[0])
            sig_c = float(sol[1])
            sig_e = max(0.0, float(sol[2]))
            ms_e = sig_e
            ms_u = max(0.0, sig_e + k0 * sig_u)
            ms_c = max(0.0, sig_e + n0 * sig_c)

        if mode == "icc":
            if ms_u == 0.0 and ms_e == 0.0:
                continue
            if agreement == "consistency":
                if unit == "single":
                    denom = ms_u + (k0 - 1.0) * ms_e
                    out[b] = (ms_u - ms_e) / denom if denom != 0 else np.nan
                else:
                    out[b] = (ms_u - ms_e) / ms_u if ms_u != 0 else np.nan
            else:
                if unit == "single":
                    denom = ms_u + (k0 - 1.0) * ms_e + (k0 / n0) * (ms_c - ms_e)
                    out[b] = (ms_u - ms_e) / denom if denom != 0 else np.nan
                else:
                    denom = ms_u + (ms_c - ms_e) / n0
                    out[b] = (ms_u - ms_e) / denom if denom != 0 else np.nan
        else:
            k_eval = float(target_k) if target_k is not None else k0
            su = max(0.0, sig_u)
            sc = max(0.0, sig_c)
            se = max(0.0, sig_e)
            if mode == "e_rho2":
                den = su + se / k_eval
                out[b] = su / den if den > 0 else np.nan
            else:
                den = su + (sc + se) / k_eval
                out[b] = su / den if den > 0 else np.nan

    return out


@dataclass(frozen=True, eq=False)
class ICCEstimate:
    """One Intraclass Correlation Coefficient estimate under a single missing-data policy."""

    name: str
    policy: str
    code: str
    model: ICCModel
    agreement: ICCAgreement
    unit: ICCUnit
    icc: float
    f_value: float
    df1: float
    df2: float
    p_value: float
    low: float
    high: float
    confidence: float
    n_units: int
    n_pairable_units: int
    n_coders: int
    n_ratings: int
    variance_components: VarianceComponents | None
    undefined_reason: str | None = None
    n_imputed: int = 0
    n_unanchored: int = 0

    @property
    def defined(self) -> bool:
        return self.undefined_reason is None

    def summary_row(self) -> dict[str, Any]:
        vc = self.variance_components
        return {
            "estimate": self.name,
            "policy": self.policy,
            "code": self.code,
            "model": self.model,
            "agreement": self.agreement,
            "unit": self.unit,
            "icc": self.icc,
            "low": self.low,
            "high": self.high,
            "f_value": self.f_value,
            "df1": self.df1,
            "df2": self.df2,
            "p_value": self.p_value,
            "confidence": self.confidence,
            "sigma2_unit": np.nan if vc is None else vc.sigma2_unit,
            "sigma2_coder": np.nan if vc is None else vc.sigma2_coder,
            "sigma2_residual": np.nan if vc is None else vc.sigma2_residual,
            "e_rho2": np.nan if vc is None else vc.e_rho2,
            "phi": np.nan if vc is None else vc.phi,
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
                "variance_components": None
                if self.variance_components is None
                else self.variance_components.to_dict(),
            }
        )


@dataclass(frozen=True, eq=False)
class ICCResult:
    """Intraclass Correlation Coefficient report across numeric missing-data policies."""

    design: str
    measurement: str
    code: str
    model: ICCModel
    agreement: ICCAgreement
    unit: ICCUnit
    complete_case: bool
    values: tuple[Any, ...]
    coders: tuple[str, ...]
    missing_policies: tuple[str, ...]
    abstain_as: str
    inapplicable_as: str
    estimates: Mapping[str, ICCEstimate]
    coverage: Coverage
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]
    provenance: Provenance | None = None
    _sample: StudySample | None = field(default=None, repr=False)

    @property
    def icc(self) -> float:
        """The reference ``as_missing`` ICC estimate."""
        return self.estimates["as_missing"].icc

    @property
    def variance_components(self) -> VarianceComponents | None:
        """G-theory variance decomposition for the reference ``as_missing`` estimate."""
        return self.estimates["as_missing"].variance_components

    def __getitem__(self, name: str) -> ICCEstimate:
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
        """Unit- or group-cluster bootstrap intervals for every ICC estimate."""
        if self._sample is None:
            raise DesignError("this ICCResult was not built by ReliabilityStudy.icc()")
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

        num_domain = _validate_numeric_domain(self.values)
        n_total_units = sample.n_units
        n_coders = sample.n_coders
        intervals: dict[str, Interval] = {}

        for name, est in self.estimates.items():
            inp = sample.inputs[name]
            y_vals = num_domain[inp.value_idx]
            z_all = sparse.csr_array(
                (np.ones(len(y_vals), dtype=float), (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()
            y_all = sparse.csr_array(
                (y_vals, (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()
            y2_all = sparse.csr_array(
                (y_vals * y_vals, (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()

            distribution = _replicate_icc_or_gtheory(
                z_all=z_all,
                y_all=y_all,
                y2_all=y2_all,
                cluster_of_unit=plan.cluster_of_unit,
                weights=weights,
                mode="icc",
                model=self.model,
                agreement=self.agreement,
                unit=self.unit,
                complete_case=self.complete_case,
            )
            if method == "percentile":
                intervals[name] = percentile_interval(
                    name, est.icc, distribution, confidence, alpha_min
                )
            else:
                ones_w = np.ones((1, n_clusters), dtype=float)
                theta_hat = float(
                    _replicate_icc_or_gtheory(
                        z_all=z_all,
                        y_all=y_all,
                        y2_all=y2_all,
                        cluster_of_unit=plan.cluster_of_unit,
                        weights=ones_w,
                        mode="icc",
                        model=self.model,
                        agreement=self.agreement,
                        unit=self.unit,
                        complete_case=self.complete_case,
                    )[0]
                )
                jack_weights = np.ones((n_clusters, n_clusters), dtype=float) - np.eye(
                    n_clusters, dtype=float
                )
                jackknife = _replicate_icc_or_gtheory(
                    z_all=z_all,
                    y_all=y_all,
                    y2_all=y2_all,
                    cluster_of_unit=plan.cluster_of_unit,
                    weights=jack_weights,
                    mode="icc",
                    model=self.model,
                    agreement=self.agreement,
                    unit=self.unit,
                    complete_case=self.complete_case,
                )
                interval = bca_interval(
                    name, est.icc, distribution, theta_hat, jackknife, confidence, alpha_min
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
            measurement=f"{self.measurement} ({self.code})",
            intervals=intervals,
            warnings=tuple(recorded),
            notes=tuple(notes),
            provenance=self.provenance,
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate."""
        return pd.DataFrame([e.summary_row() for e in self.estimates.values()])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``estimates`` and ``coverage``."""
        return {
            "estimates": self.to_frame(),
            "coverage": self.coverage.per_coder.reset_index(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                **_header("icc-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
                "design": self.design,
                "measurement": self.measurement,
                "code": self.code,
                "model": self.model,
                "agreement": self.agreement,
                "unit": self.unit,
                "complete_case": self.complete_case,
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
        """Plain-text report of ICC and its G-theory variance components."""
        c = self.coverage
        primary = self.estimates["as_missing"]
        lines = [
            f"{self.code} (model: {self.model}, {self.agreement}, {self.unit}) | "
            f"design: {self.design} | measurement: {self.measurement}",
            f"Units: {c.n_units} ({primary.n_pairable_units} used) | "
            f"Coders: {c.n_coders} | Ratings: {c.n_ratings} (density {c.coding_density:.1%})",
            "",
            f"{'estimate':<26}{self.code:>10}{'low':>9}{'high':>9}{'F':>9}"
            f"{'df1':>7}{'df2':>8}{'p':>9}",
        ]
        for e in self.estimates.values():
            lines.append(
                f"{e.name:<26}{_fmt(e.icc):>10}{_fmt(e.low):>9}{_fmt(e.high):>9}"
                f"{_fmt(e.f_value):>9}{e.df1:>7.1f}{e.df2:>8.1f}{_fmt(e.p_value):>9}"
            )
        if primary.variance_components is not None:
            vc = primary.variance_components
            lines.extend(
                [
                    "",
                    "Variance components (as_missing)",
                    f"  unit: {_fmt(vc.sigma2_unit)} ({vc.share_unit:.1%}) | "
                    f"coder: {_fmt(vc.sigma2_coder)} ({vc.share_coder:.1%}) | "
                    f"residual: {_fmt(vc.sigma2_residual)} ({vc.share_residual:.1%})",
                    f"  G-theory at k={vc.k_effective:.2f}: "
                    f"E_rho²={_fmt(vc.e_rho2)}, Phi={_fmt(vc.phi)}",
                ]
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
class GTheoryEstimate:
    """G-study variance components and D-study generalizability under one missing-data policy."""

    name: str
    policy: str
    n_units: int
    n_pairable_units: int
    n_coders: int
    n_ratings: int
    variance_components: VarianceComponents | None
    undefined_reason: str | None = None
    n_imputed: int = 0
    n_unanchored: int = 0

    @property
    def defined(self) -> bool:
        return self.undefined_reason is None and self.variance_components is not None

    @property
    def e_rho2(self) -> float:
        return np.nan if self.variance_components is None else self.variance_components.e_rho2

    @property
    def phi(self) -> float:
        return np.nan if self.variance_components is None else self.variance_components.phi

    def d_study_rows(self, k_values: Sequence[int]) -> list[dict[str, Any]]:
        vc = self.variance_components
        rows: list[dict[str, Any]] = []
        for k in k_values:
            if vc is None:
                rows.append(
                    {
                        "estimate": self.name,
                        "k": int(k),
                        "sigma2_unit": np.nan,
                        "sigma2_relative_error": np.nan,
                        "sigma2_absolute_error": np.nan,
                        "e_rho2": np.nan,
                        "phi": np.nan,
                    }
                )
            else:
                su = max(0.0, vc.sigma2_unit)
                sc = max(0.0, vc.sigma2_coder)
                se = max(0.0, vc.sigma2_residual)
                rel_err = se / k
                abs_err = (sc + se) / k
                er2 = su / (su + rel_err) if (su + rel_err) > 0 else np.nan
                phi_k = su / (su + abs_err) if (su + abs_err) > 0 else np.nan
                rows.append(
                    {
                        "estimate": self.name,
                        "k": int(k),
                        "sigma2_unit": su,
                        "sigma2_relative_error": rel_err,
                        "sigma2_absolute_error": abs_err,
                        "e_rho2": er2,
                        "phi": phi_k,
                    }
                )
        return rows

    def summary_row(self) -> dict[str, Any]:
        vc = self.variance_components
        return {
            "estimate": self.name,
            "policy": self.policy,
            "sigma2_unit": np.nan if vc is None else vc.sigma2_unit,
            "sigma2_coder": np.nan if vc is None else vc.sigma2_coder,
            "sigma2_residual": np.nan if vc is None else vc.sigma2_residual,
            "share_unit": np.nan if vc is None else vc.share_unit,
            "share_coder": np.nan if vc is None else vc.share_coder,
            "share_residual": np.nan if vc is None else vc.share_residual,
            "k_effective": np.nan if vc is None else vc.k_effective,
            "e_rho2": np.nan if vc is None else vc.e_rho2,
            "phi": np.nan if vc is None else vc.phi,
            "balanced": None if vc is None else vc.balanced,
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
                "variance_components": None
                if self.variance_components is None
                else self.variance_components.to_dict(),
            }
        )


@dataclass(frozen=True, eq=False)
class GTheoryResult:
    """Generalizability Theory (G-study and D-study) report for a declared study."""

    design: str
    measurement: str
    complete_case: bool
    k_values: tuple[int, ...]
    values: tuple[Any, ...]
    coders: tuple[str, ...]
    missing_policies: tuple[str, ...]
    abstain_as: str
    inapplicable_as: str
    estimates: Mapping[str, GTheoryEstimate]
    d_study: pd.DataFrame
    coverage: Coverage
    warnings: tuple[RecordedWarning, ...]
    notes: tuple[str, ...]
    provenance: Provenance | None = None
    _sample: StudySample | None = field(default=None, repr=False)

    @property
    def e_rho2(self) -> float:
        """The reference ``as_missing`` generalizability coefficient $E\\rho^2$."""
        return self.estimates["as_missing"].e_rho2

    @property
    def phi(self) -> float:
        """The reference ``as_missing`` dependability coefficient $\\Phi$."""
        return self.estimates["as_missing"].phi

    @property
    def variance_components(self) -> VarianceComponents | None:
        """G-study variance components for the reference ``as_missing`` estimate."""
        return self.estimates["as_missing"].variance_components

    def __getitem__(self, name: str) -> GTheoryEstimate:
        return self.estimates[name]

    def bootstrap(
        self,
        *,
        coefficient: GTheoryCoefficient = "e_rho2",
        k: int | float | None = None,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        level: Level | None = None,
        strata: bool = False,
        method: Method = "percentile",
        random_state: int | np.random.Generator | None = None,
        alpha_min: float | None = None,
        on_undefined: OnUndefined = "nan",
    ) -> BootstrapResult:
        """Unit- or group-cluster bootstrap intervals for $E\\rho^2(k)$ or $\\Phi(k)$."""
        if self._sample is None:
            raise DesignError("this GTheoryResult was not built by ReliabilityStudy.g_theory()")
        if coefficient not in GTHEORY_COEFFICIENTS:
            raise DesignError(
                f"unknown G-theory coefficient {coefficient!r}; choose from "
                f"{list(GTHEORY_COEFFICIENTS)}"
            )
        if k is not None and (isinstance(k, bool) or k <= 0 or not math.isfinite(k)):
            raise DesignError(f"k must be a positive number, got {k}")
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

        num_domain = _validate_numeric_domain(self.values)
        n_total_units = sample.n_units
        n_coders = sample.n_coders
        intervals: dict[str, Interval] = {}
        target_k = float(k) if k is not None else None

        for name, est in self.estimates.items():
            inp = sample.inputs[name]
            y_vals = num_domain[inp.value_idx]
            z_all = sparse.csr_array(
                (np.ones(len(y_vals), dtype=float), (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()
            y_all = sparse.csr_array(
                (y_vals, (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()
            y2_all = sparse.csr_array(
                (y_vals * y_vals, (inp.unit_idx, inp.coder_idx)),
                shape=(n_total_units, n_coders),
            ).toarray()

            vc = est.variance_components
            if vc is None:
                point = np.nan
            else:
                k_eval = target_k if target_k is not None else vc.k_effective
                su = max(0.0, vc.sigma2_unit)
                sc = max(0.0, vc.sigma2_coder)
                se = max(0.0, vc.sigma2_residual)
                if coefficient == "e_rho2":
                    den = su + se / k_eval
                    point = su / den if den > 0 else np.nan
                else:
                    den = su + (sc + se) / k_eval
                    point = su / den if den > 0 else np.nan

            distribution = _replicate_icc_or_gtheory(
                z_all=z_all,
                y_all=y_all,
                y2_all=y2_all,
                cluster_of_unit=plan.cluster_of_unit,
                weights=weights,
                mode=coefficient,
                complete_case=self.complete_case,
                target_k=target_k,
            )
            if method == "percentile":
                intervals[name] = percentile_interval(
                    name, point, distribution, confidence, alpha_min
                )
            else:
                ones_w = np.ones((1, n_clusters), dtype=float)
                theta_hat = float(
                    _replicate_icc_or_gtheory(
                        z_all=z_all,
                        y_all=y_all,
                        y2_all=y2_all,
                        cluster_of_unit=plan.cluster_of_unit,
                        weights=ones_w,
                        mode=coefficient,
                        complete_case=self.complete_case,
                        target_k=target_k,
                    )[0]
                )
                jack_weights = np.ones((n_clusters, n_clusters), dtype=float) - np.eye(
                    n_clusters, dtype=float
                )
                jackknife = _replicate_icc_or_gtheory(
                    z_all=z_all,
                    y_all=y_all,
                    y2_all=y2_all,
                    cluster_of_unit=plan.cluster_of_unit,
                    weights=jack_weights,
                    mode=coefficient,
                    complete_case=self.complete_case,
                    target_k=target_k,
                )
                interval = bca_interval(
                    name, point, distribution, theta_hat, jackknife, confidence, alpha_min
                )
                intervals[name] = with_fallback(interval, confidence, on_undefined, alpha_min)

        notes.append(UNITS_NOTE.format(level=level_used, within=" within strata" if strata else ""))
        if len(self.estimates) > 1:
            notes.append("All estimates use the same resamples, so their intervals are paired.")
        recorded.extend(_interval_warnings(intervals, n_resamples))

        label = "E_rho2" if coefficient == "e_rho2" else "Phi"
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
            measurement=f"{self.measurement} ({label})",
            intervals=intervals,
            warnings=tuple(recorded),
            notes=tuple(notes),
            provenance=self.provenance,
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per estimate (G-study variance components and coefficients)."""
        return pd.DataFrame([e.summary_row() for e in self.estimates.values()])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Tidy tables: ``estimates``, ``d_study``, and ``coverage``."""
        return {
            "estimates": self.to_frame(),
            "d_study": self.d_study,
            "coverage": self.coverage.per_coder.reset_index(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(  # type: ignore[no-any-return]
            {
                **_header("gtheory-result"),
                "provenance": None if self.provenance is None else self.provenance.to_dict(),
                "design": self.design,
                "measurement": self.measurement,
                "complete_case": self.complete_case,
                "k_values": list(self.k_values),
                "values": list(self.values),
                "coders": list(self.coders),
                "missing_policies": list(self.missing_policies),
                "abstain_as": self.abstain_as,
                "inapplicable_as": self.inapplicable_as,
                "coverage": self.coverage.to_dict(),
                "estimates": {k: e.to_dict() for k, e in self.estimates.items()},
                "d_study": self.d_study.to_dict(orient="records"),
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
        """Plain-text report of G-study variance components and D-study projections."""
        c = self.coverage
        primary = self.estimates["as_missing"]
        lines = [
            f"Generalizability Theory (G-study & D-study) | "
            f"design: {self.design} | measurement: {self.measurement}",
            f"Units: {c.n_units} ({primary.n_pairable_units} used) | "
            f"Coders: {c.n_coders} | Ratings: {c.n_ratings} (density {c.coding_density:.1%})",
            "",
            f"{'estimate':<26}{'var_unit':>10}{'var_coder':>10}{'var_resid':>10}"
            f"{'k_eff':>8}{'E_rho2':>9}{'Phi':>9}",
        ]
        for e in self.estimates.values():
            vc = e.variance_components
            if vc is None:
                lines.append(
                    f"{e.name:<26}{'nan':>10}{'nan':>10}{'nan':>10}{'nan':>8}{'nan':>9}{'nan':>9}"
                )
            else:
                lines.append(
                    f"{e.name:<26}{_fmt(vc.sigma2_unit):>10}{_fmt(vc.sigma2_coder):>10}"
                    f"{_fmt(vc.sigma2_residual):>10}{vc.k_effective:>8.2f}"
                    f"{_fmt(vc.e_rho2):>9}{_fmt(vc.phi):>9}"
                )
        d_ref = self.d_study[self.d_study["estimate"] == "as_missing"]
        if not d_ref.empty:
            lines.extend(
                [
                    "",
                    "D-study projections (as_missing)",
                    f"  {'k':>4}{'rel_err':>10}{'abs_err':>10}{'E_rho2':>9}{'Phi':>9}",
                ]
            )
            for row in d_ref.to_dict(orient="records"):
                lines.append(
                    f"  {int(row['k']):>4}{_fmt(float(row['sigma2_relative_error'])):>10}"
                    f"{_fmt(float(row['sigma2_absolute_error'])):>10}"
                    f"{_fmt(float(row['e_rho2'])):>9}{_fmt(float(row['phi'])):>9}"
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


def icc(
    study: ReliabilityStudy,
    *,
    model: ICCModel = "two_way_random",
    agreement: ICCAgreement = "agreement",
    unit: ICCUnit = "single",
    complete_case: bool = False,
    confidence: float = 0.95,
) -> ICCResult:
    """Compute an Intraclass Correlation Coefficient for a :class:`ReliabilityStudy`."""
    return study.icc(
        model=model,
        agreement=agreement,
        unit=unit,
        complete_case=complete_case,
        confidence=confidence,
    )


def g_theory(
    study: ReliabilityStudy,
    *,
    k: Sequence[int] | None = None,
    complete_case: bool = False,
) -> GTheoryResult:
    """Compute G-study variance components and D-study projections for a study."""
    return study.g_theory(k=k, complete_case=complete_case)
