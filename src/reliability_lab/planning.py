"""Sample-size planning for Krippendorff's alpha studies."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, sparse
from scipy.special import log_ndtr, ndtr, ndtri

from reliability_lab.alpha import Distance, per_unit_coincidences
from reliability_lab.exceptions import DesignError, UndefinedAlphaError
from reliability_lab.inference import (
    METHODS,
    EstimateInput,
    StudySample,
    _seed,
    bca_interval,
    bca_parts,
    kalpha_distribution,
    kalpha_pool,
    percentile_interval,
    replicate_alphas,
)
from reliability_lab.report import (
    _fmt,
    _fmt_any,
    _header,
    _jsonable,
    _write_json,
    _write_parquet,
)
from reliability_lab.simulate import (
    _check_count,
    _check_share,
    _LatentClassProcess,
    _prepare_process,
)

FloatArray = NDArray[np.float64]

SCHEMES: tuple[str, ...] = ("units", "kalpha")


@dataclass(frozen=True, eq=False)
class SampleSizePlan:
    """Planned sample size and the generator assumptions it holds under."""

    goal: Literal["minimum_alpha", "non_inferiority"]
    generator: Literal["krippendorff_analytical", "latent_class_simulation"]
    measurement: str
    n_units: int
    n_coders: int
    n_values: int | None
    n_clusters: int | None
    target_power: float | None
    achieved_power: float | None
    confidence: float
    significance: float
    parameters: Mapping[str, Any]
    curve: pd.DataFrame | None
    notes: tuple[str, ...]
    provenance: Mapping[str, Any]

    def to_frame(self) -> pd.DataFrame:
        """One-row summary frame of the planned design."""
        row = {
            "goal": self.goal,
            "generator": self.generator,
            "measurement": self.measurement,
            "n_units": self.n_units,
            "n_coders": self.n_coders,
            "n_values": self.n_values,
            "n_clusters": self.n_clusters,
            "target_power": self.target_power,
            "achieved_power": self.achieved_power,
            "confidence": self.confidence,
            "significance": self.significance,
        }
        return pd.DataFrame([row])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Flat tabular view of the plan and optional power curve."""
        out: dict[str, pd.DataFrame] = {"plan": self.to_frame()}
        if self.curve is not None:
            out["curve"] = self.curve.copy()
        return out

    def to_dict(self) -> dict[str, Any]:
        """Structured, JSON-safe dictionary."""
        payload = _header("plan")
        payload.update(
            {
                "goal": self.goal,
                "generator": self.generator,
                "measurement": self.measurement,
                "n_units": self.n_units,
                "n_coders": self.n_coders,
                "n_values": self.n_values,
                "n_clusters": self.n_clusters,
                "target_power": self.target_power,
                "achieved_power": self.achieved_power,
                "confidence": self.confidence,
                "significance": self.significance,
                "parameters": dict(self.parameters),
                "curve": None if self.curve is None else self.curve.to_dict(orient="records"),
                "notes": list(self.notes),
                "provenance": dict(self.provenance),
            }
        )
        return dict(_jsonable(payload))

    def to_json(self, path: str | os.PathLike[str] | None = None) -> str:
        """Serialize the plan to JSON, optionally writing it to ``path``."""
        return _write_json(self.to_dict(), path)

    def to_parquet(self, directory: str | os.PathLike[str]) -> Path:
        """Write plan tables and metadata under ``directory``."""
        return _write_parquet(self.tables(), self.to_dict(), directory)

    def summary(self) -> str:
        """Human-readable summary of the sample-size plan."""
        lines = [
            f"Sample-size plan ({self.goal})",
            f"Generator: {self.generator}",
            f"Measurement: {self.measurement}",
        ]
        size_parts = [f"Planned units: {self.n_units}", f"coders/unit: {self.n_coders}"]
        if self.n_clusters is not None and self.n_clusters != self.n_units:
            size_parts.append(f"clusters: {self.n_clusters}")
        if self.n_values is not None:
            size_parts.append(f"pairable values: {self.n_values}")
        lines.append("  " + ", ".join(size_parts))
        if self.target_power is not None:
            achieved_str = "n/a" if self.achieved_power is None else _fmt(self.achieved_power)
            lines.append(
                f"  Power: target={_fmt(self.target_power)}, "
                f"achieved={achieved_str} "
                f"(confidence={_fmt_any(self.confidence)}, "
                f"one-sided significance={_fmt_any(self.significance)})"
            )
        else:
            lines.append(
                f"  Significance: {_fmt_any(self.significance)} "
                f"(two-sided confidence={_fmt_any(self.confidence)})"
            )
        if self.parameters:
            lines.append("Parameters:")
            for key, value in self.parameters.items():
                lines.append(f"  {key}: {_fmt_any(value)}")
        if self.curve is not None and not self.curve.empty:
            lines.append("Evaluated power curve:")
            for row in self.curve.itertuples(index=False):
                n_u = int(row[0])
                n_c = int(row[1])
                emp = float(row[2])
                fit = float(row[3])
                undef = float(row[4])
                width = float(row[5])
                lines.append(
                    f"  n_units={n_u} (clusters={n_c}): "
                    f"empirical={_fmt(emp)}, "
                    f"fitted={_fmt(fit)}, "
                    f"undefined={_fmt(undef)}, "
                    f"mean_width={_fmt(width)}"
                )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def plan_krippendorff(
    *,
    alpha_min: float = 0.8,
    smallest_prevalence: float,
    significance: float = 0.05,
    n_coders: int = 2,
    missing: float = 0.0,
) -> SampleSizePlan:
    """Analytical nominal sample-size rule from Krippendorff (*Content Analysis*).

    Implements Bloch & Kraemer (1989, eq. 3.7) as tabulated in Krippendorff
    (2004, Table 11.1; 2011, *Communication Methods and Measures* 5(2):93-112,
    Table 1; 2019, Table 12.3):

    .. math::

        n_{\\text{values}} = \\left\\lceil 2 z_p^2 \\left(
            \\frac{(1 + \\alpha_{\\min})(3 - \\alpha_{\\min})}
                 {4(1 - \\alpha_{\\min}) p_c (1 - p_c)} - \\alpha_{\\min}
        \\right) \\right\\rceil

    where :math:`z_p = -\\Phi^{-1}(p)` is the upper one-sided normal quantile
    at ``significance`` :math:`p` and :math:`p_c` is ``smallest_prevalence``.
    """
    alpha_min_f = _check_unit_open(alpha_min, "alpha_min")
    p_c = _check_prevalence(smallest_prevalence)
    sig = _check_significance(significance)
    _check_count("n_coders", n_coders, minimum=2)
    coders = int(n_coders)
    _check_share("missing", missing, closed=False)
    miss = float(missing)

    z_p = -float(ndtri(sig))
    denom = 4.0 * (1.0 - alpha_min_f) * p_c * (1.0 - p_c)
    term = (1.0 + alpha_min_f) * (3.0 - alpha_min_f) / denom
    n_values = math.ceil(2.0 * (z_p**2) * (term - alpha_min_f))
    n_units = max(2, math.ceil(n_values / (coders * (1.0 - miss))))

    notes = [
        "Analytical rule derived for two-coder nominal agreement treating pairable values "
        "as independent across coders (Krippendorff, 2011, Table 1; Bloch & Kraemer, 1989)."
    ]
    if coders >= 3:
        notes.append(
            "With 3 or more coders per unit, ratings on the same unit are dependent; "
            "use plan_alpha() for cluster-aware simulation planning."
        )
    if miss > 0.0:
        notes.append(
            f"n_units={n_units} inflates required pairable values ({n_values}) by "
            f"1 / ({coders} * (1 - {miss:g}))."
        )

    return SampleSizePlan(
        goal="minimum_alpha",
        generator="krippendorff_analytical",
        measurement="nominal",
        n_units=n_units,
        n_coders=coders,
        n_values=n_values,
        n_clusters=None,
        target_power=None,
        achieved_power=None,
        confidence=1.0 - 2.0 * sig,
        significance=sig,
        parameters={
            "alpha_min": alpha_min_f,
            "smallest_prevalence": p_c,
            "missing": miss,
        },
        curve=None,
        notes=tuple(notes),
        provenance={
            "reference": "Krippendorff (2011) Table 1; Bloch & Kraemer (1989) eq. 3.7",
            "z_p": z_p,
        },
    )


def plan_alpha(
    measurement: str | Distance,
    values: Sequence[Any],
    *,
    alpha: float = 0.8,
    alpha_min: float = 0.667,
    power: float = 0.80,
    n_coders: int = 3,
    scheme: Literal["units", "kalpha"] = "units",
    method: Literal["percentile", "bca"] = "percentile",
    confidence: float = 0.95,
    errors: Literal["random", "near"] = "random",
    prevalence: Sequence[float] | None = None,
    missing: float = 0.0,
    failures: float = 0.0,
    hard_share: float = 0.0,
    hard_penalty: float = 0.25,
    cluster_size: int = 1,
    cohesion: float = 0.0,
    rho: float = 0.0,
    hard_failures: float | None = None,
    n_sim: int = 400,
    n_resamples: int = 1000,
    max_units: int = 5000,
    random_state: int | np.random.Generator | None = None,
) -> SampleSizePlan:
    """Plan unit count so a bootstrap interval's lower bound exceeds ``alpha_min``."""
    _validate_scheme_method(scheme, method)
    power_f = _check_unit_open(power, "power")
    conf_f = _check_unit_open(confidence, "confidence")
    alpha_f = _check_finite(alpha, "alpha")
    alpha_min_f = _check_finite(alpha_min, "alpha_min")
    if not (-1.0 < alpha_min_f < 1.0):
        raise DesignError(f"alpha_min must lie in (-1, 1); got {alpha_min_f}")
    if alpha_f <= alpha_min_f:
        raise DesignError(f"alpha ({alpha_f}) must exceed alpha_min ({alpha_min_f})")

    _check_count("n_sim", n_sim, minimum=10)
    _check_count("n_resamples", n_resamples, minimum=20)
    _check_count("max_units", max_units, minimum=4)
    n_sim_i = int(n_sim)
    n_res_i = int(n_resamples)
    max_units_i = int(max_units)

    proc = _prepare_process(
        measurement,
        values,
        n_coders=n_coders,
        alpha=alpha_f,
        errors=errors,
        prevalence=prevalence,
        missing=missing,
        failures=failures,
        hard_share=hard_share,
        hard_penalty=hard_penalty,
        cluster_size=cluster_size,
        cohesion=cohesion,
        rho=rho,
        hard_failures=hard_failures,
    )
    if scheme == "kalpha" and proc.cluster_size > 1:
        raise DesignError(
            "scheme='kalpha' assumes independent unit draws and does not support cluster_size > 1"
        )
    min_units, max_aligned = _unit_bounds(proc.cluster_size, max_units_i)
    base_seed = _seed(random_state)

    store = _SingleCRNStore(proc, n_sim_i, base_seed)

    def evaluator(n_units: int, sims: int) -> _GridPoint:
        return _eval_single(
            store,
            n_units=n_units,
            sims=sims,
            alpha_min=alpha_min_f,
            scheme=scheme,
            method=method,
            confidence=conf_f,
            n_resamples=n_res_i,
            base_seed=base_seed,
        )

    n_units, achieved_power, curve, notes = _search_sample_size(
        evaluator=evaluator,
        n_sim=n_sim_i,
        power=power_f,
        confidence=conf_f,
        effective_margin=alpha_f - alpha_min_f,
        cluster_size=proc.cluster_size,
        min_units=min_units,
        max_aligned=max_aligned,
        max_units=max_units_i,
    )
    if failures > 0.0 or hard_share > 0.0 or rho > 0.0:
        notes.append(
            "When failures, hard_share, or correlated errors are present, realized population "
            "alpha differs from the baseline nominal parameter according to those penalties."
        )

    q_hard = max(0.0, proc.q - proc.hard_penalty) if proc.hard_share > 0.0 else proc.q
    return SampleSizePlan(
        goal="minimum_alpha",
        generator="latent_class_simulation",
        measurement=proc.distance.name,
        n_units=n_units,
        n_coders=proc.n_coders,
        n_values=None,
        n_clusters=n_units // proc.cluster_size,
        target_power=power_f,
        achieved_power=achieved_power,
        confidence=conf_f,
        significance=(1.0 - conf_f) / 2.0,
        parameters={
            "alpha": alpha_f,
            "alpha_min": alpha_min_f,
            "scheme": scheme,
            "method": method,
            "errors": errors,
            "missing": float(missing),
            "failures": float(failures),
            "hard_share": float(hard_share),
            "hard_penalty": float(hard_penalty),
            "cluster_size": proc.cluster_size,
            "cohesion": float(cohesion),
            "rho": float(rho),
            "hard_failures": proc.hard_failures,
            "n_sim": n_sim_i,
            "n_resamples": n_res_i,
        },
        curve=curve,
        notes=tuple(notes),
        provenance={
            "seed": base_seed,
            "calibrated_easy_accuracy": proc.q,
            "calibrated_hard_accuracy": q_hard,
            "n_categories": len(proc.domain),
        },
    )


def plan_non_inferiority(
    measurement: str | Distance,
    values: Sequence[Any],
    *,
    alpha: float = 0.8,
    margin: float = 0.05,
    power: float = 0.80,
    n_coders: int = 3,
    alpha_b: float | None = None,
    method: Literal["percentile", "bca"] = "bca",
    confidence: float = 0.96,
    errors: Literal["random", "near"] = "random",
    prevalence: Sequence[float] | None = None,
    missing: float = 0.0,
    failures: float = 0.0,
    hard_share: float = 0.0,
    hard_penalty: float = 0.25,
    cluster_size: int = 1,
    cohesion: float = 0.0,
    rho: float = 0.0,
    hard_failures: float | None = None,
    n_sim: int = 400,
    n_resamples: int = 1000,
    max_units: int = 5000,
    random_state: int | np.random.Generator | None = None,
) -> SampleSizePlan:
    """Plan unit count for a paired non-inferiority comparison ``alpha_B - alpha_A > -margin``."""
    if method not in METHODS:
        raise DesignError(f"unknown bootstrap method {method!r}; expected one of {METHODS}")
    power_f = _check_unit_open(power, "power")
    conf_f = _check_unit_open(confidence, "confidence")
    alpha_f = _check_finite(alpha, "alpha")
    alpha_b_f = alpha_f if alpha_b is None else _check_finite(alpha_b, "alpha_b")
    margin_f = _check_finite(margin, "margin")
    if not (0.0 < margin_f < 2.0):
        raise DesignError(f"margin must lie in (0, 2); got {margin_f}")
    eff_margin = alpha_b_f - alpha_f + margin_f
    if eff_margin <= 0.0:
        raise DesignError(
            f"alpha_b - alpha ({alpha_b_f - alpha_f:.4f}) must exceed -margin ({-margin_f:.4f})"
        )

    _check_count("n_sim", n_sim, minimum=10)
    _check_count("n_resamples", n_resamples, minimum=20)
    _check_count("max_units", max_units, minimum=4)
    n_sim_i = int(n_sim)
    n_res_i = int(n_resamples)
    max_units_i = int(max_units)

    proc_a = _prepare_process(
        measurement,
        values,
        n_coders=n_coders,
        alpha=alpha_f,
        errors=errors,
        prevalence=prevalence,
        missing=missing,
        failures=failures,
        hard_share=hard_share,
        hard_penalty=hard_penalty,
        cluster_size=cluster_size,
        cohesion=cohesion,
        rho=rho,
        hard_failures=hard_failures,
    )
    proc_b = _prepare_process(
        measurement,
        values,
        n_coders=n_coders,
        alpha=alpha_b_f,
        errors=errors,
        prevalence=prevalence,
        missing=missing,
        failures=failures,
        hard_share=hard_share,
        hard_penalty=hard_penalty,
        cluster_size=cluster_size,
        cohesion=cohesion,
        rho=rho,
        hard_failures=hard_failures,
    )

    min_units, max_aligned = _unit_bounds(proc_a.cluster_size, max_units_i)
    base_seed = _seed(random_state)

    store = _PairedCRNStore(proc_a, proc_b, n_sim_i, base_seed)

    def evaluator(n_units: int, sims: int) -> _GridPoint:
        return _eval_paired(
            store,
            n_units=n_units,
            sims=sims,
            margin=margin_f,
            method=method,
            confidence=conf_f,
            n_resamples=n_res_i,
            base_seed=base_seed,
        )

    n_units, achieved_power, curve, notes = _search_sample_size(
        evaluator=evaluator,
        n_sim=n_sim_i,
        power=power_f,
        confidence=conf_f,
        effective_margin=eff_margin,
        cluster_size=proc_a.cluster_size,
        min_units=min_units,
        max_aligned=max_aligned,
        max_units=max_units_i,
    )
    if method == "bca" and conf_f >= 0.96:
        notes.append(
            "Uses paired unit-cluster bootstrap with BCa confidence=0.96 so a one-sided "
            "2.5% non-inferiority test avoids liberal boundary coverage."
        )

    return SampleSizePlan(
        goal="non_inferiority",
        generator="latent_class_simulation",
        measurement=proc_a.distance.name,
        n_units=n_units,
        n_coders=proc_a.n_coders,
        n_values=None,
        n_clusters=n_units // proc_a.cluster_size,
        target_power=power_f,
        achieved_power=achieved_power,
        confidence=conf_f,
        significance=(1.0 - conf_f) / 2.0,
        parameters={
            "alpha": alpha_f,
            "alpha_b": alpha_b_f,
            "margin": margin_f,
            "method": method,
            "errors": errors,
            "missing": float(missing),
            "failures": float(failures),
            "hard_share": float(hard_share),
            "hard_penalty": float(hard_penalty),
            "cluster_size": proc_a.cluster_size,
            "cohesion": float(cohesion),
            "rho": float(rho),
            "hard_failures": proc_a.hard_failures,
            "n_sim": n_sim_i,
            "n_resamples": n_res_i,
        },
        curve=curve,
        notes=tuple(notes),
        provenance={
            "seed": base_seed,
            "calibrated_easy_accuracy_a": proc_a.q,
            "calibrated_easy_accuracy_b": proc_b.q,
            "n_categories": len(proc_a.domain),
        },
    )


@dataclass
class _GridPoint:
    n_units: int
    n_clusters: int
    n_rejected: int
    n_undefined: int
    n_sims: int
    mean_width: float

    @property
    def empirical_power(self) -> float:
        return self.n_rejected / self.n_sims

    @property
    def undefined_rate(self) -> float:
        return self.n_undefined / self.n_sims


class _SingleCRNStore:
    """Lazy CRN prefix store for single-arm unit ratings."""

    def __init__(self, proc: _LatentClassProcess, n_sim: int, base_seed: int) -> None:
        self.proc = proc
        self.rngs = [np.random.default_rng([base_seed, r, 0]) for r in range(n_sim)]
        self.codes: list[NDArray[np.int16] | None] = [None] * n_sim

    def get_codes(self, rep: int, n_units: int) -> NDArray[np.int16]:
        cur = self.codes[rep]
        have = 0 if cur is None else cur.shape[0]
        if have < n_units:
            delta = n_units - have
            truth, hard, _ = self.proc.draw_structure(delta, self.rngs[rep])
            raw, kept, failed = self.proc.rate(truth, hard, self.rngs[rep])
            chunk = np.where(kept & ~failed, raw, -1).astype(np.int16, copy=False)
            cur = chunk if cur is None else np.vstack((cur, chunk))
            self.codes[rep] = cur
        assert cur is not None
        return cur[:n_units]


class _PairedCRNStore:
    """Lazy CRN prefix store for paired-arm unit ratings sharing true categories and clusters."""

    def __init__(
        self,
        proc_a: _LatentClassProcess,
        proc_b: _LatentClassProcess,
        n_sim: int,
        base_seed: int,
    ) -> None:
        self.proc_a = proc_a
        self.proc_b = proc_b
        self.rngs = [np.random.default_rng([base_seed, r, 0]) for r in range(n_sim)]
        self.codes_a: list[NDArray[np.int16] | None] = [None] * n_sim
        self.codes_b: list[NDArray[np.int16] | None] = [None] * n_sim

    def get_codes(self, rep: int, n_units: int) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        cur_a = self.codes_a[rep]
        cur_b = self.codes_b[rep]
        have = 0 if cur_a is None else cur_a.shape[0]
        if have < n_units:
            delta = n_units - have
            truth, hard, _ = self.proc_a.draw_structure(delta, self.rngs[rep])
            raw_a, kept_a, failed_a = self.proc_a.rate(truth, hard, self.rngs[rep])
            raw_b, kept_b, failed_b = self.proc_b.rate(truth, hard, self.rngs[rep])
            chunk_a = np.where(kept_a & ~failed_a, raw_a, -1).astype(np.int16, copy=False)
            chunk_b = np.where(kept_b & ~failed_b, raw_b, -1).astype(np.int16, copy=False)
            cur_a = chunk_a if cur_a is None else np.vstack((cur_a, chunk_a))
            cur_b = chunk_b if cur_b is None else np.vstack((cur_b, chunk_b))
            self.codes_a[rep] = cur_a
            self.codes_b[rep] = cur_b
        assert cur_a is not None
        assert cur_b is not None
        return cur_a[:n_units], cur_b[:n_units]


def _cluster_indicator(n_units: int, cluster_size: int) -> sparse.csr_array | None:
    if cluster_size == 1:
        return None
    n_clusters = n_units // cluster_size
    rows = np.repeat(np.arange(n_clusters, dtype=np.intp), cluster_size)
    cols = np.arange(n_units, dtype=np.intp)
    data = np.ones(n_units, dtype=np.float64)
    return sparse.csr_array((data, (rows, cols)), shape=(n_clusters, n_units))


def _coincidences_from_codes(
    codes: NDArray[np.int16],
    n_units: int,
    n_values: int,
    indicator: sparse.csr_array | None,
) -> tuple[sparse.csr_array, FloatArray]:
    mask = codes >= 0
    unit_idx, _ = np.nonzero(mask)
    value_idx = codes[mask].astype(np.intp, copy=False)
    unit_matrix, _ = per_unit_coincidences(unit_idx, value_idx, n_units, n_values)
    clusters = unit_matrix if indicator is None else indicator @ unit_matrix
    unit_totals = np.asarray(unit_matrix.sum(axis=1), dtype=np.float64)
    return clusters, unit_totals


def _eval_single(
    store: _SingleCRNStore,
    *,
    n_units: int,
    sims: int,
    alpha_min: float,
    scheme: Literal["units", "kalpha"],
    method: Literal["percentile", "bca"],
    confidence: float,
    n_resamples: int,
    base_seed: int,
) -> _GridPoint:
    proc = store.proc
    n_values = len(proc.domain)
    cluster_size = proc.cluster_size
    n_clusters = n_units // cluster_size
    indicator = _cluster_indicator(n_units, cluster_size)
    unit_weights = np.ones((1, n_clusters), dtype=np.float64)
    probs = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    unit_labels = np.arange(n_units, dtype=object)
    coder_names = np.array([f"c{i}" for i in range(proc.n_coders)], dtype=object)

    n_rejected = 0
    n_undefined = 0
    widths: list[float] = []

    for rep in range(sims):
        codes = store.get_codes(rep, n_units)
        clusters, _ = _coincidences_from_codes(codes, n_units, n_values, indicator)
        point = float(replicate_alphas(clusters, unit_weights, proc.domain, proc.distance)[0])
        if not math.isfinite(point):
            n_undefined += 1
            continue

        boot_rng = np.random.default_rng([base_seed, rep, 1, n_units])
        if scheme == "kalpha":
            mask = codes >= 0
            u_idx, c_idx = np.nonzero(mask)
            v_idx = codes[mask].astype(np.int64, copy=False)
            est_input = EstimateInput(
                unit_idx=u_idx.astype(np.int64, copy=False),
                value_idx=v_idx,
                values=proc.domain,
                coder_idx=c_idx.astype(np.int64, copy=False),
            )
            sample = StudySample(
                unit_labels=unit_labels,
                unit_groups=None,
                unit_strata=None,
                distance=proc.distance,
                inputs={"planning": est_input},
                coder_names=coder_names,
            )
            try:
                pool = kalpha_pool(sample, "planning")
            except UndefinedAlphaError:
                n_undefined += 1
                continue
            dist = kalpha_distribution(pool, boot_rng, n_resamples)
            iv = percentile_interval("planning", point, dist, confidence, alpha_min=alpha_min)
        else:
            weights = boot_rng.multinomial(n_clusters, probs, size=n_resamples).astype(np.float64)
            dist = replicate_alphas(clusters, weights, proc.domain, proc.distance)
            if method == "percentile":
                iv = percentile_interval("planning", point, dist, confidence, alpha_min=alpha_min)
            else:
                theta_hat, jack = bca_parts(clusters, proc.domain, proc.distance)
                iv = bca_interval(
                    "planning",
                    point,
                    dist,
                    theta_hat,
                    jack,
                    confidence,
                    alpha_min=alpha_min,
                )

        if not iv.defined or not math.isfinite(iv.low) or not math.isfinite(iv.high):
            n_undefined += 1
            continue
        widths.append(float(iv.high - iv.low))
        if iv.low > alpha_min:
            n_rejected += 1

    mean_width = float(np.mean(widths)) if widths else math.nan
    return _GridPoint(
        n_units=n_units,
        n_clusters=n_clusters,
        n_rejected=n_rejected,
        n_undefined=n_undefined,
        n_sims=sims,
        mean_width=mean_width,
    )


def _eval_paired(
    store: _PairedCRNStore,
    *,
    n_units: int,
    sims: int,
    margin: float,
    method: Literal["percentile", "bca"],
    confidence: float,
    n_resamples: int,
    base_seed: int,
) -> _GridPoint:
    proc = store.proc_a
    n_values = len(proc.domain)
    cluster_size = proc.cluster_size
    n_clusters = n_units // cluster_size
    indicator = _cluster_indicator(n_units, cluster_size)
    unit_weights = np.ones((1, n_clusters), dtype=np.float64)
    probs = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)

    n_rejected = 0
    n_undefined = 0
    widths: list[float] = []

    for rep in range(sims):
        codes_a, codes_b = store.get_codes(rep, n_units)
        clusters_a, _ = _coincidences_from_codes(codes_a, n_units, n_values, indicator)
        clusters_b, _ = _coincidences_from_codes(codes_b, n_units, n_values, indicator)

        point_a = float(replicate_alphas(clusters_a, unit_weights, proc.domain, proc.distance)[0])
        point_b = float(replicate_alphas(clusters_b, unit_weights, proc.domain, proc.distance)[0])
        point = point_b - point_a
        if not math.isfinite(point):
            n_undefined += 1
            continue

        boot_rng = np.random.default_rng([base_seed, rep, 1, n_units])
        weights = boot_rng.multinomial(n_clusters, probs, size=n_resamples).astype(np.float64)
        dist_a = replicate_alphas(clusters_a, weights, proc.domain, proc.distance)
        dist_b = replicate_alphas(clusters_b, weights, proc.domain, proc.distance)
        dist = dist_b - dist_a

        if method == "percentile":
            iv = percentile_interval("diff", point, dist, confidence)
        else:
            theta_a, jack_a = bca_parts(clusters_a, proc.domain, proc.distance)
            theta_b, jack_b = bca_parts(clusters_b, proc.domain, proc.distance)
            iv = bca_interval("diff", point, dist, theta_b - theta_a, jack_b - jack_a, confidence)

        if not iv.defined or not math.isfinite(iv.low) or not math.isfinite(iv.high):
            n_undefined += 1
            continue
        widths.append(float(iv.high - iv.low))
        if iv.low > -margin:
            n_rejected += 1

    mean_width = float(np.mean(widths)) if widths else math.nan
    return _GridPoint(
        n_units=n_units,
        n_clusters=n_clusters,
        n_rejected=n_rejected,
        n_undefined=n_undefined,
        n_sims=sims,
        mean_width=mean_width,
    )


def _search_sample_size(
    *,
    evaluator: Callable[[int, int], _GridPoint],
    n_sim: int,
    power: float,
    confidence: float,
    effective_margin: float,
    cluster_size: int,
    min_units: int,
    max_aligned: int,
    max_units: int,
) -> tuple[int, float, pd.DataFrame, list[str]]:
    notes: list[str] = []
    cache: dict[int, _GridPoint] = {}

    def eval_cached(n_u: int) -> _GridPoint:
        if n_u not in cache:
            cache[n_u] = evaluator(n_u, n_sim)
        return cache[n_u]

    # 1. Pilot guess based on initial interval half-width
    pilot_n = _align_up(
        max(min_units, min(80, max_aligned // 2)), cluster_size, min_units, max_aligned
    )
    pilot = evaluator(pilot_n, min(n_sim, 100))
    z_tail = -float(ndtri((1.0 - confidence) / 2.0))
    z_power = max(0.0, float(ndtri(power)))
    if math.isfinite(pilot.mean_width) and pilot.mean_width > 0.0:
        hw = 0.5 * pilot.mean_width
        scale = (hw * (1.0 + z_power / z_tail) / effective_margin) ** 2
        n_guess = _align_up(math.ceil(pilot_n * scale), cluster_size, min_units, max_aligned)
    else:
        n_guess = _align_up(
            max(min_units, min(200, max_aligned)), cluster_size, min_units, max_aligned
        )

    # 2. Dynamic bracketing so cache spans `power`
    lo = n_guess
    hi = n_guess
    pt_guess = eval_cached(n_guess)
    if pt_guess.empirical_power < power:
        while eval_cached(hi).empirical_power < power and hi < max_aligned:
            lo = hi
            next_hi = max(hi + cluster_size, math.ceil(hi * 1.8))
            hi = _align_up(next_hi, cluster_size, min_units, max_aligned)
    else:
        while lo > min_units and eval_cached(lo).empirical_power >= power:
            hi = lo
            next_lo = _align_down(int(lo * 0.55), cluster_size, min_units, max_aligned)
            lo = next_lo if next_lo < lo else max(min_units, lo - cluster_size)
            eval_cached(lo)

    if eval_cached(hi).empirical_power < power:
        hi = max_aligned
        max_pt = eval_cached(hi)
        if max_pt.empirical_power < power:
            raise DesignError(
                f"Target power {power:.3f} not reached by max_units={max_units} "
                f"(empirical power={max_pt.empirical_power:.3f}); "
                "increase max_units or relax alpha_min/margin."
            )

    # 3. Evaluate dense grid in sqrt(N) across bracket [lo, hi]
    if hi > lo:
        sqrt_grid = np.linspace(math.sqrt(lo), math.sqrt(hi), 7)
        for val in sqrt_grid:
            cand = _align_nearest(round(float(val) ** 2), cluster_size, min_units, max_aligned)
            eval_cached(cand)
    if len(cache) < 5 and max_aligned > hi:
        extra_hi = _align_up(
            min(max_aligned, math.ceil(hi * 1.6)), cluster_size, min_units, max_aligned
        )
        for val in np.linspace(math.sqrt(min_units), math.sqrt(extra_hi), 6):
            cand = _align_nearest(round(float(val) ** 2), cluster_size, min_units, max_aligned)
            eval_cached(cand)

    # 4. Fit probit + PAVA on collected grid, predict crossing, and confirm
    def fit_cache() -> tuple[
        NDArray[np.intp],
        FloatArray,
        FloatArray,
        tuple[float, float] | None,
    ]:
        pts = [cache[k] for k in sorted(cache)]
        ns = np.array([p.n_units for p in pts], dtype=np.intp)
        k_arr = np.array([p.n_rejected for p in pts], dtype=np.float64)
        emp = k_arr / n_sim
        pava = _pava_isotonic(emp)
        ab = _fit_probit(ns.astype(np.float64), k_arr, n_sim)
        if ab is not None:
            a, b = ab
            fitted = np.asarray(ndtr(a + b * np.sqrt(ns)), dtype=np.float64)
        else:
            fitted = pava
        return ns, emp, fitted, ab

    ns, emp, fitted, ab = fit_cache()
    if ab is not None:
        a, b = ab
        n_star = ((float(ndtri(power)) - a) / b) ** 2
        n_cand = _align_up(
            math.ceil(max(float(min_units), n_star)),
            cluster_size,
            min_units,
            max_aligned,
        )
    else:
        idx = int(np.searchsorted(fitted, power, side="left"))
        n_cand = int(ns[min(idx, len(ns) - 1)])

    eval_cached(n_cand)

    # Confirm: step upward if empirical or fitted power at n_cand has not reached target power yet
    chosen = n_cand
    for _ in range(25):
        ns, emp, fitted, ab = fit_cache()
        qualifying = ns[(emp >= power) & (fitted >= power)]
        if len(qualifying) > 0:
            # Prefer a candidate close to n_cand if qualifying[0] is just above n_cand, or refine
            chosen = int(qualifying[0])
            break
        if chosen >= max_aligned:
            break
        step = max(
            cluster_size,
            _align_up(
                max(cluster_size, int(0.03 * chosen)),
                cluster_size,
                cluster_size,
                max_aligned,
            ),
        )
        next_n = min(max_aligned, chosen + step)
        if next_n == chosen:
            break
        chosen = next_n
        eval_cached(chosen)

    ns, emp, fitted, ab = fit_cache()
    qualifying = ns[(emp >= power) & (fitted >= power)]
    if len(qualifying) > 0:
        chosen = int(qualifying[0])
    else:
        emp_ok = ns[emp >= power]
        if len(emp_ok) == 0:
            raise DesignError(
                f"Target power {power:.3f} not confirmed by max_units={max_units}; "
                "increase max_units or n_sim."
            )
        chosen = int(emp_ok[0])

    pts = [cache[k] for k in sorted(cache)]
    ns_f = np.array([p.n_units for p in pts], dtype=np.float64)
    k_f = np.array([p.n_rejected for p in pts], dtype=np.float64)
    pava_f = _pava_isotonic(k_f / n_sim)
    ab_f = _fit_probit(ns_f, k_f, n_sim)
    fitted_f = (
        np.asarray(ndtr(ab_f[0] + ab_f[1] * np.sqrt(ns_f)), dtype=np.float64) if ab_f else pava_f
    )

    curve = pd.DataFrame(
        {
            "n_units": [p.n_units for p in pts],
            "n_clusters": [p.n_clusters for p in pts],
            "empirical_power": [p.empirical_power for p in pts],
            "fitted_power": [float(v) for v in fitted_f],
            "undefined_rate": [p.undefined_rate for p in pts],
            "mean_width": [p.mean_width for p in pts],
        }
    )
    chosen_row = curve.loc[curve["n_units"] == chosen].iloc[0]
    achieved = float(chosen_row["fitted_power"])
    return chosen, achieved, curve, notes


def _pava_isotonic(y: FloatArray) -> FloatArray:
    """Pool Adjacent Violators Algorithm (non-decreasing)."""
    blocks: list[list[float]] = []
    for val in y:
        blocks.append([float(val), 1.0])
        while len(blocks) >= 2:
            m_prev = blocks[-2][0] / blocks[-2][1]
            m_curr = blocks[-1][0] / blocks[-1][1]
            if m_prev <= m_curr:
                break
            b_last = blocks.pop()
            blocks[-1][0] += b_last[0]
            blocks[-1][1] += b_last[1]
    out: list[float] = []
    for s, w in blocks:
        out.extend([s / w] * int(w))
    return np.asarray(out, dtype=np.float64)


def _fit_probit(n_arr: FloatArray, k_arr: FloatArray, n_sim: int) -> tuple[float, float] | None:
    """Fit monotone binomial probit ``Phi(a + b * sqrt(N))`` with ``b > 0``."""
    if len(n_arr) < 2:
        return None
    if np.all(k_arr == n_sim) or np.all(k_arr == 0.0):
        return None
    x = np.sqrt(n_arr)
    props = np.clip((k_arr + 0.5) / (n_sim + 1.0), 1e-4, 1.0 - 1e-4)
    y_z = np.asarray(ndtri(props), dtype=np.float64)
    x_var = float(np.var(x))
    if x_var <= 0.0:
        return None
    b0 = max(float(np.cov(x, y_z, bias=True)[0, 1] / x_var), 1e-3)
    a0 = float(np.mean(y_z) - b0 * np.mean(x))

    def nll(params: FloatArray) -> float:
        a, b = float(params[0]), float(params[1])
        z = a + b * x
        return -float(np.sum(k_arr * log_ndtr(z) + (n_sim - k_arr) * log_ndtr(-z)))

    res = optimize.minimize(
        nll,
        x0=np.array([a0, b0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(None, None), (1e-6, None)],
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    return float(res.x[0]), float(res.x[1])


def _validate_scheme_method(scheme: str, method: str) -> None:
    if scheme not in SCHEMES:
        raise DesignError(f"unknown bootstrap scheme {scheme!r}; expected one of {SCHEMES}")
    if method not in METHODS:
        raise DesignError(f"unknown bootstrap method {method!r}; expected one of {METHODS}")
    if scheme == "kalpha" and method != "percentile":
        raise DesignError("scheme='kalpha' only supports method='percentile'")


def _check_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignError(f"{name} must be a real number; got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise DesignError(f"{name} must be finite; got {value!r}")
    return out


def _check_unit_open(value: float, name: str) -> float:
    out = _check_finite(value, name)
    if not (0.0 < out < 1.0):
        raise DesignError(f"{name} must lie strictly between 0 and 1; got {out}")
    return out


def _check_prevalence(value: float) -> float:
    out = _check_finite(value, "smallest_prevalence")
    if not (0.0 < out <= 0.5):
        raise DesignError(f"smallest_prevalence must lie in (0, 0.5]; got {out}")
    return out


def _check_significance(value: float) -> float:
    out = _check_finite(value, "significance")
    if not (0.0 < out < 0.5):
        raise DesignError(f"significance must lie strictly between 0 and 0.5; got {out}")
    return out


def _unit_bounds(cluster_size: int, max_units: int) -> tuple[int, int]:
    raw_min = max(4, 2 * cluster_size)
    min_units = math.ceil(raw_min / cluster_size) * cluster_size
    max_aligned = (max_units // cluster_size) * cluster_size
    if max_aligned < min_units:
        raise DesignError(
            f"max_units={max_units} is too small for cluster_size={cluster_size} "
            f"(need at least {min_units} units)"
        )
    return min_units, max_aligned


def _align_up(value: int, cluster_size: int, min_units: int, max_aligned: int) -> int:
    aligned = math.ceil(value / cluster_size) * cluster_size
    return min(max_aligned, max(min_units, aligned))


def _align_down(value: int, cluster_size: int, min_units: int, max_aligned: int) -> int:
    aligned = (value // cluster_size) * cluster_size
    return min(max_aligned, max(min_units, aligned))


def _align_nearest(value: int, cluster_size: int, min_units: int, max_aligned: int) -> int:
    aligned = round(value / cluster_size) * cluster_size
    return min(max_aligned, max(min_units, aligned))


__all__ = [
    "SampleSizePlan",
    "plan_alpha",
    "plan_krippendorff",
    "plan_non_inferiority",
]
