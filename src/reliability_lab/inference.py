"""Bootstrap inference for alpha: unit/cluster resampling, KALPHA and paired comparisons.

The caller always names the scheme:

* ``units`` resamples whole units, or whole groups, optionally within strata,
  and recomputes everything in every replicate, D_e included. The coincidence
  matrix is a sum of per-unit contributions, so a replicate is a weighted sum of
  per-cluster contributions: ``o = w @ P``.
* ``kalpha`` follows Krippendorff's algorithm note (2006, revised 2016). It
  resamples pairs of values from the observed within-unit pairs, keeps each unit's
  number of pairs and their 1/(m_u - 1) weight, and holds D_e (and delta²) at the
  observed values.

Paired comparisons apply one set of resampled units or groups to both
configurations, so the interval falls on the difference alpha_b - alpha_a.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse, stats

from reliability_lab.alpha import ORDINAL, Distance, compute_alpha, per_unit_coincidences
from reliability_lab.data import preview
from reliability_lab.exceptions import BootstrapWarning, DesignError
from reliability_lab.report import (
    AlphaResult,
    BootstrapResult,
    ComparisonResult,
    Interval,
    RecordedWarning,
    emit,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SCHEMES: tuple[str, ...] = ("units", "kalpha")
METHODS: tuple[str, ...] = ("percentile", "bca")
LEVELS: tuple[str, ...] = ("unit", "group")
#: Upper bound on the floats held per batch of replicates (bounds memory use).
BATCH_FLOATS = 10_000_000

KALPHA_NOTE = (
    "KALPHA scheme (Krippendorff's bootstrap algorithm): pairs of values are resampled "
    "from the observed within-unit pairs, with D_e held at its observed value and "
    "replicates below -1 set to -1. A unit's second pick is redrawn when it repeats the "
    "first; that reading of the algorithm note follows evalica's implementation."
)
UNITS_NOTE = (
    "Units scheme: whole {level}s are resampled with replacement{within}, and D_e is "
    "recomputed in every replicate."
)


@dataclass(frozen=True, eq=False)
class EstimateInput:
    """The coded pairable values behind one estimate: one entry per rating."""

    unit_idx: IntArray
    value_idx: IntArray
    values: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class StudySample:
    """What resampling needs from a study: its units, their clusters, and the coded inputs.

    ``unit_groups`` and ``unit_strata`` hold each unit's group and stratum label, or
    None when the field was not declared. ``inputs`` is keyed by estimate name.
    """

    unit_labels: npt.NDArray[np.object_]
    unit_groups: npt.NDArray[np.object_] | None
    unit_strata: npt.NDArray[np.object_] | None
    distance: Distance
    inputs: Mapping[str, EstimateInput]

    @property
    def n_units(self) -> int:
        return len(self.unit_labels)


# -- units scheme -------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ClusterPlan:
    """Which resampling cluster each unit belongs to, and each cluster's stratum."""

    level: str
    labels: pd.Index
    cluster_of_unit: IntArray
    stratum_of_cluster: IntArray | None

    @property
    def n_clusters(self) -> int:
        return len(self.labels)


def cluster_plan(
    sample: StudySample, level: str, strata: bool, universe: pd.Index | None = None
) -> ClusterPlan:
    """Map units to clusters (themselves, or their groups), and clusters to strata.

    ``universe`` fixes the cluster order, so that two arms of a comparison share
    cluster indices.
    """
    if level == "group":
        if sample.unit_groups is None:
            raise DesignError("level='group' needs group_id declared in RatingFrame.from_long")
        labels = sample.unit_groups
    else:
        labels = sample.unit_labels
    if universe is None:
        codes, uniques = pd.factorize(pd.Series(labels, dtype=object), sort=False)
        universe = pd.Index(uniques, dtype=object)
        cluster_of_unit = np.asarray(codes, dtype=np.intp)
    else:
        cluster_of_unit = np.asarray(universe.get_indexer(pd.Index(labels, dtype=object)))
        if (cluster_of_unit < 0).any():
            raise DesignError("both configurations must share the same units and groups")

    stratum_of_cluster = None
    if strata:
        if sample.unit_strata is None:
            raise DesignError(
                "strata=True needs a stratum column declared in RatingFrame.from_long"
            )
        pairs = pd.DataFrame({"cluster": cluster_of_unit, "stratum": sample.unit_strata})
        spread = pairs.groupby("cluster")["stratum"].nunique()
        mixed = spread.index[spread > 1]
        if len(mixed):
            raise DesignError(
                f"{len(mixed)} groups span several strata {preview(universe[mixed])}; resample "
                "within strata at unit level, or give every group a single stratum"
            )
        first = pairs.drop_duplicates("cluster").sort_values("cluster")
        codes, _ = pd.factorize(first["stratum"], sort=False)
        stratum_of_cluster = np.asarray(codes, dtype=np.intp)
    return ClusterPlan(level, universe, cluster_of_unit, stratum_of_cluster)


def draw_weights(rng: np.random.Generator, plan: ClusterPlan, n: int) -> npt.NDArray[np.int64]:
    """How often each cluster is drawn in each of ``n`` resamples (n x clusters).

    Clusters are drawn with replacement, as many as there are, or within each
    stratum as many as the stratum holds.
    """
    k = plan.n_clusters
    if plan.stratum_of_cluster is None:
        return rng.multinomial(k, np.full(k, 1.0 / k), size=n)
    weights = np.zeros((n, k), dtype=np.int64)
    for stratum in np.unique(plan.stratum_of_cluster):
        members = np.flatnonzero(plan.stratum_of_cluster == stratum)
        size = len(members)
        weights[:, members] = rng.multinomial(size, np.full(size, 1.0 / size), size=n)
    return weights


def cluster_coincidences(sample: StudySample, name: str, plan: ClusterPlan) -> sparse.csr_array:
    """Per-cluster coincidence contributions (clusters x V², flattened V x V)."""
    inp = sample.inputs[name]
    per_unit, _ = per_unit_coincidences(
        inp.unit_idx, inp.value_idx, sample.n_units, len(inp.values)
    )
    n_units = sample.n_units
    indicator = sparse.csr_array(
        (np.ones(n_units), (plan.cluster_of_unit, np.arange(n_units))),
        shape=(plan.n_clusters, n_units),
    )
    return sparse.csr_array(indicator @ per_unit)


def replicate_alphas(
    clusters: sparse.csr_array,
    weights: npt.ArrayLike,
    values: tuple[Any, ...],
    distance: Distance,
) -> FloatArray:
    """Alpha for each row of cluster weights; NaN where alpha is undefined."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    n_values = len(values)
    fixed = None if distance.depends_on_marginals else distance.matrix(values, np.ones(n_values))
    out = np.empty(len(w))
    batch = max(1, BATCH_FLOATS // max(n_values * n_values, 1))
    for start in range(0, len(w), batch):
        wb = w[start : start + batch]
        o = np.asarray(clusters.T @ wb.T, dtype=float).T.reshape(len(wb), n_values, n_values)
        n_c = o.sum(axis=2)
        n = n_c.sum(axis=1)
        if fixed is not None:
            observed = np.einsum("bij,ij->b", o, fixed)
            expected = np.einsum("bi,ij,bj->b", n_c, fixed, n_c)
        else:
            delta2 = _batch_delta2(distance, values, n_c)
            observed = np.einsum("bij,bij->b", o, delta2)
            expected = np.einsum("bi,bij,bj->b", n_c, delta2, n_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_o = observed / n
            d_e = expected / (n * (n - 1))
            alpha = 1.0 - d_o / d_e
        alpha[(n < 2) | ~(d_e > 0)] = np.nan
        out[start : start + len(wb)] = alpha
    return out


def _batch_delta2(distance: Distance, values: tuple[Any, ...], n_c: FloatArray) -> FloatArray:
    """delta² for each replicate's marginals (batch x V x V)."""
    if distance is ORDINAL:
        r = np.cumsum(n_c, axis=1) - n_c / 2.0
        return np.asarray((r[:, :, None] - r[:, None, :]) ** 2, dtype=float)
    return np.stack([distance.matrix(values, row) for row in n_c])


# -- KALPHA ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class KalphaPool:
    """The observed within-unit pairs and the draw structure of the KALPHA algorithm.

    ``errors[r]`` is E(r) = 2·delta²_r / (n·D_e) for the r-th unordered pair, with
    pairs ordered by unit. Each unit draws as many pairs as it holds, so draw
    positions follow the same order. ``weights`` holds 1/(m_u - 1) per position,
    and ``second_pick`` the position of each unit's second draw.
    """

    point: float
    errors: FloatArray
    weights: FloatArray
    second_pick: IntArray
    undefined_reason: str | None = None


def kalpha_pool(sample: StudySample, name: str) -> KalphaPool:
    inp = sample.inputs[name]
    computed = compute_alpha(
        inp.unit_idx, inp.value_idx, sample.n_units, inp.values, sample.distance
    )
    c = computed.components
    empty_f = np.empty(0)
    empty_i = np.empty(0, dtype=np.intp)
    if c.undefined:
        return KalphaPool(c.alpha, empty_f, empty_f, empty_i, c.undefined_reason)
    if c.d_o == 0:
        return KalphaPool(
            c.alpha,
            empty_f,
            empty_f,
            empty_i,
            "bootstrapping does not apply when alpha = 1 (no observed disagreement)",
        )

    order = np.argsort(inp.unit_idx, kind="stable")
    units = inp.unit_idx[order]
    codes = inp.value_idx[order]
    m = np.bincount(units, minlength=sample.n_units)
    starts = np.concatenate([[0], np.cumsum(m)[:-1]])
    left_parts, right_parts, unit_parts = [], [], []
    for size in np.unique(m[m >= 2]):
        members = np.flatnonzero(m == size)
        i, j = np.triu_indices(size, k=1)
        base = starts[members][:, None]
        left_parts.append((base + i).ravel())
        right_parts.append((base + j).ravel())
        unit_parts.append(np.repeat(members, len(i)))
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    unit = np.concatenate(unit_parts)
    by_unit = np.lexsort((right, left, unit))
    left, right, unit = left[by_unit], right[by_unit], unit[by_unit]

    errors = 2.0 * c.delta2[codes[left], codes[right]] / (c.n * c.d_e)
    weights = 1.0 / (m[unit] - 1)
    block_start = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
    block_size = np.diff(np.r_[block_start, len(unit)])
    second_pick = (block_start[block_size >= 2] + 1).astype(np.intp)
    return KalphaPool(c.alpha, errors, weights, second_pick)


def kalpha_replicates(pool: KalphaPool, draws: IntArray, redraws: IntArray) -> FloatArray:
    """Alpha replicates from pair draws (replicates x positions).

    ``redraws`` (replicates x len(second_pick)) replaces a unit's second draw when
    it repeats the first.
    """
    d = np.array(draws, dtype=np.intp)
    if len(pool.second_pick):
        first = d[:, pool.second_pick - 1]
        second = d[:, pool.second_pick]
        d[:, pool.second_pick] = np.where(first == second, redraws, second)
    alpha = 1.0 - (pool.errors[d] * pool.weights).sum(axis=1)
    return np.asarray(np.maximum(alpha, -1.0), dtype=float)


def kalpha_distribution(pool: KalphaPool, rng: np.random.Generator, n: int) -> FloatArray:
    n_pairs = len(pool.errors)
    batch = max(1, BATCH_FLOATS // max(n_pairs, 1))
    parts = []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        draws = rng.integers(0, n_pairs, size=(size, n_pairs))
        redraws = rng.integers(0, n_pairs, size=(size, len(pool.second_pick)))
        parts.append(kalpha_replicates(pool, draws, redraws))
    return np.concatenate(parts)


# -- intervals ------------------------------------------------------------------------


def percentile_interval(
    name: str,
    point: float,
    distribution: FloatArray,
    confidence: float,
    alpha_min: float | None = None,
) -> Interval:
    """Percentile interval over the defined replicates; failures are counted."""
    failed = np.isnan(distribution)
    defined = distribution[~failed]
    if not len(defined):
        return Interval(
            name,
            point,
            np.nan,
            np.nan,
            len(distribution),
            int(failed.sum()),
            undefined_reason="every replicate was undefined",
            distribution=distribution,
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(defined, [tail, 1.0 - tail])
    below = float((defined < alpha_min).mean()) if alpha_min is not None else None
    return Interval(
        name,
        point,
        float(low),
        float(high),
        len(distribution),
        int(failed.sum()),
        below,
        distribution=distribution,
    )


def bca_interval(
    name: str,
    point: float,
    statistic: Callable[[IntArray], FloatArray],
    n_clusters: int,
    n_resamples: int,
    confidence: float,
    seed: int,
    alpha_min: float | None = None,
) -> Interval:
    """BCa interval through ``scipy.stats.bootstrap`` over resampled cluster indices."""

    def vectorised(idx: IntArray, axis: int = -1) -> FloatArray:
        return statistic(np.moveaxis(np.asarray(idx, dtype=np.intp), axis, -1))

    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.bootstrap(
            (np.arange(n_clusters),),
            vectorised,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="BCa",
            vectorized=True,
            rng=np.random.default_rng(seed),
        )
    distribution = np.asarray(res.bootstrap_distribution, dtype=float)
    failed = np.isnan(distribution)
    low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
    reason = None
    if not (np.isfinite(low) and np.isfinite(high)):
        reason = "the BCa interval is undefined (undefined replicates or a degenerate jackknife)"
    defined = distribution[~failed]
    below = float((defined < alpha_min).mean()) if alpha_min is not None and len(defined) else None
    return Interval(
        name, point, low, high, n_resamples, int(failed.sum()), below, reason, distribution
    )


def weights_from_indices(idx: IntArray, n_clusters: int) -> npt.NDArray[np.int64]:
    """Count cluster indices along the last axis: (..., k) -> (prod(...), n_clusters)."""
    flat = np.asarray(idx, dtype=np.intp).reshape(-1, np.shape(idx)[-1])
    offsets = (np.arange(len(flat)) * n_clusters)[:, None]
    counts = np.bincount((flat + offsets).ravel(), minlength=len(flat) * n_clusters)
    return counts.reshape(len(flat), n_clusters)


def _alpha_statistic(
    clusters: sparse.csr_array, n_clusters: int, values: tuple[Any, ...], distance: Distance
) -> Callable[[IntArray], FloatArray]:
    def statistic(idx: IntArray) -> FloatArray:
        lead = np.shape(idx)[:-1]
        w = weights_from_indices(idx, n_clusters)
        return replicate_alphas(clusters, w, values, distance).reshape(lead)

    return statistic


# -- entry points ---------------------------------------------------------------------


def bootstrap(
    result: AlphaResult,
    sample: StudySample,
    *,
    scheme: str,
    n_resamples: int,
    confidence: float,
    level: str | None,
    strata: bool,
    method: str,
    alpha_min: float | None,
    random_state: int | np.random.Generator | None,
) -> BootstrapResult:
    """Intervals for every estimate of ``result``; see :meth:`AlphaResult.bootstrap`."""
    _check_options(n_resamples, confidence, method, alpha_min)
    if scheme not in SCHEMES:
        raise DesignError(f"scheme must be one of {list(SCHEMES)}; got {scheme!r}")
    seed = _seed(random_state)
    recorded: list[RecordedWarning] = []
    notes: list[str] = []
    names = list(sample.inputs)
    points = {name: result.estimates[name].alpha for name in names}
    intervals: dict[str, Interval] = {}

    if scheme == "kalpha":
        if level is not None or strata:
            raise DesignError(
                "the kalpha scheme resamples pairs of values, not units or groups; leave "
                "level and strata unset, or use scheme='units'"
            )
        if method == "bca":
            raise DesignError("BCa intervals are available with scheme='units' only")
        if sample.unit_groups is not None:
            recorded.append(
                emit(
                    BootstrapWarning,
                    "groups are declared but the kalpha scheme ignores them: it treats pairs "
                    "of values as independent, so intervals may be too narrow. "
                    "scheme='units' resamples whole groups.",
                    stacklevel=3,
                )
            )
        for name in names:
            pool = kalpha_pool(sample, name)
            if pool.undefined_reason:
                intervals[name] = Interval(
                    name,
                    points[name],
                    np.nan,
                    np.nan,
                    n_resamples,
                    0,
                    undefined_reason=pool.undefined_reason,
                )
                continue
            distribution = kalpha_distribution(pool, np.random.default_rng(seed), n_resamples)
            intervals[name] = percentile_interval(
                name, points[name], distribution, confidence, alpha_min
            )
        notes.append(KALPHA_NOTE)
        level_used, n_clusters = None, None
    else:
        level_used, level_warning = _resolve_level(sample, level)
        if level_warning:
            recorded.append(emit(BootstrapWarning, level_warning, stacklevel=3))
        if method == "bca" and strata:
            raise DesignError(
                "BCa intervals cannot resample within strata (scipy.stats.bootstrap has no "
                "strata); use method='percentile'"
            )
        plan = cluster_plan(sample, level_used, strata)
        n_clusters = plan.n_clusters
        weights = (
            draw_weights(np.random.default_rng(seed), plan, n_resamples)
            if method == "percentile"
            else None
        )
        for name in names:
            values = sample.inputs[name].values
            clusters = cluster_coincidences(sample, name, plan)
            if weights is not None:
                distribution = replicate_alphas(clusters, weights, values, sample.distance)
                intervals[name] = percentile_interval(
                    name, points[name], distribution, confidence, alpha_min
                )
            else:
                statistic = _alpha_statistic(clusters, n_clusters, values, sample.distance)
                intervals[name] = bca_interval(
                    name,
                    points[name],
                    statistic,
                    n_clusters,
                    n_resamples,
                    confidence,
                    seed,
                    alpha_min,
                )
        notes.append(UNITS_NOTE.format(level=level_used, within=" within strata" if strata else ""))
        if len(names) > 1:
            notes.append("All estimates use the same resamples, so their intervals are paired.")

    if method == "bca":
        recorded.append(_bca_warning())
    recorded.extend(_failure_warnings(intervals, n_resamples))
    return BootstrapResult(
        scheme=scheme,
        level=level_used,
        strata=strata,
        method=method,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        alpha_min=alpha_min,
        experimental=method == "bca",
        n_clusters=n_clusters,
        design=result.design,
        measurement=result.measurement,
        intervals=intervals,
        warnings=tuple(recorded),
        notes=tuple(notes),
    )


def compare(
    result_a: AlphaResult,
    result_b: AlphaResult,
    *,
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    n_resamples: int,
    confidence: float,
    level: str | None,
    strata: bool,
    method: str,
    random_state: int | np.random.Generator | None,
) -> ComparisonResult:
    """Paired intervals on alpha_b - alpha_a; see :meth:`ReliabilityStudy.compare`."""
    _check_options(n_resamples, confidence, method, None)
    sample_a, sample_b = result_a._sample, result_b._sample
    if sample_a is None or sample_b is None:
        raise DesignError("both arms must be results of ReliabilityStudy.alpha()")
    names = list(sample_a.inputs)
    if names != list(sample_b.inputs):
        raise DesignError("both arms must report the same estimates")
    seed = _seed(random_state)
    recorded = [
        type(w)(w.category, f"arm {arm}: {w.message}")
        for arm, res in (("a", result_a), ("b", result_b))
        for w in res.warnings
    ]
    level_used, level_warning = _resolve_level(sample_a, level)
    if level_warning:
        recorded.append(emit(BootstrapWarning, level_warning, stacklevel=3))
    if method == "bca" and strata:
        raise DesignError("BCa intervals cannot resample within strata; use method='percentile'")
    plan_a = cluster_plan(sample_a, level_used, strata)
    plan_b = cluster_plan(sample_b, level_used, False, universe=plan_a.labels)
    weights = (
        draw_weights(np.random.default_rng(seed), plan_a, n_resamples)
        if method == "percentile"
        else None
    )

    arm_intervals: dict[str, dict[str, Interval]] = {"a": {}, "b": {}}
    differences: dict[str, Interval] = {}
    for name in names:
        point_a = result_a.estimates[name].alpha
        point_b = result_b.estimates[name].alpha
        clusters_a = cluster_coincidences(sample_a, name, plan_a)
        clusters_b = cluster_coincidences(sample_b, name, plan_b)
        values_a, values_b = sample_a.inputs[name].values, sample_b.inputs[name].values
        if weights is not None:
            dist_a = replicate_alphas(clusters_a, weights, values_a, sample_a.distance)
            dist_b = replicate_alphas(clusters_b, weights, values_b, sample_b.distance)
            arm_intervals["a"][name] = percentile_interval(name, point_a, dist_a, confidence)
            arm_intervals["b"][name] = percentile_interval(name, point_b, dist_b, confidence)
            differences[name] = percentile_interval(
                name, point_b - point_a, dist_b - dist_a, confidence
            )
            continue
        k = plan_a.n_clusters
        stat_a = _alpha_statistic(clusters_a, k, values_a, sample_a.distance)
        stat_b = _alpha_statistic(clusters_b, k, values_b, sample_b.distance)

        def stat_diff(
            idx: IntArray,
            stat_a: Callable[[IntArray], FloatArray] = stat_a,
            stat_b: Callable[[IntArray], FloatArray] = stat_b,
        ) -> FloatArray:
            return stat_b(idx) - stat_a(idx)

        args = (k, n_resamples, confidence, seed)
        arm_intervals["a"][name] = bca_interval(name, point_a, stat_a, *args)
        arm_intervals["b"][name] = bca_interval(name, point_b, stat_b, *args)
        differences[name] = bca_interval(name, point_b - point_a, stat_diff, *args)

    if method == "bca":
        recorded.append(_bca_warning())
    recorded.extend(_failure_warnings(differences, n_resamples))
    within = " within strata" if strata else ""
    notes = (
        "Both configurations are evaluated on the same resampled "
        f"{level_used}s in every replicate{within}; the interval is on the difference "
        "alpha_b - alpha_a.",
        UNITS_NOTE.format(level=level_used, within=within),
    )

    def arm(res: AlphaResult, ints: dict[str, Interval]) -> BootstrapResult:
        return BootstrapResult(
            scheme="units",
            level=level_used,
            strata=strata,
            method=method,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            alpha_min=None,
            experimental=method == "bca",
            n_clusters=plan_a.n_clusters,
            design=res.design,
            measurement=res.measurement,
            intervals=ints,
            warnings=(),
            notes=(),
        )

    return ComparisonResult(
        claim="configuration_difference",
        a=a,
        b=b,
        design=result_a.design,
        measurement=result_a.measurement,
        result_a=result_a,
        result_b=result_b,
        arm_a=arm(result_a, arm_intervals["a"]),
        arm_b=arm(result_b, arm_intervals["b"]),
        differences=differences,
        warnings=tuple(recorded),
        notes=notes,
    )


# -- helpers --------------------------------------------------------------------------


def _check_options(
    n_resamples: int, confidence: float, method: str, alpha_min: float | None
) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int | np.integer):
        raise DesignError(f"n_resamples must be an integer; got {n_resamples!r}")
    if n_resamples < 1:
        raise DesignError("n_resamples must be at least 1")
    if not 0 < confidence < 1:
        raise DesignError(f"confidence must be between 0 and 1; got {confidence!r}")
    if method not in METHODS:
        raise DesignError(f"method must be one of {list(METHODS)}; got {method!r}")
    if alpha_min is not None and not -1 <= alpha_min <= 1:
        raise DesignError(f"alpha_min must lie in [-1, 1]; got {alpha_min!r}")


def _resolve_level(sample: StudySample, level: str | None) -> tuple[str, str | None]:
    if level is None:
        return ("group" if sample.unit_groups is not None else "unit"), None
    if level not in LEVELS:
        raise DesignError(f"level must be one of {list(LEVELS)}; got {level!r}")
    if level == "group" and sample.unit_groups is None:
        raise DesignError("level='group' needs group_id declared in RatingFrame.from_long")
    if level == "unit" and sample.unit_groups is not None:
        return level, (
            "groups are declared but units are resampled one by one; units of one group are "
            "not independent, so intervals may be too narrow"
        )
    return level, None


def _seed(random_state: int | np.random.Generator | None) -> int:
    """The integer seed behind the resamples, so every result can be reproduced."""
    if random_state is None:
        return int(np.random.default_rng().integers(0, 2**63 - 1))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**63 - 1))
    if isinstance(random_state, bool) or not isinstance(random_state, int | np.integer):
        raise TypeError(f"random_state must be an int, a Generator or None; got {random_state!r}")
    if random_state < 0:
        raise DesignError("random_state must be non-negative")
    return int(random_state)


def _bca_warning() -> RecordedWarning:
    return emit(
        BootstrapWarning,
        "BCa intervals are experimental until coverage simulations validate them for alpha",
        stacklevel=4,
    )


def _failure_warnings(intervals: Mapping[str, Interval], n_resamples: int) -> list[RecordedWarning]:
    failed = {name: i.n_failed for name, i in intervals.items() if i.n_failed}
    if not failed:
        return []
    counts = ", ".join(f"{name} {count}" for name, count in failed.items())
    return [
        emit(
            BootstrapWarning,
            f"undefined replicates were counted and left out of the intervals ({counts}, of "
            f"{n_resamples} each)",
            stacklevel=4,
        )
    ]
