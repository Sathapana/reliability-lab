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
* ``coders`` resamples the coders with the units fixed, and ``two_way`` resamples
  both, independently. A coder drawn twice contributes two copies of its values,
  but copies of one coder are never paired with each other.

The coder jackknife leaves each coder out in turn, for a standard error and an
influence table.

Paired comparisons apply one set of resampled units or groups to both
configurations, so the interval falls on the difference alpha_b - alpha_a.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse, special, stats

from reliability_lab.alpha import ORDINAL, Distance, compute_alpha, per_unit_coincidences
from reliability_lab.data import preview
from reliability_lab.exceptions import BootstrapWarning, DesignError
from reliability_lab.report import (
    AlphaResult,
    BootstrapResult,
    ComparisonResult,
    Interval,
    JackknifeResult,
    RecordedWarning,
    emit,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SCHEMES: tuple[str, ...] = ("units", "kalpha", "coders", "two_way")
#: Coder resampling needs at least this many coders to say anything.
MIN_CODERS = 3
METHODS: tuple[str, ...] = ("percentile", "bca")
ON_UNDEFINED: tuple[str, ...] = ("nan", "percentile")
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
    "recomputed in every replicate. The coders are held fixed: the interval generalises "
    "to new units rated by these coders."
)
CODERS_NOTE = (
    "Coders scheme: the {n_coders} coders are resampled with replacement and the units "
    "held fixed; pairs between copies of one coder are left out. The interval "
    "generalises to new coders from the same pool, on these units. In simulations it "
    "was too narrow (79-92% coverage with 5-20 coders); coder_jackknife() answers the "
    "same question and covered 94-96%."
)
TWO_WAY_NOTE = (
    "Two-way scheme: whole {level}s{within} and the {n_coders} coders are resampled "
    "independently with replacement (a 'pigeonhole' bootstrap, conservative when the "
    "coders hardly differ); pairs between copies of one coder are left out. The "
    "interval generalises to new coders on new units."
)


@dataclass(frozen=True, eq=False)
class EstimateInput:
    """The coded pairable values behind one estimate: one entry per rating.

    ``coder_idx`` indexes :attr:`StudySample.coder_names`.
    """

    unit_idx: IntArray
    value_idx: IntArray
    values: tuple[Any, ...]
    coder_idx: IntArray


@dataclass(frozen=True, eq=False)
class StudySample:
    """What resampling needs from a study: its units, coders, clusters and coded inputs.

    ``unit_groups`` and ``unit_strata`` hold each unit's group and stratum label, or
    None when the field was not declared. ``inputs`` is keyed by estimate name.
    """

    unit_labels: npt.NDArray[np.object_]
    unit_groups: npt.NDArray[np.object_] | None
    unit_strata: npt.NDArray[np.object_] | None
    distance: Distance
    inputs: Mapping[str, EstimateInput]
    coder_names: npt.NDArray[np.object_]

    @property
    def n_units(self) -> int:
        return len(self.unit_labels)

    @property
    def n_coders(self) -> int:
        return len(self.coder_names)


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
    w = np.atleast_2d(np.asarray(weights))
    n_values = len(values)
    n_flat = n_values * n_values
    # A dense product is one BLAS call; large value domains stay sparse.
    dense = clusters.shape[0] * n_flat <= BATCH_FLOATS
    contributions = clusters.toarray() if dense else sparse.csr_array(clusters.T)
    out = np.empty(len(w))
    batch = max(1, BATCH_FLOATS // max(n_flat, clusters.shape[0], 1))
    for start in range(0, len(w), batch):
        wb = np.asarray(w[start : start + batch], dtype=float)
        if dense:
            flat = wb @ contributions
        else:
            flat = np.asarray(contributions @ np.asfortranarray(wb.T)).T
        o = flat.reshape(len(wb), n_values, n_values)
        out[start : start + len(wb)] = alphas_from_coincidences(o, values, distance)
    return out


def jackknife_alphas(
    clusters: sparse.csr_array, values: tuple[Any, ...], distance: Distance
) -> FloatArray:
    """Alpha without each cluster in turn: the coincidences less that cluster's."""
    n_values = len(values)
    n_flat = n_values * n_values
    total = np.asarray(clusters.sum(axis=0), dtype=float).ravel()
    out = np.empty(clusters.shape[0])
    batch = max(1, BATCH_FLOATS // max(n_flat, 1))
    for start in range(0, clusters.shape[0], batch):
        part = clusters[start : start + batch].toarray()
        o = (total - part).reshape(len(part), n_values, n_values)
        out[start : start + len(part)] = alphas_from_coincidences(o, values, distance)
    return out


def alphas_from_coincidences(
    o: FloatArray, values: tuple[Any, ...], distance: Distance
) -> FloatArray:
    """Alpha for each coincidence matrix in a batch (batch x V x V); NaN where undefined."""
    n_c = o.sum(axis=2)
    n = n_c.sum(axis=1)
    if not distance.depends_on_marginals:
        fixed = distance.matrix(values, np.ones(len(values)))
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
    return np.asarray(alpha, dtype=float)


# -- coders and two-way schemes ------------------------------------------------------


def coder_onehot(sample: StudySample, name: str) -> FloatArray:
    """The ratings of one estimate as a units x coders x values one-hot array."""
    inp = sample.inputs[name]
    onehot = np.zeros((sample.n_units, sample.n_coders, len(inp.values)))
    onehot[inp.unit_idx, inp.coder_idx, inp.value_idx] = 1.0
    return onehot


def draw_coder_weights(rng: np.random.Generator, n_coders: int, n: int) -> npt.NDArray[np.int64]:
    """How often each coder is drawn in each of ``n`` resamples (n x coders)."""
    return rng.multinomial(n_coders, np.full(n_coders, 1.0 / n_coders), size=n)


def two_way_alphas(
    onehot: FloatArray,
    unit_weights: npt.ArrayLike | None,
    coder_weights: npt.ArrayLike,
    values: tuple[Any, ...],
    distance: Distance,
) -> FloatArray:
    """Alpha for each replicate of unit weights (None: all 1) and coder weights.

    A coder drawn v times contributes v copies of each of its values, but pairs
    between copies of one coder are left out: a copy agrees perfectly with
    itself. Unit u then contributes ``(m_u / P_u) (n_u n_u' - diag S_u)``, where
    ``n_u`` counts the copies of each value, ``S_u`` the squared multiplicities,
    ``m_u`` all copies and ``P_u = m_u² - sum(S_u)`` the ordered pairs between
    different coders. With weights of 0 or 1 this is Krippendorff's per-unit
    ``(n_u n_u' - diag n_u) / (m_u - 1)``.
    """
    v = np.atleast_2d(np.asarray(coder_weights, dtype=float))
    n_units, n_coders, n_values = onehot.shape
    w = None if unit_weights is None else np.atleast_2d(np.asarray(unit_weights))
    # Every sum over coders is a matrix product with the coder weights.
    by_coder = onehot.transpose(1, 0, 2).reshape(n_coders, n_units * n_values)
    rated = onehot.sum(axis=2).T
    out = np.empty(len(v))
    per_replicate = max(n_units * n_values * 2, n_values * n_values, 1)
    batch = max(1, BATCH_FLOATS // per_replicate)
    for start in range(0, len(v), batch):
        vb = v[start : start + batch]
        v2 = vb * vb
        wb = 1.0 if w is None else np.asarray(w[start : start + batch], dtype=float)
        counts = (vb @ by_coder).reshape(len(vb), n_units, n_values)
        squares = (v2 @ by_coder).reshape(len(vb), n_units, n_values)
        m = vb @ rated
        pairs = m * m - v2 @ rated
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(pairs > 0, wb * m / pairs, 0.0)
        o = (counts * scale[:, :, None]).transpose(0, 2, 1) @ counts
        diagonal = (scale[:, None, :] @ squares)[:, 0, :]
        o[:, np.arange(n_values), np.arange(n_values)] -= diagonal
        out[start : start + len(vb)] = alphas_from_coincidences(o, values, distance)
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
    d = np.asarray(draws, dtype=np.intp)
    sp = pool.second_pick
    alpha = 1.0 - np.take(pool.errors, d) @ pool.weights
    if len(sp):
        # Repeats are rare, so their redraws correct the sums afterwards.
        rows, cols = np.nonzero(d[:, sp - 1] == d[:, sp])
        redrawn = pool.errors[np.asarray(redraws)[rows, cols]] - pool.errors[d[rows, sp[cols]]]
        np.subtract.at(alpha, rows, redrawn * pool.weights[sp[cols]])
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
            estimate=name,
            point=point,
            low=np.nan,
            high=np.nan,
            n_resamples=len(distribution),
            n_failed=int(failed.sum()),
            undefined_reason="every replicate was undefined",
            distribution=distribution,
            method_used="percentile",
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(defined, [tail, 1.0 - tail])
    below = float((defined < alpha_min).mean()) if alpha_min is not None else None
    return Interval(
        estimate=name,
        point=point,
        low=float(low),
        high=float(high),
        n_resamples=len(distribution),
        n_failed=int(failed.sum()),
        p_below_min=below,
        distribution=distribution,
        method_used="percentile",
    )


def bca_interval(
    name: str,
    point: float,
    distribution: FloatArray,
    theta_hat: float,
    jackknife: FloatArray,
    confidence: float,
    alpha_min: float | None = None,
) -> Interval:
    """BCa interval (Efron and Tibshirani 1993, 14.3 and 15.4), as ``scipy.stats.bootstrap`` does.

    ``theta_hat`` is the statistic on the full sample, computed like the replicates,
    and ``jackknife`` the statistic without each cluster in turn.
    """
    n_resamples = len(distribution)
    failed = np.isnan(distribution)
    z_alpha = float(special.ndtri((1.0 - confidence) / 2.0))
    z0, a_hat = bca_parameters(distribution, theta_hat, jackknife)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = z0 + np.array([z_alpha, -z_alpha])
        levels = special.ndtr(z0 + num / (1.0 - a_hat * num))
    low = high = np.nan
    reason: str | None = None
    if not failed.any() and np.isfinite(levels).all():
        low, high = (float(q) for q in np.quantile(distribution, levels))
    else:
        reason = _bca_failure(failed, z0, jackknife)
    defined = distribution[~failed]
    below = float((defined < alpha_min).mean()) if alpha_min is not None and len(defined) else None
    return Interval(
        estimate=name,
        point=point,
        low=low,
        high=high,
        n_resamples=n_resamples,
        n_failed=int(failed.sum()),
        p_below_min=below,
        undefined_reason=reason,
        distribution=distribution,
        method_used="bca",
    )


def bca_parameters(
    distribution: FloatArray, theta_hat: float, jackknife: FloatArray
) -> tuple[float, float]:
    """The bias correction z0 and the acceleration a_hat of a BCa interval.

    Either is non-finite when it is undefined: z0 when every replicate falls on one
    side of ``theta_hat``, a_hat when the jackknife is undefined or constant.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        below_hat = (distribution < theta_hat).sum() + (distribution <= theta_hat).sum()
        z0 = special.ndtri(below_hat / (2 * len(distribution)))
        u = jackknife.mean() - jackknife
        a_hat = (u**3).sum() / (6.0 * (u**2).sum() ** 1.5)
    return float(z0), float(a_hat)


def _bca_failure(failed: npt.NDArray[np.bool_], z0: float, jackknife: FloatArray) -> str:
    """Why a BCa interval is undefined, checked in the order the formula needs its parts."""
    if failed.any():
        return (
            f"the BCa interval is undefined: {int(failed.sum())} of {len(failed)} replicates "
            "are undefined, and BCa needs every one"
        )
    if not np.isfinite(z0):
        return (
            "the BCa interval is undefined: every replicate falls on one side of the "
            "estimate, so the bias correction is infinite"
        )
    if np.isnan(jackknife).any():
        return (
            "the BCa interval is undefined: alpha is undefined with one cluster left out, "
            "so the acceleration is undefined"
        )
    if not np.ptp(jackknife) > 0:
        return (
            "the BCa interval is undefined: alpha is the same with any one cluster left "
            "out, so the acceleration is undefined"
        )
    return "the BCa interval is undefined: its adjusted quantile levels are not finite"


def with_fallback(
    interval: Interval,
    confidence: float,
    on_undefined: str,
    alpha_min: float | None = None,
) -> Interval:
    """The percentile interval over the same replicates in place of an undefined BCa one."""
    if on_undefined == "nan" or interval.defined or interval.method_used != "bca":
        return interval
    fallback = percentile_interval(
        interval.estimate, interval.point, interval.distribution, confidence, alpha_min
    )
    return dataclasses.replace(fallback, fallback_reason=interval.undefined_reason)


def bca_weights(seed: int, n_clusters: int, n_resamples: int) -> npt.NDArray[np.int64]:
    """Cluster weights of the BCa resamples (n x clusters).

    The indices are the ones ``scipy.stats.bootstrap`` draws from the same seed,
    drawn in batches (which leaves the stream unchanged) and counted.
    """
    rng = np.random.default_rng(seed)
    batch = max(1, BATCH_FLOATS // max(n_clusters, 1))
    parts = []
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        idx = rng.integers(0, n_clusters, (size, n_clusters))
        parts.append(weights_from_indices(idx, n_clusters))
    return np.concatenate(parts)


def bca_parts(
    clusters: sparse.csr_array, values: tuple[Any, ...], distance: Distance
) -> tuple[float, FloatArray]:
    """The full-sample statistic and the leave-one-cluster-out jackknife for BCa."""
    theta_hat = replicate_alphas(clusters, np.ones(clusters.shape[0]), values, distance)[0]
    return float(theta_hat), jackknife_alphas(clusters, values, distance)


def weights_from_indices(idx: IntArray, n_clusters: int) -> npt.NDArray[np.int64]:
    """Count cluster indices along the last axis: (..., k) -> (prod(...), n_clusters)."""
    flat = np.asarray(idx, dtype=np.intp).reshape(-1, np.shape(idx)[-1])
    offsets = (np.arange(len(flat)) * n_clusters)[:, None]
    counts = np.bincount((flat + offsets).ravel(), minlength=len(flat) * n_clusters)
    return counts.reshape(len(flat), n_clusters)


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
    on_undefined: str = "nan",
) -> BootstrapResult:
    """Intervals for every estimate of ``result``; see :meth:`AlphaResult.bootstrap`."""
    _check_options(n_resamples, confidence, method, alpha_min, on_undefined)
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
                    estimate=name,
                    point=points[name],
                    low=np.nan,
                    high=np.nan,
                    n_resamples=n_resamples,
                    n_failed=0,
                    undefined_reason=pool.undefined_reason,
                    method_used="percentile",
                )
                continue
            distribution = kalpha_distribution(pool, np.random.default_rng(seed), n_resamples)
            intervals[name] = percentile_interval(
                name, points[name], distribution, confidence, alpha_min
            )
        notes.append(KALPHA_NOTE)
        level_used, n_clusters = None, None
    elif scheme in ("coders", "two_way"):
        if method == "bca":
            raise DesignError(
                f"BCa intervals are available with scheme='units' only; scheme='{scheme}' "
                "uses percentile intervals"
            )
        if sample.n_coders < MIN_CODERS:
            raise DesignError(
                f"scheme='{scheme}' resamples coders and needs at least {MIN_CODERS}; this "
                f"study has {sample.n_coders}"
            )
        rng = np.random.default_rng(seed)
        if scheme == "coders":
            if level is not None or strata:
                raise DesignError(
                    "the coders scheme holds the units fixed; leave level and strata unset, "
                    "or use scheme='two_way' to resample units and coders"
                )
            level_used, n_clusters, unit_weights = None, None, None
            notes.append(CODERS_NOTE.format(n_coders=sample.n_coders))
        else:
            level_used, level_warning = _resolve_level(sample, level)
            if level_warning:
                recorded.append(emit(BootstrapWarning, level_warning, stacklevel=3))
            plan = cluster_plan(sample, level_used, strata)
            n_clusters = plan.n_clusters
            unit_weights = draw_weights(rng, plan, n_resamples)[:, plan.cluster_of_unit]
            within = " within strata" if strata else ""
            notes.append(
                TWO_WAY_NOTE.format(level=level_used, within=within, n_coders=sample.n_coders)
            )
        coder_weights = draw_coder_weights(rng, sample.n_coders, n_resamples)
        for name in names:
            distribution = two_way_alphas(
                coder_onehot(sample, name),
                unit_weights,
                coder_weights,
                sample.inputs[name].values,
                sample.distance,
            )
            intervals[name] = percentile_interval(
                name, points[name], distribution, confidence, alpha_min
            )
        if len(names) > 1:
            notes.append("All estimates use the same resamples, so their intervals are paired.")
    else:
        level_used, level_warning = _resolve_level(sample, level)
        if level_warning:
            recorded.append(emit(BootstrapWarning, level_warning, stacklevel=3))
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
        for name in names:
            values = sample.inputs[name].values
            clusters = cluster_coincidences(sample, name, plan)
            distribution = replicate_alphas(clusters, weights, values, sample.distance)
            if method == "percentile":
                intervals[name] = percentile_interval(
                    name, points[name], distribution, confidence, alpha_min
                )
            else:
                theta_hat, jackknife = bca_parts(clusters, values, sample.distance)
                interval = bca_interval(
                    name, points[name], distribution, theta_hat, jackknife, confidence, alpha_min
                )
                intervals[name] = with_fallback(interval, confidence, on_undefined, alpha_min)
        notes.append(UNITS_NOTE.format(level=level_used, within=" within strata" if strata else ""))
        if len(names) > 1:
            notes.append("All estimates use the same resamples, so their intervals are paired.")

    recorded.extend(_interval_warnings(intervals, n_resamples))
    return BootstrapResult(
        scheme=scheme,
        level=level_used,
        strata=strata,
        method=method,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        alpha_min=alpha_min,
        experimental=False,
        n_clusters=n_clusters,
        design=result.design,
        measurement=result.measurement,
        intervals=intervals,
        warnings=tuple(recorded),
        notes=tuple(notes),
        provenance=result.provenance,
        n_coders=sample.n_coders if scheme in ("coders", "two_way") else None,
    )


def coder_jackknife(
    result: AlphaResult, sample: StudySample, *, confidence: float
) -> JackknifeResult:
    """Leave-one-coder-out alphas, jackknife standard errors and t intervals.

    See :meth:`AlphaResult.coder_jackknife`.
    """
    _check_options(1, confidence, "percentile", None)
    m = sample.n_coders
    if m < MIN_CODERS:
        raise DesignError(
            f"the coder jackknife leaves one coder out and needs at least {MIN_CODERS} "
            f"coders; this study has {m}"
        )
    t = float(stats.t.ppf(1 - (1 - confidence) / 2, m - 1))
    intervals: dict[str, Interval] = {}
    errors: dict[str, float] = {}
    rows = []
    for name, inp in sample.inputs.items():
        point = result.estimates[name].alpha
        without = np.empty(m)
        for j in range(m):
            keep = inp.coder_idx != j
            computed = compute_alpha(
                inp.unit_idx[keep], inp.value_idx[keep], sample.n_units, inp.values, sample.distance
            )
            without[j] = computed.components.alpha
            rows.append(
                {
                    "estimate": name,
                    "coder": sample.coder_names[j],
                    "units_rated": int(np.unique(inp.unit_idx[~keep]).size),
                    "alpha_without": without[j],
                    "influence": point - without[j],
                }
            )
        undefined = np.isnan(without)
        if undefined.any():
            missing = preview(sample.coder_names[undefined])
            intervals[name] = Interval(
                estimate=name,
                point=point,
                low=np.nan,
                high=np.nan,
                n_resamples=m,
                n_failed=int(undefined.sum()),
                undefined_reason=f"alpha is undefined without coders {missing}",
                distribution=without,
                method_used="t",
            )
            errors[name] = np.nan
            continue
        se = float(np.sqrt((m - 1) / m * np.sum((without - without.mean()) ** 2)))
        errors[name] = se
        # Alpha cannot exceed 1, so the upper end is capped there.
        intervals[name] = Interval(
            estimate=name,
            point=point,
            low=point - t * se,
            high=min(point + t * se, 1.0),
            n_resamples=m,
            n_failed=0,
            distribution=without,
            method_used="t",
        )
    recorded = _interval_warnings(intervals, m, replicates=False)
    return JackknifeResult(
        confidence=confidence,
        n_coders=m,
        design=result.design,
        measurement=result.measurement,
        intervals=intervals,
        standard_errors=errors,
        influence=pd.DataFrame(rows),
        warnings=tuple(recorded),
        notes=(
            f"Coder jackknife: alpha is recomputed {m} times, each time without one coder. "
            "The standard error measures how much alpha depends on which coders were "
            f"sampled, with the units held fixed; the interval uses t with {m - 1} degrees "
            "of freedom. The influence of a coder is alpha minus alpha without that coder: "
            "a large positive value means the coder raises agreement.",
        ),
        provenance=result.provenance,
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
    on_undefined: str = "nan",
) -> ComparisonResult:
    """Paired intervals on alpha_b - alpha_a; see :meth:`ReliabilityStudy.compare`."""
    _check_options(n_resamples, confidence, method, None, on_undefined)
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
        else bca_weights(seed, plan_a.n_clusters, n_resamples)
    )

    arm_intervals: dict[str, dict[str, Interval]] = {"a": {}, "b": {}}
    differences: dict[str, Interval] = {}
    for name in names:
        point_a = result_a.estimates[name].alpha
        point_b = result_b.estimates[name].alpha
        clusters_a = cluster_coincidences(sample_a, name, plan_a)
        clusters_b = cluster_coincidences(sample_b, name, plan_b)
        values_a, values_b = sample_a.inputs[name].values, sample_b.inputs[name].values
        dist_a = replicate_alphas(clusters_a, weights, values_a, sample_a.distance)
        dist_b = replicate_alphas(clusters_b, weights, values_b, sample_b.distance)
        if method == "percentile":
            arm_intervals["a"][name] = percentile_interval(name, point_a, dist_a, confidence)
            arm_intervals["b"][name] = percentile_interval(name, point_b, dist_b, confidence)
            differences[name] = percentile_interval(
                name, point_b - point_a, dist_b - dist_a, confidence
            )
            continue
        theta_a, jack_a = bca_parts(clusters_a, values_a, sample_a.distance)
        theta_b, jack_b = bca_parts(clusters_b, values_b, sample_b.distance)
        arm_intervals["a"][name] = with_fallback(
            bca_interval(name, point_a, dist_a, theta_a, jack_a, confidence),
            confidence,
            on_undefined,
        )
        arm_intervals["b"][name] = with_fallback(
            bca_interval(name, point_b, dist_b, theta_b, jack_b, confidence),
            confidence,
            on_undefined,
        )
        differences[name] = with_fallback(
            bca_interval(
                name,
                point_b - point_a,
                dist_b - dist_a,
                theta_b - theta_a,
                jack_b - jack_a,
                confidence,
            ),
            confidence,
            on_undefined,
        )

    recorded.extend(_interval_warnings(differences, n_resamples))
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
            experimental=False,
            n_clusters=plan_a.n_clusters,
            design=res.design,
            measurement=res.measurement,
            intervals=ints,
            warnings=(),
            notes=(),
            provenance=res.provenance,
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
    n_resamples: int,
    confidence: float,
    method: str,
    alpha_min: float | None,
    on_undefined: str = "nan",
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
    if on_undefined not in ON_UNDEFINED:
        raise DesignError(f"on_undefined must be one of {list(ON_UNDEFINED)}; got {on_undefined!r}")


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


def _interval_warnings(
    intervals: Mapping[str, Interval], n_resamples: int, replicates: bool = True
) -> list[RecordedWarning]:
    """Warn about undefined replicates (counted, left out) and about missing intervals.

    ``replicates=False`` (the jackknife) skips the first: there an undefined
    leave-one-out alpha leaves no interval at all.
    """
    found = []
    failed = {name: i.n_failed for name, i in intervals.items() if i.n_failed}
    if failed and replicates:
        counts = ", ".join(f"{name} {count}" for name, count in failed.items())
        found.append(
            emit(
                BootstrapWarning,
                f"undefined replicates were counted and left out of the intervals ({counts}, "
                f"of {n_resamples} each)",
                stacklevel=4,
            )
        )
    for name, interval in intervals.items():
        if interval.fallback_reason:
            cause = interval.fallback_reason.removeprefix("the BCa interval is undefined: ")
            found.append(
                emit(
                    BootstrapWarning,
                    f"no BCa interval for {name} ({cause}); the percentile interval over the "
                    "defined replicates is reported instead, as on_undefined='percentile' asks",
                    stacklevel=4,
                )
            )
        if interval.undefined_reason:
            advice = (
                "; method='percentile' or on_undefined='percentile' gives an interval from "
                "the defined replicates"
                if "BCa" in interval.undefined_reason
                else ""
            )
            found.append(
                emit(
                    BootstrapWarning,
                    f"no interval for {name}: {interval.undefined_reason}{advice}",
                    stacklevel=4,
                )
            )
    return found
