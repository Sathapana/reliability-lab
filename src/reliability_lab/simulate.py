"""Simulated coder ratings for any measurement level, at an alpha you choose.

Each unit has a true value drawn from ``prevalence``. Each coder reports it with
probability q and otherwise makes an error drawn from a kernel K(t, .):

- ``errors="random"``: K(t, .) is the prevalence itself, so an error is an
  independent draw from the population. A pair with at least one error then
  disagrees by D_e on average, so the expected alpha is q² for every distance.
- ``errors="near"``: K(t, v) is proportional to p_v exp(-delta²(t, v) / s) over the
  values other than t, with s = D_e. Errors land close to the truth as the
  metric defines closeness: neighbouring points, around the circle, near a pole.

q is chosen so that the expected alpha, in the limit of many units, equals the
target. With P(r | t) = q [r = t] + (1 - q) K(t, r):

    o   = sum_t p_t P(. | t) P(. | t)^T      (expected coincidences, normalised)
    m   = sum_t p_t P(. | t)                  (expected marginals)
    D_o = sum(o * delta2),  D_e = sum(m m^T * delta2),  alpha = 1 - D_o / D_e

Ordinal distances are taken at the expected marginals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab.alpha import Distance
from reliability_lab.data import RatingFrame, Status
from reliability_lab.exceptions import DesignError
from reliability_lab.study import _resolve_distance, _resolve_domain

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

#: Points on the q grid used to find where near-miss alpha starts to rise.
_GRID = 1001


@dataclass(frozen=True, eq=False)
class _LatentClassProcess:
    """Calibrated latent-class 'know or guess' generator for one arm."""

    distance: Distance
    domain: tuple[Any, ...]
    p: FloatArray
    kernel: FloatArray
    cumulative: FloatArray
    q: float
    n_coders: int
    missing: float
    failures: float
    hard_share: float
    hard_penalty: float
    cluster_size: int
    cohesion: float
    rho: float
    hard_failures: float

    @property
    def n_values(self) -> int:
        return len(self.domain)

    def draw_structure(
        self, n_units: int, rng: np.random.Generator
    ) -> tuple[IntArray, npt.NDArray[np.bool_], IntArray]:
        """Draw cluster ids, unit hardness, and true categories for ``n_units`` units."""
        cluster = (np.arange(n_units, dtype=np.intp) // self.cluster_size).astype(np.intp)
        n_clusters = int(cluster[-1]) + 1 if n_units else 0
        hard_cluster = rng.random(n_clusters) < self.hard_share
        hard = hard_cluster[cluster]
        if self.cohesion > 0 and self.cluster_size > 1:
            topic = rng.choice(self.n_values, size=n_clusters, p=self.p)[cluster]
            own_truth = rng.choice(self.n_values, size=n_units, p=self.p)
            truth = np.where(rng.random(n_units) < self.cohesion, topic, own_truth).astype(np.intp)
        else:
            truth = rng.choice(self.n_values, size=n_units, p=self.p).astype(np.intp)
        return truth, hard, cluster

    def rate(
        self,
        truth: IntArray,
        hard: npt.NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[IntArray, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Rate ``truth`` units, returning ``(codes, kept, failed)``."""
        n_units = len(truth)
        m = self.n_coders
        n_values = self.n_values
        cum = self.cumulative[truth]  # (units, values)
        draw = rng.random((n_units, m))
        own = np.minimum((draw[:, :, None] >= cum[:, None, :]).sum(axis=2), n_values - 1)
        if self.rho > 0:
            shared_draw = rng.random(n_units)
            shared = np.minimum((shared_draw[:, None] >= cum).sum(axis=1), n_values - 1)
            copy = rng.random((n_units, m)) < self.rho
            error = np.where(copy, shared[:, None], own)
        else:
            error = own
        if self.hard_share > 0 and self.hard_penalty > 0:
            acc = np.where(hard, max(0.0, self.q - self.hard_penalty), self.q)[:, None]
        else:
            acc = np.full((1, 1), self.q)
        knows = rng.random((n_units, m)) < acc
        codes = np.where(knows, truth[:, None], error).astype(np.intp)
        kept = rng.random((n_units, m)) >= self.missing
        if self.hard_failures != self.failures:
            f_rate = np.where(hard, self.hard_failures, self.failures)[:, None]
        else:
            f_rate = np.full((1, 1), self.failures)
        failed = rng.random((n_units, m)) < f_rate
        return codes, kept, failed


def _prepare_process(
    measurement: str | Distance,
    values: Sequence[Any],
    *,
    n_coders: int,
    alpha: float,
    errors: Literal["random", "near"],
    prevalence: Sequence[float] | None,
    missing: float,
    failures: float,
    hard_share: float,
    hard_penalty: float,
    cluster_size: int,
    cohesion: float,
    rho: float,
    hard_failures: float | None,
) -> _LatentClassProcess:
    distance = _resolve_distance(measurement)
    domain = _resolve_domain(values, distance)
    n_values = len(domain)
    _check_count("n_coders", n_coders, 2)
    _check_share("alpha", alpha, closed=True)
    _check_share("missing", missing, closed=False)
    _check_share("failures", failures, closed=False)
    _check_share("hard_share", hard_share, closed=True)
    _check_share("hard_penalty", hard_penalty, closed=True)
    _check_count("cluster_size", cluster_size, 1)
    _check_share("cohesion", cohesion, closed=True)
    _check_share("rho", rho, closed=True)
    if hard_failures is not None:
        _check_share("hard_failures", hard_failures, closed=False)
        if hard_share == 0:
            raise DesignError("hard_failures requires hard_share > 0")
    if errors not in ("random", "near"):
        raise DesignError(f"errors must be 'random' or 'near', got {errors!r}")
    p = _prevalence(prevalence, n_values)
    f_hard = failures if hard_failures is None else hard_failures

    delta2 = distance.matrix(domain, p)
    kernel = _kernel(errors, p, delta2)
    q = _calibrate_q(
        alpha,
        p,
        kernel,
        domain,
        distance,
        delta2,
        errors,
        n_coders=n_coders,
        missing=missing,
        failures=failures,
        hard_share=hard_share,
        hard_penalty=hard_penalty,
        rho=rho,
        hard_failures=f_hard,
    )
    return _LatentClassProcess(
        distance=distance,
        domain=domain,
        p=p,
        kernel=kernel,
        cumulative=np.cumsum(kernel, axis=1),
        q=q,
        n_coders=n_coders,
        missing=missing,
        failures=failures,
        hard_share=hard_share,
        hard_penalty=hard_penalty,
        cluster_size=cluster_size,
        cohesion=cohesion,
        rho=rho,
        hard_failures=f_hard,
    )


def simulate_ratings(
    measurement: str | Distance,
    values: Sequence[Any],
    *,
    n_units: int = 100,
    n_coders: int = 3,
    alpha: float = 0.8,
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
    random_state: int | np.random.Generator | None = None,
) -> RatingFrame:
    """Simulated ratings of ``n_units`` units by ``n_coders`` coders.

    ``measurement`` and ``values`` are declared as for ``ReliabilityStudy``, and the
    ratings are drawn from ``values``. Alpha computed from the ratings is ``alpha``
    on average; any one draw varies around it, the more so the fewer the units.
    ``missing`` is the share of coder-unit cells left unrated (no row), and
    ``failures`` the share of the remaining cells recorded as ``parse_error``.

    Optional latent-class factors match the coverage simulations: ``hard_share``
    of clusters are hard (accuracy lowered by ``hard_penalty``), ``cluster_size``
    groups units that share hardness and take a cluster topic with probability
    ``cohesion``, guessing coders copy a shared wrong draw with probability ``rho``,
    and ``hard_failures`` sets a separate failure rate on hard units.

    Units are ``u1, u2, ...`` and coders ``c1, c2, ...``. Pass the result to
    ``ReliabilityStudy(ratings, measurement=..., values=..., coders="rater_id",
    design="human_panel")``.
    """
    _check_count("n_units", n_units, 1)
    proc = _prepare_process(
        measurement,
        values,
        n_coders=n_coders,
        alpha=alpha,
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
    rng = np.random.default_rng(random_state)
    plain_mode = (
        hard_share == 0.0
        and cluster_size == 1
        and cohesion == 0.0
        and rho == 0.0
        and hard_failures is None
    )
    if plain_mode:
        truth = rng.choice(proc.n_values, size=n_units, p=proc.p)
        cumulative = proc.cumulative[truth]
        draw = rng.random((n_units, n_coders))
        error = np.minimum(
            (draw[:, :, None] >= cumulative[:, None, :]).sum(axis=2), proc.n_values - 1
        )
        knows = rng.random((n_units, n_coders)) < proc.q
        codes = np.where(knows, truth[:, None], error)
        kept = rng.random((n_units, n_coders)) >= missing
        failed = rng.random((n_units, n_coders)) < failures
        cluster = np.arange(n_units, dtype=np.intp)
    else:
        truth, hard, cluster = proc.draw_structure(n_units, rng)
        codes, kept, failed = proc.rate(truth, hard, rng)

    grouped = cluster_size > 1
    rows = []
    for u in range(n_units):
        for c in range(n_coders):
            if not kept[u, c]:
                continue
            row: dict[str, Any] = {
                "unit": f"u{u + 1}",
                "coder": f"c{c + 1}",
                "value": None if failed[u, c] else proc.domain[int(codes[u, c])],
                "status": Status.PARSE_ERROR.value if failed[u, c] else Status.OBSERVED.value,
            }
            if grouped:
                row["group"] = f"g{int(cluster[u]) + 1}"
            rows.append(row)
    cols = ["unit", "coder", "value", "status", *(["group"] if grouped else [])]
    frame = pd.DataFrame(rows, columns=cols)
    if grouped:
        return RatingFrame.from_long(
            frame, unit="unit", rater="coder", value="value", status="status", group="group"
        )
    return RatingFrame.from_long(frame, unit="unit", rater="coder", value="value", status="status")


def _check_count(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise DesignError(f"{name} must be a whole number of at least {minimum}, got {value!r}")


def _check_share(name: str, value: Any, *, closed: bool) -> None:
    ok = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
        and (value <= 1 if closed else value < 1)
    )
    if not ok:
        bounds = "[0, 1]" if closed else "[0, 1)"
        raise DesignError(f"{name} must be a number in {bounds}, got {value!r}")


def _prevalence(prevalence: Sequence[float] | None, n_values: int) -> FloatArray:
    if prevalence is None:
        return np.full(n_values, 1.0 / n_values)
    try:
        p = np.asarray(prevalence, dtype=float)
    except (TypeError, ValueError):
        raise DesignError("prevalence must be a sequence of numbers") from None
    if p.shape != (n_values,):
        raise DesignError(f"prevalence needs one weight per value ({n_values}), got {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any() or not math.isclose(p.sum(), 1.0):
        raise DesignError("prevalence must be non-negative weights that sum to 1")
    return p


def _kernel(errors: str, p: FloatArray, delta2: FloatArray) -> FloatArray:
    """K[t, v]: the probability that an error on a unit whose truth is t reports v."""
    n_values = len(p)
    if errors == "random":
        return np.tile(p, (n_values, 1))
    scale = float(p @ delta2 @ p)
    weights = p[None, :] * np.exp(-delta2 / scale) if scale > 0 else np.tile(p, (n_values, 1))
    np.fill_diagonal(weights, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    # A truth whose only other values have no prevalence can only be reported as itself.
    return np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), np.eye(n_values))


def _expected_alpha(q: float, p: FloatArray, kernel: FloatArray, delta2: FloatArray) -> float:
    """Alpha in the limit of many units when coders know the truth with probability q."""
    report = q * np.eye(len(p)) + (1 - q) * kernel  # report[t, r] = P(r | t)
    o = np.einsum("t,tr,ts->rs", p, report, report)
    m = p @ report
    d_e = float(m @ delta2 @ m)
    if d_e == 0:
        return math.nan
    return 1.0 - float((o * delta2).sum()) / d_e


def _expected_alpha_full(
    q: float,
    p: FloatArray,
    kernel: FloatArray,
    domain: tuple[Any, ...],
    distance: Distance,
    base_delta2: FloatArray,
    *,
    n_coders: int,
    missing: float,
    failures: float,
    hard_share: float,
    hard_penalty: float,
    rho: float,
    hard_failures: float,
) -> float:
    """Population alpha (as_missing) accounting for hard units, correlated errors and failures."""
    if hard_share == 0.0 and rho == 0.0 and not distance.depends_on_marginals:
        return _expected_alpha(q, p, kernel, base_delta2)
    n_val = len(p)
    eye = np.eye(n_val)
    p_obs = np.array([(1.0 - missing) * (1.0 - failures), (1.0 - missing) * (1.0 - hard_failures)])
    w_pairable = n_coders * p_obs * (1.0 - (1.0 - p_obs) ** (n_coders - 1))
    shares = np.array([1.0 - hard_share, hard_share]) * w_pairable
    total_w = float(shares.sum())
    if not total_w > 0:
        return math.nan
    shares = shares / total_w
    o_total = np.zeros((n_val, n_val))
    m_total = np.zeros(n_val)
    corr_term = np.diag(p @ kernel) - np.einsum("t,tr,ts->rs", p, kernel, kernel)
    for h, w_h in enumerate(shares):
        if w_h == 0:
            continue
        q_h = max(0.0, q - hard_penalty) if h == 1 else q
        report = q_h * eye + (1.0 - q_h) * kernel
        o_h = np.einsum("t,tr,ts->rs", p, report, report) + ((1.0 - q_h) * rho) ** 2 * corr_term
        o_total += w_h * o_h
        m_total += w_h * (p @ report)
    delta2 = distance.matrix(domain, m_total) if distance.depends_on_marginals else base_delta2
    d_e = float(m_total @ delta2 @ m_total)
    if d_e == 0:
        return math.nan
    return 1.0 - float((o_total * delta2).sum()) / d_e


def _accuracy(
    target: float, p: FloatArray, kernel: FloatArray, delta2: FloatArray, errors: str
) -> float:
    """The q whose expected alpha is ``target`` (plain generator)."""
    if math.isnan(_expected_alpha(1.0, p, kernel, delta2)):
        raise DesignError("alpha is undefined here: the values cannot disagree")
    if errors == "random":
        return math.sqrt(target)
    grid = np.linspace(0.0, 1.0, _GRID)
    curve = np.array([_expected_alpha(q, p, kernel, delta2) for q in grid])
    start = int(np.argmin(curve))
    lowest = float(curve[start])
    if target < lowest - 1e-12:
        raise DesignError(
            f"near-miss errors cannot bring alpha below {lowest:.3f} for these values; "
            "raise alpha or use errors='random'"
        )
    lo, hi = float(grid[start]), 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _expected_alpha(mid, p, kernel, delta2) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _calibrate_q(
    target: float,
    p: FloatArray,
    kernel: FloatArray,
    domain: tuple[Any, ...],
    distance: Distance,
    base_delta2: FloatArray,
    errors: str,
    *,
    n_coders: int,
    missing: float,
    failures: float,
    hard_share: float,
    hard_penalty: float,
    rho: float,
    hard_failures: float,
) -> float:
    """The base accuracy q whose population alpha is ``target``."""
    if hard_share == 0.0 and rho == 0.0 and not distance.depends_on_marginals:
        return _accuracy(target, p, kernel, base_delta2, errors)

    def eval_q(q: float) -> float:
        return _expected_alpha_full(
            q,
            p,
            kernel,
            domain,
            distance,
            base_delta2,
            n_coders=n_coders,
            missing=missing,
            failures=failures,
            hard_share=hard_share,
            hard_penalty=hard_penalty,
            rho=rho,
            hard_failures=hard_failures,
        )

    highest = eval_q(1.0)
    if math.isnan(highest):
        raise DesignError("alpha is undefined here: the values cannot disagree")
    if target > highest + 1e-12:
        raise DesignError(
            f"these process parameters cannot raise alpha above {highest:.3f}; "
            "lower alpha or reduce hard_penalty / hard_share"
        )
    if target >= highest - 1e-12:
        return 1.0

    grid = np.linspace(0.0, 1.0, _GRID)
    curve = np.array([eval_q(float(q)) for q in grid])
    start = int(np.nanargmin(curve))
    lowest = float(curve[start])
    if target < lowest - 1e-12:
        raise DesignError(
            f"this process cannot bring alpha below {lowest:.3f} for these values; "
            "raise alpha or use errors='random'"
        )
    lo, hi = float(grid[start]), 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if eval_q(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi
