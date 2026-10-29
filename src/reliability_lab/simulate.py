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
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab.alpha import Distance
from reliability_lab.data import RatingFrame, Status
from reliability_lab.exceptions import DesignError
from reliability_lab.study import _resolve_distance, _resolve_domain

FloatArray = npt.NDArray[np.float64]

#: Points on the q grid used to find where near-miss alpha starts to rise.
_GRID = 1001


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
    random_state: int | np.random.Generator | None = None,
) -> RatingFrame:
    """Simulated ratings of ``n_units`` units by ``n_coders`` coders.

    ``measurement`` and ``values`` are declared as for ``ReliabilityStudy``, and the
    ratings are drawn from ``values``. Alpha computed from the ratings is ``alpha``
    on average; any one draw varies around it, the more so the fewer the units.
    ``missing`` is the share of coder-unit cells left unrated (no row), and
    ``failures`` the share of the remaining cells recorded as ``parse_error``.

    Units are ``u1, u2, ...`` and coders ``c1, c2, ...``. Pass the result to
    ``ReliabilityStudy(ratings, measurement=..., values=..., coders="rater_id",
    design="human_panel")``.
    """
    distance = _resolve_distance(measurement)
    domain = _resolve_domain(values, distance)
    n_values = len(domain)
    _check_count("n_units", n_units, 1)
    _check_count("n_coders", n_coders, 2)
    _check_share("alpha", alpha, closed=True)
    _check_share("missing", missing, closed=False)
    _check_share("failures", failures, closed=False)
    if errors not in ("random", "near"):
        raise DesignError(f"errors must be 'random' or 'near', got {errors!r}")
    p = _prevalence(prevalence, n_values)

    delta2 = distance.matrix(domain, p)
    kernel = _kernel(errors, p, delta2)
    q = _accuracy(alpha, p, kernel, delta2, errors)

    rng = np.random.default_rng(random_state)
    truth = rng.choice(n_values, size=n_units, p=p)
    cumulative = np.cumsum(kernel, axis=1)[truth]  # (units, values)
    draw = rng.random((n_units, n_coders))
    error = np.minimum((draw[:, :, None] >= cumulative[:, None, :]).sum(axis=2), n_values - 1)
    knows = rng.random((n_units, n_coders)) < q
    codes = np.where(knows, truth[:, None], error)
    kept = rng.random((n_units, n_coders)) >= missing
    failed = rng.random((n_units, n_coders)) < failures

    rows = [
        {
            "unit": f"u{u + 1}",
            "coder": f"c{c + 1}",
            "value": None if failed[u, c] else domain[codes[u, c]],
            "status": Status.PARSE_ERROR.value if failed[u, c] else Status.OBSERVED.value,
        }
        for u in range(n_units)
        for c in range(n_coders)
        if kept[u, c]
    ]
    frame = pd.DataFrame(rows, columns=["unit", "coder", "value", "status"])
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


def _accuracy(
    target: float, p: FloatArray, kernel: FloatArray, delta2: FloatArray, errors: str
) -> float:
    """The q whose expected alpha is ``target``."""
    if math.isnan(_expected_alpha(1.0, p, kernel, delta2)):
        raise DesignError("alpha is undefined here: the values cannot disagree")
    if errors == "random":
        return math.sqrt(target)
    # Near misses can agree among themselves, so alpha need not rise from q = 0:
    # invert it on the stretch from its lowest point up to q = 1, where it is 1.
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
