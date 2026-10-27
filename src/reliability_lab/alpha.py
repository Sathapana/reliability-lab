"""Krippendorff's alpha from coincidence matrices.

Notation follows Krippendorff (2011), *Computing Krippendorff's alpha-reliability*:
``o`` is the V x V coincidence matrix, ``n_c`` its marginals, ``n`` the number of
pairable values and ``delta2`` the squared distances between values.

    D_o   = sum(o * delta2) / n
    D_e   = sum(n_c n_k delta2) / (n (n - 1))
    alpha = 1 - D_o / D_e

The coincidence matrix is a sum of per-unit contributions, and those do not depend
on the distance. Precomputing them once therefore serves every distance and every
bootstrap replicate. Only distances that depend on the marginals (ordinal) must be
rebuilt from each replicate's ``n_c``.

This module works on integer-coded numpy arrays only; ``study.py`` does the coding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from reliability_lab.exceptions import DesignError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

#: Above this many declared values a DomainSizeWarning is raised.
DOMAIN_SIZE_WARN = 1_000
#: Above this many declared values the study is refused.
DOMAIN_SIZE_MAX = 10_000

#: ``fn(values, marginals) -> V x V`` squared distances over the declared domain.
DistanceFn = Callable[[npt.NDArray[Any], FloatArray], FloatArray]


@dataclass(frozen=True, eq=False)
class Distance:
    """A squared-distance function delta² over a declared value domain.

    ``fn`` receives the declared values (in declared order) and the marginal count
    of each value, and returns a symmetric V x V matrix with a zero diagonal.
    ``depends_on_marginals`` marks metrics such as ordinal whose distances change
    with the category distribution. ``numeric`` marks metrics that need numbers.
    """

    name: str
    fn: DistanceFn
    depends_on_marginals: bool = False
    numeric: bool = False

    def matrix(self, values: Sequence[Any], marginals: npt.ArrayLike) -> FloatArray:
        v = np.asarray(values, dtype=float if self.numeric else object)
        m = np.asarray(marginals, dtype=float)
        d = np.asarray(self.fn(v, m), dtype=float)
        _validate_distance_matrix(d, len(values), self.name)
        return d

    @classmethod
    def from_matrix(cls, name: str, matrix: npt.ArrayLike) -> Distance:
        """A custom distance from a fixed V x V matrix of squared distances.

        Rows and columns follow the order of the declared ``values``.
        """
        fixed = np.array(matrix, dtype=float)
        if fixed.ndim != 2 or fixed.shape[0] != fixed.shape[1]:
            raise DesignError(f"distance {name!r} must be a square matrix, got {fixed.shape}")
        _validate_distance_matrix(fixed, fixed.shape[0], name)
        fixed.setflags(write=False)

        def fn(values: npt.NDArray[Any], marginals: FloatArray) -> FloatArray:
            if len(values) != fixed.shape[0]:
                raise DesignError(
                    f"distance {name!r} is {fixed.shape[0]} x {fixed.shape[0]} but the value "
                    f"domain has {len(values)} values"
                )
            return fixed

        return cls(name=name, fn=fn)


def _validate_distance_matrix(d: FloatArray, size: int, name: str) -> None:
    if d.shape != (size, size):
        raise DesignError(f"distance {name!r} returned shape {d.shape}, expected ({size}, {size})")
    if not np.isfinite(d).all():
        raise DesignError(f"distance {name!r} has non-finite entries")
    if (d < 0).any():
        raise DesignError(f"distance {name!r} has negative entries")
    if not np.allclose(d, d.T, rtol=1e-12, atol=1e-12):
        raise DesignError(f"distance {name!r} is not symmetric")
    if np.any(np.diag(d) != 0):
        raise DesignError(f"distance {name!r} must be zero on the diagonal")


def _nominal(values: npt.NDArray[Any], marginals: FloatArray) -> FloatArray:
    return 1.0 - np.eye(len(values))


def _ordinal(values: npt.NDArray[Any], marginals: FloatArray) -> FloatArray:
    # delta_ck = sum_{g=c..k} n_g - (n_c + n_k)/2, which equals r_k - r_c with
    # r = cumsum(n) - n/2 (the mid-rank of each value in declared order).
    r = np.cumsum(marginals) - marginals / 2.0
    return np.asarray((r[:, None] - r[None, :]) ** 2, dtype=float)


def _interval(values: npt.NDArray[Any], marginals: FloatArray) -> FloatArray:
    v = values.astype(float)
    return np.asarray((v[:, None] - v[None, :]) ** 2, dtype=float)


def _ratio(values: npt.NDArray[Any], marginals: FloatArray) -> FloatArray:
    v = values.astype(float)
    if (v < 0).any():
        raise DesignError("ratio distances need non-negative values")
    total = v[:, None] + v[None, :]
    diff = v[:, None] - v[None, :]
    safe = np.where(total == 0, 1.0, total)
    return np.asarray(np.where(total == 0, 0.0, (diff / safe) ** 2), dtype=float)


NOMINAL = Distance("nominal", _nominal)
ORDINAL = Distance("ordinal", _ordinal, depends_on_marginals=True)
INTERVAL = Distance("interval", _interval, numeric=True)
RATIO = Distance("ratio", _ratio, numeric=True)

BUILTIN_DISTANCES: dict[str, Distance] = {d.name: d for d in (NOMINAL, ORDINAL, INTERVAL, RATIO)}


def check_domain_size(n_values: int) -> str | None:
    """Refuse oversized domains; return a warning message for large ones."""
    megabytes = n_values * n_values * 8 / 1e6
    if n_values > DOMAIN_SIZE_MAX:
        raise DesignError(
            f"the value domain has {n_values} values (more than {DOMAIN_SIZE_MAX}); each "
            f"V x V matrix would need about {megabytes:,.0f} MB"
        )
    if n_values > DOMAIN_SIZE_WARN:
        return (
            f"the value domain has {n_values} values; each V x V matrix needs about "
            f"{megabytes:,.0f} MB"
        )
    return None


def per_unit_coincidences(
    unit_idx: IntArray, value_idx: IntArray, n_units: int, n_values: int
) -> tuple[sparse.csr_array, FloatArray]:
    """Each unit's contribution to the coincidence matrix.

    Row ``u`` of the returned ``(n_units, n_values**2)`` sparse array is unit u's
    V x V contribution, flattened in C order: ``(n_u n_u' - diag(n_u)) / (m_u - 1)``.
    Here ``n_u`` counts the unit's pairable values and ``m_u`` is their total.
    Units with fewer than two pairable values contribute nothing. Summing the rows
    (or weighting them, in a bootstrap) gives the coincidence matrix.

    Returns the sparse array and ``m_u`` for every unit.
    """
    unit_idx = np.asarray(unit_idx, dtype=np.intp)
    value_idx = np.asarray(value_idx, dtype=np.intp)
    counts = sparse.csr_array(
        (np.ones(len(unit_idx)), (unit_idx, value_idx)), shape=(n_units, n_values)
    )
    counts.sum_duplicates()
    counts.sort_indices()
    pairable = np.asarray(counts.sum(axis=1), dtype=float).ravel()

    indptr = counts.indptr.astype(np.intp)
    cols = counts.indices.astype(np.intp)
    data = counts.data.astype(float)
    distinct = np.diff(indptr)
    # Pair every non-zero entry with every non-zero entry of the same unit.
    entry_unit = np.repeat(np.arange(n_units), distinct)
    reps = distinct[entry_unit]
    left = np.repeat(np.arange(len(cols)), reps)
    block_start = np.repeat(np.cumsum(reps) - reps, reps)
    right = np.repeat(indptr[entry_unit], reps) + (np.arange(len(left)) - block_start)

    unit = entry_unit[left]
    same = cols[left] == cols[right]
    weight = data[left] * data[right] - np.where(same, data[left], 0.0)
    keep = (pairable[unit] >= 2) & (weight != 0)
    contribution = weight[keep] / (pairable[unit][keep] - 1)
    flat = cols[left][keep] * n_values + cols[right][keep]
    out = sparse.csr_array((contribution, (unit[keep], flat)), shape=(n_units, n_values * n_values))
    return out, pairable


def coincidence_matrix(per_unit: sparse.csr_array, n_values: int) -> FloatArray:
    """Sum the per-unit contributions into the V x V coincidence matrix."""
    flat = np.asarray(per_unit.sum(axis=0), dtype=float).ravel()
    return flat.reshape(n_values, n_values)


@dataclass(frozen=True, eq=False)
class AlphaComponents:
    """Alpha and the quantities it is computed from.

    When alpha is undefined, ``alpha``, ``d_o`` and ``d_e`` hold what could be
    computed (NaN otherwise) and ``undefined_reason`` says why. Nothing is raised
    here, so bootstrap code can count failed replicates.
    """

    alpha: float
    d_o: float
    d_e: float
    n: float
    marginals: FloatArray
    delta2: FloatArray
    expected: FloatArray
    undefined_reason: str | None

    @property
    def undefined(self) -> bool:
        return self.undefined_reason is not None


def alpha_from_coincidences(
    coincidence: FloatArray, values: Sequence[Any], distance: Distance
) -> AlphaComponents:
    """Alpha from a coincidence matrix over the declared ``values``."""
    o = np.asarray(coincidence, dtype=float)
    n_c = o.sum(axis=1)
    n = float(n_c.sum())
    delta2 = distance.matrix(values, n_c)
    if n < 2:
        nan = float("nan")
        return AlphaComponents(
            nan,
            nan,
            nan,
            n,
            n_c,
            delta2,
            np.zeros_like(o),
            "no unit has two or more pairable values",
        )
    expected = (np.outer(n_c, n_c) - np.diag(n_c)) / (n - 1)
    d_o = float((o * delta2).sum() / n)
    d_e = float((expected * delta2).sum() / n)
    if not d_e > 0:
        return AlphaComponents(
            float("nan"),
            d_o,
            d_e,
            n,
            n_c,
            delta2,
            expected,
            "expected disagreement is zero (every pairable value is in one category or at "
            "zero distance), so alpha is undefined",
        )
    return AlphaComponents(1.0 - d_o / d_e, d_o, d_e, n, n_c, delta2, expected, None)


@dataclass(frozen=True, eq=False)
class AlphaComputation:
    """Alpha for one set of pairable values, with its white-box decomposition.

    ``unit_disagreement[u]`` is ``sum(o_u * delta2)``. Divided by ``n`` it is unit
    u's contribution to D_o (the contributions sum to D_o). Divided by ``m_u`` it is
    the mean squared distance over the unit's ordered pairs of values.
    """

    components: AlphaComponents
    coincidence: FloatArray
    per_unit: sparse.csr_array
    pairable: FloatArray
    unit_disagreement: FloatArray


def compute_alpha(
    unit_idx: IntArray,
    value_idx: IntArray,
    n_units: int,
    values: Sequence[Any],
    distance: Distance,
) -> AlphaComputation:
    """Alpha from integer-coded pairable values (one entry per rating)."""
    n_values = len(values)
    per_unit, pairable = per_unit_coincidences(unit_idx, value_idx, n_units, n_values)
    o = coincidence_matrix(per_unit, n_values)
    components = alpha_from_coincidences(o, values, distance)
    unit_disagreement = np.asarray(per_unit @ components.delta2.ravel(), dtype=float)
    return AlphaComputation(components, o, per_unit, pairable, unit_disagreement)
