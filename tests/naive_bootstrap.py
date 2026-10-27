"""Loop oracles for the bootstrap, sharing no code with the package.

``naive_units_alphas`` expands each replicate's cluster weights into duplicated
units and recomputes alpha with the P1 reference. ``naive_kalpha`` transcribes
Krippendorff's bootstrap algorithm note line by line, consuming given draws.
``reference_pair_errors`` lists E(r) for every within-unit pair from the reference.
"""

import math
from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from reference import reference_alpha, reference_components


def naive_units_alphas(
    matrix: dict[Hashable, Sequence[Any]],
    level: str,
    values: Sequence[Any],
    cluster_of_unit: dict[int, int],
    weights: np.ndarray,
) -> np.ndarray:
    """Replicate alphas for ``weights`` (replicates x clusters).

    ``cluster_of_unit`` maps a matrix column to its cluster.
    """
    coders = list(matrix)
    n_units = len(next(iter(matrix.values())))
    out = []
    for w in weights:
        expanded: dict[Hashable, list[Any]] = {c: [] for c in coders}
        for u in range(n_units):
            if u not in cluster_of_unit:
                continue
            for _ in range(int(w[cluster_of_unit[u]])):
                for c in coders:
                    expanded[c].append(matrix[c][u])
        out.append(reference_alpha(expanded, level, values) if expanded[coders[0]] else math.nan)
    return np.array(out)


def reference_pair_errors(
    matrix: dict[Hashable, Sequence[Any]], level: str, values: Sequence[Any]
) -> list[float]:
    """E(r) = 2 delta² / (n D_e) for every unordered pair of values within a unit."""
    ref = reference_components(matrix, level, values)
    n_units = len(next(iter(matrix.values())))
    errors = []
    for u in range(n_units):
        unit = [row[u] for row in matrix.values() if row[u] is not None]
        if len(unit) < 2:
            continue
        errors.extend(2 * ref.d2(a, b) / (ref.n * ref.d_e) for a, b in combinations(unit, 2))
    return errors


def naive_kalpha(
    errors: Sequence[float],
    unit_sizes: Sequence[int],
    draws: np.ndarray,
    redraws: np.ndarray,
) -> np.ndarray:
    """The algorithm note, for given pair draws.

    For each replicate: alpha = 1; for each unit u (sizes m_u in pool order, units
    with m_u >= 2 only), draw m_u(m_u - 1)/2 pairs and subtract E(r)/(m_u - 1) for
    each. A unit's second pick is replaced by the next ``redraws`` entry when it
    repeats the first. Replicates below -1 are set to -1.
    """
    out = []
    for x in range(len(draws)):
        alpha = 1.0
        position = 0
        second_picks = 0
        for m in unit_sizes:
            first = None
            for t in range(m * (m - 1) // 2):
                r = int(draws[x, position])
                if t == 0:
                    first = r
                elif t == 1:
                    if r == first:
                        r = int(redraws[x, second_picks])
                    second_picks += 1
                alpha -= errors[r] / (m - 1)
                position += 1
        out.append(max(alpha, -1.0))
    return np.array(out)
