"""A slow, obviously correct Krippendorff's alpha, used as a test oracle.

It enumerates every ordered pair of values within each unit and applies the
textbook formulas directly. It shares no code with the package.
"""

import math
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any


@dataclass
class ReferenceComponents:
    n: float
    d_o: float
    d_e: float
    d2: Callable[[Any, Any], float]

    @property
    def alpha(self) -> float:
        if self.n == 0 or self.d_e == 0:
            return math.nan
        return 1 - self.d_o / self.d_e


def reference_components(
    matrix: dict[Hashable, Sequence[Any]], level: str, values: Sequence[Any]
) -> ReferenceComponents:
    """n, D_o, D_e and the squared distance for a coders x units matrix (``None`` = missing)."""
    n_units = len(next(iter(matrix.values())))
    units = [[row[u] for row in matrix.values() if row[u] is not None] for u in range(n_units)]

    o: Counter[tuple[Any, Any]] = Counter()
    for unit in units:
        m = len(unit)
        if m < 2:
            continue
        for a, b in permutations(unit, 2):
            o[(a, b)] += 1 / (m - 1)
    n_c: Counter[Any] = Counter()
    for (a, _), w in o.items():
        n_c[a] += w
    n = sum(n_c.values())
    order = list(values)

    def d2(a: Any, b: Any) -> float:
        if level == "nominal":
            return 0.0 if a == b else 1.0
        if level == "interval":
            return float((a - b) ** 2)
        if level == "ratio":
            return 0.0 if a + b == 0 else float(((a - b) / (a + b)) ** 2)
        if level == "ordinal":
            i, j = sorted((order.index(a), order.index(b)))
            between = sum(n_c[order[g]] for g in range(i, j + 1))
            return float((between - (n_c[a] + n_c[b]) / 2) ** 2)
        raise ValueError(level)

    if n == 0:
        return ReferenceComponents(0.0, math.nan, math.nan, d2)
    d_o = sum(w * d2(a, b) for (a, b), w in o.items()) / n
    d_e = sum(n_c[a] * n_c[b] * d2(a, b) for a in n_c for b in n_c) / (n * (n - 1))
    return ReferenceComponents(n, d_o, d_e, d2)


def reference_alpha(
    matrix: dict[Hashable, Sequence[Any]], level: str, values: Sequence[Any]
) -> float:
    """Alpha for a coders x units matrix (``None`` = missing). NaN when undefined."""
    return reference_components(matrix, level, values).alpha
