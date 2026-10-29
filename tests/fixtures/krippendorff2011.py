"""Worked examples from Krippendorff (2011), *Computing Krippendorff's alpha-reliability*.

Matrices are coders x units; ``None`` marks a missing rating. Data, coincidence
matrices, difference tables and alphas are transcribed from the paper (checked
against its text on 26 and 29 Oct 2026). The paper prints alpha to three decimals;
``exact_alpha``, ``d_o`` and ``d_e`` are the same quantities as fractions, computed
from the paper's formulas and matching its printed values.
"""

from fractions import Fraction
from typing import Any

THIRD = Fraction(1, 3)

#: Section A: binary data, two observers, no missing data.
BINARY_TWO_CODERS: dict[str, Any] = {
    "matrix": {
        "Meg": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        "Owen": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    },
    "values": [0, 1],
    "n_pairable": 20,
    "coincidence": [[10, 4], [4, 2]],
    "marginals": [14, 6],
    "alpha": {"nominal": 0.095},
    # 1 - (n - 1) o_01 / (n_0 n_1) = 1 - 19 * 4 / (14 * 6)
    "exact_alpha": {"nominal": Fraction(2, 21)},
}

#: Section B: nominal data, two observers, no missing data.
NOMINAL_TWO_CODERS: dict[str, Any] = {
    "matrix": {
        "Ben": list("aabbdcccedda"),
        "Gerry": list("babbbccceddd"),
    },
    "values": ["a", "b", "c", "d", "e"],
    "n_pairable": 24,
    "coincidence": [
        [2, 1, 0, 1, 0],
        [1, 4, 0, 1, 0],
        [0, 0, 6, 0, 0],
        [1, 1, 0, 4, 0],
        [0, 0, 0, 0, 2],
    ],
    "marginals": [4, 6, 6, 6, 2],
    "alpha": {"nominal": 0.692},
    # (23 * 18 - 104) / (24 * 23 - 104)
    "exact_alpha": {"nominal": Fraction(155, 224)},
}

#: Sections C and D: four observers, twelve units, seven missing values. Unit 12
#: has a single value (from observer B), so it cannot be paired.
FOUR_CODERS_MISSING: dict[str, Any] = {
    "matrix": {
        "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    },
    "values": [1, 2, 3, 4, 5],
    "n_pairable": 40,
    "coincidence": [
        [7, 4 * THIRD, THIRD, THIRD, 0],
        [4 * THIRD, 10, 4 * THIRD, THIRD, 0],
        [THIRD, 4 * THIRD, 8, THIRD, 0],
        [THIRD, THIRD, THIRD, 4, 0],
        [0, 0, 0, 0, 3],
    ],
    "marginals": [9, 13, 10, 5, 3],
    "alpha": {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797},
    "exact_alpha": {
        "nominal": Fraction(113, 152),
        "ordinal": Fraction(108577, 133160),
        "interval": Fraction(951, 1120),
        "ratio": Fraction(18222619, 22852465),
    },
    "d_o": {
        "nominal": Fraction(1, 5),
        "ordinal": Fraction(1891, 40),
        "interval": Fraction(13, 30),
        "ratio": Fraction(59357, 2646000),
    },
    "d_e": {
        "nominal": Fraction(152, 195),
        "ordinal": Fraction(3329, 13),
        "interval": Fraction(112, 39),
        "ratio": Fraction(4570493, 41277600),
    },
    # Section E: values (rows 1-5) by units (columns 1-12), n_uc, and the number
    # of values in each unit, n_u.
    "values_by_units": [
        [3, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2, 0],
        [0, 3, 0, 0, 4, 1, 0, 1, 4, 0, 0, 0],
        [0, 1, 4, 4, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    ],
    "values_per_unit": [3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1],
}

#: Section D, page 6: ordinal differences delta (before squaring) for the
#: frequencies of example C with an unused 5th rank added. The paper prints the
#: upper triangle as delta squared and the lower triangle as its value; its lower
#: cell (5th, 1st) reads 992, but 32.5 squared is 1056.25, as its upper triangle
#: and formula give.
ORDINAL_DIFFERENCES: dict[str, Any] = {
    "ranks": [1, 2, 3, 4, 5, 6],
    "frequencies": [9, 13, 10, 5, 0, 3],
    "delta": {
        (1, 2): 11, (1, 3): 22.5, (1, 4): 30, (1, 5): 32.5, (1, 6): 34,
        (2, 3): 11.5, (2, 4): 19, (2, 5): 21.5, (2, 6): 23,
        (3, 4): 7.5, (3, 5): 10, (3, 6): 11.5,
        (4, 5): 2.5, (4, 6): 4,
        (5, 6): 1.5,
    },
}  # fmt: skip

#: Section D, page 7: ratio differences ((c - k) / (c + k))^2, printed to two decimals.
RATIO_DIFFERENCES: dict[str, Any] = {
    "values": [0, 1, 2, 3, 4, 5],
    "printed": {
        (1, 0): 1, (2, 0): 1, (3, 0): 1, (4, 0): 1, (5, 0): 1,
        (2, 1): 0.11, (3, 1): 0.25, (4, 1): 0.36, (5, 1): 0.44,
        (3, 2): 0.04, (4, 2): 0.11, (5, 2): 0.18,
        (4, 3): 0.02, (5, 3): 0.06,
        (5, 4): 0.01,
    },
}  # fmt: skip

#: Section D, page 7: circular differences sin^2(180 (c - k) / U) with U = 6.
CIRCULAR_DIFFERENCES: dict[str, Any] = {
    "values": [0, 1, 2, 3, 4, 5],
    "circumference": 6,
    "first_row": [0, 0.25, 0.75, 1, 0.75, 0.25],
}

#: Section D, page 7: bipolar differences (c - k)^2 / ((c + k - 2 c_min)(2 c_max - c - k)),
#: printed to three decimals.
BIPOLAR_DIFFERENCES: dict[str, Any] = {
    "values": [-2, -1, 0, 1, 2],
    "printed": {
        (-1, -2): 0.143, (0, -2): 0.333, (1, -2): 0.600, (2, -2): 1.0,
        (0, -1): 0.067, (1, -1): 0.250, (2, -1): 0.600,
        (1, 0): 0.067, (2, 0): 0.333,
        (2, 1): 0.143,
    },
}  # fmt: skip

EXAMPLES = {
    "binary_two_coders": BINARY_TWO_CODERS,
    "nominal_two_coders": NOMINAL_TWO_CODERS,
    "four_coders_missing": FOUR_CODERS_MISSING,
}
