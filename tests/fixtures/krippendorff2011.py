"""Worked examples from Krippendorff (2011), *Computing Krippendorff's alpha-reliability*.

Matrices are coders x units; ``None`` marks a missing rating. Data and expected
values are transcribed from the paper (checked against its text on 26 Oct 2026).
The paper prints alpha to three decimals. The ratio value for example C did not
extract cleanly from the PDF, so it is also covered by the parity tests against
the ``krippendorff`` package.
"""

from typing import Any

#: Section A: binary data, two observers, no missing data.
BINARY_TWO_CODERS: dict[str, Any] = {
    "matrix": {
        "Meg": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        "Owen": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    },
    "values": [0, 1],
    "n_pairable": 20,
    "alpha": {"nominal": 0.095},
}

#: Section B: nominal data, two observers, no missing data.
NOMINAL_TWO_CODERS: dict[str, Any] = {
    "matrix": {
        "Ben": list("aabbdcccedda"),
        "Gerry": list("babbbccceddd"),
    },
    "values": ["a", "b", "c", "d", "e"],
    "n_pairable": 24,
    "alpha": {"nominal": 0.692},
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
    "alpha": {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797},
}

EXAMPLES = {
    "binary_two_coders": BINARY_TWO_CODERS,
    "nominal_two_coders": NOMINAL_TWO_CODERS,
    "four_coders_missing": FOUR_CODERS_MISSING,
}
