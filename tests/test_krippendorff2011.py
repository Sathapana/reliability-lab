"""The intermediate quantities Krippendorff (2011) prints, not just its alphas:
coincidence matrices, per-unit contributions, difference tables, D_o and D_e, and
the values-by-units form of section E."""

from itertools import combinations
from typing import Any

import numpy as np
import pytest
from fixtures.krippendorff2011 import (
    BIPOLAR_DIFFERENCES,
    CIRCULAR_DIFFERENCES,
    EXAMPLES,
    FOUR_CODERS_MISSING,
    ORDINAL_DIFFERENCES,
    RATIO_DIFFERENCES,
)
from helpers import human_study

from reliability_lab import Distance
from reliability_lab.alpha import BUILTIN_DISTANCES, INTERVAL, ORDINAL, RATIO, per_unit_coincidences

F4 = FOUR_CODERS_MISSING
LEVELS = ["nominal", "ordinal", "interval", "ratio"]
EXACT = [
    (name, level, exact)
    for name, example in EXAMPLES.items()
    for level, exact in example["exact_alpha"].items()
]


def _as_float(matrix: list[list[Any]]) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in matrix])


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_coincidence_matrix_matches_the_paper(name: str) -> None:
    example = EXAMPLES[name]
    est = human_study(example["matrix"], "nominal", example["values"]).alpha()["as_missing"]
    np.testing.assert_allclose(
        est.coincidence.to_numpy(), _as_float(example["coincidence"]), atol=1e-12
    )
    np.testing.assert_allclose(est.coincidence.to_numpy().sum(axis=1), example["marginals"])


@pytest.mark.parametrize(("name", "level", "exact"), EXACT)
def test_alpha_matches_the_exact_value(name: str, level: str, exact: Any) -> None:
    example = EXAMPLES[name]
    result = human_study(example["matrix"], level, example["values"]).alpha()
    assert result.alpha == pytest.approx(float(exact), abs=1e-12)
    assert round(float(exact), 3) == example["alpha"][level]


@pytest.mark.parametrize("level", LEVELS)
def test_observed_and_expected_disagreement(level: str) -> None:
    est = human_study(F4["matrix"], level, F4["values"]).alpha()["as_missing"]
    assert est.d_o == pytest.approx(float(F4["d_o"][level]), rel=1e-12)
    assert est.d_e == pytest.approx(float(F4["d_e"][level]), rel=1e-12)


def test_unit_contributions_described_in_the_text() -> None:
    units, codes = [], []
    for u, column in enumerate(zip(*F4["matrix"].values(), strict=True)):
        for value in column:
            if value is not None:
                units.append(u)
                codes.append(F4["values"].index(value))
    per_unit, pairable = per_unit_coincidences(np.array(units), np.array(codes), 12, 5)
    o = per_unit.toarray().reshape(12, 5, 5)
    # Unit 1: three 1s, 6 pairs, adds 6 / (3 - 1) = 3 to o_11.
    assert o[0, 0, 0] == pytest.approx(3) and o[0].sum() == pytest.approx(3)
    # Unit 2: adds 2 to o_22, 1 to o_23 and 1 to o_32, accounting for its 4 values.
    assert (o[1, 1, 1], o[1, 1, 2], o[1, 2, 1]) == pytest.approx((2, 1, 1))
    assert o[1].sum() == pytest.approx(4)
    # Unit 6: 12 mismatching pairs, each adding 1/3 to a different cell.
    mismatches = o[5][~np.eye(5, dtype=bool)]
    assert np.count_nonzero(mismatches) == 12
    np.testing.assert_allclose(mismatches[mismatches > 0], 1 / 3)
    # Unit 12: a lone value, no pairs.
    assert pairable[11] == 1 and o[11].sum() == 0


@pytest.mark.parametrize("level", LEVELS)
def test_values_by_units_form(level: str) -> None:
    """Section E: alpha and each unit's share of D_o from the values-by-units counts."""
    est = human_study(F4["matrix"], level, F4["values"]).alpha()["as_missing"]
    table = est.per_unit.set_index("unit_id").sort_index()
    counts = np.array(F4["values_by_units"], dtype=float)
    per_unit = np.array(F4["values_per_unit"], dtype=float)
    np.testing.assert_array_equal(table["pairable_values"], per_unit)
    pairable = per_unit >= 2
    totals = counts[:, pairable].sum(axis=1)
    np.testing.assert_array_equal(totals, F4["marginals"])
    n = totals.sum()
    delta2 = BUILTIN_DISTANCES[level].matrix(F4["values"], totals)
    pairs = list(combinations(range(len(totals)), 2))
    within = np.array(
        [
            sum(counts[c, u] * counts[k, u] * delta2[c, k] for c, k in pairs) / (per_unit[u] - 1)
            if pairable[u]
            else 0.0
            for u in range(counts.shape[1])
        ]
    )
    # A unit's share of D_o counts both orders of each pair.
    np.testing.assert_allclose(table["d_o_contribution"], 2 * within / n, atol=1e-12)
    between = sum(totals[c] * totals[k] * delta2[c, k] for c, k in pairs)
    alpha = 1 - (n - 1) * within.sum() / between
    assert alpha == pytest.approx(float(F4["exact_alpha"][level]), abs=1e-12)


def test_ordinal_differences_table() -> None:
    table = ORDINAL_DIFFERENCES
    delta2 = ORDINAL.matrix(table["ranks"], table["frequencies"])
    for (c, k), delta in table["delta"].items():
        assert delta2[c - 1, k - 1] == pytest.approx(delta**2)
        assert delta2[k - 1, c - 1] == pytest.approx(delta**2)


def test_an_unused_rank_does_not_change_ordinal_alpha() -> None:
    shifted = {coder: [6 if v == 5 else v for v in col] for coder, col in F4["matrix"].items()}
    alpha = human_study(shifted, "ordinal", [1, 2, 3, 4, 5, 6]).alpha().alpha
    assert alpha == pytest.approx(float(F4["exact_alpha"]["ordinal"]), abs=1e-12)


def test_standardised_ordinal_differences_give_the_same_alpha() -> None:
    def standardised(values: Any, marginals: Any) -> Any:
        d = ORDINAL.fn(values, marginals)
        return d / d.max()

    distance = Distance("ordinal_standardised", standardised, depends_on_marginals=True)
    alpha = human_study(F4["matrix"], distance, F4["values"]).alpha().alpha
    assert alpha == pytest.approx(float(F4["exact_alpha"]["ordinal"]), abs=1e-12)


def test_interval_and_ratio_tables() -> None:
    values = np.arange(-1, 5)
    expected = (values[:, None] - values[None, :]) ** 2
    np.testing.assert_array_equal(INTERVAL.matrix(list(values), np.ones(6)), expected)
    ratio = RATIO.matrix(RATIO_DIFFERENCES["values"], np.ones(6))
    for (c, k), printed in RATIO_DIFFERENCES["printed"].items():
        assert round(ratio[c, k], 2) == printed
        assert ratio[k, c] == ratio[c, k]


def test_circular_and_bipolar_tables_as_custom_distances() -> None:
    circumference = CIRCULAR_DIFFERENCES["circumference"]
    circular = Distance(
        "circular",
        lambda v, m: np.sin(np.pi * (v[:, None] - v[None, :]) / circumference) ** 2,
        numeric=True,
    )
    d = circular.matrix(CIRCULAR_DIFFERENCES["values"], np.ones(6))
    np.testing.assert_allclose(d[0], CIRCULAR_DIFFERENCES["first_row"], atol=1e-12)
    # Every row is the first row rotated: only the distance around the circle counts.
    for c in range(6):
        np.testing.assert_allclose(d[c], np.roll(d[0], c), atol=1e-12)

    def bipolar(v: Any, m: Any) -> Any:
        c, k = v[:, None], v[None, :]
        denominator = (c + k - 2 * v.min()) * (2 * v.max() - c - k)
        return np.where(c == k, 0.0, (c - k) ** 2 / np.where(denominator == 0, 1, denominator))

    values = BIPOLAR_DIFFERENCES["values"]
    d = Distance("bipolar", bipolar, numeric=True).matrix(values, np.ones(5))
    for (c, k), printed in BIPOLAR_DIFFERENCES["printed"].items():
        assert round(d[values.index(c), values.index(k)], 3) == printed
