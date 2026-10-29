import math

import numpy as np
import pytest
from fixtures.krippendorff2011 import EXAMPLES, FOUR_CODERS_MISSING
from helpers import human_study, random_matrix
from reference import reference_alpha

from reliability_lab import NOMINAL, DesignError, Distance, UndefinedAlphaError
from reliability_lab.alpha import (
    DOMAIN_SIZE_MAX,
    DOMAIN_SIZE_WARN,
    check_domain_size,
    compute_alpha,
    per_unit_coincidences,
)

LEVELS = ["nominal", "ordinal", "interval", "ratio", "bipolar"]

PUBLISHED = [
    (name, level, expected)
    for name, example in EXAMPLES.items()
    for level, expected in example["alpha"].items()
]


@pytest.mark.parametrize(("name", "level", "expected"), PUBLISHED)
def test_reproduces_published_alpha(name: str, level: str, expected: float) -> None:
    example = EXAMPLES[name]
    result = human_study(example["matrix"], level, example["values"]).alpha()
    assert round(result.alpha, 3) == expected
    assert result["as_missing"].n_pairable_values == example["n_pairable"]


def test_four_coder_example_white_box() -> None:
    result = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5]).alpha()
    est = result["as_missing"]
    o = est.coincidence.to_numpy()
    # Published coincidence marginals for example C: n_1..n_5 = 9, 13, 10, 5, 3.
    np.testing.assert_allclose(o.sum(axis=1), [9, 13, 10, 5, 3])
    np.testing.assert_allclose(o, o.T)
    np.testing.assert_allclose(est.per_unit["d_o_contribution"].sum(), est.d_o)
    assert est.n_unpairable_units == 1
    unit12 = est.per_unit.set_index("unit_id").loc[11]
    assert unit12["pairable_values"] == 1
    assert math.isnan(unit12["disagreement"])
    assert unit12["d_o_contribution"] == 0
    # Observed agreement: the share of pairable value pairs that match exactly.
    assert est.observed_agreement == pytest.approx(np.trace(o) / 40)


@pytest.mark.parametrize("level", LEVELS)
def test_engine_matches_reference_on_random_data(level: str) -> None:
    rng = np.random.default_rng(2026)
    values = [1, 2, 3, 4, 5]
    for _ in range(25):
        matrix = random_matrix(
            rng, int(rng.integers(2, 6)), int(rng.integers(3, 30)), values, p_missing=0.25
        )
        expected = reference_alpha(matrix, level, values)
        if math.isnan(expected):
            continue
        assert human_study(matrix, level, values).alpha().alpha == pytest.approx(
            expected, abs=1e-12
        )


def test_perfect_agreement_gives_one() -> None:
    matrix = {"a": [1, 2, 3, 1], "b": [1, 2, 3, 1], "c": [1, None, 3, 1]}
    for level in LEVELS:
        assert human_study(matrix, level, [1, 2, 3]).alpha().alpha == pytest.approx(1.0)


def test_single_value_is_undefined() -> None:
    matrix = {"a": [2, 2, 2], "b": [2, 2, 2]}
    with pytest.raises(UndefinedAlphaError, match="expected disagreement is zero"):
        human_study(matrix, "nominal", [1, 2, 3]).alpha()


def test_no_pairable_values_is_undefined() -> None:
    matrix = {"a": [1, None, 3], "b": [None, 2, None]}
    with pytest.raises(UndefinedAlphaError, match="no unit has two or more"):
        human_study(matrix, "nominal", [1, 2, 3]).alpha()


def test_per_unit_coincidences_counts_repeated_values() -> None:
    # One unit with values [0, 0, 1] (m_u = 3), one unit with a single value.
    per_unit, pairable = per_unit_coincidences(
        np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), n_units=2, n_values=2
    )
    np.testing.assert_allclose(pairable, [3, 1])
    unit0 = per_unit.toarray()[0].reshape(2, 2)
    # o_00 = 2*1/2, o_01 = o_10 = 2*1/2, o_11 = 0.
    np.testing.assert_allclose(unit0, [[1.0, 1.0], [1.0, 0.0]])
    assert per_unit.toarray()[1].sum() == 0


def test_interval_alpha_is_scale_invariant() -> None:
    matrix = FOUR_CODERS_MISSING["matrix"]
    scaled = {c: [None if v is None else 10 * v + 7 for v in row] for c, row in matrix.items()}
    a = human_study(matrix, "interval", [1, 2, 3, 4, 5]).alpha().alpha
    b = human_study(scaled, "interval", [17, 27, 37, 47, 57]).alpha().alpha
    assert a == pytest.approx(b)


def test_custom_matrix_distance_equals_builtin() -> None:
    custom = Distance.from_matrix("my_nominal", 1 - np.eye(5))
    matrix = FOUR_CODERS_MISSING["matrix"]
    a = human_study(matrix, "nominal", [1, 2, 3, 4, 5]).alpha().alpha
    result = human_study(matrix, custom, [1, 2, 3, 4, 5]).alpha()
    assert result.alpha == pytest.approx(a)
    assert result.measurement == "my_nominal"


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[0, 1], [2, 0]], "not symmetric"),
        ([[1, 1], [1, 0]], "zero on the diagonal"),
        ([[0, -1], [-1, 0]], "negative"),
        ([[0, np.inf], [np.inf, 0]], "non-finite"),
        ([[0, 1, 1]], "square"),
    ],
)
def test_custom_distance_validation(matrix: list[list[float]], message: str) -> None:
    with pytest.raises(DesignError, match=message):
        Distance.from_matrix("bad", matrix)


def test_custom_distance_must_match_domain_size() -> None:
    custom = Distance.from_matrix("two", 1 - np.eye(2))
    with pytest.raises(DesignError, match="2 x 2"):
        human_study({"a": [1, 2], "b": [1, 3]}, custom, [1, 2, 3])


def test_compute_alpha_never_raises_on_degenerate_input() -> None:
    computed = compute_alpha(np.array([0, 0]), np.array([1, 1]), 1, [0, 1], NOMINAL)
    assert computed.components.undefined
    assert math.isnan(computed.components.alpha)


def test_domain_size_guard() -> None:
    assert check_domain_size(DOMAIN_SIZE_WARN) is None
    assert "MB" in (check_domain_size(DOMAIN_SIZE_WARN + 1) or "")
    with pytest.raises(DesignError, match="more than"):
        check_domain_size(DOMAIN_SIZE_MAX + 1)
