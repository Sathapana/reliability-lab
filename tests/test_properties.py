import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from helpers import long_from_matrix
from hypothesis import given, settings
from hypothesis import strategies as st
from reference import reference_alpha

from reliability_lab import RatingFrame, ReliabilityStudy, UndefinedAlphaError

LEVELS = ["nominal", "ordinal", "interval", "ratio", "bipolar"]

# Random data often has one dominant category; the warning is expected here.
pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


@st.composite
def matrices(draw: st.DrawFn) -> tuple[dict[str, list[Any]], list[int]]:
    n_values = draw(st.integers(2, 5))
    values = list(range(1, n_values + 1))
    n_coders = draw(st.integers(2, 5))
    n_units = draw(st.integers(1, 12))
    cell = st.one_of(st.none(), st.sampled_from(values))
    matrix = {
        f"c{i}": draw(st.lists(cell, min_size=n_units, max_size=n_units)) for i in range(n_coders)
    }
    return matrix, values


def study(
    matrix: dict[str, list[Any]], level: str, values: list[Any], df: pd.DataFrame | None = None
) -> ReliabilityStudy:
    frame = long_from_matrix(matrix) if df is None else df
    ratings = RatingFrame.from_long(frame, unit="unit", rater="coder", value="value")
    return ReliabilityStudy(
        ratings, measurement=level, values=values, coders="rater_id", design="human_panel"
    )


def alpha_or_nan(s: ReliabilityStudy) -> float:
    try:
        return s.alpha().alpha
    except UndefinedAlphaError:
        return math.nan


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-10)


def has_two_coders(matrix: dict[str, list[Any]]) -> bool:
    return sum(any(v is not None for v in row) for row in matrix.values()) >= 2


@settings(max_examples=150, deadline=None)
@given(matrices(), st.sampled_from(LEVELS))
def test_matches_reference_oracle(data: tuple[dict[str, list[Any]], list[int]], level: str) -> None:
    matrix, values = data
    if not has_two_coders(matrix):
        return
    assert same(alpha_or_nan(study(matrix, level, values)), reference_alpha(matrix, level, values))


@settings(max_examples=75, deadline=None)
@given(matrices(), st.sampled_from(LEVELS), st.randoms(use_true_random=False))
def test_invariant_to_row_and_coder_order(
    data: tuple[dict[str, list[Any]], list[int]], level: str, rnd: Any
) -> None:
    matrix, values = data
    if not has_two_coders(matrix):
        return
    base = alpha_or_nan(study(matrix, level, values))
    shuffled_rows = long_from_matrix(matrix).sample(frac=1.0, random_state=rnd.randint(0, 9999))
    assert same(alpha_or_nan(study(matrix, level, values, shuffled_rows)), base)
    coders = list(matrix)
    rnd.shuffle(coders)
    reordered = {c: matrix[c] for c in coders}
    assert same(alpha_or_nan(study(reordered, level, values)), base)


@settings(max_examples=75, deadline=None)
@given(matrices(), st.permutations(range(5)))
def test_nominal_alpha_invariant_to_relabelling(
    data: tuple[dict[str, list[Any]], list[int]], perm: list[int]
) -> None:
    matrix, values = data
    if not has_two_coders(matrix):
        return
    relabel = {v: f"label{perm[v - 1]}" for v in values}
    renamed = {c: [None if v is None else relabel[v] for v in row] for c, row in matrix.items()}
    a = alpha_or_nan(study(matrix, "nominal", values))
    b = alpha_or_nan(study(renamed, "nominal", [relabel[v] for v in values]))
    assert same(a, b)


@settings(max_examples=75, deadline=None)
@given(matrices(), st.sampled_from(LEVELS))
def test_alpha_at_most_one_and_unpairable_unit_changes_nothing(
    data: tuple[dict[str, list[Any]], list[int]], level: str
) -> None:
    matrix, values = data
    if not has_two_coders(matrix):
        return
    a = alpha_or_nan(study(matrix, level, values))
    assert math.isnan(a) or a <= 1 + 1e-12
    coders = list(matrix)
    extended = {c: [*row, values[0] if c == coders[0] else None] for c, row in matrix.items()}
    assert same(alpha_or_nan(study(extended, level, values)), a)


@settings(max_examples=50, deadline=None)
@given(matrices(), st.sampled_from(LEVELS))
def test_white_box_decomposition_is_consistent(
    data: tuple[dict[str, list[Any]], list[int]], level: str
) -> None:
    matrix, values = data
    if not has_two_coders(matrix):
        return
    try:
        est = study(matrix, level, values).alpha()["as_missing"]
    except UndefinedAlphaError:
        return
    o = est.coincidence.to_numpy()
    np.testing.assert_allclose(o, o.T, atol=1e-12)
    assert o.sum() == pytest.approx(est.n_pairable_values)
    assert est.per_unit["d_o_contribution"].sum() == pytest.approx(est.d_o, abs=1e-12)
    assert est.expected.to_numpy().sum() == pytest.approx(est.n_pairable_values)
