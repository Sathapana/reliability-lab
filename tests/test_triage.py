"""The category_skew diagnostic and the per-unit status counts used for triage."""

import json
import math

import numpy as np
import pytest
from helpers import human_study
from test_retries import ratings_of, retried_log, study_of
from test_status_policies import (
    OBS,
    PARSE,
    PROVIDER,
    abstaining,
    make,
    nominal_with_errors,
    ratings_from,
)

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def test_even_use_of_the_scale_has_entropy_one() -> None:
    matrix = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
    skew = human_study(matrix, "ordinal", [1, 2, 3, 4]).alpha().diagnostics["category_skew"]
    assert skew.values["n_categories"] == 4
    assert skew.values["normalised_entropy"] == pytest.approx(1.0)
    assert skew.values["effective_categories"] == pytest.approx(4.0)
    assert skew.values["largest_share"] == pytest.approx(0.25)
    assert skew.table is None


def test_unused_declared_values_lower_the_entropy() -> None:
    # Two of four declared values used equally: H = ln 2, so H / ln 4 = 1/2.
    matrix = {"a": [1, 2, 1, 2, 1], "b": [1, 2, 1, 2, 2]}
    values = human_study(matrix, "nominal", [1, 2, 3, 4]).alpha().diagnostics["category_skew"]
    assert values.values["effective_categories"] == pytest.approx(2.0)
    assert values.values["normalised_entropy"] == pytest.approx(0.5)


def test_one_dominant_value() -> None:
    matrix = {"a": ["x"] * 9 + ["y"], "b": ["x"] * 10}
    result = human_study(matrix, "nominal", ["x", "y"]).alpha()
    skew = result.diagnostics["category_skew"].values
    p = np.array([19, 1]) / 20
    assert skew["normalised_entropy"] == pytest.approx(-(p * np.log(p)).sum() / math.log(2))
    assert skew["largest_value"] == "x" and skew["largest_share"] == pytest.approx(0.95)
    payload = json.loads(result.to_json())["diagnostics"]["category_skew"]
    assert payload["values"]["largest_value"] == "x"
    assert "category_skew" in result.summary()
    # The diagnostic adds no table, so tables and Parquet output keep their files.
    assert "diagnostic_category_skew" not in result.tables()


def test_per_unit_counts_abstentions_and_inapplicable_ratings() -> None:
    est = make(
        abstaining(), "nominal", ["a", "b"], abstain_as="missing", inapplicable_as="missing"
    ).alpha()["as_missing"]
    table = est.per_unit.set_index("unit_id").sort_index()
    assert list(table["abstained"]) == [0, 0, 1, 0]
    assert list(table["inapplicable"]) == [0, 0, 0, 1]
    assert list(table["failed"]) == [0, 0, 0, 0]


def test_per_unit_counts_failures_under_every_policy() -> None:
    result = make(
        nominal_with_errors(), "nominal", ["a", "b"], missing_policies=["as_missing", "as_category"]
    ).alpha()
    for estimate in result.estimates.values():
        table = estimate.per_unit.set_index("unit_id").sort_index()
        assert list(table["failed"]) == [0, 0, 1, 2, 0]
        assert list(table["abstained"]) == [0] * 5
    assert list(result.tables()["per_unit"].columns[-3:]) == ["abstained", "inapplicable", "failed"]


def test_retried_attempts_are_counted_once() -> None:
    est = study_of(ratings_of(retried_log()), retries="last").alpha()["as_missing"]
    table = est.per_unit.set_index("unit_id").sort_index()
    # u0's parse error was retried successfully; u3's two failed attempts are one call.
    assert dict(table["failed"]) == {"u0": 0, "u1": 0, "u2": 0, "u3": 1}


def test_the_adjudication_list() -> None:
    ratings = ratings_from(
        {
            "c1": [("a", OBS), ("a", OBS), ("b", OBS), (None, PARSE)],
            "c2": [("a", OBS), ("b", OBS), ("b", OBS), ("a", OBS)],
            "c3": [("a", OBS), ("b", OBS), ("a", OBS), (None, PROVIDER)],
        }
    )
    per_unit = make(ratings, "nominal", ["a", "b"]).alpha()["as_missing"].per_unit
    triage = per_unit.sort_values("d_o_contribution", ascending=False)
    assert triage.iloc[0]["d_o_contribution"] > 0
    assert per_unit.set_index("unit_id").loc[3, "failed"] == 2
