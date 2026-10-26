from typing import Any

import pandas as pd
import pytest
from reference import reference_alpha

from reliability_lab import DesignError, RatingFrame, ReliabilityStudy

E = "<error>"


def ratings_from(cells: dict[str, list[tuple[Any, str] | None]]) -> RatingFrame:
    """Coders x units of (value, status) cells; None means no attempt."""
    rows = [
        {"unit": u, "coder": coder, "value": cell[0], "status": cell[1]}
        for coder, unit_cells in cells.items()
        for u, cell in enumerate(unit_cells)
        if cell is not None
    ]
    df = pd.DataFrame(rows)
    return RatingFrame.from_long(df, unit="unit", rater="coder", value="value", status="status")


def make(ratings: RatingFrame, level: str, values: list[Any], **kwargs: Any) -> ReliabilityStudy:
    return ReliabilityStudy(
        ratings, measurement=level, values=values, coders="rater_id", design="human_panel", **kwargs
    )


OBS = "observed"
PARSE = "parse_error"
PROVIDER = "provider_error"


def nominal_with_errors() -> RatingFrame:
    return ratings_from(
        {
            "c1": [("a", OBS), ("b", OBS), ("a", OBS), (None, PROVIDER), ("b", OBS)],
            "c2": [("a", OBS), ("b", OBS), (None, PARSE), (None, PARSE), ("a", OBS)],
        }
    )


def test_as_category_treats_failures_as_one_extra_category() -> None:
    result = make(nominal_with_errors(), "nominal", ["a", "b"]).alpha()
    assert result.missing_policies == ("as_missing", "as_category")
    expected_missing = reference_alpha(
        {"c1": ["a", "b", "a", None, "b"], "c2": ["a", "b", None, None, "a"]}, "nominal", ["a", "b"]
    )
    expected_category = reference_alpha(
        {"c1": ["a", "b", "a", E, "b"], "c2": ["a", "b", E, E, "a"]}, "nominal", ["a", "b", E]
    )
    assert result["as_missing"].alpha == pytest.approx(expected_missing)
    assert result["as_category"].alpha == pytest.approx(expected_category)
    assert list(result["as_category"].coincidence.index) == ["a", "b", E]


def test_sensitivity_imputes_pessimistically_and_optimistically() -> None:
    ratings = ratings_from(
        {
            "c1": [(1, OBS), (5, OBS), (3, OBS), (2, OBS), (4, OBS), (None, PROVIDER)],
            "c2": [(1, OBS), (5, OBS), (None, PARSE), (2, OBS), (4, OBS), None],
        }
    )
    result = make(ratings, "interval", [1, 2, 3, 4, 5]).alpha()
    assert set(result.estimates) == {
        "as_missing",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    }
    pessimistic = result["sensitivity_pessimistic"]
    optimistic = result["sensitivity_optimistic"]
    # Unit 2's failure is imputed to the farthest value (1; ties go to the first
    # value in domain order) or the closest (3). Unit 5's failure has no other value
    # in its unit and stays missing.
    base = {"c1": [1, 5, 3, 2, 4, None]}
    assert pessimistic.alpha == pytest.approx(
        reference_alpha({**base, "c2": [1, 5, 1, 2, 4, None]}, "interval", [1, 2, 3, 4, 5])
    )
    assert optimistic.alpha == pytest.approx(
        reference_alpha({**base, "c2": [1, 5, 3, 2, 4, None]}, "interval", [1, 2, 3, 4, 5])
    )
    assert pessimistic.alpha < result.alpha <= optimistic.alpha
    assert (pessimistic.n_imputed, pessimistic.n_unanchored) == (1, 1)
    assert pessimistic.policy == "sensitivity"
    assert "stay missing" in result.summary()


def test_ordinal_defaults_to_sensitivity() -> None:
    ratings = ratings_from(
        {"c1": [(1, OBS), (2, OBS), (3, OBS)], "c2": [(1, OBS), (None, PARSE), (3, OBS)]}
    )
    assert make(ratings, "ordinal", [1, 2, 3]).missing_policies == ("as_missing", "sensitivity")


@pytest.mark.parametrize(
    ("policies", "level", "message"),
    [
        (["as_missing"], "nominal", "at least two"),
        (["as_category"], "nominal", "always reported"),
        (["as_missing", "as_category"], "ordinal", "nominal data only"),
        (["as_missing", "bogus"], "nominal", "unknown missing policies"),
        (["as_missing", "as_missing"], "nominal", "repeat"),
    ],
)
def test_policy_declaration_errors(policies: list[str], level: str, message: str) -> None:
    values = ["a", "b"] if level == "nominal" else [1, 2]
    ratings = ratings_from(
        {
            "c1": [(values[0], OBS), (values[1], OBS)],
            "c2": [(values[0], OBS), (None, PARSE)],
        }
    )
    with pytest.raises(DesignError, match=message):
        make(ratings, level, values, missing_policies=policies)


def test_single_policy_is_fine_without_failures() -> None:
    ratings = ratings_from({"c1": [("a", OBS), ("b", OBS)], "c2": [("a", OBS), ("b", OBS)]})
    result = make(ratings, "nominal", ["a", "b"], missing_policies=["as_missing"]).alpha()
    assert list(result.estimates) == ["as_missing"]


def test_policies_coincide_note_without_failures() -> None:
    ratings = ratings_from(
        {"c1": [("a", OBS), ("b", OBS), ("a", OBS)], "c2": [("a", OBS), ("b", OBS), ("b", OBS)]}
    )
    result = make(ratings, "nominal", ["a", "b"]).alpha()
    assert result["as_category"].alpha == pytest.approx(result.alpha)
    assert any("coincides" in note for note in result.notes)


def abstaining() -> RatingFrame:
    return ratings_from(
        {
            "c1": [("a", OBS), ("b", OBS), (None, "abstained"), ("a", OBS)],
            "c2": [("a", OBS), ("b", OBS), ("b", OBS), (None, "inapplicable")],
        }
    )


def test_abstentions_must_be_declared() -> None:
    with pytest.raises(DesignError, match="status 'abstained'"):
        make(abstaining(), "nominal", ["a", "b", "skip"], inapplicable_as="missing")
    with pytest.raises(DesignError, match="status 'inapplicable'"):
        make(abstaining(), "nominal", ["a", "b", "skip"], abstain_as="missing")


def test_abstention_as_missing_or_as_value() -> None:
    as_missing = make(
        abstaining(), "nominal", ["a", "b", "skip"], abstain_as="missing", inapplicable_as="missing"
    ).alpha()
    assert as_missing.alpha == pytest.approx(
        reference_alpha(
            {"c1": ["a", "b", None, "a"], "c2": ["a", "b", "b", None]}, "nominal", ["a", "b"]
        )
    )
    as_value = make(
        abstaining(), "nominal", ["a", "b", "skip"], abstain_as="skip", inapplicable_as="skip"
    ).alpha()
    assert as_value.alpha == pytest.approx(
        reference_alpha(
            {"c1": ["a", "b", "skip", "a"], "c2": ["a", "b", "b", "skip"]},
            "nominal",
            ["a", "b", "skip"],
        )
    )
    assert as_value.abstain_as == "skip"
    assert as_value.coverage.status_counts["abstained"] == 1


def test_abstention_value_must_be_in_domain() -> None:
    with pytest.raises(DesignError, match="neither 'missing' nor a declared value"):
        make(abstaining(), "nominal", ["a", "b"], abstain_as="skip", inapplicable_as="missing")


@pytest.mark.parametrize(
    ("values", "level", "message"),
    [
        ("ab", "nominal", "sequence"),
        (["a"], "nominal", "at least two"),
        (["a", "a"], "nominal", "repeat"),
        (["a", None], "nominal", "missing values"),
        (["a", E], "nominal", "reserved"),
        (["a", "b"], "interval", "numeric"),
        ([0, float("inf")], "interval", "finite"),
        ([-1, 1], "ratio", "non-negative"),
        ([1, 2], "cosine", "measurement must be"),
    ],
)
def test_domain_declaration_errors(values: Any, level: str, message: str) -> None:
    ratings = ratings_from({"c1": [("a", OBS)], "c2": [("a", OBS)]})
    with pytest.raises(DesignError, match=message):
        make(ratings, level, values)


def test_observed_values_outside_domain_are_refused() -> None:
    ratings = ratings_from({"c1": [("a", OBS), ("c", OBS)], "c2": [("a", OBS), ("b", OBS)]})
    with pytest.raises(DesignError, match=r"outside the declared domain: values \['c'\]"):
        make(ratings, "nominal", ["a", "b"])
