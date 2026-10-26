import json

import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study

import reliability_lab
from reliability_lab import CategorySkewWarning, RatingFrame, ReliabilityStudy


def test_to_dict_is_strict_json() -> None:
    result = human_study(FOUR_CODERS_MISSING["matrix"], "ordinal", [1, 2, 3, 4, 5]).alpha()
    payload = result.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["estimates"]["as_missing"]["alpha"] == pytest.approx(result.alpha)
    assert payload["design"] == "human_panel"
    assert payload["measurement"] == "ordinal"
    assert payload["reliability_lab_version"] == reliability_lab.__version__
    assert payload["coverage"]["n_units"] == 12
    assert len(payload["estimates"]["as_missing"]["per_unit"]) == 12
    assert payload["estimates"]["as_missing"]["coincidence"]["values"] == [1, 2, 3, 4, 5]


def test_to_frame_has_one_row_per_estimate() -> None:
    result = human_study(FOUR_CODERS_MISSING["matrix"], "interval", [1, 2, 3, 4, 5]).alpha()
    table = result.to_frame()
    assert list(table["estimate"]) == [
        "as_missing",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    assert table.loc[0, "alpha"] == pytest.approx(0.849, abs=5e-4)


def test_summary_names_design_measurement_and_alpha() -> None:
    result = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5]).alpha()
    text = result.summary()
    assert "design: human_panel" in text
    assert "measurement: nominal" in text
    assert "0.743" in text
    assert "1 of 12 units" in text


def test_coverage_per_coder() -> None:
    df = pd.DataFrame(
        {
            "unit": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "coder": ["a", "b"] * 3,
            "value": [1, 1, 2, None, 2, 2],
            "status": ["observed", "observed", "observed", "parse_error", "observed", "observed"],
        }
    )
    ratings = RatingFrame.from_long(df, unit="unit", rater="coder", value="value", status="status")
    study = ReliabilityStudy(
        ratings, measurement="nominal", values=[1, 2], coders="rater_id", design="human_panel"
    )
    cov = study.alpha().coverage
    assert (cov.n_units, cov.n_ratings, cov.n_coders, cov.n_observed) == (3, 6, 2, 5)
    assert cov.coding_density == pytest.approx(5 / 6)
    assert cov.status_counts["parse_error"] == 1
    b = cov.per_coder.loc["b"]
    assert b["ratings"] == 3
    assert b["coverage"] == pytest.approx(2 / 3)
    assert b["error_rate"] == pytest.approx(1 / 3)
    assert cov.per_coder.loc["a", "error_rate"] == 0


def test_skew_warning_is_raised_and_recorded() -> None:
    matrix = {"a": ["x"] * 19 + ["y"], "b": ["x"] * 19 + ["x"]}
    study = human_study(matrix, "nominal", ["x", "y"])
    with pytest.warns(CategorySkewWarning, match="holds 98% of pairable values"):
        result = study.alpha()
    assert result.warnings[-1].category == "CategorySkewWarning"
    assert result["as_missing"].observed_agreement == pytest.approx(0.95)
