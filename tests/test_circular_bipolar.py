"""The built-in bipolar and circular distances (Krippendorff 2011, section D)."""

import json
import math
from typing import Any

import numpy as np
import pytest
from helpers import human_study, long_from_matrix, random_matrix
from reference import reference_alpha

from reliability_lab import BIPOLAR, DesignError, RatingFrame, ReliabilityStudy, circular

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")

HOURS = list(range(24))
#: (oracle level, measurement, values)
CASES = [
    ("bipolar", "bipolar", [-2, -1, 0, 1, 2]),
    ("circular(12)", circular(12), list(range(1, 13))),
]


@pytest.mark.parametrize(("level", "measurement", "values"), CASES, ids=["bipolar", "circular"])
def test_engine_matches_reference_on_random_data(
    level: str, measurement: Any, values: list[int]
) -> None:
    rng = np.random.default_rng(2029)
    checked = 0
    for _ in range(25):
        matrix = random_matrix(
            rng, int(rng.integers(2, 6)), int(rng.integers(3, 30)), values, p_missing=0.25
        )
        expected = reference_alpha(matrix, level, values)
        if math.isnan(expected):
            continue
        assert human_study(matrix, measurement, values).alpha().alpha == pytest.approx(
            expected, abs=1e-12
        )
        checked += 1
    assert checked >= 20


def test_circular_needs_its_circumference() -> None:
    with pytest.raises(DesignError, match=r"circular\(U\)"):
        human_study({"a": [1, 2], "b": [1, 2]}, "circular", [1, 2])


@pytest.mark.parametrize("circumference", [0, -1, math.nan, math.inf, True, "24", None])
def test_circumference_must_be_a_positive_number(circumference: Any) -> None:
    with pytest.raises(DesignError, match="circumference"):
        circular(circumference)


def test_the_circumference_is_recorded() -> None:
    matrix = {"a": [0, 6, 12, 23], "b": [1, 6, 13, 0]}
    result = human_study(matrix, circular(24), HOURS).alpha()
    assert result.measurement == "circular(24)"
    payload = json.loads(result.to_json())
    assert payload["measurement"] == "circular(24)"
    assert payload["provenance"]["declarations"]["measurement"] == "circular(24)"
    assert circular(6.5).name == "circular(6.5)"


def test_hours_wrap_around_midnight() -> None:
    d = circular(24).matrix(HOURS, np.ones(24))
    assert d[23, 0] == pytest.approx(d[0, 1])
    assert d[0, 12] == pytest.approx(1.0)  # opposite hours are the farthest apart
    assert d.max() == pytest.approx(1.0)


def test_rotating_every_value_leaves_circular_alpha_unchanged() -> None:
    rng = np.random.default_rng(7)
    matrix = random_matrix(rng, 4, 30, HOURS, p_missing=0.2)
    rotated = {
        coder: [None if v is None else (v + 5) % 24 for v in column]
        for coder, column in matrix.items()
    }
    a = human_study(matrix, circular(24), HOURS).alpha().alpha
    b = human_study(rotated, circular(24), HOURS).alpha().alpha
    assert a == pytest.approx(b, abs=1e-12)


def test_bipolar_takes_its_poles_from_the_declared_values() -> None:
    narrow = BIPOLAR.matrix([-2, -1, 0, 1, 2], np.ones(5))
    wide = BIPOLAR.matrix([-3, -2, -1, 0, 1, 2, 3], np.ones(7))
    # -2 and 2 are the poles of the narrow scale, the farthest apart possible ...
    assert narrow[0, 4] == pytest.approx(1.0)
    # ... but on a -3..3 scale they are 16 / (6 * 6) apart.
    assert wide[1, 5] == pytest.approx(16 / 36)
    with pytest.raises(DesignError, match="numeric"):
        human_study({"a": ["x", "y"], "b": ["x", "y"]}, "bipolar", ["x", "y"])


@pytest.mark.parametrize(("level", "measurement", "values"), CASES, ids=["bipolar", "circular"])
def test_bootstraps_and_sensitivity_run(level: str, measurement: Any, values: list[int]) -> None:
    rng = np.random.default_rng(3)
    # Six coders: with four, a two-way resample now and then draws one coder four
    # times, leaving no pairs (counted and warned about, but noise in this test).
    frame = long_from_matrix(random_matrix(rng, 6, 40, values, p_missing=0.1))
    frame["status"] = "observed"
    frame.loc[0, ["value", "status"]] = [None, "parse_error"]
    ratings = RatingFrame.from_long(
        frame, unit="unit", rater="coder", value="value", status="status"
    )
    study = ReliabilityStudy(
        ratings, measurement=measurement, values=values, coders="rater_id", design="human_panel"
    )
    result = study.alpha()
    assert set(result.estimates) == {
        "as_missing",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    }
    pessimistic, optimistic = (
        result.estimates[f"sensitivity_{v}"].alpha for v in ("pessimistic", "optimistic")
    )
    assert pessimistic <= optimistic
    for scheme in ("units", "two_way"):
        boot = result.bootstrap(scheme=scheme, n_resamples=200, random_state=1)  # type: ignore[arg-type]
        assert boot["as_missing"].defined
