"""simulate_ratings: synthetic coders for any measurement level, at a chosen alpha."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reliability_lab import (
    DesignError,
    Distance,
    RatingFrame,
    ReliabilityStudy,
    circular,
    simulate_ratings,
)
from reliability_lab.alpha import BUILTIN_DISTANCES
from reliability_lab.simulate import _expected_alpha, _kernel

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")

CUSTOM = Distance.from_matrix("custom", [[0, 1, 4], [1, 0, 1], [4, 1, 0]])
#: (id, measurement, values)
METRICS = [
    ("nominal", "nominal", ["a", "b", "c", "d"]),
    ("ordinal", "ordinal", [1, 2, 3, 4, 5]),
    ("interval", "interval", [1, 2, 3, 4, 5]),
    ("ratio", "ratio", [0, 1, 2, 3, 4, 5]),
    ("bipolar", "bipolar", [-3, -2, -1, 0, 1, 2, 3]),
    ("circular", circular(24), list(range(24))),
    ("custom", CUSTOM, ["x", "y", "z"]),
]
IDS = [m[0] for m in METRICS]


def _study(ratings: RatingFrame, measurement: Any, values: list[Any]) -> ReliabilityStudy:
    return ReliabilityStudy(
        ratings, measurement=measurement, values=values, coders="rater_id", design="human_panel"
    )


def _alpha(measurement: Any, values: list[Any], seed: int, **kwargs: Any) -> float:
    ratings = simulate_ratings(measurement, values, random_state=seed, **kwargs)
    return _study(ratings, measurement, values).alpha().alpha


@pytest.mark.parametrize(("name", "measurement", "values"), METRICS, ids=IDS)
def test_every_metric_runs_end_to_end(name: str, measurement: Any, values: list[Any]) -> None:
    ratings = simulate_ratings(measurement, values, n_units=30, n_coders=3, random_state=0)
    frame = ratings.to_frame()
    assert set(frame["value"]) <= set(values)
    assert frame["unit_id"].nunique() == 30 and frame["rater_id"].nunique() == 3
    assert math.isfinite(_study(ratings, measurement, values).alpha().alpha)


@pytest.mark.parametrize("errors", ["random", "near"])
@pytest.mark.parametrize(("name", "measurement", "values"), METRICS, ids=IDS)
def test_alpha_is_the_target_on_average(
    name: str, measurement: Any, values: list[Any], errors: str
) -> None:
    alphas = [
        _alpha(measurement, values, seed, n_units=200, n_coders=4, alpha=0.7, errors=errors)
        for seed in range(15)
    ]
    assert np.mean(alphas) == pytest.approx(0.7, abs=0.03)


@pytest.mark.parametrize(("name", "measurement", "values"), METRICS, ids=IDS)
def test_random_errors_give_q_squared(name: str, measurement: Any, values: list[Any]) -> None:
    distance = measurement if isinstance(measurement, Distance) else BUILTIN_DISTANCES[measurement]
    p = np.random.default_rng(1).dirichlet(np.ones(len(values)))
    delta2 = distance.matrix(values, p)
    kernel = _kernel("random", p, delta2)
    for q in (0.0, 0.3, 0.9, 1.0):
        assert _expected_alpha(q, p, kernel, delta2) == pytest.approx(q**2, abs=1e-12)


def test_alpha_one_is_perfect_agreement_and_zero_is_chance() -> None:
    ratings = simulate_ratings("ordinal", [1, 2, 3, 4, 5], alpha=1.0, random_state=3)
    assert ratings.to_frame().groupby("unit_id")["value"].nunique().max() == 1
    chance = [_alpha("interval", [1, 2, 3, 4, 5], s, n_units=200, alpha=0.0) for s in range(15)]
    assert np.mean(chance) == pytest.approx(0.0, abs=0.03)


def test_near_misses_are_nearer_than_random_errors() -> None:
    values = np.arange(1, 10)
    p = np.full(9, 1 / 9)
    delta2 = BUILTIN_DISTANCES["interval"].matrix(list(values), p)
    gap = np.abs(values[:, None] - values[None, :])
    near, random = (_kernel(errors, p, delta2) for errors in ("near", "random"))
    # Mean distance of a wrong report from the truth, over the truths.
    off = ~np.eye(9, dtype=bool)
    near_gap = (near * gap).sum() / 9
    random_gap = (random * gap * off).sum() / (random * off).sum()
    assert near_gap < random_gap - 1
    assert np.allclose(np.diag(near), 0)  # a near miss is always a miss


def test_near_misses_follow_the_circle() -> None:
    hours = list(range(24))
    delta2 = circular(24).matrix(hours, np.ones(24))
    kernel = _kernel("near", np.full(24, 1 / 24), delta2)
    assert kernel[0, 23] == pytest.approx(kernel[0, 1])  # 23:00 is next to midnight
    assert kernel[0, 23] > kernel[0, 12]


def test_missing_and_failed_cells() -> None:
    ratings = simulate_ratings(
        "ordinal",
        [1, 2, 3, 4, 5],
        n_units=500,
        n_coders=4,
        missing=0.2,
        failures=0.1,
        random_state=11,
    )
    frame = ratings.to_frame()
    assert len(frame) / 2000 == pytest.approx(0.8, abs=0.03)
    failed = frame["status"] == "parse_error"
    assert failed.mean() == pytest.approx(0.1, abs=0.02)
    assert frame.loc[failed, "value"].isna().all()
    assert frame.loc[~failed, "value"].notna().all()
    result = _study(ratings, "ordinal", [1, 2, 3, 4, 5]).alpha()
    assert {"sensitivity_pessimistic", "sensitivity_optimistic"} <= set(result.estimates)


def test_same_seed_same_ratings() -> None:
    a = simulate_ratings("bipolar", [-2, -1, 0, 1, 2], missing=0.1, random_state=42).to_frame()
    b = simulate_ratings("bipolar", [-2, -1, 0, 1, 2], missing=0.1, random_state=42).to_frame()
    pd.testing.assert_frame_equal(a, b)
    c = simulate_ratings("bipolar", [-2, -1, 0, 1, 2], missing=0.1, random_state=43).to_frame()
    assert not a.equals(c)


def test_prevalence_shapes_the_truths() -> None:
    frame = simulate_ratings(
        "nominal", ["yes", "no"], n_units=2000, alpha=1.0, prevalence=[0.9, 0.1], random_state=2
    ).to_frame()
    assert (frame["value"] == "yes").mean() == pytest.approx(0.9, abs=0.02)


def test_near_misses_cannot_go_below_their_floor() -> None:
    with pytest.raises(DesignError, match=r"cannot bring alpha below 0\.\d+"):
        simulate_ratings("interval", [1, 2, 3, 4, 5], alpha=0.05, errors="near")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"measurement": "circular", "values": [1, 2]}, r"circular\(U\)"),
        ({"measurement": "interval", "values": ["a", "b"]}, "numeric"),
        ({"measurement": "ratio", "values": [-1, 1]}, "non-negative"),
        ({"values": [1]}, "at least two"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": math.nan}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"n_units": 0}, "n_units"),
        ({"n_coders": 1}, "n_coders"),
        ({"n_coders": 2.0}, "n_coders"),
        ({"missing": 1.0}, "missing"),
        ({"failures": -0.1}, "failures"),
        ({"errors": "close"}, "errors"),
        ({"prevalence": [0.5, 0.5]}, "one weight per value"),
        ({"prevalence": [0.5, 0.6, -0.1]}, "non-negative"),
        ({"prevalence": [0.2, 0.2, 0.2]}, "sum to 1"),
    ],
)
def test_bad_arguments_are_refused(kwargs: dict[str, Any], match: str) -> None:
    arguments: dict[str, Any] = {"measurement": "ordinal", "values": [1, 2, 3]} | kwargs
    with pytest.raises(DesignError, match=match):
        simulate_ratings(**arguments)
