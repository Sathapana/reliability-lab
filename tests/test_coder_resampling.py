import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study, long_from_matrix, random_matrix
from naive_bootstrap import naive_two_way_alpha

from reliability_lab import DesignError, RatingFrame, ReliabilityStudy
from reliability_lab.inference import (
    cluster_coincidences,
    cluster_plan,
    coder_onehot,
    draw_coder_weights,
    draw_weights,
    replicate_alphas,
    two_way_alphas,
)

LEVELS = ["nominal", "ordinal", "interval", "ratio"]
VALUES = [1, 2, 3, 4]
F4 = FOUR_CODERS_MISSING

pytestmark = pytest.mark.filterwarnings(
    "ignore::reliability_lab.CategorySkewWarning", "ignore::reliability_lab.BootstrapWarning"
)


def sample_of(matrix: dict[Any, list[Any]], level: str, values: list[Any]) -> Any:
    sample = human_study(matrix, level, values).alpha()._sample
    assert sample is not None
    return sample


def test_without_duplicated_coders_the_engine_is_the_units_engine() -> None:
    sample = sample_of(F4["matrix"], "ordinal", F4["values"])
    plan = cluster_plan(sample, "unit", strata=False)
    unit_weights = draw_weights(np.random.default_rng(1), plan, 50)
    expected = replicate_alphas(
        cluster_coincidences(sample, "as_missing", plan),
        unit_weights,
        sample.inputs["as_missing"].values,
        sample.distance,
    )
    ones = np.ones((50, sample.n_coders))
    got = two_way_alphas(
        coder_onehot(sample, "as_missing"),
        unit_weights[:, plan.cluster_of_unit],
        ones,
        sample.inputs["as_missing"].values,
        sample.distance,
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("level", LEVELS)
def test_with_all_weights_one_the_engine_gives_the_point_alpha(level: str) -> None:
    result = human_study(F4["matrix"], level, F4["values"]).alpha()
    sample = result._sample
    assert sample is not None
    got = two_way_alphas(
        coder_onehot(sample, "as_missing"),
        None,
        np.ones((1, sample.n_coders)),
        sample.inputs["as_missing"].values,
        sample.distance,
    )
    assert got[0] == pytest.approx(result.alpha, abs=1e-12)


@pytest.mark.parametrize("level", LEVELS)
def test_engine_matches_naive_copies_and_pairs(level: str) -> None:
    rng = np.random.default_rng(4)
    matrix = random_matrix(rng, 5, 15, VALUES, p_missing=0.2)
    sample = sample_of(matrix, level, VALUES)
    # Map the study's unit and coder order back to matrix columns and rows.
    unit_cols = [int(label) for label in sample.unit_labels]
    coder_rows = list(sample.coder_names)
    onehot = coder_onehot(sample, "as_missing")
    for _ in range(15):
        w = rng.multinomial(len(unit_cols), np.full(len(unit_cols), 1 / len(unit_cols)))
        v = rng.multinomial(len(coder_rows), np.full(len(coder_rows), 1 / len(coder_rows)))
        fast = two_way_alphas(onehot, w[None, :], v[None, :], tuple(VALUES), sample.distance)[0]
        unit_weights = [0] * 15
        for i, col in enumerate(unit_cols):
            unit_weights[col] = int(w[i])
        coder_weights = [int(v[coder_rows.index(name)]) for name in matrix]
        slow = naive_two_way_alpha(matrix, level, VALUES, unit_weights, coder_weights)
        if np.isnan(slow):
            assert np.isnan(fast)
        else:
            assert fast == pytest.approx(slow, abs=1e-12)


def test_a_coder_drawn_twice_is_not_paired_with_itself() -> None:
    # Coder c disagrees with a and b everywhere; drawing a twice and b, c once must
    # not count a's copies as agreeing with each other.
    matrix = {"a": [1, 2, 1, 2], "b": [1, 2, 1, 2], "c": [2, 1, 2, 1]}
    sample = sample_of(matrix, "nominal", [1, 2])
    order = [list(sample.coder_names).index(c) for c in ("a", "b", "c")]
    v = np.zeros((1, 3))
    v[0, order] = [2, 1, 1]
    got = two_way_alphas(coder_onehot(sample, "as_missing"), None, v, (1, 2), sample.distance)[0]
    expected = naive_two_way_alpha(matrix, "nominal", [1, 2], [1, 1, 1, 1], [2, 1, 1])
    assert got == pytest.approx(expected)
    with_self_pairs = naive_two_way_alpha(
        {"a1": matrix["a"], "a2": matrix["a"], "b": matrix["b"], "c": matrix["c"]},
        "nominal",
        [1, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    assert got < with_self_pairs


def test_coders_and_two_way_schemes_report_what_they_resample() -> None:
    result = human_study(F4["matrix"], "ordinal", F4["values"]).alpha()
    coders = result.bootstrap(scheme="coders", n_resamples=300, random_state=3)
    assert (coders.n_coders, coders.level, coders.n_clusters) == (4, None, None)
    assert "scheme coders (resampling 4 coders; units fixed)" in coders.describe()
    assert any("new coders from the same pool, on these units" in n for n in coders.notes)
    two_way = result.bootstrap(scheme="two_way", n_resamples=300, random_state=3)
    assert "scheme two_way (resampling 12 units and 4 coders)" in two_way.describe()
    assert any("new coders on new units" in n for n in two_way.notes)
    units = result.bootstrap(scheme="units", n_resamples=300, random_state=3)
    assert any("generalises to new units rated by these coders" in n for n in units.notes)
    payload = json.loads(two_way.to_json())
    assert (payload["scheme"], payload["n_coders"]) == ("two_way", 4)


def test_two_way_is_reproducible_and_wider_than_units() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    a = result.bootstrap(scheme="two_way", n_resamples=500, random_state=9)
    b = result.bootstrap(scheme="two_way", n_resamples=500, random_state=9)
    np.testing.assert_array_equal(a["as_missing"].distribution, b["as_missing"].distribution)
    units = result.bootstrap(scheme="units", n_resamples=500, random_state=9)["as_missing"]
    both = a["as_missing"]
    assert both.high - both.low > units.high - units.low


def test_two_way_resamples_groups_when_declared() -> None:
    df = long_from_matrix(F4["matrix"])
    df["group"] = df["unit"] // 3
    ratings = RatingFrame.from_long(df, unit="unit", rater="coder", value="value", group="group")
    study = ReliabilityStudy(
        ratings, measurement="nominal", values=F4["values"], coders="rater_id", design="human_panel"
    )
    boot = study.alpha().bootstrap(scheme="two_way", n_resamples=100, random_state=0)
    assert (boot.level, boot.n_clusters, boot.n_coders) == ("group", 4, 4)


@pytest.mark.parametrize(
    ("matrix", "kwargs", "message"),
    [
        ({"a": [1, 2, 1], "b": [1, 2, 2]}, {"scheme": "coders"}, "at least 3"),
        ({"a": [1, 2, 1], "b": [1, 2, 2]}, {"scheme": "two_way"}, "at least 3"),
        (F4["matrix"], {"scheme": "coders", "method": "bca"}, "scheme='units' only"),
        (F4["matrix"], {"scheme": "two_way", "method": "bca"}, "scheme='units' only"),
        (F4["matrix"], {"scheme": "coders", "level": "unit"}, "holds the units fixed"),
    ],
)
def test_coder_resampling_refusals(
    matrix: dict[Any, list[Any]], kwargs: dict[str, Any], message: str
) -> None:
    values = sorted({v for row in matrix.values() for v in row if v is not None})
    result = human_study(matrix, "nominal", values).alpha()
    with pytest.raises(DesignError, match=message):
        result.bootstrap(n_resamples=10, **kwargs)


def test_draw_coder_weights_sum_to_the_number_of_coders() -> None:
    weights = draw_coder_weights(np.random.default_rng(0), 7, 100)
    assert weights.shape == (100, 7)
    assert (weights.sum(axis=1) == 7).all()
    assert isinstance(pd.DataFrame(weights), pd.DataFrame)
