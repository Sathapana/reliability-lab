import json
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study, long_from_matrix, random_matrix
from naive_bootstrap import naive_units_alphas

from reliability_lab import (
    ORDINAL,
    BootstrapWarning,
    DesignError,
    Distance,
    RatingFrame,
    ReliabilityStudy,
)
from reliability_lab.inference import (
    cluster_coincidences,
    cluster_plan,
    draw_weights,
    replicate_alphas,
)

LEVELS = ["nominal", "ordinal", "interval", "ratio"]
VALUES = [1, 2, 3, 4]
F4 = FOUR_CODERS_MISSING

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def grouped_study(
    matrix: dict[Hashable, Sequence[Any]],
    level: Any,
    values: Sequence[Any],
    groups: dict[int, str] | None = None,
    strata: dict[int, str] | None = None,
) -> ReliabilityStudy:
    df = long_from_matrix(matrix)
    kwargs: dict[str, str] = {}
    if groups is not None:
        df["group"] = df["unit"].map(groups)
        kwargs["group"] = "group"
    if strata is not None:
        df["stratum"] = df["unit"].map(strata)
        kwargs["stratum"] = "stratum"
    ratings = RatingFrame.from_long(df, unit="unit", rater="coder", value="value", **kwargs)
    return ReliabilityStudy(
        ratings, measurement=level, values=values, coders="rater_id", design="human_panel"
    )


def nested_study(level: Any, seed: int = 5) -> tuple[dict[Hashable, list[Any]], ReliabilityStudy]:
    """16 units in 8 groups of two, the groups nested in 2 strata."""
    matrix = random_matrix(np.random.default_rng(seed), 3, 16, VALUES, p_missing=0.2)
    groups = {u: f"g{u // 2}" for u in range(16)}
    strata = {u: f"s{u // 8}" for u in range(16)}
    return matrix, grouped_study(matrix, level, VALUES, groups, strata)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("resample", ["unit", "group", "group_within_strata", "unit_within_strata"])
def test_vectorised_units_scheme_matches_naive_loop(level: str, resample: str) -> None:
    matrix, study = nested_study(level)
    sample = study.alpha()._sample
    assert sample is not None
    plan = cluster_plan(sample, resample.split("_")[0], strata=resample.endswith("strata"))
    weights = draw_weights(np.random.default_rng(11), plan, 40)
    clusters = cluster_coincidences(sample, "as_missing", plan)
    fast = replicate_alphas(clusters, weights, sample.inputs["as_missing"].values, sample.distance)
    mapping = {
        int(label): int(plan.cluster_of_unit[i]) for i, label in enumerate(sample.unit_labels)
    }
    slow = naive_units_alphas(matrix, level, VALUES, mapping, weights)
    np.testing.assert_allclose(fast, slow, atol=1e-12, equal_nan=True)


def test_custom_marginal_dependent_distance_matches_naive_loop() -> None:
    custom = Distance("my_ordinal", ORDINAL.fn, depends_on_marginals=True)
    matrix, study = nested_study(custom)
    sample = study.alpha()._sample
    assert sample is not None
    plan = cluster_plan(sample, "unit", strata=False)
    weights = draw_weights(np.random.default_rng(3), plan, 25)
    fast = replicate_alphas(
        cluster_coincidences(sample, "as_missing", plan), weights, tuple(VALUES), custom
    )
    mapping = {
        int(label): int(plan.cluster_of_unit[i]) for i, label in enumerate(sample.unit_labels)
    }
    slow = naive_units_alphas(matrix, "ordinal", VALUES, mapping, weights)
    np.testing.assert_allclose(fast, slow, atol=1e-12, equal_nan=True)


def test_seed_reproduces_resamples_and_is_recorded() -> None:
    result = human_study(F4["matrix"], "ordinal", F4["values"]).alpha()
    a = result.bootstrap(scheme="units", n_resamples=200, random_state=3)
    b = result.bootstrap(scheme="units", n_resamples=200, random_state=3)
    np.testing.assert_array_equal(a["as_missing"].distribution, b["as_missing"].distribution)
    assert a.seed == 3
    fresh = result.bootstrap(scheme="units", n_resamples=200)
    again = result.bootstrap(scheme="units", n_resamples=200, random_state=fresh.seed)
    np.testing.assert_array_equal(
        fresh["as_missing"].distribution, again["as_missing"].distribution
    )
    gen = result.bootstrap(scheme="units", n_resamples=50, random_state=np.random.default_rng(1))
    replay = result.bootstrap(scheme="units", n_resamples=50, random_state=gen.seed)
    np.testing.assert_array_equal(gen["as_missing"].distribution, replay["as_missing"].distribution)


def test_estimates_share_resamples() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    boot = result.bootstrap(scheme="units", n_resamples=100, random_state=0)
    # No failed attempts, so every policy sees identical data and identical draws.
    np.testing.assert_array_equal(
        boot["as_missing"].distribution, boot["sensitivity_pessimistic"].distribution
    )
    assert any("paired" in note for note in boot.notes)


def test_interval_brackets_point_and_records_quantiles() -> None:
    result = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    i = result.bootstrap(scheme="units", n_resamples=2000, random_state=7)["as_missing"]
    assert i.low < i.point < i.high
    np.testing.assert_allclose([i.low, i.high], np.quantile(i.distribution, [0.025, 0.975]))
    assert i.n_resamples == 2000 and i.n_failed == 0


def test_group_level_is_the_default_when_groups_are_declared() -> None:
    _, study = nested_study("nominal")
    result = study.alpha()
    boot = result.bootstrap(scheme="units", n_resamples=50, random_state=0)
    assert (boot.level, boot.n_clusters) == ("group", 8)
    with pytest.warns(BootstrapWarning, match="resampled one by one"):
        by_unit = result.bootstrap(scheme="units", level="unit", n_resamples=50, random_state=0)
    assert (by_unit.level, by_unit.n_clusters) == ("unit", 16)
    assert by_unit.warnings[-1].category == "BootstrapWarning"


def test_strata_keep_their_sizes() -> None:
    _, study = nested_study("nominal")
    sample = study.alpha()._sample
    assert sample is not None
    plan = cluster_plan(sample, "group", strata=True)
    assert plan.stratum_of_cluster is not None
    weights = draw_weights(np.random.default_rng(0), plan, 100)
    for stratum in np.unique(plan.stratum_of_cluster):
        members = plan.stratum_of_cluster == stratum
        assert (weights[:, members].sum(axis=1) == members.sum()).all()
    plain = draw_weights(np.random.default_rng(0), cluster_plan(sample, "group", False), 100)
    assert (plain.sum(axis=1) == 8).all()


def test_groups_spanning_strata_are_refused() -> None:
    matrix = random_matrix(np.random.default_rng(2), 3, 8, VALUES, p_missing=0.0)
    study = grouped_study(
        matrix,
        "nominal",
        VALUES,
        groups={u: f"g{u // 4}" for u in range(8)},
        strata={u: f"s{u % 2}" for u in range(8)},
    )
    with pytest.raises(DesignError, match="span several strata"):
        study.alpha().bootstrap(scheme="units", strata=True, n_resamples=10)


def test_failed_replicates_are_counted_and_warned() -> None:
    matrix = {"a": ["x", "x", "x"], "b": ["x", "x", "y"]}
    result = human_study(matrix, "nominal", ["x", "y"]).alpha()
    with pytest.warns(BootstrapWarning, match="undefined replicates"):
        boot = result.bootstrap(scheme="units", n_resamples=500, random_state=1)
    i = boot["as_missing"]
    assert 0 < i.n_failed < 500
    assert np.isnan(i.distribution).sum() == i.n_failed
    assert np.isfinite([i.low, i.high]).all()


def test_p_below_min() -> None:
    result = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    boot = result.bootstrap(scheme="units", n_resamples=300, alpha_min=0.8, random_state=2)
    i = boot["as_missing"]
    assert i.p_below_min == pytest.approx(float((i.distribution < 0.8).mean()))
    assert "P(<min)" in boot.summary()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": "bogus"}, "scheme must be"),
        ({"scheme": "units", "n_resamples": 0}, "at least 1"),
        ({"scheme": "units", "confidence": 1.5}, "confidence"),
        ({"scheme": "units", "method": "studentized"}, "method must be"),
        ({"scheme": "units", "alpha_min": 2}, "alpha_min"),
        ({"scheme": "units", "level": "group"}, "needs group_id"),
        ({"scheme": "units", "level": "cluster"}, "level must be"),
        ({"scheme": "units", "strata": True}, "needs a stratum"),
        ({"scheme": "units", "random_state": -1}, "non-negative"),
        ({"scheme": "kalpha", "level": "unit"}, "pairs of values"),
        ({"scheme": "kalpha", "method": "bca"}, "scheme='units' only"),
    ],
)
def test_option_errors(kwargs: dict[str, Any], message: str) -> None:
    result = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    with pytest.raises(DesignError, match=message):
        result.bootstrap(**kwargs)


def test_scheme_is_required() -> None:
    result = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    with pytest.raises(TypeError, match="scheme"):
        result.bootstrap(n_resamples=10)  # type: ignore[call-arg]


def test_report_names_scheme_and_serialises() -> None:
    result = human_study(F4["matrix"], "ordinal", F4["values"]).alpha()
    boot = result.bootstrap(scheme="units", n_resamples=100, random_state=4)
    text = boot.summary()
    assert "scheme units (resampling 12 units; coders fixed)" in text
    assert "method percentile" in text
    assert "seed 4" in text
    payload = json.loads(json.dumps(boot.to_dict(), allow_nan=False))
    assert payload["scheme"] == "units" and payload["level"] == "unit"
    assert "distribution" not in payload["intervals"]["as_missing"]
    full = boot.to_dict(include_distribution=True)
    assert len(full["intervals"]["as_missing"]["distribution"]) == 100
    assert list(boot.to_frame()["estimate"]) == list(result.estimates)
