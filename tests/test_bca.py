import json
import warnings

import numpy as np
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study
from scipy import stats

from reliability_lab import AlphaResult, BootstrapWarning, DesignError
from reliability_lab.inference import (
    bca_interval,
    cluster_coincidences,
    cluster_plan,
    jackknife_alphas,
    replicate_alphas,
    weights_from_indices,
)

F4 = FOUR_CODERS_MISSING


def direct_scipy_bca(
    result: AlphaResult, name: str, n_resamples: int, seed: int
) -> tuple[float, float]:
    """The BCa interval of a direct ``scipy.stats.bootstrap`` call over cluster indices."""
    sample = result._sample
    assert sample is not None
    plan = cluster_plan(sample, "unit", strata=False)
    clusters = cluster_coincidences(sample, name, plan)
    values = sample.inputs[name].values

    def statistic(idx: np.ndarray, axis: int = -1) -> np.ndarray:
        idx = np.moveaxis(idx, axis, -1)
        w = weights_from_indices(idx, plan.n_clusters)
        return replicate_alphas(clusters, w, values, sample.distance).reshape(idx.shape[:-1])

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        direct = stats.bootstrap(
            (np.arange(plan.n_clusters),),
            statistic,
            n_resamples=n_resamples,
            confidence_level=0.95,
            method="BCa",
            vectorized=True,
            rng=np.random.default_rng(seed),
        )
    return float(direct.confidence_interval.low), float(direct.confidence_interval.high)


def test_bca_matches_a_direct_scipy_call() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    with warnings.catch_warnings():
        warnings.simplefilter("error", BootstrapWarning)  # validated: no warning when it works
        boot = result.bootstrap(scheme="units", method="bca", n_resamples=999, random_state=5)
    assert not boot.experimental
    assert "[experimental]" not in boot.summary()

    low, high = direct_scipy_bca(result, "as_missing", 999, 5)
    interval = boot["as_missing"]
    assert interval.low == pytest.approx(low)
    assert interval.high == pytest.approx(high)
    assert interval.low < interval.point < interval.high


@pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")
@pytest.mark.filterwarnings("ignore::reliability_lab.BootstrapWarning")
@pytest.mark.parametrize(
    ("matrix", "n_resamples", "seed"),
    [
        # Undefined replicates: scipy gives no interval.
        ({"a": ["x"] * 19 + ["y"], "b": ["x"] * 20}, 500, 1),
        ({"a": ["x"] * 9 + ["y"], "b": ["x"] * 9 + ["y"]}, 50, 3),
        # Every replicate defined in a short run, undefined ones in a long run.
        (
            {"a": ["x"] * 5 + ["y"] * 5, "b": ["x"] * 4 + ["y"] * 6, "c": ["x"] * 6 + ["y"] * 4},
            50,
            4,
        ),
        (
            {"a": ["x"] * 5 + ["y"] * 5, "b": ["x"] * 4 + ["y"] * 6, "c": ["x"] * 6 + ["y"] * 4},
            500,
            4,
        ),
    ],
)
def test_bca_matches_scipy_on_degenerate_data(
    matrix: dict[str, list[str]], n_resamples: int, seed: int
) -> None:
    result = human_study(matrix, "nominal", ["x", "y"]).alpha()
    boot = result.bootstrap(
        scheme="units", method="bca", n_resamples=n_resamples, random_state=seed
    )
    low, high = direct_scipy_bca(result, "as_missing", n_resamples, seed)
    interval = boot["as_missing"]
    np.testing.assert_allclose([interval.low, interval.high], [low, high], equal_nan=True)
    assert interval.defined == np.isfinite(low)


@pytest.mark.parametrize("budget", [None, 20])
def test_jackknife_alphas_leave_out_each_cluster(
    budget: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if budget is not None:
        monkeypatch.setattr("reliability_lab.inference.BATCH_FLOATS", budget)
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    sample = result._sample
    assert sample is not None
    plan = cluster_plan(sample, "unit", strata=False)
    clusters = cluster_coincidences(sample, "as_missing", plan)
    values = sample.inputs["as_missing"].values
    k = plan.n_clusters
    left_out = replicate_alphas(clusters, np.ones((k, k)) - np.eye(k), values, sample.distance)
    jackknife = jackknife_alphas(clusters, values, sample.distance)
    np.testing.assert_allclose(jackknife, left_out, atol=1e-12, equal_nan=True)


@pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")
def test_undefined_bca_replicates_and_intervals_are_warned() -> None:
    # One unit holds the only disagreement: about a third of resamples miss it and
    # see a single category, so alpha is undefined there.
    matrix = {"a": ["x"] * 19 + ["y"], "b": ["x"] * 20}
    result = human_study(matrix, "nominal", ["x", "y"]).alpha()
    with pytest.warns(BootstrapWarning) as caught:
        boot = result.bootstrap(scheme="units", method="bca", n_resamples=500, random_state=1)
    interval = boot["as_missing"]
    assert interval.n_failed > 0
    messages = [str(w.message) for w in caught]
    assert any("undefined replicates" in m for m in messages)
    if not interval.defined:
        assert any("method='percentile'" in m for m in messages)
    assert [w.category for w in boot.warnings] == ["BootstrapWarning"] * len(boot.warnings)
    with pytest.warns(BootstrapWarning, match="undefined replicates"):
        fallback = result.bootstrap(scheme="units", n_resamples=500, random_state=1)
    percentile = fallback["as_missing"]
    assert percentile.defined and percentile.n_failed > 0


def test_weights_from_indices_counts_draws() -> None:
    idx = np.array([[0, 0, 2], [1, 2, 2]])
    np.testing.assert_array_equal(weights_from_indices(idx, 3), [[2, 0, 1], [0, 1, 2]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": "kalpha", "method": "bca"}, "scheme='units' only"),
        ({"scheme": "units", "method": "bca", "strata": True}, "within strata"),
    ],
)
def test_bca_refusals(kwargs: dict[str, object], message: str) -> None:
    import pandas as pd

    from reliability_lab import RatingFrame, ReliabilityStudy

    df = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1, 2, 2, 3, 3],
            "coder": ["a", "b"] * 4,
            "value": [1, 1, 2, 2, 1, 2, 2, 2],
            "stratum": ["x", "x", "x", "x", "y", "y", "y", "y"],
        }
    )
    ratings = RatingFrame.from_long(
        df, unit="unit", rater="coder", value="value", stratum="stratum"
    )
    study = ReliabilityStudy(
        ratings, measurement="nominal", values=[1, 2], coders="rater_id", design="human_panel"
    )
    with pytest.raises(DesignError, match=message):
        study.alpha().bootstrap(**kwargs)  # type: ignore[arg-type]


# -- why a BCa interval is undefined, and the opt-in fallback -------------------------

SKEWED = {"a": ["x"] * 19 + ["y"], "b": ["x"] * 20}


def _varied(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.5, 0.1, n)


@pytest.mark.parametrize(
    ("distribution", "theta_hat", "jackknife", "cause"),
    [
        (np.array([0.4, np.nan, 0.5, 0.6, np.nan]), 0.5, _varied(10, 1), "2 of 5 replicates"),
        (np.linspace(0.6, 0.9, 50), 0.1, _varied(10, 2), "one side of the estimate"),
        (_varied(50, 3), 0.5, np.array([0.4, np.nan, 0.6, 0.5]), "one cluster left out"),
        (_varied(50, 4), 0.5, np.full(10, 0.5), "the same with any one cluster"),
    ],
    ids=["undefined replicates", "one side", "undefined jackknife", "constant jackknife"],
)
def test_bca_names_why_it_is_undefined(
    distribution: np.ndarray, theta_hat: float, jackknife: np.ndarray, cause: str
) -> None:
    interval = bca_interval("as_missing", 0.5, distribution, theta_hat, jackknife, 0.95)
    assert not interval.defined
    assert interval.undefined_reason is not None
    assert interval.undefined_reason.startswith("the BCa interval is undefined: ")
    assert cause in interval.undefined_reason
    assert interval.method_used == "bca"
    assert np.isnan(interval.low) and np.isnan(interval.high)


def test_a_defined_bca_interval_has_no_reason() -> None:
    interval = bca_interval("as_missing", 0.5, _varied(200, 5), 0.5, _varied(10, 6), 0.95)
    assert interval.defined and interval.undefined_reason is None
    assert interval.method_used == "bca" and interval.fallback_reason is None


@pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")
def test_on_undefined_percentile_reports_the_percentile_interval() -> None:
    result = human_study(SKEWED, "nominal", ["x", "y"]).alpha()
    with pytest.warns(BootstrapWarning):
        default = result.bootstrap(scheme="units", method="bca", n_resamples=500, random_state=1)
    assert not default["as_missing"].defined
    assert default["as_missing"].fallback_reason is None

    with pytest.warns(BootstrapWarning) as caught:
        boot = result.bootstrap(
            scheme="units",
            method="bca",
            n_resamples=500,
            random_state=1,
            on_undefined="percentile",
        )
    interval = boot["as_missing"]
    assert interval.defined and interval.method_used == "percentile"
    assert interval.fallback_reason == default["as_missing"].undefined_reason
    assert "replicates are undefined" in interval.fallback_reason
    # The same BCa resamples, read as percentiles over the defined replicates.
    replicates = default["as_missing"].distribution
    low, high = np.quantile(replicates[~np.isnan(replicates)], [0.025, 0.975])
    assert (interval.low, interval.high) == pytest.approx((low, high))
    assert interval.n_failed == default["as_missing"].n_failed
    messages = [str(w.message) for w in caught]
    assert any("no BCa interval for as_missing" in m and "reported instead" in m for m in messages)

    payload = json.loads(boot.to_json())["intervals"]["as_missing"]
    assert payload["method_used"] == "percentile"
    assert payload["fallback_reason"] == interval.fallback_reason
    assert set(boot.to_frame().columns) >= {"method_used", "fallback_reason"}


@pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")
def test_on_undefined_applies_to_comparisons() -> None:
    matrix = {**SKEWED, "c": ["x"] * 18 + ["y"] * 2, "d": ["x"] * 20}
    study = human_study(matrix, "nominal", ["x", "y"])
    arms = {"a": {"rater_id": ["a", "b"]}, "b": {"rater_id": ["c", "d"]}}
    with pytest.warns(BootstrapWarning):
        default = study.compare(**arms, method="bca", n_resamples=300, random_state=2)
    with pytest.warns(BootstrapWarning):
        diff = study.compare(
            **arms, method="bca", n_resamples=300, random_state=2, on_undefined="percentile"
        )
    for old, new in [
        (default["as_missing"], diff["as_missing"]),
        (default.arm_a["as_missing"], diff.arm_a["as_missing"]),
        (default.arm_b["as_missing"], diff.arm_b["as_missing"]),
    ]:
        if old.defined:
            assert new.method_used == "bca" and new.low == old.low
        else:
            assert new.method_used == "percentile" and new.defined
            assert new.fallback_reason == old.undefined_reason
    assert not default["as_missing"].defined  # the case this test is about
    assert "diff_method_used" in diff.to_frame().columns


def test_percentile_and_jackknife_intervals_record_their_method() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    boot = result.bootstrap(scheme="units", n_resamples=200, random_state=0)
    assert boot["as_missing"].method_used == "percentile"
    assert result.coder_jackknife()["as_missing"].method_used == "t"


def test_on_undefined_is_checked() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    with pytest.raises(DesignError, match="on_undefined"):
        result.bootstrap(scheme="units", on_undefined="drop")  # type: ignore[arg-type]
