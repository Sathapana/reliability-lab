import warnings

import numpy as np
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study
from scipy import stats

from reliability_lab import BootstrapWarning, DesignError
from reliability_lab.inference import (
    cluster_coincidences,
    cluster_plan,
    replicate_alphas,
    weights_from_indices,
)

F4 = FOUR_CODERS_MISSING


def test_bca_matches_a_direct_scipy_call() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    with warnings.catch_warnings():
        warnings.simplefilter("error", BootstrapWarning)  # validated: no warning when it works
        boot = result.bootstrap(scheme="units", method="bca", n_resamples=999, random_state=5)
    assert not boot.experimental
    assert "[experimental]" not in boot.summary()

    sample = result._sample
    assert sample is not None
    plan = cluster_plan(sample, "unit", strata=False)
    clusters = cluster_coincidences(sample, "as_missing", plan)
    values = sample.inputs["as_missing"].values

    def statistic(idx: np.ndarray, axis: int = -1) -> np.ndarray:
        idx = np.moveaxis(idx, axis, -1)
        w = weights_from_indices(idx, plan.n_clusters)
        return replicate_alphas(clusters, w, values, sample.distance).reshape(idx.shape[:-1])

    direct = stats.bootstrap(
        (np.arange(plan.n_clusters),),
        statistic,
        n_resamples=999,
        confidence_level=0.95,
        method="BCa",
        vectorized=True,
        rng=np.random.default_rng(5),
    )
    interval = boot["as_missing"]
    assert interval.low == pytest.approx(direct.confidence_interval.low)
    assert interval.high == pytest.approx(direct.confidence_interval.high)
    assert interval.low < interval.point < interval.high


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
