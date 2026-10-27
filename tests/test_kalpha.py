from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import BINARY_TWO_CODERS, FOUR_CODERS_MISSING
from helpers import human_study, long_from_matrix, random_matrix
from naive_bootstrap import naive_kalpha, reference_pair_errors
from scipy import stats

from reliability_lab import BootstrapWarning, RatingFrame, ReliabilityStudy
from reliability_lab.inference import kalpha_pool, kalpha_replicates

LEVELS = ["nominal", "ordinal", "interval", "ratio"]
F4 = FOUR_CODERS_MISSING

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def pool_for(matrix: dict[Hashable, Sequence[Any]], level: str, values: Sequence[Any]) -> Any:
    result = human_study(matrix, level, values).alpha()
    assert result._sample is not None
    return result, result._sample, kalpha_pool(result._sample, "as_missing")


@pytest.mark.parametrize("level", LEVELS)
def test_identity_draws_reproduce_observed_alpha(level: str) -> None:
    result, _, pool = pool_for(F4["matrix"], level, F4["values"])
    draws = np.arange(len(pool.errors))[None, :]
    redraws = np.zeros((1, len(pool.second_pick)), dtype=int)
    assert kalpha_replicates(pool, draws, redraws)[0] == pytest.approx(result.alpha, abs=1e-12)


@pytest.mark.parametrize("level", LEVELS)
def test_pool_matches_reference_pairs(level: str) -> None:
    _, _, pool = pool_for(F4["matrix"], level, F4["values"])
    expected = reference_pair_errors(F4["matrix"], level, F4["values"])
    np.testing.assert_allclose(np.sort(pool.errors), np.sort(expected), atol=1e-12)
    # Each unit draws m_u(m_u - 1)/2 pairs weighted 1/(m_u - 1): the weights sum to n/2.
    assert pool.weights.sum() == pytest.approx(F4["n_pairable"] / 2)


@pytest.mark.parametrize("level", LEVELS)
def test_vectorised_kalpha_matches_algorithm_note(level: str) -> None:
    _, sample, pool = pool_for(F4["matrix"], level, F4["values"])
    m = np.bincount(sample.inputs["as_missing"].unit_idx, minlength=sample.n_units)
    sizes = [int(x) for x in m if x >= 2]
    rng = np.random.default_rng(9)
    n_pairs, n_second = len(pool.errors), len(pool.second_pick)
    draws = rng.integers(0, n_pairs, size=(200, n_pairs))
    # Force half the replicates to repeat the first pick, exercising the redraw rule.
    draws[:100, pool.second_pick] = draws[:100, pool.second_pick - 1]
    redraws = rng.integers(0, n_pairs, size=(200, n_second))
    fast = kalpha_replicates(pool, draws, redraws)
    slow = naive_kalpha(pool.errors, sizes, draws, redraws)
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_replicates_below_minus_one_are_truncated() -> None:
    _, _, pool = pool_for(BINARY_TWO_CODERS["matrix"], "nominal", [0, 1])
    worst = int(np.argmax(pool.errors))
    draws = np.full((1, len(pool.errors)), worst)
    redraws = np.full((1, len(pool.second_pick)), worst)
    assert 1 - (pool.errors[draws] * pool.weights).sum() < -1
    assert kalpha_replicates(pool, draws, redraws)[0] == -1


def test_kalpha_does_not_apply_at_perfect_agreement() -> None:
    matrix = {"a": [1, 2, 3, 1], "b": [1, 2, 3, 1]}
    result = human_study(matrix, "nominal", [1, 2, 3]).alpha()
    with pytest.warns(BootstrapWarning, match="no interval for as_missing"):
        boot = result.bootstrap(scheme="kalpha", n_resamples=100, random_state=0)
    interval = boot["as_missing"]
    assert "alpha = 1" in (interval.undefined_reason or "")
    assert np.isnan(interval.low)
    assert "no interval" in boot.summary()


def test_kalpha_result_metadata_and_bounds() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    boot = result.bootstrap(scheme="kalpha", n_resamples=1000, random_state=5)
    assert (boot.scheme, boot.level, boot.n_clusters) == ("kalpha", None, None)
    assert "scheme kalpha" in boot.describe()
    dist = boot["as_missing"].distribution
    assert dist.min() >= -1 and dist.max() <= 1
    assert any("evalica" in note for note in boot.notes)


def test_kalpha_warns_that_it_ignores_groups() -> None:
    df = long_from_matrix(F4["matrix"])
    df["group"] = df["unit"] // 3
    ratings = RatingFrame.from_long(df, unit="unit", rater="coder", value="value", group="group")
    study = ReliabilityStudy(
        ratings, measurement="nominal", values=F4["values"], coders="rater_id", design="human_panel"
    )
    with pytest.warns(BootstrapWarning, match="kalpha scheme ignores them"):
        study.alpha().bootstrap(scheme="kalpha", n_resamples=50, random_state=0)


def as_array(matrix: dict[Hashable, Sequence[Any]]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in matrix.values()], float)


@pytest.mark.parity
@pytest.mark.parametrize("level", LEVELS)
def test_kalpha_distribution_matches_evalica(level: str) -> None:
    evalica = pytest.importorskip("evalica")
    values = [1, 2, 3, 4, 5]
    cases = [F4["matrix"], random_matrix(np.random.default_rng(21), 4, 30, values, 0.2)]
    for matrix in cases:
        ours = (
            human_study(matrix, level, values)
            .alpha()
            .bootstrap(scheme="kalpha", n_resamples=20_000, random_state=1)["as_missing"]
            .distribution
        )
        theirs = evalica.alpha_bootstrap(
            pd.DataFrame(as_array(matrix)), distance=level, n_resamples=20_000, random_state=2
        ).distribution.to_numpy()
        # The distributions are discrete. The same atoms come out a few ulps apart
        # from the two implementations, so round before comparing ECDFs. Quantiles
        # may land one atom apart (about 0.011 for the fixture) by chance.
        ours, theirs = np.round(ours, 9), np.round(theirs, 9)
        assert stats.ks_2samp(ours, theirs).pvalue > 1e-3
        assert ours.mean() == pytest.approx(theirs.mean(), abs=0.005)
        assert ours.std() == pytest.approx(theirs.std(), abs=0.005)
        quantiles = [0.025, 0.5, 0.975]
        np.testing.assert_allclose(
            np.quantile(ours, quantiles), np.quantile(theirs, quantiles), atol=0.02
        )
