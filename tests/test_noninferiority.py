"""Non-inferiority and equivalence verdicts in compare(), and the p-values behind them."""

import json
import math
from typing import Any

import numpy as np
import pytest
from helpers import human_study
from hypothesis import given, settings
from hypothesis import strategies as st

from reliability_lab import DesignError, Interval, ReliabilityStudy
from reliability_lab.inference import (
    REJECTS,
    bca_interval,
    bca_parameters,
    decision_p_value,
    percentile_interval,
    verdict,
)

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def _interval(low: float, high: float, defined: bool = True) -> Interval:
    return Interval(
        estimate="as_missing",
        point=(low + high) / 2,
        low=low,
        high=high,
        n_resamples=1,
        n_failed=0,
        undefined_reason=None if defined else "every replicate was undefined",
    )


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.02, 0.04, "non_inferior"),
        (-0.03, 0.04, "inconclusive"),  # the lower end on the margin rejects nothing
        (-0.10, 0.01, "inconclusive"),
        (-0.10, -0.03, "inconclusive"),
        (-0.10, -0.04, "inferior"),
    ],
)
def test_non_inferiority_verdicts(low: float, high: float, expected: str) -> None:
    assert verdict("non_inferiority", 0.03, _interval(low, high)) == expected


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.02, 0.02, "equivalent"),
        (-0.03, 0.02, "inconclusive"),
        (-0.02, 0.03, "inconclusive"),
        (-0.05, 0.05, "inconclusive"),
        (-0.10, -0.04, "not_equivalent"),
        (0.04, 0.10, "not_equivalent"),
        (-0.10, -0.03, "inconclusive"),
    ],
)
def test_equivalence_verdicts(low: float, high: float, expected: str) -> None:
    assert verdict("equivalence", 0.03, _interval(low, high)) == expected


@pytest.mark.parametrize("test", ["non_inferiority", "equivalence"])
def test_an_undefined_interval_decides_nothing(test: str) -> None:
    interval = _interval(math.nan, math.nan, defined=False)
    assert verdict(test, 0.03, interval) == "inconclusive"
    assert math.isnan(decision_p_value(test, 0.03, interval, 0.95))


def test_percentile_p_values_on_a_known_distribution() -> None:
    # 101 evenly spaced replicates from -0.5 to 0.5: the u-quantile is -0.5 + u.
    replicates = np.linspace(-0.5, 0.5, 101)
    interval = percentile_interval("as_missing", 0.0, replicates, 0.95)
    assert decision_p_value("non_inferiority", 0.4, interval, 0.95) == pytest.approx(0.1)
    assert decision_p_value("equivalence", 0.45, interval, 0.95) == pytest.approx(0.05)
    assert decision_p_value("non_inferiority", 0.6, interval, 0.95) == 0.0
    # Undefined replicates are left out, as the interval leaves them out.
    with_failures = np.concatenate([replicates, [np.nan] * 7])
    interval = percentile_interval("as_missing", 0.0, with_failures, 0.95)
    assert decision_p_value("non_inferiority", 0.4, interval, 0.95) == pytest.approx(0.1)


def _check_consistency(
    test: str, margin: float, interval: Interval, confidence: float, p: float
) -> None:
    tail = (1 - confidence) / 2
    rejected = verdict(test, margin, interval) == REJECTS[test]
    if math.isnan(p):
        return
    assert 0 <= p <= 1
    # Exact ties sit on a measure-zero boundary; float rounding may break them.
    if abs(p - tail) > 1e-9:
        assert (p < tail) == rejected


@settings(max_examples=300, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    size=st.integers(2, 400),
    margin=st.floats(0.001, 0.5),
    confidence=st.sampled_from([0.8, 0.9, 0.95, 0.99]),
    test=st.sampled_from(["non_inferiority", "equivalence"]),
)
def test_percentile_p_values_never_contradict_the_verdict(
    seed: int, size: int, margin: float, confidence: float, test: str
) -> None:
    rng = np.random.default_rng(seed)
    replicates = rng.normal(rng.uniform(-0.3, 0.3), rng.uniform(0.01, 0.3), size)
    interval = percentile_interval("as_missing", 0.0, replicates, confidence)
    p = decision_p_value(test, margin, interval, confidence)
    assert not math.isnan(p)
    _check_consistency(test, margin, interval, confidence, p)


@settings(max_examples=300, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    size=st.integers(20, 400),
    margin=st.floats(0.001, 0.5),
    confidence=st.sampled_from([0.8, 0.9, 0.95, 0.99]),
    test=st.sampled_from(["non_inferiority", "equivalence"]),
)
def test_bca_p_values_never_contradict_the_verdict(
    seed: int, size: int, margin: float, confidence: float, test: str
) -> None:
    rng = np.random.default_rng(seed)
    centre = rng.uniform(-0.3, 0.3)
    replicates = rng.normal(centre, rng.uniform(0.01, 0.3), size)
    theta_hat = centre + rng.normal(0, 0.05)
    jackknife = rng.gamma(2.0, 0.01, int(rng.integers(5, 60))) + centre
    interval = bca_interval("as_missing", theta_hat, replicates, theta_hat, jackknife, confidence)
    if not interval.defined:
        return
    bca = bca_parameters(replicates, theta_hat, jackknife)
    p = decision_p_value(test, margin, interval, confidence, bca)
    _check_consistency(test, margin, interval, confidence, p)


def _panel(rng: np.random.Generator, n_units: int, accuracy_b: float) -> ReliabilityStudy:
    """Four coders on the same units: r1, r2 at accuracy 0.9, r3, r4 at ``accuracy_b``."""
    truth = rng.integers(0, 4, n_units)

    def coder(accuracy: float) -> list[int]:
        guess = rng.integers(0, 4, n_units)
        return list(np.where(rng.random(n_units) < accuracy, truth, guess))

    matrix = {"r1": coder(0.9), "r2": coder(0.9), "r3": coder(accuracy_b), "r4": coder(accuracy_b)}
    return human_study(matrix, "nominal", [0, 1, 2, 3])


A, B = {"rater_id": ["r1", "r2"]}, {"rater_id": ["r3", "r4"]}


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_compare_reports_a_verdict_and_p_value_for_every_estimate(method: Any) -> None:
    study = _panel(np.random.default_rng(1), 300, 0.9)
    diff = study.compare(
        A, B, margin=0.15, test="non_inferiority", method=method, n_resamples=500, random_state=3
    )
    assert diff.test == "non_inferiority" and diff.margin == 0.15
    assert set(diff.verdicts) == set(diff.differences) == set(diff.p_values)
    assert diff.verdicts["as_missing"] == "non_inferior"
    assert diff.p_values["as_missing"] < 0.025
    for name, interval in diff.differences.items():
        assert diff.verdicts[name] == verdict("non_inferiority", 0.15, interval)

    frame = diff.to_frame()
    assert list(frame["verdict"]) == [diff.verdicts[n] for n in frame["estimate"]]
    payload = json.loads(diff.to_json())
    assert payload["test"] == "non_inferiority" and payload["margin"] == 0.15
    assert payload["verdicts"] == dict(diff.verdicts)
    assert payload["provenance"]["declarations"] == {"margin": 0.15, "test": "non_inferiority"}
    text = diff.summary()
    assert "test: non-inferiority at margin 0.15" in text
    assert "non_inferior" in text and "one-sided test at level 0.025" in text


def test_a_worse_configuration_is_inferior() -> None:
    study = _panel(np.random.default_rng(2), 300, 0.4)
    diff = study.compare(A, B, margin=0.05, test="non_inferiority", n_resamples=500, random_state=4)
    assert diff.verdicts["as_missing"] == "inferior"
    assert diff.p_values["as_missing"] > 0.5


def test_equivalence_at_five_percent_reads_the_ninety_percent_interval() -> None:
    study = _panel(np.random.default_rng(5), 400, 0.9)
    diff = study.compare(
        A, B, margin=0.15, test="equivalence", confidence=0.90, n_resamples=500, random_state=6
    )
    interval = diff["as_missing"]
    assert diff.verdicts["as_missing"] == "equivalent"
    assert interval.low > -0.15 and interval.high < 0.15
    assert diff.p_values["as_missing"] < 0.05
    assert "two one-sided tests at level 0.05" in diff.summary()


def test_without_a_test_there_are_no_verdicts() -> None:
    study = _panel(np.random.default_rng(7), 60, 0.9)
    diff = study.compare(A, B, n_resamples=50, random_state=1)
    assert diff.test is None and diff.margin is None
    assert not diff.verdicts and not diff.p_values
    assert diff.to_frame()["verdict"].isna().all()
    assert "verdict" not in diff.summary()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"margin": 0.05}, "go together"),
        ({"test": "non_inferiority"}, "go together"),
        ({"margin": 0.05, "test": "superiority"}, "test must be one of"),
        ({"margin": 0, "test": "equivalence"}, "margin must be"),
        ({"margin": -0.1, "test": "equivalence"}, "margin must be"),
        ({"margin": math.nan, "test": "equivalence"}, "margin must be"),
        ({"margin": True, "test": "equivalence"}, "margin must be"),
        ({"margin": "0.05", "test": "equivalence"}, "margin must be"),
    ],
)
def test_bad_tests_are_refused(kwargs: dict[str, Any], match: str) -> None:
    study = _panel(np.random.default_rng(8), 40, 0.9)
    with pytest.raises(DesignError, match=match):
        study.compare(A, B, n_resamples=20, **kwargs)
