import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from reference import reference_alpha

from reliability_lab import DesignError, RatingFrame, ReliabilityStudy

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")

N_UNITS = 40


def panel_matrix(seed: int = 0) -> dict[str, list[int]]:
    """Configuration a is r1, r2; b is r3, r4: copies of r1, r2 with a few changes."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(1, 5, N_UNITS)

    def rater() -> list[int]:
        noise = rng.integers(1, 5, N_UNITS)
        return list(np.where(rng.random(N_UNITS) < 0.75, truth, noise).astype(int))

    r1, r2 = rater(), rater()
    r3, r4 = list(r1), list(r2)
    for u in rng.choice(N_UNITS, 4, replace=False):
        r3[u] = int(rng.integers(1, 5))
    return {"r1": r1, "r2": r2, "r3": r3, "r4": r4}


def study_of(matrix: dict[str, list[Any]], groups: bool = False) -> ReliabilityStudy:
    rows = [
        {"unit": u, "rater": r, "score": v}
        for r, values in matrix.items()
        for u, v in enumerate(values)
        if v is not None
    ]
    df = pd.DataFrame(rows)
    kwargs = {}
    if groups:
        df["conversation"] = df["unit"] // 4
        kwargs["group"] = "conversation"
    ratings = RatingFrame.from_long(df, unit="unit", rater="rater", value="score", **kwargs)
    return ReliabilityStudy(
        ratings,
        measurement="ordinal",
        values=[1, 2, 3, 4],
        coders="rater_id",
        design="human_panel",
    )


A = {"rater_id": ["r1", "r2"]}
B = {"rater_id": ["r3", "r4"]}


def test_difference_is_b_minus_a_on_identical_resamples() -> None:
    matrix = panel_matrix()
    diff = study_of(matrix).compare(A, B, n_resamples=300, random_state=3)
    alpha_a = reference_alpha({k: matrix[k] for k in ("r1", "r2")}, "ordinal", [1, 2, 3, 4])
    alpha_b = reference_alpha({k: matrix[k] for k in ("r3", "r4")}, "ordinal", [1, 2, 3, 4])
    d = diff["as_missing"]
    assert diff.arm_a["as_missing"].point == pytest.approx(alpha_a)
    assert diff.arm_b["as_missing"].point == pytest.approx(alpha_b)
    assert d.point == pytest.approx(alpha_b - alpha_a)
    np.testing.assert_allclose(
        d.distribution,
        diff.arm_b["as_missing"].distribution - diff.arm_a["as_missing"].distribution,
    )
    assert diff.claim == "configuration_difference"


def test_paired_interval_is_narrower_than_independent_one() -> None:
    matrix = panel_matrix()
    study = study_of(matrix)
    paired = study.compare(A, B, n_resamples=2000, random_state=1)["as_missing"]
    alone_a = study_of({k: matrix[k] for k in ("r1", "r2")}).alpha()
    alone_b = study_of({k: matrix[k] for k in ("r3", "r4")}).alpha()
    dist_a = alone_a.bootstrap(scheme="units", n_resamples=2000, random_state=1)["as_missing"]
    dist_b = alone_b.bootstrap(scheme="units", n_resamples=2000, random_state=2)["as_missing"]
    independent = np.quantile(dist_b.distribution - dist_a.distribution, [0.025, 0.975])
    assert paired.high - paired.low < (independent[1] - independent[0]) / 2


def test_group_level_is_used_when_groups_are_declared() -> None:
    diff = study_of(panel_matrix(), groups=True).compare(A, B, n_resamples=100, random_state=0)
    assert (diff.arm_a.level, diff.arm_a.n_clusters) == ("group", N_UNITS // 4)


def test_report_and_serialisation() -> None:
    diff = study_of(panel_matrix()).compare(A, B, n_resamples=200, random_state=5)
    text = diff.summary()
    assert "claim: configuration_difference" in text
    assert "scheme units" in text and "difference = b - a" in text
    payload = json.loads(json.dumps(diff.to_dict(), allow_nan=False))
    assert payload["a"] == A and payload["b"] == B
    assert payload["scheme"] == "units"
    table = diff.to_frame()
    assert {"alpha_a", "alpha_b", "diff_point", "diff_low", "diff_high"} <= set(table.columns)


def test_unequal_unit_sets_are_refused() -> None:
    matrix = panel_matrix()
    matrix["r3"][0] = None  # type: ignore[call-overload]
    matrix["r4"][0] = None  # type: ignore[call-overload]
    with pytest.raises(DesignError, match="same units"):
        study_of(matrix).compare(A, B, n_resamples=10)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ({"rater_id": ["r1", "r2"]}, {"rater_id": ["r2", "r3"]}, "selected by both"),
        ({"value": 1}, B, "selectors may use"),
        ({"model_id": "gpt"}, B, "not mapped"),
        ({"rater_id": "nobody"}, B, "selects no rows"),
        ({}, B, "non-empty mapping"),
    ],
)
def test_selector_errors(a: dict[str, Any], b: dict[str, Any], message: str) -> None:
    with pytest.raises(DesignError, match=message):
        study_of(panel_matrix()).compare(a, b, n_resamples=10)


def test_kalpha_has_no_paired_form() -> None:
    with pytest.raises(DesignError, match="no paired form"):
        study_of(panel_matrix()).compare(A, B, scheme="kalpha")


def test_bca_comparison() -> None:
    diff = study_of(panel_matrix()).compare(A, B, method="bca", n_resamples=500, random_state=2)
    assert diff.arm_a.method == "bca"
    assert not diff.arm_a.experimental
    assert not [w for w in diff.warnings if w.category == "BootstrapWarning"]
    d = diff["as_missing"]
    assert d.low <= d.point <= d.high
