"""Tests for analytical and simulation-based sample-size planning."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reliability_lab import (
    Design,
    DesignError,
    RatingFrame,
    ReliabilityStudy,
    SampleSizePlan,
    circular,
    plan_alpha,
    plan_krippendorff,
    plan_non_inferiority,
    simulate_ratings,
)
from reliability_lab.simulate import _prepare_process

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


@pytest.mark.parametrize(
    ("alpha_min", "expected"),
    [
        (2 / 3, 81),
        (0.800, 139),
        (0.900, 283),
    ],
)
def test_plan_krippendorff_matches_table_1_at_pc_025(alpha_min: float, expected: int) -> None:
    plan = plan_krippendorff(
        alpha_min=alpha_min,
        smallest_prevalence=0.25,
        significance=0.05,
        n_coders=2,
    )
    assert isinstance(plan, SampleSizePlan)
    assert plan.goal == "minimum_alpha"
    assert plan.generator == "krippendorff_analytical"
    assert plan.measurement == "nominal"
    assert plan.n_values == expected
    assert plan.n_units == math.ceil(expected / 2)
    assert plan.curve is None
    assert plan.target_power is None
    assert plan.achieved_power is None


@pytest.mark.parametrize(
    ("alpha_min", "significance", "table_val"),
    [
        (2 / 3, 0.10, 104),
        (2 / 3, 0.05, 172),
        (2 / 3, 0.01, 344),
        (2 / 3, 0.005, 421),
        (0.800, 0.10, 178),
        (0.800, 0.05, 293),
        (0.800, 0.01, 587),
        (0.800, 0.005, 719),
        (0.900, 0.10, 361),
        (0.900, 0.05, 595),
        (0.900, 0.01, 1190),
        (0.900, 0.005, 1459),
    ],
)
def test_plan_krippendorff_matches_table_1_at_pc_010(
    alpha_min: float,
    significance: float,
    table_val: int,
) -> None:
    plan = plan_krippendorff(
        alpha_min=alpha_min,
        smallest_prevalence=0.10,
        significance=significance,
        n_coders=2,
    )
    assert plan.n_values is not None
    assert abs(plan.n_values - table_val) <= 1


def test_plan_krippendorff_scales_coders_and_missing_and_warns_multi_coder() -> None:
    base = plan_krippendorff(alpha_min=0.8, smallest_prevalence=0.25, significance=0.05)
    assert base.n_values == 139
    assert base.n_units == 70

    three_coders = plan_krippendorff(
        alpha_min=0.8, smallest_prevalence=0.25, significance=0.05, n_coders=3, missing=0.2
    )
    assert three_coders.n_values == 139
    assert three_coders.n_units == math.ceil(139 / (3 * 0.8))
    assert any("plan_alpha" in note for note in three_coders.notes)
    assert any("inflates" in note for note in three_coders.notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_min": 0.0, "smallest_prevalence": 0.25},
        {"alpha_min": 1.0, "smallest_prevalence": 0.25},
        {"alpha_min": 0.8, "smallest_prevalence": 0.0},
        {"alpha_min": 0.8, "smallest_prevalence": 0.6},
        {"alpha_min": 0.8, "smallest_prevalence": 0.25, "significance": 0.0},
        {"alpha_min": 0.8, "smallest_prevalence": 0.25, "significance": 0.5},
        {"alpha_min": 0.8, "smallest_prevalence": 0.25, "n_coders": 1},
        {"alpha_min": 0.8, "smallest_prevalence": 0.25, "missing": 1.0},
    ],
)
def test_plan_krippendorff_refuses_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(DesignError):
        plan_krippendorff(**kwargs)


def test_plan_alpha_nominal_meets_target_power_on_fresh_seeds() -> None:
    plan = plan_alpha(
        "nominal",
        ["a", "b", "c"],
        alpha=0.80,
        alpha_min=0.667,
        power=0.80,
        n_coders=3,
        scheme="units",
        method="percentile",
        n_sim=200,
        n_resamples=300,
        random_state=10,
    )
    assert plan.curve is not None
    assert plan.curve["fitted_power"].is_monotonic_increasing
    assert plan.achieved_power is not None
    assert plan.achieved_power >= 0.78

    wins = 0
    n_check = 150
    for i in range(n_check):
        rf = simulate_ratings(
            "nominal",
            ["a", "b", "c"],
            n_units=plan.n_units,
            n_coders=3,
            alpha=0.80,
            random_state=10_000 + i,
        )
        study = ReliabilityStudy(
            rf,
            measurement="nominal",
            values=["a", "b", "c"],
            coders="rater_id",
            design=Design.HUMAN_PANEL,
        )
        iv = study.alpha().bootstrap(
            scheme="units",
            method="percentile",
            n_resamples=300,
            confidence=0.95,
            alpha_min=0.667,
            random_state=20_000 + i,
        )["as_missing"]
        if iv.defined and iv.low > 0.667:
            wins += 1
    realized = wins / n_check
    assert 0.71 <= realized <= 0.90


def test_plan_alpha_clustered_ordinal_near_meets_power_on_fresh_seeds() -> None:
    plan = plan_alpha(
        "ordinal",
        [1, 2, 3, 4, 5],
        alpha=0.82,
        alpha_min=0.667,
        power=0.80,
        n_coders=3,
        errors="near",
        cluster_size=4,
        cohesion=0.5,
        n_sim=120,
        n_resamples=250,
        random_state=21,
    )
    assert plan.n_units % 4 == 0
    assert plan.n_clusters == plan.n_units // 4
    assert plan.curve is not None
    assert plan.curve["fitted_power"].is_monotonic_increasing

    wins = 0
    n_check = 100
    for i in range(n_check):
        rf = simulate_ratings(
            "ordinal",
            [1, 2, 3, 4, 5],
            n_units=plan.n_units,
            n_coders=3,
            alpha=0.82,
            errors="near",
            cluster_size=4,
            cohesion=0.5,
            random_state=30_000 + i,
        )
        study = ReliabilityStudy(
            rf,
            measurement="ordinal",
            values=[1, 2, 3, 4, 5],
            coders="rater_id",
            design=Design.HUMAN_PANEL,
        )
        iv = study.alpha().bootstrap(
            scheme="units",
            level="group",
            method="percentile",
            n_resamples=250,
            confidence=0.95,
            alpha_min=0.667,
            random_state=40_000 + i,
        )["as_missing"]
        if iv.defined and iv.low > 0.667:
            wins += 1
    realized = wins / n_check
    assert 0.69 <= realized <= 0.93


def test_plan_alpha_supports_circular_and_kalpha_and_serialization(tmp_path: Any) -> None:
    circ_plan = plan_alpha(
        circular(24),
        list(range(24)),
        alpha=0.85,
        alpha_min=0.70,
        power=0.80,
        n_coders=3,
        errors="near",
        n_sim=60,
        n_resamples=200,
        random_state=5,
    )
    assert circ_plan.measurement == "circular(24)"
    assert circ_plan.n_units >= 4

    kalpha_plan = plan_alpha(
        "nominal",
        ["a", "b", "c"],
        alpha=0.80,
        alpha_min=0.667,
        scheme="kalpha",
        n_sim=80,
        n_resamples=200,
        random_state=6,
    )
    assert kalpha_plan.parameters["scheme"] == "kalpha"
    assert kalpha_plan.curve is not None
    assert kalpha_plan.curve["fitted_power"].is_monotonic_increasing

    frame = kalpha_plan.to_frame()
    assert len(frame) == 1
    tables = kalpha_plan.tables()
    assert set(tables) == {"plan", "curve"}
    text_summary = kalpha_plan.summary()
    assert "Sample-size plan (minimum_alpha)" in text_summary
    assert "Evaluated power curve:" in text_summary

    json_path = tmp_path / "plan.json"
    written = kalpha_plan.to_json(json_path)
    parsed = json.loads(written)
    assert parsed["schema"] == "reliability-lab/plan"
    assert parsed["n_units"] == kalpha_plan.n_units
    assert json_path.is_file()

    pq = pytest.importorskip("pyarrow.parquet")
    pq_dir = tmp_path / "plan_pq"
    out_dir = kalpha_plan.to_parquet(pq_dir)
    assert (out_dir / "plan.parquet").is_file()
    assert (out_dir / "curve.parquet").is_file()
    tbl = pq.read_table(out_dir / "plan.parquet").to_pandas()
    assert int(tbl.loc[0, "n_units"]) == kalpha_plan.n_units


def test_plan_non_inferiority_meets_target_power_on_fresh_seeds() -> None:
    values = ["a", "b", "c"]
    plan = plan_non_inferiority(
        "nominal",
        values,
        alpha=0.80,
        margin=0.09,
        power=0.80,
        n_coders=3,
        method="bca",
        confidence=0.96,
        n_sim=100,
        n_resamples=250,
        random_state=33,
    )
    assert plan.goal == "non_inferiority"
    assert plan.curve is not None
    assert plan.curve["fitted_power"].is_monotonic_increasing

    proc_a = _prepare_process(
        "nominal",
        values,
        n_coders=3,
        alpha=0.80,
        errors="random",
        prevalence=None,
        missing=0.0,
        failures=0.0,
        hard_share=0.0,
        hard_penalty=0.25,
        cluster_size=1,
        cohesion=0.0,
        rho=0.0,
        hard_failures=None,
    )
    proc_b = _prepare_process(
        "nominal",
        values,
        n_coders=3,
        alpha=0.80,
        errors="random",
        prevalence=None,
        missing=0.0,
        failures=0.0,
        hard_share=0.0,
        hard_penalty=0.25,
        cluster_size=1,
        cohesion=0.0,
        rho=0.0,
        hard_failures=None,
    )

    wins = 0
    n_check = 80
    unit_ids = [f"u{u}" for u in range(plan.n_units)]
    for rep in range(n_check):
        rng = np.random.default_rng(50_000 + rep)
        truth, hard, _ = proc_a.draw_structure(plan.n_units, rng)
        raw_a, _, _ = proc_a.rate(truth, hard, rng)
        raw_b, _, _ = proc_b.rate(truth, hard, rng)
        rows = []
        for cond, codes in (("a", raw_a), ("b", raw_b)):
            for u_idx, u_lbl in enumerate(unit_ids):
                for c_idx in range(3):
                    rows.append(
                        {
                            "unit_id": u_lbl,
                            "rater_id": f"c{c_idx}",
                            "condition_id": cond,
                            "value": values[int(codes[u_idx, c_idx])],
                        }
                    )
        rf = RatingFrame.from_long(
            pd.DataFrame(rows),
            unit="unit_id",
            rater="rater_id",
            value="value",
            condition_id="condition_id",
        )
        study = ReliabilityStudy(
            rf,
            measurement="nominal",
            values=values,
            coders=["rater_id", "condition_id"],
            design=Design.HUMAN_PANEL,
        )
        diff = study.compare(
            {"condition_id": "a"},
            {"condition_id": "b"},
            method="bca",
            confidence=0.96,
            margin=0.09,
            test="non_inferiority",
            n_resamples=250,
            random_state=60_000 + rep,
        )
        if diff.verdicts["as_missing"] == "non_inferior":
            wins += 1
    realized = wins / n_check
    assert 0.69 <= realized <= 0.93


def test_planning_validation_refuses_impossible_or_invalid_inputs() -> None:
    with pytest.raises(DesignError, match="exceed alpha_min"):
        plan_alpha("nominal", ["a", "b"], alpha=0.6, alpha_min=0.7)

    with pytest.raises(DesignError, match="only supports method='percentile'"):
        plan_alpha("nominal", ["a", "b"], scheme="kalpha", method="bca")

    with pytest.raises(DesignError, match="does not support cluster_size > 1"):
        plan_alpha("nominal", ["a", "b"], scheme="kalpha", cluster_size=3)

    with pytest.raises(DesignError, match="must exceed -margin"):
        plan_non_inferiority("nominal", ["a", "b"], alpha=0.8, alpha_b=0.7, margin=0.05)

    with pytest.raises(DesignError, match="max_units"):
        plan_alpha(
            "nominal",
            ["a", "b", "c"],
            alpha=0.70,
            alpha_min=0.667,
            power=0.95,
            n_sim=30,
            n_resamples=50,
            max_units=20,
            random_state=1,
        )
