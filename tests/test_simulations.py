"""Sanity checks on the coverage simulation harness (simulations/, not the package)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulations import coders, paired
from simulations.coverage import (
    Scenario,
    alpha_of,
    draw_panel,
    run_cell,
    simulate,
    study_of,
)
from simulations.grid import scenarios


def test_perfect_coders_give_alpha_one_and_guessers_zero() -> None:
    rng = np.random.default_rng(0)
    # Hard units lose 0.25 accuracy, so 1.25 keeps every coder always right.
    perfect = Scenario("perfect", 0, accuracy=1.25)
    assert alpha_of(simulate(perfect, 2000, rng).codes, perfect) == pytest.approx(1.0)
    guessing = Scenario("guess", 0, accuracy=0.0)
    assert alpha_of(simulate(guessing, 50_000, rng).codes, guessing) == pytest.approx(0.0, abs=0.02)


def test_informative_failures_concentrate_on_hard_units() -> None:
    s = Scenario("informative", 0, failures="informative")
    sim = simulate(s, 20_000, np.random.default_rng(1))
    failed = (sim.codes < 0).mean(axis=1)
    assert failed[sim.hard].mean() == pytest.approx(0.35, abs=0.02)
    assert failed[~sim.hard].mean() == pytest.approx(0.05, abs=0.01)


def test_clusters_share_hardness_and_have_the_intended_size() -> None:
    s = Scenario("clustered", 0, cluster_size=5, cohesion=0.7)
    sim = simulate(s, 1000, np.random.default_rng(2))
    sizes = np.bincount(sim.cluster)
    assert (sizes == 5).all()
    hardness = [set(sim.hard[sim.cluster == c]) for c in range(len(sizes))]
    assert all(len(h) == 1 for h in hardness)


def test_a_seed_reproduces_the_data() -> None:
    s = Scenario("ordinal", 50, measurement="ordinal", prevalence=(0.15, 0.2, 0.3, 0.2, 0.15))
    a = simulate(s, 50, np.random.default_rng([7, 3]))
    b = simulate(s, 50, np.random.default_rng([7, 3]))
    np.testing.assert_array_equal(a.codes, b.codes)


def test_study_of_declares_failures_and_groups() -> None:
    s = Scenario("x", 40, failures="mcar", cluster_size=5)
    study = study_of(simulate(s, 40, np.random.default_rng(3)), s)
    assert study.missing_policies == ("as_missing", "as_category")
    result = study.alpha()
    assert result.coverage.status_counts["parse_error"] > 0
    assert result._sample is not None and result._sample.unit_groups is not None


def test_run_cell_smoke() -> None:
    s = Scenario("clustered", 40, cluster_size=5, cohesion=0.7)
    cell = run_cell(s, n_sim=3, n_resamples=50, base_seed=1, cell_index=0)
    summary = cell.summary()
    assert list(summary["scheme"]) == ["units", "units_group", "kalpha", "bca"]
    expected = {
        "coverage",
        "coverage_mc_se",
        "miss_below",
        "miss_above",
        "mean_width",
        "median_width",
        "undefined_rate",
        "bias",
        "true_alpha",
        "true_alpha_mc_error",
        "complete_alpha",
        "sensitivity_covers_complete",
    }
    assert expected <= set(summary.columns)
    assert (summary["n_sim"] == 3).all()


def test_panels_with_spread_give_coders_different_accuracies() -> None:
    s = Scenario("x", 0, n_coders=10)
    panel = draw_panel(s, 10, 0.2, np.random.default_rng(1))
    assert panel.accuracy.std() > 0.1
    assert panel.tilt == pytest.approx(0.4)
    same = draw_panel(s, 10, 0.0, np.random.default_rng(1))
    assert (same.accuracy == 0.8).all() and same.tilt == 0


def test_a_panel_without_spread_reproduces_the_p5_data() -> None:
    s = Scenario("baseline", 50)
    plain = simulate(s, 50, np.random.default_rng([7, 1]))
    panel = draw_panel(s, s.n_coders, 0.0, np.random.default_rng(99))
    with_panel = simulate(s, 50, np.random.default_rng([7, 1]), panel)
    np.testing.assert_array_equal(plain.codes, with_panel.codes)


def test_coder_grid_smoke() -> None:
    cs = coders.CoderScenario(Scenario("nominal", 30, n_coders=4), 0.2)
    cell = coders.run_cell(cs, n_sim=2, n_resamples=50, base_seed=1, index=0)
    summary = cell.summary()
    assert list(summary["scheme"]) == ["units", "coders", "two_way", "jackknife"]
    expected = {
        "coverage_own",
        "coverage_coders_fixed",
        "coverage_questions_fixed",
        "coverage_both_random",
        "mean_width",
        "both_random",
        "both_random_mc_error",
    }
    assert expected <= set(summary.columns)
    assert len(coders.scenarios()) == 19


def test_grid_has_26_distinct_cells() -> None:
    cells = [s.cell for s in scenarios()]
    assert len(cells) == 26 == len(set(cells))


# -- the paired (non-inferiority) grid ---------------------------------------------

PAIRED_RESULTS = Path(__file__).resolve().parent.parent / "simulations" / "results" / "paired"


def test_paired_arms_share_their_units() -> None:
    ps = paired.PairedScenario(Scenario("nominal", 200), 0.6)
    a, b = paired.rate_pair(ps, 200, np.random.default_rng(3))
    np.testing.assert_array_equal(a.truth, b.truth)
    np.testing.assert_array_equal(a.hard, b.hard)
    assert alpha_of(b.codes, ps.arm_b) < alpha_of(a.codes, ps.base)


def test_equal_cells_have_no_true_difference() -> None:
    ps = paired.PairedScenario(Scenario("nominal", 50), 0.8)
    assert ps.kind == "equal" and ps.cell == "nominal-equal-n50"
    truth = paired.paired_truth(ps, seed=5)
    assert truth["true_delta"] == 0.0
    assert abs(truth["simulated_delta"]) < 5 * truth["true_delta_mc_error"] + 0.002


@pytest.mark.parametrize("accuracy_b", [paired.ACCURACY_B, 0.8])
def test_paired_grid_smoke(accuracy_b: float) -> None:
    ps = paired.PairedScenario(Scenario("nominal", 40), accuracy_b)
    summary = paired.run_cell(ps, n_sim=3, n_resamples=50, base_seed=1, index=0).summary()
    assert list(zip(summary["method"], summary["confidence"], strict=True)) == [
        ("percentile", 0.95),
        ("percentile", 0.96),
        ("bca", 0.95),
        ("bca", 0.96),
    ]
    if ps.kind == "margin":
        assert summary["margin"].iloc[0] == pytest.approx(-summary["true_delta"].iloc[0])
        assert summary["verdict_agrees"].all()
        assert summary[["rejection_rate", "rejection_mc_se"]].notna().all().all()
    else:
        assert summary[[f"power_{m:g}" for m in paired.POWER_MARGINS]].notna().all().all()
    cells = [s.cell for s in paired.scenarios()]
    assert len(cells) == 18 == len(set(cells))


def test_the_non_inferiority_gate() -> None:
    """Release gate for 0.2.0, as revised on 30 Oct 2026 (docs/design/roadmap.md).

    At confidence 0.96 the BCa test must hold 2.5%: on average over the margin
    cells within two pooled Monte Carlo standard errors (2,000 replications per
    cell), and in every cell within two of that cell's. At 0.95 no cell may exceed
    4%. Every verdict agrees with its interval.
    """
    data = pd.read_csv(PAIRED_RESULTS / "coverage.csv")
    margin = data[(data["kind"] == "margin") & (data["method"] == "bca")]
    assert len(margin) == 20 and (margin["n_sim"] == 2000).all()
    assert margin["verdict_agrees"].all()
    strict = margin[margin["confidence"] == 0.96]
    assert len(strict) == 10
    pooled_se = (0.025 * 0.975 / (2000 * len(strict))) ** 0.5
    assert strict["rejection_rate"].mean() <= 0.025 + 2 * pooled_se
    cell_se = (0.025 * 0.975 / 2000) ** 0.5
    over = strict[strict["rejection_rate"] > 0.025 + 2 * cell_se]
    assert over.empty, over[["cell", "rejection_rate"]].to_string()
    usual = margin[margin["confidence"] == 0.95]
    over = usual[usual["rejection_rate"] > 0.04]
    assert over.empty, over[["cell", "rejection_rate"]].to_string()
