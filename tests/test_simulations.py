"""Sanity checks on the coverage simulation harness (simulations/, not the package)."""

import numpy as np
import pytest

from simulations import coders
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
