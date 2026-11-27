"""Tests for core text interval bars and matplotlib figures in reliability_lab.plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study

from reliability_lab import (
    DesignError,
    Interval,
    RatingFrame,
    ReliabilityStudy,
    format_interval_bars,
    plot_coder_influence,
    plot_coincidence,
    plot_policy_forest,
)


def test_format_interval_bars_and_summaries() -> None:
    intervals = {
        "as_missing": Interval(
            estimate="as_missing",
            point=0.75,
            low=0.60,
            high=0.88,
            n_resamples=100,
            n_failed=0,
        ),
        "undefined": Interval(
            estimate="undefined",
            point=np.nan,
            low=np.nan,
            high=np.nan,
            n_resamples=100,
            n_failed=100,
            undefined_reason="all failed",
        ),
    }
    bars = format_interval_bars(intervals, width=20, reference=0.70)
    assert len(bars["as_missing"]) == 22
    assert bars["as_missing"].startswith("|") and bars["as_missing"].endswith("|")
    assert "●" in bars["as_missing"]
    assert "─" in bars["as_missing"]
    assert bars["undefined"].count("●") == 0

    with pytest.raises(ValueError, match="at least 4"):
        format_interval_bars(intervals, width=2)

    study = human_study(FOUR_CODERS_MISSING["matrix"], "ordinal", [1, 2, 3, 4, 5])
    boot = study.alpha().bootstrap(scheme="units", n_resamples=40, random_state=7, alpha_min=0.70)
    boot_summary = boot.summary()
    assert "interval" in boot_summary
    assert "●" in boot_summary


def test_plot_policy_forest_structure_and_modes(tmp_path: Path) -> None:
    rows = []
    for cond in ("v1", "v2"):
        for u in range(1, 21):
            val = "a" if u % 2 == 0 else "b"
            for r in ("c1", "c2", "c3"):
                if cond == "v2" and u == 1 and r == "c3":
                    rows.append(
                        {
                            "u": f"u{u}",
                            "r": r,
                            "cond": cond,
                            "v": None,
                            "status": "parse_error",
                        }
                    )
                else:
                    rows.append(
                        {"u": f"u{u}", "r": r, "cond": cond, "v": val, "status": "observed"}
                    )
    df = pd.DataFrame(rows)
    rf = RatingFrame.from_long(
        df, unit="u", rater="r", value="v", status="status", condition_id="cond"
    )
    study_cmp = ReliabilityStudy(
        rf,
        measurement="nominal",
        values=["a", "b"],
        coders=["rater_id", "condition_id"],
        design="human_panel",
        missing_policies=["as_missing", "as_category", "sensitivity"],
    )
    cmp_res = study_cmp.compare(
        {"condition_id": "v1"},
        {"condition_id": "v2"},
        n_resamples=40,
        random_state=11,
        margin=0.05,
        test="non_inferiority",
    )
    assert "●" in cmp_res.summary()

    # BootstrapResult forest plot
    png_boot = tmp_path / "forest_boot.png"
    fig_boot = plot_policy_forest(cmp_res.arm_b, mode="light", path=png_boot)
    assert png_boot.exists()
    ax_boot = fig_boot.axes[0]
    tick_labels = [t.get_text() for t in ax_boot.get_yticklabels()]
    assert tick_labels == [
        "as_missing",
        "as_category",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    # 4 intervals -> 4 horizontal bars + 4 point markers = 8 line objects
    assert len(ax_boot.lines) == 8
    plt.close(fig_boot)

    # ComparisonResult forest plot in dark mode
    fig_cmp = plot_policy_forest(cmp_res, mode="dark")
    ax_cmp = fig_cmp.axes[0]
    assert [t.get_text() for t in ax_cmp.get_yticklabels()] == [
        "as_missing",
        "as_category",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    plt.close(fig_cmp)

    # GwetResult forest plot
    study_single = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    gwet_res = study_single.gwet()
    fig_gwet = plot_policy_forest(gwet_res)
    assert len(fig_gwet.axes) == 1
    plt.close(fig_gwet)

    # JackknifeResult forest plot
    jack_res = study_single.alpha().coder_jackknife()
    fig_jack = plot_policy_forest(jack_res)
    assert len(fig_jack.axes) == 1
    plt.close(fig_jack)

    # Direct AlphaResult raises helpful DesignError
    with pytest.raises(DesignError, match="point estimates only"):
        plot_policy_forest(study_single.alpha())


def test_plot_coincidence_side_by_side_and_residual(tmp_path: Path) -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    res = study.alpha()

    png_coinc = tmp_path / "coincidence.png"
    fig_sbs = plot_coincidence(res, view="side_by_side", mode="light", path=png_coinc)
    assert png_coinc.exists()
    assert len(fig_sbs.axes) == 2
    assert "Observed coincidences" in fig_sbs.axes[0].get_title()
    assert "Expected coincidences" in fig_sbs.axes[1].get_title()
    assert [t.get_text() for t in fig_sbs.axes[0].get_xticklabels()] == ["1", "2", "3", "4", "5"]
    plt.close(fig_sbs)

    fig_res = plot_coincidence(res["as_missing"], view="residual", mode="dark")
    assert len(fig_res.axes) == 1
    assert "Observed - Expected" in fig_res.axes[0].get_title()
    plt.close(fig_res)

    with pytest.raises(DesignError, match=r"not in result\.estimates"):
        plot_coincidence(res, policy="nonexistent")


def test_plot_coder_influence_structure(tmp_path: Path) -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    res = study.alpha()

    png_inf = tmp_path / "influence.png"
    fig = plot_coder_influence(res, mode="light", path=png_inf)
    assert png_inf.exists()
    ax = fig.axes[0]
    labels = {t.get_text() for t in ax.get_yticklabels()}
    assert labels == {"A", "B", "C", "D"}
    # 1 vertical reference line + 4 stems + 4 markers = 9 lines
    assert len(ax.lines) == 9
    plt.close(fig)

    with pytest.raises(DesignError, match="not in jackknife"):
        plot_coder_influence(res, policy="nonexistent")
