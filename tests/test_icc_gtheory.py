"""Tests for Intraclass Correlation Coefficients (ICC) and Generalizability Theory (G-theory)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study

from reliability_lab import (
    DesignError,
    GTheoryResult,
    ICCResult,
    RatingFrame,
    ReliabilityStudy,
    compute_icc,
    g_theory,
    icc,
    plot_policy_forest,
)

# Shrout & Fleiss (1979, Table 2, p. 423): 6 targets rated by 4 judges on a 1..10 scale.
SHROUT_FLEISS_MATRIX: dict[str, list[int]] = {
    "j1": [9, 6, 8, 7, 10, 6],
    "j2": [2, 1, 4, 1, 5, 2],
    "j3": [5, 3, 6, 2, 6, 4],
    "j4": [8, 2, 8, 6, 9, 7],
}
SHROUT_FLEISS_VALUES = list(range(1, 11))


def test_shrout_fleiss_1979_table2_all_six_iccs_and_variance_components() -> None:
    study = human_study(SHROUT_FLEISS_MATRIX, "interval", SHROUT_FLEISS_VALUES)

    # 1. ICC(1,1) and ICC(1,k)
    r_11 = study.icc(model="one_way", agreement="agreement", unit="single")
    r_1k = study.icc(model="one_way", agreement="agreement", unit="average")
    assert r_11.code == "ICC(1,1)"
    assert r_1k.code == "ICC(1,k)"
    assert r_11.icc == pytest.approx(0.16574, abs=1e-4)
    assert r_1k.icc == pytest.approx(0.44277, abs=1e-4)
    vc1 = r_11.variance_components
    assert vc1 is not None
    assert vc1.ms_unit == pytest.approx(11.24167, abs=1e-4)
    assert vc1.ms_residual == pytest.approx(6.26389, abs=1e-4)
    assert r_11["as_missing"].low == pytest.approx(-0.1329, abs=1e-3)
    assert r_11["as_missing"].high == pytest.approx(0.7226, abs=1e-3)

    # 2. ICC(2,1) and ICC(2,k)
    r_21 = icc(study, model="two_way_random", agreement="agreement", unit="single")
    r_2k = study.icc(model="two_way_random", agreement="agreement", unit="average")
    assert isinstance(r_21, ICCResult)
    assert r_21.code == "ICC(2,1)"
    assert r_2k.code == "ICC(2,k)"
    assert r_21.icc == pytest.approx(0.28976, abs=1e-4)
    assert r_2k.icc == pytest.approx(0.62005, abs=1e-4)
    vc2 = r_21.variance_components
    assert vc2 is not None
    assert vc2.ms_unit == pytest.approx(11.24167, abs=1e-4)
    assert vc2.ms_coder == pytest.approx(32.48611, abs=1e-4)
    assert vc2.ms_residual == pytest.approx(1.01944, abs=1e-4)
    assert vc2.sigma2_unit == pytest.approx((11.24167 - 1.01944) / 4.0, abs=1e-4)
    assert vc2.sigma2_coder == pytest.approx((32.48611 - 1.01944) / 6.0, abs=1e-4)
    assert vc2.sigma2_residual == pytest.approx(1.01944, abs=1e-4)
    assert r_21["as_missing"].low == pytest.approx(0.0188, abs=1e-3)
    assert r_21["as_missing"].high == pytest.approx(0.7611, abs=1e-3)

    # 3. ICC(3,1) and ICC(3,k)
    r_31 = study.icc(model="two_way_mixed", agreement="consistency", unit="single")
    r_3k = study.icc(model="two_way_mixed", agreement="consistency", unit="average")
    assert r_31.code == "ICC(3,1)"
    assert r_3k.code == "ICC(3,k)"
    assert r_31.icc == pytest.approx(0.71484, abs=1e-4)
    assert r_3k.icc == pytest.approx(0.90935, abs=1e-4)
    assert r_31["as_missing"].low == pytest.approx(0.3425, abs=1e-3)
    assert r_31["as_missing"].high == pytest.approx(0.9459, abs=1e-3)

    # On balanced data with positive variance components:
    # Phi(4) == ICC(2,4) and E_rho2(4) == ICC(3,4)
    assert vc2.phi == pytest.approx(r_2k.icc, abs=1e-12)
    assert vc2.e_rho2 == pytest.approx(r_3k.icc, abs=1e-12)


def test_g_theory_d_study_and_serialization(tmp_path: Path) -> None:
    study = human_study(SHROUT_FLEISS_MATRIX, "interval", SHROUT_FLEISS_VALUES)
    gt = g_theory(study, k=[1, 2, 4, 8])
    assert isinstance(gt, GTheoryResult)
    assert gt.k_values == (1, 2, 4, 8)

    d_ref = gt.d_study[gt.d_study["estimate"] == "as_missing"].reset_index(drop=True)
    assert list(d_ref["k"]) == [1, 2, 4, 8]
    # Both E_rho2(k) and Phi(k) increase monotonically with k, and E_rho2(k) >= Phi(k)
    assert d_ref["e_rho2"].is_monotonic_increasing
    assert d_ref["phi"].is_monotonic_increasing
    assert (d_ref["e_rho2"] >= d_ref["phi"]).all()

    # At k=1, Phi(1) == ICC(2,1) and E_rho2(1) == ICC(3,1)
    assert d_ref.loc[d_ref["k"] == 1, "phi"].iloc[0] == pytest.approx(
        study.icc(model="two_way_random", agreement="agreement", unit="single").icc, abs=1e-12
    )
    assert d_ref.loc[d_ref["k"] == 1, "e_rho2"].iloc[0] == pytest.approx(
        study.icc(model="two_way_mixed", agreement="consistency", unit="single").icc, abs=1e-12
    )

    payload = json.loads(gt.to_json(tmp_path / "gtheory.json"))
    assert payload["schema"] == "reliability-lab/gtheory-result"
    assert len(payload["d_study"]) == 12  # 3 estimates x 4 k values
    assert "Generalizability Theory" in gt.summary()

    pq_dir = tmp_path / "gt_pq"
    gt.to_parquet(pq_dir)
    assert (pq_dir / "estimates.parquet").exists()
    assert (pq_dir / "d_study.parquet").exists()


def test_icc_and_gtheory_unbalanced_henderson_vs_complete_case_and_policies(
    tmp_path: Path,
) -> None:
    # Unbalanced dataset with failed attempts: check as_missing + sensitivity policies
    # and check that as_category (on nominal numeric ratings) is gracefully skipped with a note.
    rows: list[dict[str, object]] = []
    for u in range(1, 13):
        base_score = (u % 5) + 1
        for idx_r, r in enumerate(("c1", "c2", "c3")):
            if u == 1 and r == "c3":
                rows.append(
                    {"unit_id": f"u{u}", "rater_id": r, "value": None, "status": "parse_error"}
                )
            else:
                val = min(5, max(1, base_score + (1 if idx_r == 2 and u % 3 == 0 else 0)))
                rows.append({"unit_id": f"u{u}", "rater_id": r, "value": val, "status": "observed"})
    df = pd.DataFrame(rows)
    rf = RatingFrame.from_long(df, unit="unit_id", rater="rater_id", value="value", status="status")
    study = ReliabilityStudy(
        rf,
        measurement="interval",
        values=[1, 2, 3, 4, 5],
        coders="rater_id",
        design="human_panel",
        missing_policies=["as_missing", "sensitivity"],
    )

    res_unbal = study.icc()
    assert list(res_unbal.estimates) == [
        "as_missing",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    assert res_unbal["as_missing"].n_pairable_units == 12
    assert res_unbal["as_missing"].variance_components is not None
    assert not res_unbal["as_missing"].variance_components.balanced
    assert res_unbal["sensitivity_pessimistic"].icc <= res_unbal["sensitivity_optimistic"].icc

    study_nom_num = ReliabilityStudy(
        rf,
        measurement="nominal",
        values=[1, 2, 3, 4, 5],
        coders="rater_id",
        design="human_panel",
        missing_policies=["as_missing", "as_category"],
    )
    res_nom_num = study_nom_num.icc()
    assert list(res_nom_num.estimates) == ["as_missing"]
    assert any("as_category" in n for n in res_nom_num.notes)

    res_cc = study.icc(complete_case=True)
    assert res_cc["as_missing"].n_pairable_units == 11
    assert res_cc["as_missing"].variance_components is not None
    assert res_cc["as_missing"].variance_components.balanced

    payload = json.loads(res_unbal.to_json(tmp_path / "icc.json"))
    assert payload["schema"] == "reliability-lab/icc-result"
    assert "Variance components" in res_unbal.summary()
    fig = plot_policy_forest(res_unbal)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_icc_and_gtheory_bootstrap_matches_naive_unit_resampling() -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "interval", [1, 2, 3, 4, 5])
    res_icc = study.icc(model="two_way_random", agreement="agreement", unit="single")
    boot_icc = res_icc.bootstrap(n_resamples=40, method="percentile", random_state=41)

    sample = res_icc._sample
    assert sample is not None
    inp = sample.inputs["as_missing"]
    num_domain = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(boot_icc.seed)
    n_units = sample.n_units
    weights = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=40)

    naive_icc = []
    for b in range(40):
        u_list = []
        c_list = []
        y_list = []
        new_u = 0
        for u_idx, count in enumerate(weights[b]):
            if count == 0:
                continue
            mask = inp.unit_idx == u_idx
            u_vals = num_domain[inp.value_idx[mask]]
            u_coders = inp.coder_idx[mask]
            for _ in range(int(count)):
                if len(u_vals):
                    u_list.append(np.full(len(u_vals), new_u, dtype=np.intp))
                    c_list.append(u_coders)
                    y_list.append(u_vals)
                new_u += 1
        comp = compute_icc(
            np.concatenate(u_list),
            np.concatenate(c_list),
            np.concatenate(y_list),
            sample.n_coders,
            model="two_way_random",
            agreement="agreement",
            unit="single",
        )
        naive_icc.append(comp.icc)

    np.testing.assert_allclose(boot_icc["as_missing"].distribution, np.array(naive_icc), rtol=1e-12)

    boot_bca = res_icc.bootstrap(n_resamples=50, method="bca", random_state=43)
    assert boot_bca["as_missing"].defined

    gt = study.g_theory(k=[2, 4])
    boot_gt = gt.bootstrap(coefficient="phi", k=4, n_resamples=40, random_state=47)
    assert boot_gt["as_missing"].defined


def test_icc_and_gtheory_validation_errors() -> None:
    study_str = human_study({"c1": ["a", "b", "a"], "c2": ["a", "b", "b"]}, "nominal", ["a", "b"])
    with pytest.raises(DesignError, match="numeric values"):
        study_str.icc()
    with pytest.raises(DesignError, match="numeric values"):
        study_str.g_theory()

    study_num = human_study(SHROUT_FLEISS_MATRIX, "interval", SHROUT_FLEISS_VALUES)
    with pytest.raises(DesignError, match="does not separate coder main effects"):
        study_num.icc(model="one_way", agreement="consistency")
    with pytest.raises(DesignError, match="positive integer"):
        study_num.g_theory(k=[0])
