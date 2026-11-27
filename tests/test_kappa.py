"""Tests for Conger's, Fleiss's, and Cohen's kappa coefficients in reliability_lab.metrics."""

from __future__ import annotations

import json
import math
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
    KappaResult,
    RatingFrame,
    ReliabilityStudy,
    UndefinedAlphaError,
    compute_kappa,
    kappa,
    plot_policy_forest,
)


def test_kappa_variants_cohen_conger_fleiss() -> None:
    # Two-coder study: Cohen's kappa equals Conger's kappa, and matches manual 2x2 formula.
    # 10 units: c1=[a,a,a,a,a,b,b,b,b,b], c2=[a,a,a,a,b,a,b,b,b,b]
    # pa = 8/10 = 0.80. p1 = (0.5, 0.5), p2 = (0.5, 0.5) -> pe = 0.50 -> kappa = 0.60
    matrix_2 = {
        "c1": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        "c2": ["a", "a", "a", "a", "b", "a", "b", "b", "b", "b"],
    }
    study_2 = human_study(matrix_2, "nominal", ["a", "b"])
    r_cohen = study_2.kappa(variant="cohen")
    r_conger = kappa(study_2, variant="conger")
    r_fleiss = study_2.kappa(variant="fleiss")

    assert isinstance(r_cohen, KappaResult)
    assert r_cohen.coefficient_name == "Cohen's kappa"
    assert r_conger.coefficient_name == "Conger's kappa"
    assert r_fleiss.coefficient_name == "Fleiss' kappa"
    assert r_cohen.kappa == pytest.approx(0.60, abs=1e-12)
    assert r_conger.kappa == pytest.approx(0.60, abs=1e-12)
    assert r_fleiss.kappa == pytest.approx(0.60, abs=1e-12)
    assert r_cohen["as_missing"].se == pytest.approx(r_conger["as_missing"].se, abs=1e-12)

    # With marginal asymmetry between two raters, Conger/Cohen differs from Fleiss (Scott's pi):
    # c1 rates 'a' 8/10, c2 rates 'a' 4/10; agree on 6/10 -> pa = 0.60
    # Cohen/Conger pe = 0.8*0.4 + 0.2*0.6 = 0.44 -> kappa = (0.60 - 0.44) / 0.56 = 2/7
    # Fleiss pi = (0.6, 0.4) -> pe = 0.36 + 0.16 = 0.52 -> kappa = (0.60 - 0.52) / 0.48 = 1/6
    matrix_asym = {
        "c1": ["a", "a", "a", "a", "a", "a", "a", "a", "b", "b"],
        "c2": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
    }
    study_asym = human_study(matrix_asym, "nominal", ["a", "b"])
    assert study_asym.kappa(variant="cohen").kappa == pytest.approx(2.0 / 7.0, abs=1e-12)
    assert study_asym.kappa(variant="conger").kappa == pytest.approx(2.0 / 7.0, abs=1e-12)
    assert study_asym.kappa(variant="fleiss").kappa == pytest.approx(1.0 / 6.0, abs=1e-12)

    # Cohen refused when n_coders > 2
    study_4 = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    with pytest.raises(DesignError, match="exactly 2 coders"):
        study_4.kappa(variant="cohen")


def test_kappa_missing_policies_serialization_and_plot(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for u in range(1, 16):
        v = 1 if u % 2 == 0 else 3
        for r in ("c1", "c2", "c3"):
            if u == 2 and r == "c3":
                rows.append(
                    {"unit_id": f"u{u}", "rater_id": r, "value": None, "status": "parse_error"}
                )
            else:
                rows.append({"unit_id": f"u{u}", "rater_id": r, "value": v, "status": "observed"})
    df = pd.DataFrame(rows)
    rf = RatingFrame.from_long(df, unit="unit_id", rater="rater_id", value="value", status="status")
    study = ReliabilityStudy(
        rf,
        measurement="interval",
        values=[1, 2, 3],
        coders="rater_id",
        design="human_panel",
        missing_policies=["as_missing", "sensitivity"],
    )
    res = study.kappa(variant="conger")
    assert res.weights_name == "quadratic"
    assert list(res.estimates) == [
        "as_missing",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    assert res["sensitivity_pessimistic"].kappa <= res["sensitivity_optimistic"].kappa

    payload = json.loads(res.to_json(tmp_path / "kappa.json"))
    assert payload["schema"] == "reliability-lab/kappa-result"
    assert payload["variant"] == "conger"
    assert "Conger's kappa" in res.summary()

    pq_dir = tmp_path / "kappa_pq"
    res.to_parquet(pq_dir)
    assert (pq_dir / "estimates.parquet").exists()
    assert (pq_dir / "weights.parquet").exists()

    fig = plot_policy_forest(res)
    assert len(fig.axes) == 1
    plt.close(fig)


@pytest.mark.parametrize("variant", ["fleiss", "conger"])
def test_kappa_bootstrap_matches_naive_unit_resampling_and_bca(variant: str) -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "interval", [1, 2, 3, 4, 5])
    res = study.kappa(variant=variant)  # type: ignore[arg-type]
    boot = res.bootstrap(n_resamples=50, method="percentile", random_state=31)
    assert "as_missing" in boot.intervals

    sample = res._sample
    assert sample is not None
    inp = sample.inputs["as_missing"]
    w_mat = res["as_missing"].weights_matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(boot.seed)
    n_units = sample.n_units
    weights = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=50)

    naive = []
    for b in range(50):
        u_list = []
        v_list = []
        c_list = []
        new_u = 0
        for u_idx, count in enumerate(weights[b]):
            if count == 0:
                continue
            mask = inp.unit_idx == u_idx
            u_vals = inp.value_idx[mask]
            u_coders = inp.coder_idx[mask]
            for _ in range(int(count)):
                if len(u_vals):
                    u_list.append(np.full(len(u_vals), new_u, dtype=np.intp))
                    v_list.append(u_vals)
                    c_list.append(u_coders)
                new_u += 1
        comp = compute_kappa(
            np.concatenate(u_list),
            np.concatenate(v_list),
            np.concatenate(c_list),
            sample.n_coders,
            inp.values,
            w_mat,
            variant=variant,  # type: ignore[arg-type]
        )
        naive.append(comp.kappa)

    np.testing.assert_allclose(boot["as_missing"].distribution, np.array(naive), rtol=1e-12)

    boot_bca = res.bootstrap(n_resamples=60, method="bca", random_state=37)
    assert boot_bca["as_missing"].defined
    assert math.isfinite(boot_bca["as_missing"].low) and math.isfinite(boot_bca["as_missing"].high)


def test_kappa_validation_errors() -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    with pytest.raises(DesignError, match="unknown kappa variant"):
        study.kappa(variant="invalid")  # type: ignore[arg-type]

    df = pd.DataFrame({"u": ["u1", "u2"], "r": ["c1", "c2"], "v": ["a", "b"]})
    rf = RatingFrame.from_long(df, unit="u", rater="r", value="v")
    s_undef = ReliabilityStudy(
        rf, measurement="nominal", values=["a", "b"], coders="rater_id", design="human_panel"
    )
    with pytest.raises(UndefinedAlphaError, match="fewer than two ratings"):
        s_undef.kappa()
