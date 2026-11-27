"""Tests for Gwet's AC1 and AC2 companion coefficients in reliability_lab.metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study

from reliability_lab import (
    DesignError,
    GwetResult,
    RatingFrame,
    ReliabilityStudy,
    UndefinedAlphaError,
    circular,
    compute_gwet,
    gwet,
)


def test_gwet_ac1_prevalence_paradox_and_top_level_helper() -> None:
    # Under extreme 95/5 category skew with 95% agreement, alpha drops near 0 while AC1 stays high.
    matrix = {
        "c1": ["ok"] * 19 + ["bad"],
        "c2": ["ok"] * 19 + ["ok"],
    }
    study = human_study(matrix, "nominal", ["ok", "bad"])
    with pytest.warns(match="holds 98%"):
        alpha_res = study.alpha()
    gwet_res = gwet(study)
    assert isinstance(gwet_res, GwetResult)
    assert gwet_res.coefficient_name == "AC1"
    assert gwet_res.weights_name == "identity"
    assert alpha_res.alpha < 0.1
    assert gwet_res.ac > 0.90
    assert gwet_res["as_missing"].pa == pytest.approx(0.95)


def test_gwet_ac2_distance_defaults_and_weight_overrides() -> None:
    matrix = FOUR_CODERS_MISSING["matrix"]
    values = [1, 2, 3, 4, 5]

    study_nom = human_study(matrix, "nominal", values)
    r_nom = study_nom.gwet()
    assert r_nom.coefficient_name == "AC1"
    assert r_nom.weights_name == "identity"

    study_int = human_study(matrix, "interval", values)
    r_int = study_int.gwet()
    assert r_int.coefficient_name == "AC2"
    assert r_int.weights_name == "quadratic"
    assert r_int.ac == pytest.approx(study_nom.gwet(weights="quadratic").ac, abs=1e-12)

    study_ord = human_study(matrix, "ordinal", values)
    r_ord_default = study_ord.gwet()
    assert r_ord_default.coefficient_name == "AC2"
    assert r_ord_default.weights_name == "ordinal"
    r_ord_gwet = study_ord.gwet(weights="ordinal")
    assert r_ord_gwet.coefficient_name == "AC2"
    assert math.isfinite(r_ord_default.ac) and math.isfinite(r_ord_gwet.ac)

    study_circ = human_study(matrix, circular(5), values)
    r_circ = study_circ.gwet()
    assert r_circ.coefficient_name == "AC2"
    assert math.isfinite(r_circ.ac)

    # Custom weight matrix
    custom_w = np.array(
        [
            [1.0, 0.8, 0.5, 0.2, 0.0],
            [0.8, 1.0, 0.8, 0.5, 0.2],
            [0.5, 0.8, 1.0, 0.8, 0.5],
            [0.2, 0.5, 0.8, 1.0, 0.8],
            [0.0, 0.2, 0.5, 0.8, 1.0],
        ]
    )
    r_custom = study_nom.gwet(weights=custom_w)
    assert r_custom.coefficient_name == "AC2"
    assert r_custom.weights_name == "custom"
    assert math.isfinite(r_custom.ac)


def test_gwet_missing_policies_and_serialization(tmp_path: Path) -> None:
    rows = []
    for u in range(1, 16):
        v = "a" if u % 2 == 0 else "b"
        for r in ("c1", "c2", "c3"):
            if u == 3 and r == "c3":
                rows.append(
                    {"unit_id": f"u{u}", "rater_id": r, "value": None, "status": "parse_error"}
                )
            else:
                rows.append({"unit_id": f"u{u}", "rater_id": r, "value": v, "status": "observed"})
    df = pd.DataFrame(rows)
    rf = RatingFrame.from_long(df, unit="unit_id", rater="rater_id", value="value", status="status")
    study = ReliabilityStudy(
        rf,
        measurement="nominal",
        values=["a", "b"],
        coders="rater_id",
        design="human_panel",
        missing_policies=["as_missing", "as_category", "sensitivity"],
    )
    res = study.gwet()
    assert list(res.estimates) == [
        "as_missing",
        "as_category",
        "sensitivity_pessimistic",
        "sensitivity_optimistic",
    ]
    assert res["sensitivity_pessimistic"].ac <= res["sensitivity_optimistic"].ac

    frame = res.to_frame()
    assert len(frame) == 4
    tables = res.tables()
    assert set(tables) == {"estimates", "weights", "coverage"}

    json_path = tmp_path / "gwet.json"
    text = res.to_json(json_path)
    payload = json.loads(text)
    assert payload["schema"] == "reliability-lab/gwet-result"
    assert payload["coefficient"] == "AC1"
    assert "Gwet's AC1" in res.summary()

    pq_dir = tmp_path / "gwet_pq"
    res.to_parquet(pq_dir)
    assert (pq_dir / "estimates.parquet").exists()
    assert (pq_dir / "weights.parquet").exists()
    assert (pq_dir / "result.json").exists()


def test_gwet_bootstrap_matches_naive_unit_resampling_and_supports_bca() -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "interval", [1, 2, 3, 4, 5])
    res = study.gwet()
    boot = res.bootstrap(n_resamples=60, method="percentile", random_state=19)
    assert "as_missing" in boot.intervals
    assert "│" in boot.summary() or "|" in boot.summary()

    # Check vectorized replicate distribution against naive unit resampling loop
    sample = res._sample
    assert sample is not None
    inp = sample.inputs["as_missing"]
    w_mat = res["as_missing"].weights_matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(boot.seed)
    n_units = sample.n_units
    weights = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=60)

    naive = []
    for b in range(60):
        u_list = []
        v_list = []
        new_u = 0
        for u_idx, count in enumerate(weights[b]):
            if count == 0:
                continue
            u_vals = inp.value_idx[inp.unit_idx == u_idx]
            for _ in range(int(count)):
                if len(u_vals):
                    u_list.append(np.full(len(u_vals), new_u, dtype=np.intp))
                    v_list.append(u_vals)
                new_u += 1
        comp = compute_gwet(
            np.concatenate(u_list),
            np.concatenate(v_list),
            inp.values,
            w_mat,
        )
        naive.append(comp.ac)

    np.testing.assert_allclose(boot["as_missing"].distribution, np.array(naive), rtol=1e-12)

    boot_bca = res.bootstrap(n_resamples=80, method="bca", random_state=23)
    assert boot_bca["as_missing"].defined
    assert boot_bca["as_missing"].low < boot_bca["as_missing"].point < boot_bca["as_missing"].high


def test_gwet_validation_errors() -> None:
    study = human_study(FOUR_CODERS_MISSING["matrix"], "nominal", [1, 2, 3, 4, 5])
    with pytest.raises(DesignError, match="unknown Gwet weights"):
        study.gwet(weights="unknown_scheme")
    with pytest.raises(DesignError, match="shape"):
        study.gwet(weights=np.eye(3))
    bad_diag = np.eye(5)
    bad_diag[0, 0] = 0.5
    with pytest.raises(DesignError, match="diagonal"):
        study.gwet(weights=bad_diag)

    # Single rater per unit -> undefined Gwet AC1
    df = pd.DataFrame({"u": ["u1", "u2"], "r": ["c1", "c2"], "v": ["a", "b"]})
    rf = RatingFrame.from_long(df, unit="u", rater="r", value="v")
    s_undef = ReliabilityStudy(
        rf, measurement="nominal", values=["a", "b"], coders="rater_id", design="human_panel"
    )
    with pytest.raises(UndefinedAlphaError, match="fewer than two ratings"):
        s_undef.gwet()
