"""Tests for v0.4.0 family alpha, UnitSidecar diagnostics, and call telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reliability_lab import (
    Design,
    RatingFrame,
    ReliabilityStudy,
    SchemaError,
    TraceSidecar,
    UnitSidecar,
)
from reliability_lab.alpha import NOMINAL, compute_alpha
from reliability_lab.cli import main

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def _simulate_family_correlated_ratings(
    n_units: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Two model families (m1,m2 in fam_a; m3,m4 in fam_b) with strong within-family error shock."""
    rng = np.random.default_rng(seed)
    categories = ["a", "b", "c", "d"]
    truth = rng.integers(0, 4, size=n_units)
    rows: list[dict[str, object]] = []

    family_models = (("fam_a", ("m1", "m2")), ("fam_b", ("m3", "m4")))
    for fam, models in family_models:
        # Shared family mistake rate and shared wrong category whenever the family errs together
        shared_err = rng.random(n_units) < 0.35
        fam_wrong = (truth + rng.integers(1, 4, size=n_units)) % 4
        for model_id in models:
            ind_err = rng.random(n_units) < 0.05
            ind_wrong = (truth + rng.integers(1, 4, size=n_units)) % 4
            for u in range(n_units):
                if shared_err[u]:
                    code = int(fam_wrong[u])
                elif ind_err[u]:
                    code = int(ind_wrong[u])
                else:
                    code = int(truth[u])
                rows.append(
                    {
                        "unit_id": f"u{u}",
                        "rater_id": model_id,
                        "model_id": model_id,
                        "model_family": fam,
                        "condition_id": "std",
                        "value": categories[code],
                    }
                )
    return pd.DataFrame(rows)


def test_family_alpha_detects_same_family_correlation_and_matches_naive_bootstrap() -> None:
    df = _simulate_family_correlated_ratings(n_units=120, seed=42)
    rf = RatingFrame.from_long(
        df,
        unit="unit_id",
        rater="rater_id",
        value="value",
        model_id="model_id",
        model_family="model_family",
        condition_id="condition_id",
    )
    study = ReliabilityStudy(
        rf,
        measurement="nominal",
        values=["a", "b", "c", "d"],
        coders="model_id",
        design=Design.INTER_MODEL,
    )
    res = study.alpha()
    assert "family_alpha" in res.diagnostics
    fam = res.diagnostics["family_alpha"]
    assert fam.values["n_families"] == 2
    assert fam.values["within_pairs"] == 2
    assert fam.values["between_pairs"] == 4
    assert fam.values["within_family_alpha"] > fam.values["between_family_alpha"] + 0.15
    assert fam.values["family_gap"] == pytest.approx(
        fam.values["within_family_alpha"] - fam.values["between_family_alpha"]
    )

    boot = res.bootstrap(scheme="units", method="percentile", n_resamples=60, random_state=7)
    assert set(boot.family_intervals) == {"within_family", "between_family", "family_gap"}
    assert boot.family_intervals["family_gap"].low > 0.0
    assert "family_intervals" in boot.tables()
    assert "Family intervals" in boot.summary()
    payload = json.loads(boot.to_json(include_distribution=True))
    assert set(payload["family_intervals"]) == {"within_family", "between_family", "family_gap"}

    # Check paired family_gap distribution against an explicit naive unit loop from the same seed
    unit_list = [f"u{u}" for u in range(120)]
    rng = np.random.default_rng(7)
    weights = rng.multinomial(120, np.full(120, 1.0 / 120), size=60)
    by_unit = {u: grp for u, grp in df.groupby("unit_id", sort=False)}
    cat_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    naive_gaps = []

    for w_row in weights:
        rep_units: list[int] = []
        rep_vals: list[int] = []
        rep_models: list[str] = []
        new_u = 0
        for u_pos, mult in enumerate(w_row):
            if mult == 0:
                continue
            grp = by_unit[unit_list[u_pos]]
            for _ in range(int(mult)):
                for row in grp.itertuples(index=False):
                    rep_units.append(new_u)
                    rep_vals.append(cat_idx[str(row.value)])
                    rep_models.append(str(row.model_id))
                new_u += 1
        rep_df = pd.DataFrame({"u": rep_units, "v": rep_vals, "m": rep_models})
        within_a = []
        between_a = []
        for ma, mb, rel in (
            ("m1", "m2", "within"),
            ("m3", "m4", "within"),
            ("m1", "m3", "between"),
            ("m1", "m4", "between"),
            ("m2", "m3", "between"),
            ("m2", "m4", "between"),
        ):
            sub = rep_df[rep_df["m"].isin((ma, mb))]
            a_val = compute_alpha(
                sub["u"].to_numpy(),
                sub["v"].to_numpy(),
                new_u,
                ["a", "b", "c", "d"],
                NOMINAL,
            ).components.alpha
            if rel == "within":
                within_a.append(a_val)
            else:
                between_a.append(a_val)
        naive_gaps.append(float(np.mean(within_a) - np.mean(between_a)))

    np.testing.assert_allclose(
        boot.family_intervals["family_gap"].distribution,
        np.array(naive_gaps),
        atol=1e-12,
    )

    # Also verify BCa runs cleanly on family_intervals
    boot_bca = res.bootstrap(scheme="units", method="bca", n_resamples=60, random_state=7)
    assert boot_bca.family_intervals["family_gap"].defined


def test_unit_sidecar_validation_arrow_and_provenance(tmp_path: Path) -> None:
    df_u = pd.DataFrame(
        {"uid": ["u2", "u1", "u3"], "len": [10, 25, 40], "ofam": ["f1", "f2", "f1"]}
    )
    u1 = UnitSidecar.from_frame(df_u, unit="uid", length="len", output_family="ofam")
    u2 = UnitSidecar.from_frame(
        df_u.iloc[::-1].reset_index(drop=True), unit="uid", length="len", output_family="ofam"
    )
    assert u1.fingerprint() == u2.fingerprint()
    assert u1.value("u1", "length") == 25.0
    assert u1.value("u1", "output_family") == "f2"
    assert u1.value("missing", "length") is None

    # Arrow PyCapsule table yields identical fingerprint
    pa = pytest.importorskip("pyarrow")
    pa_tbl = pa.Table.from_pandas(df_u, preserve_index=False)
    u_arrow = UnitSidecar.from_frame(pa_tbl, unit="uid", length="len", output_family="ofam")
    assert u_arrow.fingerprint() == u1.fingerprint()

    # Error checks
    with pytest.raises(SchemaError, match="at least one covariate"):
        UnitSidecar.from_frame(df_u, unit="uid")
    with pytest.raises(SchemaError, match="cannot combine pointwise 'length'"):
        UnitSidecar.from_frame(
            pd.DataFrame({"uid": ["u1"], "l": [1], "la": [1], "lb": [2]}),
            unit="uid",
            length="l",
            length_a="la",
            length_b="lb",
        )
    with pytest.raises(SchemaError, match="declared together"):
        UnitSidecar.from_frame(pd.DataFrame({"uid": ["u1"], "la": [1]}), unit="uid", length_a="la")
    with pytest.raises(SchemaError, match="declared together"):
        UnitSidecar.from_frame(
            pd.DataFrame({"uid": ["u1"], "fa": ["x"]}), unit="uid", output_family_a="fa"
        )
    with pytest.raises(SchemaError, match="non-negative"):
        UnitSidecar.from_frame(pd.DataFrame({"uid": ["u1"], "l": [-5]}), unit="uid", length="l")
    with pytest.raises(SchemaError, match="several rows"):
        UnitSidecar.from_frame(
            pd.DataFrame({"uid": ["u1", "u1"], "l": [1, 2]}), unit="uid", length="l"
        )

    # Provenance gaps when rated unit is missing from UnitSidecar
    ratings_df = pd.DataFrame(
        {
            "unit_id": ["u1", "u1", "u2", "u2", "u4", "u4"],
            "rater_id": ["r1", "r2"] * 3,
            "value": [2, 3, 1, 2, 4, 4],
        }
    )
    rf = RatingFrame.from_long(ratings_df, unit="unit_id", rater="rater_id", value="value")
    study = ReliabilityStudy(
        rf,
        measurement="ordinal",
        values=[1, 2, 3, 4],
        coders="rater_id",
        design="human_panel",
        units=u1,
    )
    res = study.alpha()
    assert res.provenance is not None
    assert res.provenance.units is not None
    assert any("1 units have no covariate row" in g for g in res.provenance.gaps)


def test_length_bias_pointwise_and_pairwise_order_robustness() -> None:
    # 1. Pointwise ordinal length correlation
    units = [f"u{i}" for i in range(1, 21)]
    u_df = pd.DataFrame({"unit_id": units, "length": [i * 10 for i in range(1, 21)]})
    u_side = UnitSidecar.from_frame(u_df, unit="unit_id", length="length")
    rows = []
    for idx, u in enumerate(units):
        score = min(5, 1 + idx // 4)
        for r in ("c1", "c2"):
            rows.append({"unit_id": u, "rater_id": r, "value": score})
    rf_pt = RatingFrame.from_long(
        pd.DataFrame(rows), unit="unit_id", rater="rater_id", value="value"
    )
    res_pt = ReliabilityStudy(
        rf_pt,
        measurement="ordinal",
        values=[1, 2, 3, 4, 5],
        coders="rater_id",
        design="human_panel",
        units=u_side,
    ).alpha()
    lb = res_pt.diagnostics["length_bias"]
    assert lb.values["mode"] == "pointwise"
    assert lb.values["spearman_r"] > 0.9
    assert lb.values["p_value"] < 1e-6
    assert any("causal claim" in n for n in lb.notes)

    # Nominal skips pointwise length_bias cleanly
    res_nom = ReliabilityStudy(
        rf_pt,
        measurement="nominal",
        values=[1, 2, 3, 4, 5],
        coders="rater_id",
        design="human_panel",
        units=u_side,
    ).alpha()
    assert "length_bias" not in res_nom.diagnostics

    # 2. Pairwise order_robustness: longer candidate wins consistently across both orders
    pw_units = [f"p{i}" for i in range(1, 21)]
    la = [300 if i % 2 == 0 else 50 for i in range(1, 21)]
    lb_list = [50 if i % 2 == 0 else 300 for i in range(1, 21)]
    u_pw = UnitSidecar.from_frame(
        pd.DataFrame({"unit_id": pw_units, "length_a": la, "length_b": lb_list}),
        unit="unit_id",
        length_a="length_a",
        length_b="length_b",
    )
    pw_rows = []
    for i, u in enumerate(pw_units, start=1):
        winner = "A" if i % 2 == 0 else "B"  # longer side wins
        for cond in ("ab", "ba"):
            # Flip 2 units when order flips to check consistent subset separates reversals
            val = ("B" if winner == "A" else "A") if (cond == "ba" and i in (1, 2)) else winner
            pw_rows.append(
                {
                    "unit_id": u,
                    "rater_id": "judge",
                    "model_id": "m1",
                    "condition_id": cond,
                    "value": val,
                }
            )
    rf_pw = RatingFrame.from_long(
        pd.DataFrame(pw_rows),
        unit="unit_id",
        rater="rater_id",
        value="value",
        model_id="model_id",
        condition_id="condition_id",
    )
    trace = TraceSidecar.from_frame(
        pd.DataFrame(
            [
                {"condition_id": "ab", "candidate_order": "A,B"},
                {"condition_id": "ba", "candidate_order": "B,A"},
            ]
        ),
        condition="condition_id",
        candidate_order="candidate_order",
    )
    res_pw = ReliabilityStudy(
        rf_pw,
        measurement="nominal",
        values=["A", "tie", "B"],
        coders="condition_id",
        design=Design.ORDER_ROBUSTNESS,
        trace=trace,
        units=u_pw,
    ).alpha()
    lb_pw = res_pw.diagnostics["length_bias"]
    assert lb_pw.values["mode"] == "pairwise"
    assert lb_pw.values["spearman_r"] > 0.7
    assert lb_pw.values["consistent_units_with_length_diff"] == 18
    assert lb_pw.values["consistent_longer_wins"] == 18
    assert lb_pw.values["consistent_longer_wins_share"] == 1.0


def test_self_enhancement_pointwise_and_pairwise() -> None:
    # 1. Pointwise self-enhancement: model m1 (fam_a) rates fam_a outputs higher than fam_b
    units = [f"u{i}" for i in range(1, 21)]
    out_fam = ["fam_a" if i <= 10 else "fam_b" for i in range(1, 21)]
    u_pt = UnitSidecar.from_frame(
        pd.DataFrame({"unit_id": units, "output_family": out_fam}),
        unit="unit_id",
        output_family="output_family",
    )
    rows = []
    for i, u in enumerate(units, start=1):
        rows.append(
            {
                "unit_id": u,
                "rater_id": "m1",
                "model_id": "m1",
                "model_family": "fam_a",
                "condition_id": "c1",
                "value": 5 if i <= 10 else 3,
            }
        )
        rows.append(
            {
                "unit_id": u,
                "rater_id": "m2",
                "model_id": "m2",
                "model_family": "fam_b",
                "condition_id": "c1",
                "value": 3 if i <= 10 else 4,
            }
        )
    rf = RatingFrame.from_long(
        pd.DataFrame(rows),
        unit="unit_id",
        rater="rater_id",
        value="value",
        model_id="model_id",
        model_family="model_family",
        condition_id="condition_id",
    )
    res = ReliabilityStudy(
        rf,
        measurement="ordinal",
        values=[1, 2, 3, 4, 5],
        coders="model_id",
        design="inter_model",
        units=u_pt,
    ).alpha()
    se = res.diagnostics["self_enhancement"]
    assert se.values["mode"] == "pointwise"
    assert se.values["own_mean"] > se.values["other_mean"]
    assert se.values["mean_shift"] == pytest.approx(1.5)

    # 2. Pairwise self-enhancement: judge picks its own family when facing another family
    u_pw = UnitSidecar.from_frame(
        pd.DataFrame(
            {
                "unit_id": units,
                "output_family_a": ["fam_a" if i % 2 == 0 else "fam_b" for i in range(1, 21)],
                "output_family_b": ["fam_b" if i % 2 == 0 else "fam_a" for i in range(1, 21)],
            }
        ),
        unit="unit_id",
        output_family_a="output_family_a",
        output_family_b="output_family_b",
    )
    pw_rows = []
    for i, u in enumerate(units, start=1):
        # Judge fam_a picks whichever candidate is fam_a 18/20 times
        own_label = "A" if i % 2 == 0 else "B"
        other_label = "B" if own_label == "A" else "A"
        pick = own_label if i <= 18 else other_label
        for cond in ("ab", "ba"):
            pw_rows.append(
                {
                    "unit_id": u,
                    "rater_id": "judge",
                    "model_id": "m1",
                    "model_family": "fam_a",
                    "condition_id": cond,
                    "value": pick,
                }
            )
    rf_pw = RatingFrame.from_long(
        pd.DataFrame(pw_rows),
        unit="unit_id",
        rater="rater_id",
        value="value",
        model_id="model_id",
        model_family="model_family",
        condition_id="condition_id",
    )
    trace = TraceSidecar.from_frame(
        pd.DataFrame(
            [
                {"condition_id": "ab", "candidate_order": "A,B"},
                {"condition_id": "ba", "candidate_order": "B,A"},
            ]
        ),
        condition="condition_id",
        candidate_order="candidate_order",
    )
    res_pw = ReliabilityStudy(
        rf_pw,
        measurement="nominal",
        values=["A", "tie", "B"],
        coders="condition_id",
        design=Design.ORDER_ROBUSTNESS,
        trace=trace,
        units=u_pw,
    ).alpha()
    se_pw = res_pw.diagnostics["self_enhancement"]
    assert se_pw.values["mode"] == "pairwise"
    assert se_pw.values["own_family_win_share"] == pytest.approx(0.9)
    assert se_pw.values["wilson_low"] > 0.7
    assert se_pw.values["consistent_own_family_win_share"] == pytest.approx(0.9)


def test_call_telemetry_retries_comparison_and_validation() -> None:
    # Retries included in attempt totals vs counted ratings
    rows = []
    for cond, base_tok, base_cost, err_on_u1 in (("v1", 100, 0.01, True), ("v2", 50, 0.004, False)):
        for u in range(1, 11):
            val = "a" if u % 2 == 0 else "b"
            for r in ("c1", "c2"):
                if err_on_u1 and u == 1 and r == "c1":
                    rows.append(
                        {
                            "unit_id": f"u{u}",
                            "rater_id": r,
                            "condition_id": cond,
                            "attempt": 1,
                            "status": "parse_error",
                            "value": None,
                            "prompt_tokens": base_tok,
                            "completion_tokens": 0,
                            "latency_ms": 200.0,
                            "cost": base_cost,
                        }
                    )
                    rows.append(
                        {
                            "unit_id": f"u{u}",
                            "rater_id": r,
                            "condition_id": cond,
                            "attempt": 2,
                            "status": "observed",
                            "value": val,
                            "prompt_tokens": base_tok,
                            "completion_tokens": 10,
                            "latency_ms": 150.0,
                            "cost": base_cost,
                        }
                    )
                else:
                    rows.append(
                        {
                            "unit_id": f"u{u}",
                            "rater_id": r,
                            "condition_id": cond,
                            "attempt": 1,
                            "status": "observed",
                            "value": val,
                            "prompt_tokens": base_tok,
                            "completion_tokens": 10,
                            "latency_ms": 120.0,
                            "cost": base_cost,
                        }
                    )
    df = pd.DataFrame(rows)
    rf = RatingFrame.from_long(
        df,
        unit="unit_id",
        rater="rater_id",
        value="value",
        status="status",
        condition_id="condition_id",
        attempt="attempt",
        prompt_tokens="prompt_tokens",
        completion_tokens="completion_tokens",
        latency_ms="latency_ms",
        cost="cost",
    )

    # Single-arm v1: c1 had 11 attempts (10 counted) -> total_prompt_tokens = 1100, mean = 110.0
    rf_v1 = rf._subset((rf.to_frame()["condition_id"] == "v1").to_numpy())
    res_v1 = ReliabilityStudy(
        rf_v1,
        measurement="nominal",
        values=["a", "b"],
        coders="rater_id",
        design="human_panel",
        retries="last",
    ).alpha()
    assert res_v1.telemetry is not None
    c1_row = res_v1.telemetry.loc[res_v1.telemetry["coder"] == "c1"].iloc[0]
    assert c1_row["attempts"] == 11
    assert c1_row["counted_ratings"] == 10
    assert c1_row["total_prompt_tokens"] == 1100.0
    assert c1_row["counted_prompt_tokens"] == 1000.0
    assert c1_row["mean_prompt_tokens_per_rating"] == pytest.approx(110.0)
    assert "telemetry" in res_v1.tables()
    assert "Telemetry" in res_v1.summary()

    # Comparison side-by-side telemetry
    study_cmp = ReliabilityStudy(
        rf,
        measurement="nominal",
        values=["a", "b"],
        coders=["rater_id", "condition_id"],
        design="human_panel",
        retries="last",
    )
    diff = study_cmp.compare(
        {"condition_id": "v1"}, {"condition_id": "v2"}, n_resamples=30, random_state=1
    )
    telem = diff.telemetry_table()
    assert telem is not None
    assert set(telem["arm"]) == {"a", "b"}
    assert "telemetry" in diff.tables()
    assert "Telemetry" in diff.summary()
    payload = json.loads(diff.to_json())
    assert len(payload["telemetry"]) == 4

    # Invalid telemetry validation errors
    bad_neg = df.iloc[:2].copy()
    bad_neg.loc[0, "cost"] = -0.01
    with pytest.raises(SchemaError, match="non-negative"):
        RatingFrame.from_long(
            bad_neg,
            unit="unit_id",
            rater="rater_id",
            value="value",
            status="status",
            cost="cost",
        )
    bad_frac = df.iloc[:2].copy()
    bad_frac["prompt_tokens"] = bad_frac["prompt_tokens"].astype(float)
    bad_frac.loc[0, "prompt_tokens"] = 12.5
    with pytest.raises(SchemaError, match="whole number"):
        RatingFrame.from_long(
            bad_frac,
            unit="unit_id",
            rater="rater_id",
            value="value",
            status="status",
            prompt_tokens="prompt_tokens",
        )


def test_cli_units_sidecar_and_telemetry_integration(tmp_path: Path) -> None:
    df_ratings = _simulate_family_correlated_ratings(n_units=40, seed=5)
    df_ratings["prompt_tokens"] = 120
    df_ratings["cost"] = 0.002
    ratings_csv = tmp_path / "ratings.csv"
    df_ratings.to_csv(ratings_csv, index=False)

    df_units = pd.DataFrame(
        {
            "unit_id": [f"u{u}" for u in range(40)],
            "output_family": ["fam_a" if u % 2 == 0 else "fam_b" for u in range(40)],
        }
    )
    units_csv = tmp_path / "units.csv"
    df_units.to_csv(units_csv, index=False)

    toml_path = tmp_path / "study.toml"
    toml_path.write_text(
        """
[study]
measurement = "nominal"
values = ["a", "b", "c", "d"]
coders = "model_id"
design = "inter_model"
ratings = "ratings.csv"

[units]
path = "units.csv"
unit = "unit_id"
output_family = "output_family"

[bootstrap]
scheme = "units"
n_resamples = 40
seed = 3
""",
        encoding="utf-8",
    )
    out_json = tmp_path / "report.json"
    code = main(["evaluate", str(toml_path), "--output", str(out_json)])
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "family_alpha" in payload["diagnostics"]
    assert payload["telemetry"] is not None
    assert payload["provenance"]["units"] is not None
    assert "family_gap" in payload["family_intervals"]
