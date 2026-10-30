import json
from pathlib import Path

import pandas as pd
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study, long_from_matrix

import reliability_lab
from reliability_lab import RatingFrame, ReliabilityStudy
from reliability_lab import report as report_module

F4 = FOUR_CODERS_MISSING


def failing_study() -> ReliabilityStudy:
    df = pd.DataFrame(
        {
            "unit": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
            "coder": ["a", "b"] * 4,
            "value": [1, 1, 2, 2, 1, None, 2, 1],
            "status": ["observed"] * 5 + ["parse_error"] + ["observed"] * 2,
        }
    )
    ratings = RatingFrame.from_long(df, unit="unit", rater="coder", value="value", status="status")
    return ReliabilityStudy(
        ratings, measurement="nominal", values=[1, 2], coders="rater_id", design="human_panel"
    )


def test_to_json_is_strict_versioned_and_has_provenance(tmp_path: Path) -> None:
    result = human_study(F4["matrix"], "ordinal", F4["values"]).alpha()
    path = tmp_path / "alpha.json"
    text = result.to_json(path)
    assert path.read_text(encoding="utf-8") == text
    payload = json.loads(text)
    assert payload["schema"] == "reliability-lab/alpha-result"
    assert payload["schema_version"] == 1
    assert payload["reliability_lab_version"] == reliability_lab.__version__
    provenance = payload["provenance"]
    assert set(provenance["versions"]) == {"reliability_lab", "python", "numpy", "scipy", "pandas"}
    assert provenance["n_rows"] == 41
    assert provenance["declarations"]["measurement"] == "ordinal"
    assert provenance["split"] is None
    assert len(provenance["ratings_fingerprint"]) == 64


def test_fingerprint_ignores_row_order_and_dtype_but_not_values() -> None:
    df = long_from_matrix(F4["matrix"])

    def fingerprint(frame: pd.DataFrame) -> str:
        return RatingFrame.from_long(frame, unit="unit", rater="coder", value="value").fingerprint()

    base = fingerprint(df)
    assert fingerprint(df.sample(frac=1.0, random_state=3)) == base
    assert fingerprint(df.assign(value=df["value"].astype(float))) == base
    changed = df.copy()
    changed.loc[changed.index[0], "value"] = 5
    assert fingerprint(changed) != base


def test_provenance_lists_gaps() -> None:
    human = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    assert human.provenance is not None
    gaps = " ".join(human.provenance.gaps)
    assert "No status column" in gaps
    assert "No group_id" in gaps

    df = long_from_matrix(F4["matrix"]).assign(prompt="p1")
    ratings = RatingFrame.from_long(
        df, unit="unit", rater="coder", value="value", model_id="coder", condition_id="prompt"
    )
    with pytest.warns(reliability_lab.CoderDependenceWarning, match="model_family is missing"):
        llm = ReliabilityStudy(
            ratings,
            measurement="nominal",
            values=F4["values"],
            coders="model_id",
            design="inter_model",
        )
    result = llm.alpha()
    assert result.provenance is not None
    gaps = " ".join(result.provenance.gaps)
    assert "model_family is missing" in gaps
    assert "No trace sidecar" in gaps


def test_alpha_tables() -> None:
    result = failing_study().alpha()
    tables = result.tables()
    assert set(tables) == {
        "estimates",
        "per_unit",
        "coincidence",
        "coverage",
        "diagnostic_coder_agreement",
    }
    assert len(tables["estimates"]) == 2
    assert len(tables["per_unit"]) == 2 * 4
    assert list(tables["per_unit"].columns[:2]) == ["estimate", "unit_id"]
    # as_missing has 2 values, as_category 3 (with "<error>").
    assert len(tables["coincidence"]) == 2 * 2 + 3 * 3
    assert {"value_a", "value_b", "coincidence", "expected"} <= set(tables["coincidence"].columns)
    assert "coder" in tables["coverage"].columns


def test_to_parquet_round_trips(tmp_path: Path) -> None:
    result = failing_study().alpha()
    out = result.to_parquet(tmp_path / "alpha")
    assert {p.name for p in out.iterdir()} == {
        "estimates.parquet",
        "per_unit.parquet",
        "coincidence.parquet",
        "coverage.parquet",
        "diagnostic_coder_agreement.parquet",
        "result.json",
    }
    estimates = pd.read_parquet(out / "estimates.parquet")
    pd.testing.assert_frame_equal(estimates, result.tables()["estimates"], check_dtype=False)
    coincidence = pd.read_parquet(out / "coincidence.parquet")
    assert "<error>" in set(coincidence["value_a"])
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["schema"].endswith(
        "alpha-result"
    )


def test_to_parquet_names_the_extra_when_pyarrow_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_module, "_has_pyarrow", lambda: False)
    result = failing_study().alpha()
    with pytest.raises(ImportError, match=r"reliability-lab\[parquet\]"):
        result.to_parquet(tmp_path)


def test_bootstrap_output_carries_provenance_and_seed(tmp_path: Path) -> None:
    result = human_study(F4["matrix"], "nominal", F4["values"]).alpha()
    boot = result.bootstrap(scheme="units", n_resamples=40, random_state=8)
    payload = json.loads(boot.to_json(tmp_path / "boot.json"))
    assert payload["schema"] == "reliability-lab/bootstrap-result"
    assert payload["seed"] == 8
    assert payload["provenance"]["ratings_fingerprint"] == result.provenance.ratings_fingerprint  # type: ignore[union-attr]
    full = json.loads(boot.to_json(include_distribution=True))
    assert len(full["intervals"]["as_missing"]["distribution"]) == 40
    tables = boot.tables()
    assert len(tables["replicates"]) == 40 * len(result.estimates)
    assert list(tables["replicates"].columns) == ["estimate", "replicate", "alpha"]
    boot.to_parquet(tmp_path / "boot")
    assert (tmp_path / "boot" / "replicates.parquet").exists()


def test_comparison_output(tmp_path: Path) -> None:
    matrix = {
        "r1": [1, 2, 3, 1, 2, 3, 1, 2],
        "r2": [1, 2, 3, 1, 2, 2, 1, 2],
        "r3": [1, 2, 3, 1, 3, 3, 1, 2],
        "r4": [1, 2, 2, 1, 2, 3, 1, 3],
    }
    study = human_study(matrix, "nominal", [1, 2, 3])
    diff = study.compare(
        {"rater_id": ["r1", "r2"]}, {"rater_id": ["r3", "r4"]}, n_resamples=30, random_state=1
    )
    payload = json.loads(diff.to_json(tmp_path / "diff.json"))
    assert payload["schema"] == "reliability-lab/comparison-result"
    assert set(payload["provenance"]) == {"a", "b", "declarations"}
    assert payload["provenance"]["declarations"] == {"margin": None, "test": None}
    replicates = diff.tables()["replicates"]
    assert list(replicates.columns) == ["estimate", "replicate", "alpha_a", "alpha_b", "difference"]
    pd.testing.assert_series_equal(
        replicates["difference"],
        replicates["alpha_b"] - replicates["alpha_a"],
        check_names=False,
    )
