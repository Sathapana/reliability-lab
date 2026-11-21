"""End-to-end tests for the reliability-lab CLI and quality gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from reliability_lab import circular, simulate_ratings
from reliability_lab.cli import main

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def _krippendorff_four_coder_df() -> pd.DataFrame:
    # Classic Krippendorff (1980/2011) 4-coder nominal table (12 units, alpha ~= 0.7434).
    table: list[list[int | None]] = [
        [1, 1, None, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [1, 1, 2, 1],
        [2, 2, 2, 2],
        [None, 5, 5, 5],
        [None, None, 1, 1],
        [None, None, 3, None],
    ]
    rows = []
    for u_idx, unit_vals in enumerate(table, start=1):
        for c_idx, val in enumerate(unit_vals, start=1):
            if val is not None:
                rows.append({"item": f"u{u_idx}", "coder": f"c{c_idx}", "code": val})
    return pd.DataFrame(rows)


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


def test_cli_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "evaluate" in capsys.readouterr().out
    assert main(["evaluate", "--help"]) == 0
    assert "--min-alpha" in capsys.readouterr().out
    assert main(["compare", "--help"]) == 0
    assert "--margin" in capsys.readouterr().out
    assert main(["--version"]) == 0
    assert "reliability-lab" in capsys.readouterr().out


def test_cli_evaluate_krippendorff_fixture_and_thresholds(tmp_path: Path) -> None:
    csv_path = tmp_path / "krippendorff.csv"
    _krippendorff_four_coder_df().to_csv(csv_path, index=False)

    toml_path = _write_toml(
        tmp_path / "study.toml",
        """
[study]
measurement = "nominal"
values = "observed"
coders = "rater_id"
design = "human_panel"
ratings = "krippendorff.csv"

[columns]
unit = "item"
rater = "coder"
value = "code"
""",
    )

    out_pass = tmp_path / "reports" / "pass.json"
    code_pass = main(["evaluate", str(toml_path), "--min-alpha", "0.70", "--output", str(out_pass)])
    assert code_pass == 0
    report = json.loads(out_pass.read_text(encoding="utf-8"))
    assert report["schema"] == "reliability-lab/alpha-result"
    assert report["schema_version"] == 1
    assert report["estimates"]["as_missing"]["alpha"] == pytest.approx(0.7434, abs=1e-3)
    assert report["gate"]["passed"] is True
    assert report["gate"]["exit_code"] == 0

    out_fail = tmp_path / "fail.json"
    code_fail = main(["evaluate", str(toml_path), "--min-alpha", "0.80", "--output", str(out_fail)])
    assert code_fail == 2
    fail_report = json.loads(out_fail.read_text(encoding="utf-8"))
    assert fail_report["gate"]["passed"] is False
    assert fail_report["gate"]["exit_code"] == 2
    assert any("alpha=" in msg for msg in fail_report["gate"]["failures"])


def test_cli_evaluate_policy_selection_and_min_low_bootstrap(tmp_path: Path) -> None:
    rf = simulate_ratings(
        "nominal",
        ["a", "b", "c"],
        n_units=80,
        n_coders=3,
        alpha=0.82,
        failures=0.25,
        random_state=12,
    )
    csv_path = tmp_path / "with_failures.csv"
    rf.to_frame().to_csv(csv_path, index=False)

    toml_path = _write_toml(
        tmp_path / "study.toml",
        """
[study]
measurement = "nominal"
values = ["a", "b", "c"]
coders = "rater_id"
design = "human_panel"
missing_policies = ["as_missing", "as_category", "sensitivity"]

[bootstrap]
scheme = "units"
method = "percentile"
confidence = 0.95
n_resamples = 200
seed = 5
""",
    )

    # All policies: sensitivity_pessimistic / as_category fall below min_low=0.60.
    all_out = tmp_path / "all.json"
    assert (
        main(
            [
                "evaluate",
                str(toml_path),
                str(csv_path),
                "--min-low",
                "0.60",
                "--output",
                str(all_out),
            ]
        )
        == 2
    )
    all_data = json.loads(all_out.read_text(encoding="utf-8"))
    assert "intervals" in all_data
    assert all_data["intervals"]["as_missing"]["low"] >= 0.60
    assert all_data["gate"]["exit_code"] == 2

    # Gating explicitly on as_missing passes with exit code 0.
    assert (
        main(
            [
                "evaluate",
                str(toml_path),
                str(csv_path),
                "--policy",
                "as_missing",
                "--min-low",
                "0.60",
            ]
        )
        == 0
    )

    # Gating on sensitivity group fails with exit code 2.
    assert (
        main(
            [
                "evaluate",
                str(toml_path),
                str(csv_path),
                "--policy",
                "sensitivity",
                "--min-alpha",
                "0.70",
            ]
        )
        == 2
    )


def test_cli_evaluate_multifile_csv_circular_custom_matrix_and_parquet(tmp_path: Path) -> None:
    rf_circ = simulate_ratings(
        circular(24),
        list(range(24)),
        n_units=60,
        n_coders=3,
        alpha=0.85,
        errors="near",
        random_state=4,
    )
    df_circ = rf_circ.to_frame()
    mid = len(df_circ) // 2
    p1 = tmp_path / "part1.csv"
    p2 = tmp_path / "part2.csv"
    df_circ.iloc[:mid].to_csv(p1, index=False)
    df_circ.iloc[mid:].to_csv(p2, index=False)

    toml_c1 = _write_toml(
        tmp_path / "circ1.toml",
        """
[study]
measurement = "circular(24)"
values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
coders = "rater_id"
design = "human_panel"
""",
    )
    toml_c2 = _write_toml(
        tmp_path / "circ2.toml",
        """
[study]
measurement = "circular"
circumference = 24
values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
coders = "rater_id"
design = "human_panel"
""",
    )
    out1 = tmp_path / "c1.json"
    out2 = tmp_path / "c2.json"
    assert main(["evaluate", str(toml_c1), str(p1), str(p2), "--output", str(out1)]) == 0
    assert (
        main(
            ["evaluate", str(toml_c2), "--ratings", str(p1), "--ratings", str(p2), "-o", str(out2)]
        )
        == 0
    )
    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))
    assert d1["gate"]["ratings_count"] == len(df_circ)
    assert d1["estimates"]["as_missing"]["alpha"] == pytest.approx(
        d2["estimates"]["as_missing"]["alpha"], abs=1e-12
    )

    # Custom distance matrix + Parquet input
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rf_cust = simulate_ratings(
        "ordinal", ["low", "mid", "high"], n_units=50, n_coders=3, alpha=0.8, random_state=7
    )
    pq_path = tmp_path / "custom.parquet"
    pq.write_table(pa.Table.from_pandas(rf_cust.to_frame(), preserve_index=False), pq_path)

    toml_mat = _write_toml(
        tmp_path / "matrix.toml",
        f"""
[study]
measurement = "weighted"
distance_matrix = [[0, 1, 4], [1, 0, 1], [4, 1, 0]]
values = ["low", "mid", "high"]
coders = "rater_id"
design = "human_panel"
ratings = "{pq_path.name}"
""",
    )
    assert main(["evaluate", str(toml_mat), "--min-alpha", "0.60"]) == 0


def test_cli_strict_exit_3_on_trace_and_coder_dependence_and_exit_2_precedence(
    tmp_path: Path,
) -> None:
    # Deterministic contradiction triggers TraceWarning / trace_findings.
    rows = []
    for u in range(1, 21):
        v_a = "yes" if u % 2 == 0 else "no"
        v_b = ("yes" if v_a == "no" else "no") if u == 1 else v_a
        rows.append(
            {
                "unit_id": f"u{u}",
                "rater_id": "judge_r1",
                "model_id": "m1",
                "condition_id": "det",
                "run_id": "r1",
                "attempt": 1,
                "status": "parse_error" if u == 5 else "observed",
                "value": None if u == 5 else v_a,
            }
        )
        if u == 5:
            rows.append(
                {
                    "unit_id": f"u{u}",
                    "rater_id": "judge_r1",
                    "model_id": "m1",
                    "condition_id": "det",
                    "run_id": "r1",
                    "attempt": 2,
                    "status": "observed",
                    "value": v_a,
                }
            )
        rows.append(
            {
                "unit_id": f"u{u}",
                "rater_id": "judge_r2",
                "model_id": "m1",
                "condition_id": "det",
                "run_id": "r2",
                "attempt": 1,
                "status": "observed",
                "value": v_b,
            }
        )
    ratings_csv = tmp_path / "repeat.csv"
    pd.DataFrame(rows).to_csv(ratings_csv, index=False)

    trace_csv = tmp_path / "trace.csv"
    pd.DataFrame(
        [
            {
                "condition_id": "det",
                "provider": "test-llm",
                "prompt_hash": "h1",
                "rubric_version": "v1",
                "temperature": 0.0,
                "seed": 42,
                "deterministic": True,
                "parser_version": "p1",
                "candidate_order": "yes,no",
                "request_time": "2026-01-01T00:00:00Z",
                "retry_policy": "exp_backoff",
            }
        ]
    ).to_csv(trace_csv, index=False)

    toml_path = _write_toml(
        tmp_path / "repeat.toml",
        """
[study]
measurement = "nominal"
values = ["yes", "no"]
coders = "run_id"
design = "repeatability"
retries = "last"
ratings = "repeat.csv"

[trace]
path = "trace.csv"
condition = "condition_id"
""",
    )

    # Without --strict, passes with exit code 0
    assert main(["evaluate", str(toml_path), "--min-alpha", "0.70"]) == 0

    # With --strict and passing threshold, fails with exit code 3 due to trace finding
    strict_out = tmp_path / "strict.json"
    assert (
        main(
            [
                "evaluate",
                str(toml_path),
                "--min-alpha",
                "0.70",
                "--strict",
                "--output",
                str(strict_out),
            ]
        )
        == 3
    )
    strict_data = json.loads(strict_out.read_text(encoding="utf-8"))
    assert strict_data["gate"]["exit_code"] == 3
    assert strict_data["gate"]["strict_findings"]

    # When threshold ALSO fails, exit code 2 takes precedence over 3
    assert main(["evaluate", str(toml_path), "--min-alpha", "0.99", "--strict"]) == 2


def test_cli_compare_non_inferiority_equivalence_and_numeric_coercion(tmp_path: Path) -> None:
    rf_a = simulate_ratings(
        "nominal", ["a", "b", "c"], n_units=80, n_coders=3, alpha=0.82, random_state=1
    )
    rf_b_good = simulate_ratings(
        "nominal", ["a", "b", "c"], n_units=80, n_coders=3, alpha=0.81, random_state=2
    )
    rf_b_bad = simulate_ratings(
        "nominal", ["a", "b", "c"], n_units=80, n_coders=3, alpha=0.45, random_state=3
    )

    df_a = rf_a.to_frame().assign(condition_id="baseline", run_id=1)
    df_b_good = rf_b_good.to_frame().assign(condition_id="candidate", run_id=2)
    df_b_bad = rf_b_bad.to_frame().assign(condition_id="degraded", run_id=3)
    combined = pd.concat([df_a, df_b_good, df_b_bad], ignore_index=True)

    csv_path = tmp_path / "compare.csv"
    combined.to_csv(csv_path, index=False)

    toml_path = _write_toml(
        tmp_path / "compare.toml",
        """
[study]
measurement = "nominal"
values = ["a", "b", "c"]
coders = ["rater_id", "condition_id", "run_id"]
design = "human_panel"
ratings = "compare.csv"

[bootstrap]
method = "bca"
confidence = 0.96
n_resamples = 200
seed = 7
""",
    )

    cmp_out = tmp_path / "compare.json"
    code_ok = main(
        [
            "compare",
            str(toml_path),
            "--a",
            "condition_id=baseline",
            "--a",
            "run_id=1",
            "--b",
            "condition_id=candidate",
            "--b",
            "run_id=2",
            "--margin",
            "0.20",
            "--test",
            "non_inferiority",
            "--output",
            str(cmp_out),
        ]
    )
    assert code_ok == 0
    cmp_data = json.loads(cmp_out.read_text(encoding="utf-8"))
    assert cmp_data["schema"] == "reliability-lab/comparison-result"
    assert cmp_data["verdicts"]["as_missing"] == "non_inferior"
    assert cmp_data["gate"]["passed"] is True

    # Degraded arm fails non-inferiority and equivalence (exit code 2)
    code_ni_fail = main(
        [
            "compare",
            str(toml_path),
            "--a",
            "condition_id=baseline",
            "--b",
            "condition_id=degraded",
            "--margin",
            "0.05",
            "--test",
            "non_inferiority",
        ]
    )
    assert code_ni_fail == 2

    code_eq_fail = main(
        [
            "compare",
            str(toml_path),
            "--a",
            "condition_id=baseline",
            "--b",
            "condition_id=degraded",
            "--margin",
            "0.05",
            "--test",
            "equivalence",
        ]
    )
    assert code_eq_fail == 2


def test_cli_usage_schema_and_toml_errors_return_exit_code_1(tmp_path: Path) -> None:
    csv_path = tmp_path / "ok.csv"
    _krippendorff_four_coder_df().to_csv(csv_path, index=False)

    # Usage errors map to exit code 1 (not argparse default 2)
    assert main(["unknown_cmd"]) == 1
    assert main(["evaluate"]) == 1

    # Missing study TOML
    assert main(["evaluate", str(tmp_path / "missing.toml")]) == 1

    # Bad TOML syntax
    bad_toml = _write_toml(tmp_path / "bad.toml", "[study\nmeasurement = 'nominal'")
    assert main(["evaluate", str(bad_toml), str(csv_path)]) == 1

    # Unknown TOML section and unknown TOML key
    unk_sec = _write_toml(
        tmp_path / "unk_sec.toml",
        """
[study]
measurement = "nominal"
values = [1, 2]
coders = "rater_id"
design = "human_panel"
[unknown]
x = 1
""",
    )
    assert main(["evaluate", str(unk_sec), str(csv_path)]) == 1

    unk_key = _write_toml(
        tmp_path / "unk_key.toml",
        """
[study]
measurement = "nominal"
values = [1, 2]
coders = "rater_id"
design = "human_panel"
extra_key = true
""",
    )
    assert main(["evaluate", str(unk_key), str(csv_path)]) == 1

    # Valid study TOML, bad extensions / missing files / bad columns / bad policy / bad selector
    ok_toml = _write_toml(
        tmp_path / "ok.toml",
        """
[study]
measurement = "nominal"
values = [1, 2, 3, 4, 5]
coders = "rater_id"
design = "human_panel"
ratings = "ok.csv"

[columns]
unit = "item"
rater = "coder"
value = "code"
""",
    )
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("x", encoding="utf-8")
    assert main(["evaluate", str(ok_toml), str(txt_file)]) == 1
    assert main(["evaluate", str(ok_toml), str(tmp_path / "nope.csv")]) == 1
    assert main(["evaluate", str(ok_toml), "--policy", "nonexistent"]) == 1
    assert main(["compare", str(ok_toml), "--a", "bad_selector", "--b", "condition_id=y"]) == 1


def test_cli_subprocess_entrypoint(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    _krippendorff_four_coder_df().to_csv(csv_path, index=False)
    toml_path = _write_toml(
        tmp_path / "study.toml",
        """
[study]
measurement = "nominal"
values = "observed"
coders = "rater_id"
design = "human_panel"
ratings = "data.csv"

[columns]
unit = "item"
rater = "coder"
value = "code"
""",
    )
    proc_pass = subprocess.run(
        [
            sys.executable,
            "-m",
            "reliability_lab",
            "evaluate",
            str(toml_path),
            "--min-alpha",
            "0.70",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_pass.returncode == 0
    assert "Gate status: PASS" in proc_pass.stdout

    proc_fail = subprocess.run(
        [
            sys.executable,
            "-m",
            "reliability_lab",
            "evaluate",
            str(toml_path),
            "--min-alpha",
            "0.90",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc_fail.returncode == 2
    assert "Gate status: FAIL" in proc_fail.stdout

    console_bin = Path(sys.executable).parent / "reliability-lab"
    if console_bin.exists():
        proc_bin = subprocess.run(
            [str(console_bin), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc_bin.returncode == 0


def test_cli_toml_gate_table_and_coder_dependence_warning(tmp_path: Path) -> None:
    rf = simulate_ratings(
        "nominal",
        ["a", "b", "c"],
        n_units=60,
        n_coders=3,
        alpha=0.80,
        random_state=9,
    )
    dep_df = rf.to_frame().copy()
    dep_df["model_id"] = dep_df["rater_id"]
    dep_df["model_family"] = "family_shared"
    dep_df["condition_id"] = "c0"
    dep_csv = tmp_path / "dep_family.csv"
    dep_df.to_csv(dep_csv, index=False)

    # Models sharing a model_family emit CoderDependenceWarning -> strict exit code 3
    toml_dep = _write_toml(
        tmp_path / "dep.toml",
        """
[study]
measurement = "nominal"
values = ["a", "b", "c"]
coders = "model_id"
design = "inter_model"
ratings = "dep_family.csv"

[bootstrap]
scheme = "units"
n_resamples = 100
seed = 2

[gate]
min_alpha = 0.60
strict = true
policy = "as_missing"
""",
    )
    assert main(["evaluate", str(toml_dep)]) == 3

    # Using [gate] comparison selectors directly in TOML
    rf_a = simulate_ratings(
        "nominal", ["a", "b"], n_units=50, n_coders=3, alpha=0.82, random_state=1
    )
    rf_b = simulate_ratings(
        "nominal", ["a", "b"], n_units=50, n_coders=3, alpha=0.82, random_state=2
    )
    df_cmp = pd.concat(
        [
            rf_a.to_frame().assign(model_id="m1", condition_id="v1"),
            rf_b.to_frame().assign(model_id="m1", condition_id="v2"),
        ],
        ignore_index=True,
    )
    cmp_csv = tmp_path / "gate_cmp.csv"
    df_cmp.to_csv(cmp_csv, index=False)
    toml_cmp = _write_toml(
        tmp_path / "gate_cmp.toml",
        """
[study]
measurement = "nominal"
values = ["a", "b"]
coders = ["rater_id", "condition_id"]
design = "prompt_robustness"
ratings = "gate_cmp.csv"

[gate]
a = { condition_id = "v1" }
b = { condition_id = "v2" }
margin = 0.20
test = "non_inferiority"
method = "percentile"
n_resamples = 100
seed = 4
""",
    )
    assert main(["compare", str(toml_cmp)]) == 0
