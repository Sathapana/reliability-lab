"""Command-line interface and CI quality gates for reliability-lab."""

from __future__ import annotations

import argparse
import math
import re
import sys
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import pandas as pd

from reliability_lab._version import __version__
from reliability_lab.alpha import Distance, circular
from reliability_lab.data import RatingFrame, _arrow_to_pandas, _import_pyarrow
from reliability_lab.design import Design
from reliability_lab.exceptions import (
    CoderDependenceWarning,
    ReliabilityLabError,
    TraceWarning,
)
from reliability_lab.llm import TRACE_FIELDS, UNIT_FIELDS, TraceSidecar, UnitSidecar
from reliability_lab.report import AlphaResult, BootstrapResult, ComparisonResult, _write_json
from reliability_lab.study import SELECTOR_FIELDS, MissingPolicy, ReliabilityStudy, Retries

TOP_SECTIONS = frozenset({"study", "columns", "trace", "units", "bootstrap", "gate"})
STUDY_KEYS = frozenset(
    {
        "measurement",
        "circumference",
        "distance_matrix",
        "values",
        "coders",
        "design",
        "missing_policies",
        "abstain_as",
        "inapplicable_as",
        "retries",
        "ratings",
    }
)
COLUMN_MAP: dict[str, str] = {
    "unit": "unit_id",
    "rater": "rater_id",
    "value": "value",
    "status": "status",
    "group": "group_id",
    "stratum": "stratum",
    "model_id": "model_id",
    "model_family": "model_family",
    "condition_id": "condition_id",
    "run_id": "run_id",
    "attempt": "attempt",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "latency_ms": "latency_ms",
    "cost": "cost",
}
REQUIRED_COLUMNS = ("unit", "rater", "value")
OPTIONAL_COLUMNS = tuple(k for k in COLUMN_MAP if k not in REQUIRED_COLUMNS)
TRACE_KEYS = frozenset({"path", "condition", *TRACE_FIELDS})
UNITS_KEYS = frozenset({"path", "unit", *UNIT_FIELDS})
BOOTSTRAP_KEYS = frozenset(
    {"scheme", "method", "confidence", "n_resamples", "level", "strata", "seed", "on_undefined"}
)
GATE_KEYS = frozenset(
    {
        "min_alpha",
        "min_low",
        "policy",
        "strict",
        "margin",
        "test",
        "a",
        "b",
        *BOOTSTRAP_KEYS,
    }
)
CIRCULAR_RE = re.compile(r"^\s*circular\(\s*([^)]+)\s*\)\s*$", re.IGNORECASE)


class _CliError(Exception):
    """Usage, configuration, or input error mapped to exit code 1."""


class _CliParser(argparse.ArgumentParser):
    """ArgumentParser whose parse failures map cleanly to exit code 1."""

    def error(self, message: str) -> NoReturn:
        raise _CliError(message)


def _build_parser() -> _CliParser:
    parser = _CliParser(
        prog="reliability-lab",
        description=(
            "Evaluate Krippendorff's alpha reliability studies and enforce CI quality gates."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=_CliParser)

    common = _CliParser(add_help=False)
    common.add_argument("study", help="Path to the TOML study specification.")
    common.add_argument(
        "ratings",
        nargs="*",
        help="Optional paths to CSV, TSV, or Parquet ratings files.",
    )
    common.add_argument(
        "--ratings",
        dest="ratings_opt",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional CSV, TSV, or Parquet ratings file path (repeatable).",
    )
    common.add_argument("--trace", default=None, help="Path to CSV, TSV, or Parquet trace file.")
    common.add_argument(
        "--units", default=None, help="Path to CSV, TSV, or Parquet unit covariates."
    )
    common.add_argument(
        "--policy",
        default=None,
        help="Missing-data policy estimate to gate on (default: all reported estimates).",
    )
    common.add_argument("--scheme", default=None, help="Bootstrap scheme (e.g. 'units', 'kalpha').")
    common.add_argument(
        "--method",
        choices=("percentile", "bca"),
        default=None,
        help="Bootstrap interval method.",
    )
    common.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Bootstrap confidence level.",
    )
    common.add_argument("--n-resamples", type=int, default=None, help="Number of resamples.")
    common.add_argument("--level", default=None, help="Resampling level ('unit' or 'group').")
    common.add_argument(
        "--strata",
        action="store_true",
        default=None,
        help="Enable stratified resampling within unit strata.",
    )
    common.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap.")
    common.add_argument(
        "--on-undefined",
        choices=("nan", "percentile"),
        default=None,
        help="Fallback when BCa acceleration is undefined.",
    )
    common.add_argument(
        "--min-low",
        type=float,
        default=None,
        help="Minimum allowed lower confidence bound.",
    )
    common.add_argument(
        "--strict",
        action="store_true",
        default=None,
        help="Exit with code 3 when design/trace warnings or trace findings occur.",
    )
    common.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the versioned JSON report to this path.",
    )

    eval_p = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate a reliability study and enforce alpha/lower-bound gates.",
    )
    eval_p.add_argument(
        "--min-alpha",
        type=float,
        default=None,
        help="Minimum allowed point alpha estimate.",
    )

    cmp_p = subparsers.add_parser(
        "compare",
        parents=[common],
        help="Compare two study arms and enforce non-inferiority/equivalence or lower-bound gates.",
    )
    cmp_p.add_argument(
        "--a",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Baseline arm selector (repeatable, e.g. --a condition_id=baseline).",
    )
    cmp_p.add_argument(
        "--b",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Candidate arm selector (repeatable, e.g. --b condition_id=candidate).",
    )
    cmp_p.add_argument("--margin", type=float, default=None, help="Comparison margin.")
    cmp_p.add_argument(
        "--test",
        choices=("non_inferiority", "equivalence"),
        default=None,
        help="Hypothesis test verdict to require.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reliability-lab CLI and return the structured exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        if args.command == "evaluate":
            return _run_evaluate(args)
        if args.command == "compare":
            return _run_compare(args)
        raise _CliError(f"unknown subcommand {args.command!r}")
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    except (
        _CliError,
        ReliabilityLabError,
        NotImplementedError,
        OSError,
        ImportError,
        tomllib.TOMLDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_evaluate(args: argparse.Namespace) -> int:
    study_path = Path(args.study)
    cfg = _load_toml(study_path)
    boot_opts = _merge_bootstrap_opts(cfg, args)
    gate_cfg = _table_section(cfg, "gate", GATE_KEYS)

    min_alpha = _pick_opt_float(args.min_alpha, gate_cfg.get("min_alpha"), "min_alpha")
    min_low = _pick_opt_float(args.min_low, gate_cfg.get("min_low"), "min_low")
    policy = _pick_opt_str(args.policy, gate_cfg.get("policy"), "policy")
    strict = _pick_opt_bool(args.strict, gate_cfg.get("strict"), "strict", default=False)

    if gate_cfg.get("margin") is not None or gate_cfg.get("test") is not None:
        raise _CliError("[gate] keys 'margin' and 'test' are only valid for 'compare'")
    if gate_cfg.get("a") is not None or gate_cfg.get("b") is not None:
        raise _CliError("[gate] keys 'a' and 'b' are only valid for 'compare'")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study_obj, rf, base_dir = _build_study(cfg, study_path, args)
        alpha_res = study_obj.alpha()
        boot_res: BootstrapResult | None = None
        if min_low is not None or _has_bootstrap_config(boot_opts):
            boot_kwargs = _bootstrap_kwargs(boot_opts, default_scheme="units")
            if boot_kwargs.get("alpha_min") is None and min_low is not None:
                boot_kwargs["alpha_min"] = min_low
            boot_res = alpha_res.bootstrap(**boot_kwargs)

    available = list(alpha_res.estimates)
    target_policies = _resolve_policies(policy, available)

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in target_policies:
        est = alpha_res.estimates[name]
        est_alpha = est.alpha
        row_check: dict[str, Any] = {
            "policy": name,
            "alpha": est_alpha,
            "undefined_reason": est.undefined_reason,
        }
        if min_alpha is not None:
            ok_alpha = (
                est.undefined_reason is None and math.isfinite(est_alpha) and est_alpha >= min_alpha
            )
            row_check["alpha_ok"] = ok_alpha
            if not ok_alpha:
                failures.append(
                    f"{name}: alpha={est_alpha:.4f} < min_alpha={min_alpha:.4f}"
                    if math.isfinite(est_alpha)
                    else f"{name}: alpha is undefined ({est.undefined_reason})"
                )
        if min_low is not None:
            assert boot_res is not None
            iv = boot_res.intervals[name]
            ok_low = iv.defined and math.isfinite(iv.low) and iv.low >= min_low
            row_check["low"] = iv.low
            row_check["high"] = iv.high
            row_check["low_ok"] = ok_low
            if not ok_low:
                failures.append(
                    f"{name}: low={iv.low:.4f} < min_low={min_low:.4f}"
                    if iv.defined and math.isfinite(iv.low)
                    else f"{name}: interval is undefined ({iv.undefined_reason})"
                )
        checks.append(row_check)

    strict_findings = _collect_strict_findings(caught, alpha_res=alpha_res, boot_res=boot_res)
    exit_code = _determine_exit_code(failures, strict=strict, strict_findings=strict_findings)

    gate_payload = {
        "command": "evaluate",
        "study": str(study_path),
        "ratings_count": rf.to_frame().shape[0],
        "base_dir": str(base_dir),
        "policies": target_policies,
        "min_alpha": min_alpha,
        "min_low": min_low,
        "strict": strict,
        "strict_findings": strict_findings,
        "checks": checks,
        "failures": failures,
        "passed": len(failures) == 0,
        "exit_code": exit_code,
    }

    if args.output is not None:
        out_payload = _evaluate_json_payload(alpha_res, boot_res, gate_payload)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_payload, out_path)

    print(alpha_res.summary())
    if boot_res is not None:
        print()
        print(boot_res.summary())
    _print_gate_status(gate_payload)
    return exit_code


def _run_compare(args: argparse.Namespace) -> int:
    study_path = Path(args.study)
    cfg = _load_toml(study_path)
    boot_opts = _merge_bootstrap_opts(cfg, args)
    gate_cfg = _table_section(cfg, "gate", GATE_KEYS)

    min_low = _pick_opt_float(args.min_low, gate_cfg.get("min_low"), "min_low")
    margin = _pick_opt_float(args.margin, gate_cfg.get("margin"), "margin")
    test = _pick_opt_str(args.test, gate_cfg.get("test"), "test")
    policy = _pick_opt_str(args.policy, gate_cfg.get("policy"), "policy")
    strict = _pick_opt_bool(args.strict, gate_cfg.get("strict"), "strict", default=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study_obj, rf, base_dir = _build_study(cfg, study_path, args)
        rf_df = rf.to_frame()
        sel_a = _resolve_selector(args.a, gate_cfg.get("a"), "a", rf_df)
        sel_b = _resolve_selector(args.b, gate_cfg.get("b"), "b", rf_df)
        cmp_kwargs = _compare_kwargs(boot_opts)
        diff_res = study_obj.compare(
            sel_a,
            sel_b,
            margin=margin,
            test=cast(Any, test),
            **cmp_kwargs,
        )

    available = list(diff_res.differences)
    target_policies = _resolve_policies(policy, available)

    expected_verdict: str | None = None
    if test is not None:
        expected_verdict = "non_inferior" if test == "non_inferiority" else "equivalent"

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in target_policies:
        iv = diff_res.differences[name]
        verdict_val = diff_res.verdicts.get(name)
        p_val = diff_res.p_values.get(name)
        row_check: dict[str, Any] = {
            "policy": name,
            "diff": iv.point,
            "low": iv.low,
            "high": iv.high,
            "verdict": verdict_val,
            "p_value": p_val,
        }
        if expected_verdict is not None:
            ok_verdict = verdict_val == expected_verdict
            row_check["verdict_ok"] = ok_verdict
            if not ok_verdict:
                failures.append(
                    f"{name}: verdict={verdict_val!r} != expected {expected_verdict!r} "
                    f"(diff={iv.point:.4f}, CI=[{iv.low:.4f}, {iv.high:.4f}])"
                )
        if min_low is not None:
            ok_low = iv.defined and math.isfinite(iv.low) and iv.low >= min_low
            row_check["low_ok"] = ok_low
            if not ok_low:
                failures.append(
                    f"{name}: diff low={iv.low:.4f} < min_low={min_low:.4f}"
                    if iv.defined and math.isfinite(iv.low)
                    else f"{name}: diff interval is undefined ({iv.undefined_reason})"
                )
        checks.append(row_check)

    strict_findings = _collect_strict_findings(caught, comp_res=diff_res)
    exit_code = _determine_exit_code(failures, strict=strict, strict_findings=strict_findings)

    gate_payload = {
        "command": "compare",
        "study": str(study_path),
        "ratings_count": rf_df.shape[0],
        "base_dir": str(base_dir),
        "a": dict(sel_a),
        "b": dict(sel_b),
        "policies": target_policies,
        "margin": margin,
        "test": test,
        "min_low": min_low,
        "strict": strict,
        "strict_findings": strict_findings,
        "checks": checks,
        "failures": failures,
        "passed": len(failures) == 0,
        "exit_code": exit_code,
    }

    if args.output is not None:
        out_payload = dict(diff_res.to_dict())
        out_payload["gate"] = gate_payload
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_payload, out_path)

    print(diff_res.summary())
    _print_gate_status(gate_payload)
    return exit_code


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise _CliError(f"study specification not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - TOP_SECTIONS
    if unknown:
        raise _CliError(f"unknown TOML section(s): {sorted(unknown)}")
    return data


def _table_section(
    cfg: Mapping[str, Any],
    name: str,
    allowed_keys: frozenset[str],
) -> dict[str, Any]:
    raw = cfg.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise _CliError(f"TOML section [{name}] must be a table")
    unknown = set(raw) - allowed_keys
    if unknown:
        raise _CliError(f"unknown key(s) in [{name}]: {sorted(unknown)}")
    return dict(raw)


def _build_study(
    cfg: Mapping[str, Any],
    study_path: Path,
    args: argparse.Namespace,
) -> tuple[ReliabilityStudy, RatingFrame, Path]:
    base_dir = study_path.parent
    study_cfg = _table_section(cfg, "study", STUDY_KEYS)
    if not study_cfg:
        raise _CliError("TOML file must contain a [study] section")
    cols_cfg = _table_section(cfg, "columns", frozenset(COLUMN_MAP))
    trace_cfg = _table_section(cfg, "trace", TRACE_KEYS)
    units_cfg = _table_section(cfg, "units", UNITS_KEYS)

    # 1. Load ratings tables
    rating_paths: list[Path] = []
    if args.ratings_opt or args.ratings:
        for p in [*args.ratings_opt, *args.ratings]:
            rating_paths.append(Path(p))
    elif "ratings" in study_cfg:
        raw_r = study_cfg["ratings"]
        if isinstance(raw_r, str):
            rating_paths.append(_resolve_toml_path(base_dir, raw_r))
        elif isinstance(raw_r, list) and all(isinstance(item, str) for item in raw_r):
            rating_paths.extend(_resolve_toml_path(base_dir, item) for item in raw_r)
        else:
            raise _CliError("[study].ratings must be a path string or list of path strings")
    if not rating_paths:
        raise _CliError("no ratings files provided on CLI or in [study].ratings")

    frames = [_read_table(p) for p in rating_paths]
    df_ratings = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    # 2. Construct RatingFrame
    col_kwargs: dict[str, Any] = {}
    for key in REQUIRED_COLUMNS:
        val = cols_cfg.get(key, COLUMN_MAP[key])
        if not isinstance(val, str):
            raise _CliError(f"[columns].{key} must be a string column name")
        col_kwargs[key] = val
    for key in OPTIONAL_COLUMNS:
        if key in cols_cfg:
            val = cols_cfg[key]
            if not isinstance(val, str):
                raise _CliError(f"[columns].{key} must be a string column name")
            col_kwargs[key] = val
        else:
            canonical = COLUMN_MAP[key]
            if canonical in df_ratings.columns and df_ratings[canonical].notna().any():
                col_kwargs[key] = canonical
    rf = RatingFrame.from_long(df_ratings, **col_kwargs)

    # 3. Resolve values domain
    if "values" not in study_cfg:
        raise _CliError("[study].values is required (list of domain values or 'observed')")
    raw_values = study_cfg["values"]
    if raw_values == "observed":
        values_domain: Sequence[Any] = rf.observed_values()
        if not values_domain:
            raise _CliError("[study].values = 'observed' found no observed ratings")
    elif isinstance(raw_values, list):
        values_domain = list(raw_values)
    else:
        raise _CliError("[study].values must be a list of values or 'observed'")

    # 4. Resolve measurement / distance
    measurement = _resolve_cli_measurement(study_cfg, values_domain)

    # 5. Coders & Design
    if "coders" not in study_cfg:
        raise _CliError("[study].coders is required")
    coders = study_cfg["coders"]
    valid_coders = isinstance(coders, str) or (
        isinstance(coders, list) and all(isinstance(x, str) for x in coders)
    )
    if not valid_coders:
        raise _CliError("[study].coders must be a field name or list of field names")

    if "design" not in study_cfg:
        raise _CliError("[study].design is required")
    raw_design = study_cfg["design"]
    if not isinstance(raw_design, str):
        raise _CliError("[study].design must be a design string")
    design_val = _parse_design(raw_design)

    # 6. Trace sidecar
    trace_sidecar: TraceSidecar | None = None
    trace_path_str = args.trace if args.trace is not None else trace_cfg.get("path")
    if trace_path_str is not None:
        if not isinstance(trace_path_str, str):
            raise _CliError("[trace].path must be a file path string")
        trace_path = (
            Path(trace_path_str)
            if args.trace is not None
            else _resolve_toml_path(base_dir, trace_path_str)
        )
        df_trace = _read_table(trace_path)
        trace_kwargs: dict[str, Any] = {}
        if "condition" in trace_cfg:
            if not isinstance(trace_cfg["condition"], str):
                raise _CliError("[trace].condition must be a string column name")
            trace_kwargs["condition"] = trace_cfg["condition"]
        elif "condition_id" in df_trace.columns:
            trace_kwargs["condition"] = "condition_id"
        elif "condition" in df_trace.columns:
            trace_kwargs["condition"] = "condition"
        else:
            raise _CliError(
                "trace file requires [trace].condition or a 'condition_id' / 'condition' column"
            )
        for key in TRACE_FIELDS:
            if key in trace_cfg:
                if not isinstance(trace_cfg[key], str):
                    raise _CliError(f"[trace].{key} must be a string column name")
                trace_kwargs[key] = trace_cfg[key]
            elif key in df_trace.columns:
                trace_kwargs[key] = key
        trace_sidecar = TraceSidecar.from_frame(df_trace, **trace_kwargs)
    elif trace_cfg:
        raise _CliError(
            "[trace] section is present but neither [trace].path nor --trace was provided"
        )

    # 6b. Unit covariate sidecar
    unit_sidecar: UnitSidecar | None = None
    units_path_str = args.units if args.units is not None else units_cfg.get("path")
    if units_path_str is not None:
        if not isinstance(units_path_str, str):
            raise _CliError("[units].path must be a file path string")
        units_path = (
            Path(units_path_str)
            if args.units is not None
            else _resolve_toml_path(base_dir, units_path_str)
        )
        df_units = _read_table(units_path)
        units_kwargs: dict[str, Any] = {}
        if "unit" in units_cfg:
            if not isinstance(units_cfg["unit"], str):
                raise _CliError("[units].unit must be a string column name")
            units_kwargs["unit"] = units_cfg["unit"]
        elif "unit_id" in df_units.columns:
            units_kwargs["unit"] = "unit_id"
        elif "unit" in df_units.columns:
            units_kwargs["unit"] = "unit"
        else:
            raise _CliError("units file requires [units].unit or a 'unit_id' / 'unit' column")
        for key in UNIT_FIELDS:
            if key in units_cfg:
                if not isinstance(units_cfg[key], str):
                    raise _CliError(f"[units].{key} must be a string column name")
                units_kwargs[key] = units_cfg[key]
            elif key in df_units.columns:
                units_kwargs[key] = key
        unit_sidecar = UnitSidecar.from_frame(df_units, **units_kwargs)
    elif units_cfg:
        raise _CliError(
            "[units] section is present but neither [units].path nor --units was provided"
        )

    # 7. Missing policies & retries
    missing_policies: Sequence[MissingPolicy] | None = None
    if "missing_policies" in study_cfg:
        raw_mp = study_cfg["missing_policies"]
        if not (isinstance(raw_mp, list) and all(isinstance(x, str) for x in raw_mp)):
            raise _CliError("[study].missing_policies must be a list of policy names")
        missing_policies = cast(Sequence[MissingPolicy], list(raw_mp))

    retries = cast(Retries | None, study_cfg.get("retries"))
    if retries is not None and not isinstance(retries, str):
        raise _CliError("[study].retries must be a string ('last' or 'first_observed')")

    study_obj = ReliabilityStudy(
        rf,
        measurement=measurement,
        values=values_domain,
        coders=coders,
        design=design_val,
        missing_policies=missing_policies,
        abstain_as=study_cfg.get("abstain_as"),
        inapplicable_as=study_cfg.get("inapplicable_as"),
        trace=trace_sidecar,
        retries=retries,
        units=unit_sidecar,
    )
    return study_obj, rf, base_dir


def _resolve_cli_measurement(
    study_cfg: Mapping[str, Any],
    values_domain: Sequence[Any],
) -> str | Distance:
    has_meas = "measurement" in study_cfg
    has_circ = "circumference" in study_cfg
    has_mat = "distance_matrix" in study_cfg

    if not has_meas and not has_mat:
        raise _CliError("[study].measurement (or distance_matrix) is required")

    raw_meas = study_cfg.get("measurement")
    if has_meas and not isinstance(raw_meas, str):
        raise _CliError("[study].measurement must be a string")

    if has_mat:
        if has_circ:
            raise _CliError("[study].circumference cannot be combined with distance_matrix")
        matrix = np.asarray(study_cfg["distance_matrix"], dtype=np.float64)
        n_vals = len(values_domain)
        if matrix.ndim != 2 or matrix.shape != (n_vals, n_vals):
            raise _CliError(
                f"[study].distance_matrix must be square with shape ({n_vals}, {n_vals})"
            )
        name = str(raw_meas) if has_meas else "custom"
        return Distance.from_matrix(name, matrix)

    assert isinstance(raw_meas, str)
    match = CIRCULAR_RE.fullmatch(raw_meas)
    if match is not None:
        if has_circ:
            raise _CliError(
                "specify circumference either inside circular(U) or via 'circumference', not both"
            )
        try:
            circ_u = float(match.group(1))
        except ValueError as exc:
            raise _CliError(f"invalid circumference in {raw_meas!r}") from exc
        return circular(circ_u)

    if raw_meas.lower() == "circular":
        if not has_circ:
            raise _CliError("[study].measurement = 'circular' requires [study].circumference")
        raw_circ = study_cfg["circumference"]
        if isinstance(raw_circ, bool) or not isinstance(raw_circ, (int, float)):
            raise _CliError("[study].circumference must be a positive number")
        return circular(float(raw_circ))

    if has_circ:
        raise _CliError("[study].circumference is only valid when measurement is 'circular'")
    return raw_meas


def _parse_design(raw_design: str) -> Design | str:
    lower = raw_design.strip().lower()
    for member in Design:
        if lower == member.value or lower == member.name.lower():
            return member
    return raw_design


def _resolve_toml_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise _CliError(f"data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        import importlib

        _import_pyarrow("reading Parquet input")
        pq: Any = importlib.import_module("pyarrow.parquet")
        return _arrow_to_pandas(pq.read_table(path))
    raise _CliError(
        f"unsupported table extension {path.suffix!r} for {path}; expected .csv, .tsv, or .parquet"
    )


def _merge_bootstrap_opts(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    boot_cfg = _table_section(cfg, "bootstrap", BOOTSTRAP_KEYS)
    gate_cfg = _table_section(cfg, "gate", GATE_KEYS)
    merged: dict[str, Any] = {}
    for key in BOOTSTRAP_KEYS:
        if key in boot_cfg:
            merged[key] = boot_cfg[key]
        if key in gate_cfg:
            merged[key] = gate_cfg[key]
    cli_overrides = {
        "scheme": args.scheme,
        "method": args.method,
        "confidence": args.confidence,
        "n_resamples": args.n_resamples,
        "level": args.level,
        "strata": args.strata,
        "seed": args.seed,
        "on_undefined": args.on_undefined,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            merged[key] = val
    return merged


def _has_bootstrap_config(opts: Mapping[str, Any]) -> bool:
    return any(opts.get(key) is not None for key in BOOTSTRAP_KEYS)


def _bootstrap_kwargs(opts: Mapping[str, Any], *, default_scheme: str) -> dict[str, Any]:
    scheme = opts.get("scheme", default_scheme)
    if not isinstance(scheme, str):
        raise _CliError("bootstrap 'scheme' must be a string")
    kwargs: dict[str, Any] = {"scheme": scheme}
    if opts.get("method") is not None:
        kwargs["method"] = _as_str(opts["method"], "method")
    if opts.get("confidence") is not None:
        kwargs["confidence"] = _as_float(opts["confidence"], "confidence")
    if opts.get("n_resamples") is not None:
        kwargs["n_resamples"] = _as_int(opts["n_resamples"], "n_resamples")
    if opts.get("level") is not None:
        kwargs["level"] = _as_str(opts["level"], "level")
    if opts.get("strata") is not None:
        kwargs["strata"] = _as_bool_opt(opts["strata"], "strata")
    if opts.get("seed") is not None:
        kwargs["random_state"] = _as_int(opts["seed"], "seed")
    if opts.get("on_undefined") is not None:
        kwargs["on_undefined"] = _as_str(opts["on_undefined"], "on_undefined")
    return kwargs


def _compare_kwargs(opts: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if opts.get("scheme") is not None:
        kwargs["scheme"] = _as_str(opts["scheme"], "scheme")
    if opts.get("method") is not None:
        kwargs["method"] = _as_str(opts["method"], "method")
    if opts.get("confidence") is not None:
        kwargs["confidence"] = _as_float(opts["confidence"], "confidence")
    if opts.get("n_resamples") is not None:
        kwargs["n_resamples"] = _as_int(opts["n_resamples"], "n_resamples")
    if opts.get("level") is not None:
        kwargs["level"] = _as_str(opts["level"], "level")
    if opts.get("strata") is not None:
        kwargs["strata"] = _as_bool_opt(opts["strata"], "strata")
    if opts.get("seed") is not None:
        kwargs["random_state"] = _as_int(opts["seed"], "seed")
    if opts.get("on_undefined") is not None:
        kwargs["on_undefined"] = _as_str(opts["on_undefined"], "on_undefined")
    return kwargs


def _resolve_selector(
    cli_items: Sequence[str],
    toml_sel: Any,
    name: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if toml_sel is not None:
        if not isinstance(toml_sel, Mapping):
            raise _CliError(f"[gate].{name} must be a table of field = value pairs")
        for key, val in toml_sel.items():
            if not isinstance(key, str):
                raise _CliError(f"[gate].{name} keys must be strings")
            out[key] = val
    for item in cli_items:
        if "=" not in item:
            raise _CliError(f"invalid --{name} selector {item!r}; expected FIELD=VALUE")
        field, raw_val = item.split("=", 1)
        field = field.strip()
        if not field:
            raise _CliError(f"invalid --{name} selector {item!r}; empty field name")
        out[field] = _coerce_selector_value(field, raw_val, df)
    if not out:
        raise _CliError(
            f"missing selector for arm '{name}'; pass --{name} FIELD=VALUE or set [gate].{name}"
        )
    unknown = set(out) - set(SELECTOR_FIELDS)
    if unknown:
        raise _CliError(f"unknown selector field(s) for arm '{name}': {sorted(unknown)}")
    return out


def _coerce_selector_value(field: str, raw: str, df: pd.DataFrame) -> Any:
    if field not in df.columns:
        return raw
    series = df[field].dropna()
    if series.empty:
        return raw
    if pd.api.types.is_bool_dtype(series.dtype) or all(isinstance(v, bool) for v in series):
        low = raw.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        return raw
    if pd.api.types.is_integer_dtype(series.dtype) or all(
        isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in series
    ):
        try:
            return int(raw)
        except ValueError:
            return raw
    if pd.api.types.is_numeric_dtype(series.dtype) or all(
        isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in series
    ):
        try:
            f_val = float(raw)
            return int(raw) if f_val.is_integer() and raw.strip().lstrip("+-").isdigit() else f_val
        except ValueError:
            return raw
    return raw


def _resolve_policies(policy: str | None, available: Sequence[str]) -> list[str]:
    if policy is None:
        return list(available)
    if policy == "sensitivity":
        candidates = ("sensitivity_pessimistic", "sensitivity_optimistic")
        selected = [p for p in candidates if p in available]
        if not selected:
            raise _CliError(
                f"policy 'sensitivity' was not reported; available estimates: {list(available)}"
            )
        return selected
    if policy not in available:
        raise _CliError(
            f"policy {policy!r} was not reported; available estimates: {list(available)}"
        )
    return [policy]


def _collect_strict_findings(
    caught: Sequence[warnings.WarningMessage],
    *,
    alpha_res: AlphaResult | None = None,
    boot_res: BootstrapResult | None = None,
    comp_res: ComparisonResult | None = None,
) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    def add(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            findings.append(msg)

    for w in caught:
        if issubclass(w.category, (TraceWarning, CoderDependenceWarning)):
            add(f"{w.category.__name__}: {w.message}")

    if alpha_res is not None:
        for rw in alpha_res.warnings:
            if rw.category in ("TraceWarning", "CoderDependenceWarning"):
                add(f"{rw.category}: {rw.message}")
        for tf in alpha_res.trace_findings:
            add(f"trace_finding: {tf}")

    if boot_res is not None:
        for rw in boot_res.warnings:
            if rw.category in ("TraceWarning", "CoderDependenceWarning"):
                add(f"{rw.category}: {rw.message}")

    if comp_res is not None:
        for rw in comp_res.warnings:
            if rw.category in ("TraceWarning", "CoderDependenceWarning"):
                add(f"{rw.category}: {rw.message}")
        for arm in (comp_res.result_a, comp_res.result_b):
            for rw in arm.warnings:
                if rw.category in ("TraceWarning", "CoderDependenceWarning"):
                    add(f"{rw.category}: {rw.message}")
            for tf in arm.trace_findings:
                add(f"trace_finding: {tf}")

    return findings


def _determine_exit_code(
    failures: Sequence[str],
    *,
    strict: bool,
    strict_findings: Sequence[str],
) -> int:
    if failures:
        return 2
    if strict and strict_findings:
        return 3
    return 0


def _evaluate_json_payload(
    alpha_res: AlphaResult,
    boot_res: BootstrapResult | None,
    gate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(alpha_res.to_dict())
    if boot_res is not None:
        boot_dict = boot_res.to_dict()
        payload["intervals"] = boot_dict["intervals"]
        if boot_dict.get("family_intervals"):
            payload["family_intervals"] = boot_dict["family_intervals"]
        payload["bootstrap"] = {
            key: boot_dict[key]
            for key in (
                "scheme",
                "level",
                "strata",
                "method",
                "confidence",
                "n_resamples",
                "seed",
                "alpha_min",
                "n_clusters",
                "n_coders",
                "experimental",
            )
            if key in boot_dict
        }
        payload["warnings"] = list(alpha_res.to_dict()["warnings"]) + [
            w for w in boot_dict["warnings"] if w not in alpha_res.to_dict()["warnings"]
        ]
        payload["notes"] = list(alpha_res.to_dict()["notes"]) + [
            n for n in boot_dict["notes"] if n not in alpha_res.to_dict()["notes"]
        ]
    payload["gate"] = dict(gate_payload)
    return payload


def _print_gate_status(gate: Mapping[str, Any]) -> None:
    print()
    status = (
        "PASS"
        if gate["exit_code"] == 0
        else ("FAIL (strict warnings/findings)" if gate["exit_code"] == 3 else "FAIL (gate)")
    )
    print(f"Gate status: {status} (exit code {gate['exit_code']})")
    for item in gate["checks"]:
        parts = [f"policy={item['policy']}"]
        if "alpha" in item:
            alpha_val = item["alpha"]
            parts.append(f"alpha={alpha_val:.4f}" if math.isfinite(alpha_val) else "alpha=nan")
        if "diff" in item:
            diff_val = item["diff"]
            parts.append(f"diff={diff_val:.4f}" if math.isfinite(diff_val) else "diff=nan")
        if "low" in item:
            parts.append(
                f"CI=[{item['low']:.4f}, {item['high']:.4f}]"
                if math.isfinite(item["low"]) and math.isfinite(item["high"])
                else "CI=[nan, nan]"
            )
        if item.get("verdict") is not None:
            parts.append(f"verdict={item['verdict']}")
        print("  " + ", ".join(parts))
    for msg in gate["failures"]:
        print(f"  failure: {msg}")
    if gate["strict_findings"]:
        label = "strict finding" if gate["strict"] else "finding"
        for msg in gate["strict_findings"]:
            print(f"  {label}: {msg}")


def _pick_opt_float(cli_val: Any, toml_val: Any, name: str) -> float | None:
    val = cli_val if cli_val is not None else toml_val
    return None if val is None else _as_float(val, name)


def _pick_opt_str(cli_val: Any, toml_val: Any, name: str) -> str | None:
    val = cli_val if cli_val is not None else toml_val
    return None if val is None else _as_str(val, name)


def _pick_opt_bool(cli_val: Any, toml_val: Any, name: str, *, default: bool) -> bool:
    val = cli_val if cli_val is not None else toml_val
    return default if val is None else _as_bool_opt(val, name)


def _as_float(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise _CliError(f"{name} must be numeric; got {val!r}")
    out = float(val)
    if not math.isfinite(out):
        raise _CliError(f"{name} must be finite; got {val!r}")
    return out


def _as_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise _CliError(f"{name} must be an integer; got {val!r}")
    return int(val)


def _as_bool_opt(val: Any, name: str) -> bool:
    if not isinstance(val, bool):
        raise _CliError(f"{name} must be a boolean; got {val!r}")
    return val


def _as_str(val: Any, name: str) -> str:
    if not isinstance(val, str):
        raise _CliError(f"{name} must be a string; got {val!r}")
    return val


if __name__ == "__main__":
    raise SystemExit(main())
