"""Provider-neutral end-to-end runs of each LLM workflow: the P4 gate.

Each test goes from a synthetic judge log and its trace sidecar to alpha,
diagnostics, a bootstrap and JSON, and checks that planted effects are recovered.
"""

import json

import pandas as pd
import pytest
from fixtures.llm_logs import (
    configuration_change_log,
    inter_model_log,
    order_swap_log,
    repeatability_log,
)

from reliability_lab import (
    CoderDependenceWarning,
    CoderSpec,
    DesignError,
    RatingFrame,
    ReliabilityStudy,
    TraceSidecar,
    TraceWarning,
)

SCORES = [1, 2, 3, 4, 5]
pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def trace_of(conditions: pd.DataFrame, **fields: str) -> TraceSidecar:
    mapping = {f: f for f in fields.get("extra", "").split() if f}
    for name in ("provider", "prompt_hash", "rubric_version", "temperature", "parser_version"):
        if name in conditions:
            mapping[name] = name
    if "retry_policy" in conditions:
        mapping["retry_policy"] = "retry_policy"
    if "order" in conditions:
        mapping["candidate_order"] = "order"
    return TraceSidecar.from_frame(conditions, condition="condition", **mapping)


def test_repeatability_with_retries() -> None:
    log, conditions = repeatability_log()
    trace = trace_of(conditions)
    ratings = RatingFrame.from_long(
        log,
        unit="unit",
        rater="model",
        value="score",
        status="status",
        group="conversation",
        model_id="model",
        model_family="family",
        condition_id="condition",
        run_id="run",
        attempt="attempt",
    )
    declared = {
        "measurement": "ordinal",
        "values": SCORES,
        "coders": "run_id",
        "design": "repeatability",
        "trace": trace,
    }
    with pytest.raises(DesignError, match="retries="):
        ReliabilityStudy(ratings, **declared)  # type: ignore[arg-type]

    result = ReliabilityStudy(ratings, retries="last", **declared).alpha()  # type: ignore[arg-type]
    assert 0 < result.alpha < 1
    runs = result.diagnostics["run_agreement"].values
    assert runs["units_with_two_or_more_runs"] > 50
    assert 0 < runs["unanimous_share"] < 1
    assert any("Temperature is 0.7" in f for f in result.trace_findings)
    assert any(n.startswith("Retries:") for n in result.notes)

    per_coder = result.coverage.per_coder
    assert (per_coder["attempts"] > per_coder["ratings"]).all()
    assert (per_coder["attempt_error_rate"] > per_coder["error_rate"]).all()
    assert (per_coder["error_rate"] > 0).all()  # calls that failed on every attempt

    provenance = result.provenance
    assert provenance is not None and provenance.trace is not None
    assert "temperature" in provenance.trace["fields"]
    assert [row["condition_id"] for row in provenance.trace["rows"]] == ["cfg-1"]
    assert any("does not record seed" in g for g in provenance.gaps)
    assert provenance.declarations["retries"] == "last"

    boot = result.bootstrap(scheme="units", n_resamples=200, random_state=1)
    assert boot.level == "group"
    payload = json.loads(result.to_json())
    assert payload["diagnostics"]["run_agreement"]["values"]["units_with_disagreement"] > 0
    assert payload["trace_findings"]


def test_inter_model_flags_family_and_prompt_confound() -> None:
    log, conditions = inter_model_log()
    ratings = RatingFrame.from_long(
        log,
        unit="unit",
        rater="model",
        value="score",
        status="status",
        model_id="model",
        model_family="family",
        condition_id="condition",
    )
    with (
        pytest.warns(CoderDependenceWarning, match="share model family 'fam-2'"),
        pytest.warns(TraceWarning, match="different prompt_hash"),
    ):
        study = ReliabilityStudy(
            ratings,
            measurement="ordinal",
            values=SCORES,
            coders="model_id",
            design="inter_model",
            trace=trace_of(conditions),
        )
    result = study.alpha()
    agreement = result.diagnostics["coder_agreement"]
    assert agreement.values["pairs"] == 3
    assert agreement.table is not None
    assert (agreement.table["shared_units"] == 80).all()
    assert agreement.values["mean_exact_agreement"] == pytest.approx(
        agreement.table["exact_agreement"].mean()
    )
    assert 0.3 < agreement.values["min_exact_agreement"] < 0.9
    assert any("different prompt_hash" in f for f in result.trace_findings)


def test_configuration_change_recovers_the_better_prompt() -> None:
    log, conditions = configuration_change_log()
    ratings = RatingFrame.from_long(
        log,
        unit="unit",
        rater="model",
        value="score",
        status="status",
        model_id="model",
        model_family="family",
        condition_id="condition",
    )
    # The pooled study mixes both prompts, and its warnings say so; the comparison
    # below is what separates them.
    with (
        pytest.warns(CoderDependenceWarning, match="use the same model"),
        pytest.warns(TraceWarning, match="different prompt_hash"),
    ):
        study = ReliabilityStudy(
            ratings,
            measurement="ordinal",
            values=SCORES,
            coders=CoderSpec(by=["model_id", "condition_id"]),
            design="inter_model",
            trace=trace_of(conditions),
        )
    diff = study.compare(
        a={"condition_id": "v1"}, b={"condition_id": "v2"}, n_resamples=1000, random_state=4
    )
    change = diff["as_missing"]
    assert change.point > 0
    assert change.low > 0  # the better prompt is reliably better on these units
    assert diff.configuration_diff == {"prompt_hash": {"a": ["h-v1"], "b": ["h-v2"]}}
    assert "configuration changes: prompt_hash" in diff.summary()
    payload = json.loads(diff.to_json())
    assert payload["configuration_diff"]["prompt_hash"]["b"] == ["h-v2"]


def order_study(
    trace: TraceSidecar | None, judges: tuple[tuple[str, float], ...] = (("judge-x", 0.4),)
) -> ReliabilityStudy:
    log, _ = order_swap_log(judges=judges)
    ratings = RatingFrame.from_long(
        log,
        unit="item",
        rater="model",
        value="label",
        status="status",
        model_id="model",
        model_family="family",
        condition_id="condition",
    )
    return ReliabilityStudy(
        ratings,
        measurement="nominal",
        values=["A", "tie", "B"],
        coders="condition_id",
        design="order_robustness",
        trace=trace,
    )


def test_order_swap_recovers_planted_position_bias() -> None:
    _, conditions = order_swap_log()
    result = order_study(trace_of(conditions)).alpha()
    swap = result.diagnostics["order_swap"]
    v = swap.values
    assert v["p_first_shown_wins_low"] <= 0.7 <= v["p_first_shown_wins_high"]
    assert v["p_first_shown_wins_low"] > 0.5
    assert v["units_compared"] == 200
    assert 0 < v["reversal_rate"] <= v["disagreement_rate"] < 1
    assert swap.table is not None
    assert list(swap.table["coder"]) == ["ab", "ba", "pooled"]
    assert list(swap.table["candidate_order"][:2]) == ["A,B", "B,A"]
    # Position bias shows up as low agreement between the two orders.
    assert result.alpha < 0.6
    assert (
        json.loads(result.to_json())["diagnostics"]["order_swap"]["values"]["units_compared"] == 200
    )
    assert "diagnostic_order_swap" in result.tables()


def test_order_swap_needs_candidate_order() -> None:
    _, conditions = order_swap_log()
    with pytest.raises(DesignError, match="needs a trace sidecar"):
        order_study(None)
    no_order = TraceSidecar.from_frame(conditions, condition="condition", prompt_hash="prompt_hash")
    with pytest.raises(DesignError, match="needs candidate_order"):
        order_study(no_order)


# -- configuration changes within single-configuration designs ----------------------


def test_repeatability_compared_across_two_configurations() -> None:
    log, conditions = repeatability_log(
        configs=(("cfg-hot", 0.7, 0.6), ("cfg-cold", 0.0, 0.95)), seed=5
    )
    ratings = RatingFrame.from_long(
        log,
        unit="unit",
        rater="model",
        value="score",
        status="status",
        group="conversation",
        model_id="model",
        model_family="family",
        condition_id="condition",
        run_id="run",
        attempt="attempt",
    )
    study = ReliabilityStudy(
        ratings,
        measurement="ordinal",
        values=SCORES,
        coders="run_id",
        design="repeatability",
        trace=trace_of(conditions),
        retries="last",
    )
    with pytest.raises(DesignError, match="holds 2 judge configurations"):
        study.alpha()
    diff = study.compare(
        a={"condition_id": "cfg-hot"}, b={"condition_id": "cfg-cold"}, random_state=6
    )
    change = diff["as_missing"]
    assert change.point > 0 and change.low > 0  # the cold configuration repeats itself more
    assert diff.arm_a.level == "group"
    assert diff.configuration_diff == {"temperature": {"a": ["0.7"], "b": ["0"]}}
    hot = diff.result_a.diagnostics["run_agreement"].values["unanimous_share"]
    cold = diff.result_b.diagnostics["run_agreement"].values["unanimous_share"]
    assert cold > hot
    assert any("Temperature is 0.7" in f for f in diff.result_a.trace_findings)


def test_order_robustness_compared_across_two_judges() -> None:
    _, conditions = order_swap_log()
    study = order_study(trace_of(conditions), judges=(("judge-x", 0.4), ("judge-y", 0.0)))
    with pytest.raises(DesignError, match="holds 2 judge configurations"):
        study.alpha()
    diff = study.compare(a={"model_id": "judge-x"}, b={"model_id": "judge-y"}, random_state=7)
    change = diff["as_missing"]
    assert change.point > 0 and change.low > 0  # the unbiased judge is more order-robust
    assert diff.configuration_diff == {"model_id": {"a": ["judge-x"], "b": ["judge-y"]}}
    biased = diff.result_a.diagnostics["order_swap"].values
    fair = diff.result_b.diagnostics["order_swap"].values
    assert biased["p_first_shown_wins_low"] <= 0.7 <= biased["p_first_shown_wins_high"]
    assert fair["p_first_shown_wins_low"] <= 0.5 <= fair["p_first_shown_wins_high"]
    assert "configuration changes: model_id" in diff.summary()
