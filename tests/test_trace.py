import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reliability_lab import (
    DesignError,
    RatingFrame,
    ReliabilityStudy,
    SchemaError,
    TraceSidecar,
    TraceWarning,
)
from reliability_lab.llm import configuration_diff, wilson

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def conditions_frame(**columns: list[Any]) -> pd.DataFrame:
    return pd.DataFrame({"condition": ["p1", "p2"], **columns})


def llm_ratings(
    conditions: tuple[str, ...] = ("p1", "p2"), models: tuple[str, ...] = ("m1",), runs: int = 1
) -> RatingFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(12):
        truth = int(rng.integers(1, 4))
        for model in models:
            for condition in conditions:
                for run in range(1, runs + 1):
                    value = truth if rng.random() < 0.8 else int(rng.integers(1, 4))
                    rows.append((f"u{u}", model, f"fam-{model}", condition, run, value))
    df = pd.DataFrame(rows, columns=["unit", "model", "family", "condition", "run", "value"])
    return RatingFrame.from_long(
        df,
        unit="unit",
        rater="model",
        value="value",
        model_id="model",
        model_family="family",
        condition_id="condition",
        run_id="run",
    )


def make(
    ratings: RatingFrame, design: str, coders: Any, trace: TraceSidecar | None
) -> ReliabilityStudy:
    return ReliabilityStudy(
        ratings, measurement="ordinal", values=[1, 2, 3], coders=coders, design=design, trace=trace
    )


# -- sidecar ---------------------------------------------------------------------------


def test_sidecar_maps_and_normalises_fields() -> None:
    frame = conditions_frame(
        h=["a", "b"], t=["0.0", 0.7], det=["true", False], order=["A,B", ["B", "A"]]
    )
    trace = TraceSidecar.from_frame(
        frame,
        condition="condition",
        prompt_hash="h",
        temperature="t",
        deterministic="det",
        candidate_order="order",
    )
    assert trace.fields == ("prompt_hash", "temperature", "deterministic", "candidate_order")
    assert trace.value("p1", "temperature") == 0.0
    assert trace.value("p1", "deterministic") is True
    assert trace.value("p2", "candidate_order") == ("B", "A")
    assert trace.value("p2", "seed") is None
    assert trace.value("unknown", "temperature") is None
    assert "p1" in trace and len(trace) == 2
    assert list(trace.rows(["p2"]).index) == ["p2"]
    payload = trace.to_dict(["p1"])
    assert payload["rows"] == [
        {
            "condition_id": "p1",
            "prompt_hash": "a",
            "temperature": 0.0,
            "deterministic": True,
            "candidate_order": ["A", "B"],
        }
    ]
    shuffled = TraceSidecar.from_frame(
        frame.iloc[::-1],
        condition="condition",
        prompt_hash="h",
        temperature="t",
        deterministic="det",
        candidate_order="order",
    )
    assert shuffled.fingerprint() == trace.fingerprint()


@pytest.mark.parametrize(
    ("frame", "kwargs", "message"),
    [
        (conditions_frame(), {"prompt_hash": "nope"}, "columns not found"),
        (pd.DataFrame({"condition": ["p1", "p1"]}), {}, "one row per condition_id"),
        (pd.DataFrame({"condition": ["p1", None]}), {}, "condition_id is missing"),
        (conditions_frame(t=["warm", 0.1]), {"temperature": "t"}, "numeric"),
        (conditions_frame(d=["maybe", True]), {"deterministic": "d"}, "true or false"),
        (conditions_frame(o=["A", "A,B"]), {"candidate_order": "o"}, "two or more distinct"),
        (conditions_frame(o=["A,A", "A,B"]), {"candidate_order": "o"}, "two or more distinct"),
    ],
)
def test_sidecar_validation(frame: pd.DataFrame, kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        TraceSidecar.from_frame(frame, condition="condition", **kwargs)


# -- design checks ---------------------------------------------------------------------


def test_prompt_robustness_confounds() -> None:
    clean = TraceSidecar.from_frame(
        conditions_frame(h=["a", "b"], t=[0.0, 0.0]),
        condition="condition",
        prompt_hash="h",
        temperature="t",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", TraceWarning)
        make(llm_ratings(), "prompt_robustness", "condition_id", clean)

    warm = TraceSidecar.from_frame(
        conditions_frame(h=["a", "b"], t=[0.0, 0.7]),
        condition="condition",
        prompt_hash="h",
        temperature="t",
    )
    with pytest.warns(TraceWarning, match="differ in temperature as well as the prompt"):
        study = make(llm_ratings(), "prompt_robustness", "condition_id", warm)
    assert any("temperature" in f for f in study.alpha().trace_findings)

    same_prompt = TraceSidecar.from_frame(
        conditions_frame(h=["a", "a"]), condition="condition", prompt_hash="h"
    )
    with pytest.warns(TraceWarning, match="share a prompt_hash"):
        make(llm_ratings(), "prompt_robustness", "condition_id", same_prompt)


def test_missing_conditions_and_fields_become_gaps() -> None:
    partial = TraceSidecar.from_frame(
        pd.DataFrame({"condition": ["p1"], "h": ["a"]}), condition="condition", prompt_hash="h"
    )
    result = make(llm_ratings(), "prompt_robustness", "condition_id", partial).alpha()
    assert result.provenance is not None
    gaps = " ".join(result.provenance.gaps)
    assert "1 conditions have no trace row ['p2']" in gaps
    assert "does not record provider, rubric_version, temperature" in gaps
    assert [r["condition_id"] for r in result.provenance.trace["rows"]] == ["p1"]  # type: ignore[index]


def test_trace_needs_llm_rows() -> None:
    df = pd.DataFrame({"unit": ["u1", "u1", "u2", "u2"], "who": ["a", "b"] * 2, "v": [1, 2, 2, 2]})
    human = RatingFrame.from_long(df, unit="unit", rater="who", value="v")
    trace = TraceSidecar.from_frame(conditions_frame(), condition="condition")
    with pytest.raises(DesignError, match="no LLM rows"):
        ReliabilityStudy(
            human,
            measurement="nominal",
            values=[1, 2],
            coders="rater_id",
            design="human_panel",
            trace=trace,
        )
    with pytest.raises(TypeError, match="TraceSidecar"):
        ReliabilityStudy(
            human,
            measurement="nominal",
            values=[1, 2],
            coders="rater_id",
            design="human_panel",
            trace="trace.csv",  # type: ignore[arg-type]
        )


def test_repeatability_findings() -> None:
    frame = pd.DataFrame({"condition": ["p1"], "det": [True]})
    trace = TraceSidecar.from_frame(frame, condition="condition", deterministic="det")
    result = make(llm_ratings(conditions=("p1",), runs=3), "repeatability", "run_id", trace).alpha()
    assert result.diagnostics["run_agreement"].values["units_with_disagreement"] > 0
    assert any("declared deterministic" in f for f in result.trace_findings)

    warm = TraceSidecar.from_frame(
        pd.DataFrame({"condition": ["p1"], "t": [0.5]}), condition="condition", temperature="t"
    )
    findings = make(llm_ratings(conditions=("p1",), runs=3), "repeatability", "run_id", warm)
    joined = " ".join(findings.alpha().trace_findings)
    assert "Temperature is 0.5" in joined
    assert "No seed or determinism setting" in joined


def order_trace(orders: list[str], **extra: list[Any]) -> TraceSidecar:
    frame = conditions_frame(o=orders, **extra)
    kwargs = {"temperature": "t"} if "t" in extra else {}
    return TraceSidecar.from_frame(frame, condition="condition", candidate_order="o", **kwargs)


def pairwise_ratings() -> RatingFrame:
    rows = [
        (f"i{i}", "m1", c, ["A", "B", "tie"][(i + k) % 3])
        for i in range(10)
        for k, c in enumerate(("p1", "p2"))
    ]
    df = pd.DataFrame(rows, columns=["item", "model", "condition", "label"])
    return RatingFrame.from_long(
        df, unit="item", rater="model", value="label", model_id="model", condition_id="condition"
    )


def order_study(trace: TraceSidecar, values: list[str] | None = None) -> ReliabilityStudy:
    return ReliabilityStudy(
        pairwise_ratings(),
        measurement="nominal",
        values=values or ["A", "tie", "B"],
        coders="condition_id",
        design="order_robustness",
        trace=trace,
    )


def test_order_robustness_checks() -> None:
    with pytest.raises(DesignError, match="same order"):
        order_study(order_trace(["A,B", "A,B"]))
    with pytest.raises(DesignError, match="different candidates"):
        order_study(order_trace(["A,B", "B,C"]))
    with pytest.raises(DesignError, match="not declared values"):
        order_study(order_trace(["A,B", "B,A"]), values=["A", "tie", "C"])
    with pytest.warns(TraceWarning, match="temperature as well as candidate order"):
        order_study(order_trace(["A,B", "B,A"], t=[0.0, 0.3]))
    result = order_study(order_trace(["A,B", "B,A"])).alpha()
    assert "order_swap" in result.diagnostics


def test_order_robustness_is_per_judge_and_needs_condition_coders() -> None:
    trace = order_trace(["A,B", "B,A"])
    rows = [
        (f"i{i}", m, c, ["A", "B"][(i + k) % 2])
        for i in range(6)
        for m in ("m1", "m2")
        for k, c in enumerate(("p1", "p2"))
    ]
    df = pd.DataFrame(rows, columns=["item", "model", "condition", "label"])
    two_models = RatingFrame.from_long(
        df, unit="item", rater="model", value="label", model_id="model", condition_id="condition"
    )
    study = ReliabilityStudy(
        two_models,
        measurement="nominal",
        values=["A", "B"],
        coders="condition_id",
        design="order_robustness",
        trace=trace,
    )
    with pytest.raises(DesignError, match="holds 2 judge configurations"):
        study.alpha()
    with pytest.raises(DesignError, match="include 'condition_id'"):
        ReliabilityStudy(
            pairwise_ratings(),
            measurement="nominal",
            values=["A", "tie", "B"],
            coders="model_id",
            design="order_robustness",
            trace=trace,
        )


def test_configuration_diff_and_wilson() -> None:
    trace = TraceSidecar.from_frame(
        pd.DataFrame({"condition": ["a1", "a2", "b1"], "h": ["x", "x", "y"], "t": [0, 0, 0]}),
        condition="condition",
        prompt_hash="h",
        temperature="t",
    )
    assert configuration_diff(trace, ["a1", "a2"], ["b1"]) == {
        "prompt_hash": {"a": ["x"], "b": ["y"]}
    }
    assert configuration_diff(trace, ["a1"], ["a2"]) == {}
    p, low, high = wilson(70, 100)
    assert p == 0.7
    assert low == pytest.approx(0.6041, abs=1e-4)
    assert high == pytest.approx(0.7810, abs=1e-4)
    assert all(np.isnan(x) for x in wilson(0, 0))
