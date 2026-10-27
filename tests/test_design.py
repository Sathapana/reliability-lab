import warnings

import pandas as pd
import pytest

from reliability_lab import (
    CoderDependenceWarning,
    CoderSpec,
    Design,
    DesignError,
    RatingFrame,
    ReliabilityStudy,
)


def llm_frame(rows: list[tuple[str, str, str, str, int, int]]) -> RatingFrame:
    """Rows of (unit, model, family, condition, run, score)."""
    df = pd.DataFrame(rows, columns=["unit", "model", "family", "condition", "run", "score"])
    return RatingFrame.from_long(
        df,
        unit="unit",
        rater="model",
        value="score",
        model_id="model",
        model_family="family",
        condition_id="condition",
        run_id="run",
    )


def grid(models: list[tuple[str, str]], conditions: list[str], runs: list[int]) -> RatingFrame:
    rows = [
        (f"u{u}", model, family, cond, run, (u + run) % 3 + 1)
        for u in range(6)
        for model, family in models
        for cond in conditions
        for run in runs
    ]
    return llm_frame(rows)


def study(ratings: RatingFrame, design: str, by: list[str]) -> ReliabilityStudy:
    return ReliabilityStudy(
        ratings, measurement="ordinal", values=[1, 2, 3], coders=CoderSpec(by=by), design=design
    )


def human() -> RatingFrame:
    df = pd.DataFrame({"unit": ["u1", "u1", "u2", "u2"], "who": ["a", "b"] * 2, "v": [1, 2, 2, 2]})
    return RatingFrame.from_long(df, unit="unit", rater="who", value="v")


# -- CoderSpec -------------------------------------------------------------------


def test_coderspec_normalises_and_validates() -> None:
    assert CoderSpec("rater_id").by == ("rater_id",)
    assert CoderSpec(["model_id", "run_id"]).by == ("model_id", "run_id")
    with pytest.raises(DesignError, match="at least one"):
        CoderSpec([])
    with pytest.raises(DesignError, match="must be among"):
        CoderSpec("value")
    with pytest.raises(DesignError, match="repeat"):
        CoderSpec(["rater_id", "rater_id"])


def test_coderspec_needs_mapped_fields() -> None:
    with pytest.raises(DesignError, match="not mapped"):
        CoderSpec("run_id").labels(human())


def test_coder_labels_combine_fields() -> None:
    ratings = grid([("m1", "f1")], ["p1", "p2"], [1])
    labels = CoderSpec(["model_id", "condition_id"]).labels(ratings)
    assert set(labels) == {"m1 | p1", "m1 | p2"}


# -- design labels ---------------------------------------------------------------


def test_unknown_and_future_designs() -> None:
    with pytest.raises(DesignError, match="unknown design"):
        ReliabilityStudy(
            human(), measurement="nominal", values=[1, 2], coders="rater_id", design="x"
        )
    for design in ("human_llm_alignment", "configuration_change"):
        with pytest.raises(NotImplementedError):
            ReliabilityStudy(
                human(), measurement="nominal", values=[1, 2], coders="rater_id", design=design
            )
    with pytest.raises(NotImplementedError, match=r"study\.compare"):
        ReliabilityStudy(
            human(),
            measurement="nominal",
            values=[1, 2],
            coders="rater_id",
            design="configuration_change",
        )
    with pytest.raises(DesignError, match="order_robustness needs LLM rows"):
        ReliabilityStudy(
            human(),
            measurement="nominal",
            values=[1, 2],
            coders="rater_id",
            design="order_robustness",
        )


def test_mixed_human_and_llm_rows_are_refused() -> None:
    df = pd.DataFrame(
        {
            "unit": ["u1", "u1", "u2", "u2"],
            "who": ["ann", "gpt", "ann", "gpt"],
            "v": [1, 2, 2, 2],
            "model": [None, "gpt", None, "gpt"],
            "cond": [None, "p1", None, "p1"],
        }
    )
    ratings = RatingFrame.from_long(
        df, unit="unit", rater="who", value="v", model_id="model", condition_id="cond"
    )
    with pytest.raises(DesignError, match=r"v0\.2"):
        ReliabilityStudy(
            ratings, measurement="nominal", values=[1, 2], coders="rater_id", design="human_panel"
        )


def test_human_panel_refuses_llm_rows_and_vice_versa() -> None:
    with pytest.raises(DesignError, match="carry a model_id"):
        study(grid([("m1", "f1"), ("m2", "f2")], ["p1"], [1]), "human_panel", ["model_id"])
    with pytest.raises(DesignError, match="needs LLM rows"):
        ReliabilityStudy(
            human(), measurement="nominal", values=[1, 2], coders="rater_id", design="inter_model"
        )


def test_needs_two_coders() -> None:
    df = pd.DataFrame({"unit": ["u1", "u2"], "who": ["a", "a"], "v": [1, 2]})
    ratings = RatingFrame.from_long(df, unit="unit", rater="who", value="v")
    with pytest.raises(DesignError, match="at least two coders"):
        ReliabilityStudy(
            ratings, measurement="nominal", values=[1, 2], coders="rater_id", design="human_panel"
        )


def test_repeatability() -> None:
    ok = study(grid([("m1", "f1")], ["p1"], [1, 2, 3]), "repeatability", ["run_id"])
    assert ok.design is Design.REPEATABILITY
    assert "not inter-coder reliability" in ok.alpha().notes[0]
    with pytest.raises(DesignError, match="one judge configuration"):
        study(
            grid([("m1", "f1")], ["p1", "p2"], [1, 2]), "repeatability", ["condition_id", "run_id"]
        )
    with pytest.raises(DesignError, match="include 'run_id'"):
        study(grid([("m1", "f1")], ["p1"], [1, 2]), "repeatability", ["model_id"])


def test_prompt_robustness() -> None:
    ok = study(grid([("m1", "f1")], ["p1", "p2"], [1]), "prompt_robustness", ["condition_id"])
    assert ok.design is Design.PROMPT_ROBUSTNESS
    with pytest.raises(DesignError, match="one model"):
        study(
            grid([("m1", "f1"), ("m2", "f2")], ["p1", "p2"], [1]),
            "prompt_robustness",
            ["model_id", "condition_id"],
        )
    with pytest.raises(DesignError, match="include 'condition_id'"):
        study(grid([("m1", "f1")], ["p1", "p2"], [1]), "prompt_robustness", ["model_id"])


def test_inter_model_without_shared_family_is_quiet() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = study(grid([("m1", "f1"), ("m2", "f2")], ["p1"], [1]), "inter_model", ["model_id"])
    assert ok.warnings == ()


def test_inter_model_warns_on_shared_family() -> None:
    with pytest.warns(CoderDependenceWarning, match="share model family 'f1'"):
        s = study(grid([("m1", "f1"), ("m2", "f1")], ["p1"], [1]), "inter_model", ["model_id"])
    result = s.alpha()
    assert [w.category for w in result.warnings] == ["CoderDependenceWarning"]


def test_inter_model_warns_when_coders_share_a_model() -> None:
    ratings = grid([("m1", "f1"), ("m2", "f2")], ["p1", "p2"], [1])
    with pytest.warns(CoderDependenceWarning, match="use the same model"):
        study(ratings, "inter_model", ["model_id", "condition_id"])


def test_inter_model_warns_when_family_unknown() -> None:
    rows = [(f"u{u}", m, "", "p1", 1, u % 3 + 1) for u in range(4) for m in ("m1", "m2")]
    df = pd.DataFrame(rows, columns=["unit", "model", "family", "condition", "run", "score"])
    ratings = RatingFrame.from_long(
        df, unit="unit", rater="model", value="score", model_id="model", condition_id="condition"
    )
    with pytest.warns(CoderDependenceWarning, match="cannot be checked"):
        study(ratings, "inter_model", ["model_id"])


def test_inter_model_requires_model_coders() -> None:
    with pytest.raises(DesignError, match="include 'model_id'"):
        study(grid([("m1", "f1"), ("m2", "f2")], ["p1"], [1]), "inter_model", ["condition_id"])
    with pytest.raises(DesignError, match="two distinct model_id"):
        study(grid([("m1", "f1")], ["p1"], [1, 2]), "inter_model", ["model_id", "run_id"])


def test_one_rating_per_unit_and_coder() -> None:
    ratings = grid([("m1", "f1"), ("m2", "f2")], ["p1"], [1, 2])
    with pytest.raises(DesignError, match="more than one row"):
        study(ratings, "inter_model", ["model_id"])
