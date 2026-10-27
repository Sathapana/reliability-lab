from typing import Any

import pandas as pd
import pytest
from reference import reference_alpha
from sklearn.model_selection import KFold

from reliability_lab import DesignError, RatingFrame, ReliabilityStudy, SchemaError

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")

OBS, PARSE = "observed", "parse_error"


def retried_log() -> pd.DataFrame:
    """Two models on four units. m1 retries u0 after a parse error; m2 is asked twice on u1."""
    rows = [
        ("u0", "m1", 1, PARSE, None),
        ("u0", "m1", 2, OBS, "x"),
        ("u0", "m2", 1, OBS, "x"),
        ("u1", "m1", 1, OBS, "y"),
        ("u1", "m2", 1, OBS, "y"),
        ("u1", "m2", 2, OBS, "x"),
        ("u2", "m1", 1, OBS, "x"),
        ("u2", "m2", 1, OBS, "y"),
        ("u3", "m1", 1, OBS, "y"),
        ("u3", "m2", 1, PARSE, None),
        ("u3", "m2", 2, PARSE, None),
    ]
    df = pd.DataFrame(rows, columns=["unit", "model", "attempt", "status", "label"])
    return df.assign(condition="p1", family=df["model"].map({"m1": "f1", "m2": "f2"}))


def ratings_of(df: pd.DataFrame, attempt: bool = True, run: bool = False) -> RatingFrame:
    kwargs = {"attempt": "attempt"} if attempt else {}
    if run:
        kwargs["run_id"] = "run"
    return RatingFrame.from_long(
        df,
        unit="unit",
        rater="model",
        value="label",
        status="status",
        model_id="model",
        model_family="family",
        condition_id="condition",
        **kwargs,
    )


def study_of(ratings: RatingFrame, retries: Any) -> ReliabilityStudy:
    return ReliabilityStudy(
        ratings,
        measurement="nominal",
        values=["x", "y"],
        coders="model_id",
        design="inter_model",
        retries=retries,
    )


def test_last_and_first_observed_count_different_attempts() -> None:
    last = study_of(ratings_of(retried_log()), "last").alpha()
    first = study_of(ratings_of(retried_log()), "first_observed").alpha()
    # u3/m2 failed on both attempts, so both rules count a failure there.
    expected_last = reference_alpha(
        {"m1": ["x", "y", "x", "y"], "m2": ["x", "x", "y", None]}, "nominal", ["x", "y"]
    )
    expected_first = reference_alpha(
        {"m1": ["x", "y", "x", "y"], "m2": ["x", "y", "y", None]}, "nominal", ["x", "y"]
    )
    assert last.alpha == pytest.approx(expected_last)
    assert first.alpha == pytest.approx(expected_first)
    assert last.coverage.n_ratings == 8
    assert any("Retries: 3 calls" in n for n in last.notes)


def test_attempt_and_final_error_rates() -> None:
    per_coder = study_of(ratings_of(retried_log()), "last").alpha().coverage.per_coder
    m1, m2 = per_coder.loc["m1"], per_coder.loc["m2"]
    assert (m1["attempts"], m1["ratings"], m1["retries"]) == (5, 4, 1)
    assert m1["attempt_error_rate"] == pytest.approx(1 / 5)
    assert m1["error_rate"] == 0
    assert (m2["attempts"], m2["retries"]) == (6, 2)
    assert m2["attempt_error_rate"] == pytest.approx(2 / 6)
    assert m2["error_rate"] == pytest.approx(1 / 4)


def test_retries_must_be_declared_and_well_formed() -> None:
    with pytest.raises(DesignError, match="retries='last' or retries='first_observed'"):
        study_of(ratings_of(retried_log()), None)
    with pytest.raises(DesignError, match="needs an attempt column"):
        study_of(
            ratings_of(retried_log().drop_duplicates(["unit", "model"]), attempt=False), "last"
        )
    with pytest.raises(DesignError, match="retries must be"):
        study_of(ratings_of(retried_log()), "first")
    repeated = retried_log()
    repeated.loc[1, "attempt"] = 1
    with pytest.raises(DesignError, match="repeat an attempt number"):
        study_of(ratings_of(repeated), "last")
    bad = retried_log().astype({"attempt": object})
    bad.loc[0, "attempt"] = "first"
    with pytest.raises(SchemaError, match=r"attempt .* must be numeric"):
        ratings_of(bad)


def test_runs_are_not_mistaken_for_retries() -> None:
    df = retried_log().assign(run=lambda d: d["attempt"])
    with pytest.raises(DesignError, match="more than one row"):
        study_of(ratings_of(df, run=True), "last")


def test_splits_keep_all_attempts_of_a_unit_together() -> None:
    study = study_of(ratings_of(retried_log()), "last")
    dev, test = study.holdout(KFold(n_splits=2, shuffle=True, random_state=0))
    assert len(dev.ratings) + len(test.ratings) == len(study.ratings)
    for part in (dev, test):
        rows = part.ratings.to_frame()
        source = retried_log()
        expected = source[source["unit"].isin(rows["unit_id"])]
        assert len(rows) == len(expected)
        assert part.retries == "last"
