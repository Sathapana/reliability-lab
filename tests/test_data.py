import numpy as np
import pandas as pd
import pytest

from reliability_lab import RatingFrame, SchemaError, Status


def frame(**overrides: list[object]) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "item": ["u1", "u1", "u2", "u2"],
        "who": ["a", "b", "a", "b"],
        "score": [1, 2, 2, 2],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_maps_columns_to_canonical_fields() -> None:
    ratings = RatingFrame.from_long(frame(), unit="item", rater="who", value="score")
    table = ratings.to_frame()
    assert list(table.columns[:4]) == ["unit_id", "rater_id", "value", "status"]
    assert (table["status"] == "observed").all()
    assert ratings.implicit_status
    assert ratings.n_units == 2
    assert len(ratings) == 4
    assert "status" in ratings.declared and "group_id" not in ratings.declared
    assert "RatingFrame(rows=4" in repr(ratings)


def test_to_frame_returns_a_copy() -> None:
    ratings = RatingFrame.from_long(frame(), unit="item", rater="who", value="score")
    copy = ratings.to_frame()
    copy.loc[copy.index[0], "value"] = 99
    assert 99 not in ratings.to_frame()["value"].to_numpy()


def test_source_dataframe_is_not_aliased() -> None:
    df = frame()
    ratings = RatingFrame.from_long(df, unit="item", rater="who", value="score")
    df.loc[0, "score"] = 99
    assert 99 not in ratings.to_frame()["value"].to_numpy()


def test_observed_values_are_sorted_and_skip_failures() -> None:
    df = frame(score=[3, None, 1, 2], status=["observed", "parse_error", "observed", "observed"])
    ratings = RatingFrame.from_long(df, unit="item", rater="who", value="score", status="status")
    assert ratings.observed_values() == [1, 2, 3]


def test_accepts_status_enum_members() -> None:
    df = frame(status=[Status.OBSERVED, Status.OBSERVED, Status.ABSTAINED, Status.OBSERVED])
    ratings = RatingFrame.from_long(df, unit="item", rater="who", value="score", status="status")
    assert list(ratings.to_frame()["status"]) == ["observed", "observed", "abstained", "observed"]


@pytest.mark.parametrize(
    ("df", "kwargs", "message"),
    [
        (frame(), {"unit": "nope"}, "columns not found"),
        (frame(item=["u1", None, "u2", "u2"]), {}, "unit_id"),
        (frame(who=["a", "b", None, "b"]), {}, "rater_id"),
        (frame(score=[1, None, 2, 2]), {}, "Pass status="),
        (
            frame(st=["observed", "oops", "observed", "observed"]),
            {"status": "st"},
            "unknown status",
        ),
        (frame(st=["observed", None, "observed", "observed"]), {"status": "st"}, "status"),
        (
            frame(score=[1, None, 2, 2], st=["observed"] * 4),
            {"status": "st"},
            "'observed' but no value",
        ),
        (frame(g=["x", "y", "z", "z"]), {"group": "g"}, "exactly one group_id"),
        (frame(g=["x", None, "z", "z"]), {"group": "g"}, "group_id"),
        (frame(s=["x", "y", "z", "z"]), {"stratum": "s"}, "exactly one stratum"),
        (frame(m=["gpt", "gpt", "gpt", "gpt"]), {"model_id": "m"}, "need condition_id"),
        (
            frame(m=["gpt"] * 4, c=["v1", None, "v1", "v1"]),
            {"model_id": "m", "condition_id": "c"},
            "no condition_id",
        ),
        (
            frame(m=["gpt"] * 4, c=["v1"] * 4, f=["x", "y", "x", "x"]),
            {"model_id": "m", "condition_id": "c", "model_family": "f"},
            "exactly one model_family",
        ),
    ],
)
def test_schema_errors(df: pd.DataFrame, kwargs: dict[str, str], message: str) -> None:
    args = {"unit": "item", "rater": "who", "value": "score", **kwargs}
    with pytest.raises(SchemaError, match=message):
        RatingFrame.from_long(df, **args)


def test_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        RatingFrame.from_long(np.zeros((2, 2)), unit=0, rater=1, value=0)  # type: ignore[arg-type]


def test_llm_mask() -> None:
    df = frame(m=["gpt", "gpt", "gpt", "gpt"], c=["v1"] * 4)
    ratings = RatingFrame.from_long(
        df, unit="item", rater="who", value="score", model_id="m", condition_id="c"
    )
    assert ratings.is_llm.all()
