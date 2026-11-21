"""Arrow tables (pyarrow, polars, DuckDB) give the same ratings as pandas."""

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reliability_lab import RatingFrame, ReliabilityStudy, TraceSidecar
from reliability_lab import data as data_module

pytestmark = [
    pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning"),
    # The trace has missing settings on purpose, so the study reports them.
    pytest.mark.filterwarnings("ignore::reliability_lab.TraceWarning"),
]

COLUMNS = {
    "unit": "item",
    "rater": "judge",
    "value": "score",
    "status": "outcome",
    "group": "source",
    "stratum": "topic",
    "model_id": "model",
    "model_family": "family",
    "condition_id": "prompt",
    "run_id": "run",
    "attempt": "try",
}
TRACE_COLUMNS = {
    "condition": "prompt",
    "provider": "provider",
    "prompt_hash": "hash",
    "temperature": "temperature",
    "seed": "seed",
    "deterministic": "deterministic",
    "candidate_order": "order",
    "retry_policy": "retries",
}


def ratings_columns() -> dict[str, list[Any]]:
    """Every field, with failed attempts (no value), retries and an abstention."""
    rng = np.random.default_rng(3)
    rows: list[tuple[int, str, int | None, str, int]] = []
    for u in range(12):
        truth = int(rng.integers(1, 4))
        for prompt in ("p1", "p2", "p3"):
            value = truth if rng.random() < 0.8 else int(rng.integers(1, 4))
            if (u, prompt) == (2, "p2"):
                rows.append((u, prompt, None, "parse_error", 1))
                rows.append((u, prompt, value, "observed", 2))
            elif (u, prompt) == (5, "p3"):
                rows.append((u, prompt, None, "provider_error", 1))
            elif (u, prompt) == (7, "p1"):
                rows.append((u, prompt, None, "abstained", 1))
            else:
                rows.append((u, prompt, value, "observed", 1))
    return {
        "item": [f"u{u}" for u, *_ in rows],
        "judge": [f"m1/{p}" for _, p, *_ in rows],
        "score": [v for _, _, v, _, _ in rows],
        "outcome": [s for *_, s, _ in rows],
        "source": [f"g{u // 3}" for u, *_ in rows],
        "topic": [f"t{u // 6}" for u, *_ in rows],
        "model": ["m1"] * len(rows),
        "family": ["fam"] * len(rows),
        "prompt": [p for _, p, *_ in rows],
        "run": [1] * len(rows),
        "try": [a for *_, a in rows],
    }


def trace_columns() -> dict[str, list[Any]]:
    return {
        "prompt": ["p1", "p2", "p3"],
        "provider": ["acme", "acme", "acme"],
        "hash": ["h1", "h2", "h3"],
        "temperature": [0.0, 0.0, None],
        "seed": [7, None, 7],
        "deterministic": [True, None, True],
        "order": ["A,B", "A,B", "A,B"],
        "retries": ["last", "last", "last"],
    }


def as_pyarrow(columns: dict[str, list[Any]]) -> Any:
    pa = pytest.importorskip("pyarrow")
    return pa.table(columns)


def as_polars(columns: dict[str, list[Any]]) -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame(columns)


def as_duckdb(columns: dict[str, list[Any]]) -> Any:
    duckdb = pytest.importorskip("duckdb")
    return duckdb.from_arrow(as_pyarrow(columns))


def as_reader(columns: dict[str, list[Any]]) -> Any:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(columns)
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=10))


def as_polars_categorical(columns: dict[str, list[Any]]) -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame(columns).with_columns(pl.col("prompt").cast(pl.Categorical))


MAKERS: dict[str, Callable[[dict[str, list[Any]]], Any]] = {
    "pyarrow": as_pyarrow,
    "polars": as_polars,
    "duckdb": as_duckdb,
    "record_batch_reader": as_reader,
    "polars_categorical": as_polars_categorical,
}


def study_of(ratings: Any, trace: Any) -> ReliabilityStudy:
    return ReliabilityStudy(
        RatingFrame.from_long(ratings, **COLUMNS),
        measurement="ordinal",
        values=[1, 2, 3],
        coders="condition_id",
        design="prompt_robustness",
        missing_policies=["as_missing", "sensitivity"],
        abstain_as="missing",
        retries="last",
        trace=TraceSidecar.from_frame(trace, **TRACE_COLUMNS),
    )


@pytest.mark.parametrize("source", list(MAKERS))
def test_arrow_tables_give_the_same_ratings_as_pandas(source: str) -> None:
    make = MAKERS[source]
    expected_ratings = RatingFrame.from_long(pd.DataFrame(ratings_columns()), **COLUMNS)
    expected_trace = TraceSidecar.from_frame(pd.DataFrame(trace_columns()), **TRACE_COLUMNS)

    ratings = RatingFrame.from_long(make(ratings_columns()), **COLUMNS)
    trace = TraceSidecar.from_frame(make(trace_columns()), **TRACE_COLUMNS)

    assert ratings.fingerprint() == expected_ratings.fingerprint()
    assert trace.fingerprint() == expected_trace.fingerprint()
    observed = ratings.observed_values()
    assert observed == [1, 2, 3]
    assert all(type(v) is int for v in observed)
    assert type(trace.value("p1", "seed")) is int
    assert trace.value("p1", "seed") == 7
    assert trace.value("p2", "seed") is None
    assert trace.value("p2", "deterministic") is None
    assert trace.value("p1", "candidate_order") == ("A", "B")


@pytest.mark.parametrize("source", ["pyarrow", "polars", "duckdb"])
def test_a_study_from_arrow_tables_gives_the_same_alpha(source: str) -> None:
    make = MAKERS[source]
    expected = study_of(pd.DataFrame(ratings_columns()), pd.DataFrame(trace_columns())).alpha()
    result = study_of(make(ratings_columns()), make(trace_columns())).alpha()
    for name, estimate in expected.estimates.items():
        assert result[name].alpha == estimate.alpha
    assert expected.provenance is not None and result.provenance is not None
    assert result.provenance.ratings_fingerprint == expected.provenance.ratings_fingerprint


def test_unsigned_dictionary_indices_are_converted() -> None:
    pa = pytest.importorskip("pyarrow")
    columns = ratings_columns()
    prompts = ["p1", "p2", "p3"]
    codes = pa.array([prompts.index(p) for p in columns["prompt"]], type=pa.uint32())
    table = pa.table(columns).set_column(
        8, "prompt", pa.DictionaryArray.from_arrays(codes, pa.array(prompts))
    )
    expected = RatingFrame.from_long(pd.DataFrame(columns), **COLUMNS)
    assert RatingFrame.from_long(table, **COLUMNS).fingerprint() == expected.fingerprint()


def test_pandas_frames_do_not_go_through_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(purpose: str) -> Any:
        raise AssertionError(f"pyarrow was imported for {purpose}")

    monkeypatch.setattr(data_module, "_import_pyarrow", unavailable)
    RatingFrame.from_long(pd.DataFrame(ratings_columns()), **COLUMNS)
    TraceSidecar.from_frame(pd.DataFrame(trace_columns()), **TRACE_COLUMNS)


@pytest.mark.parametrize("bad", [[{"item": "u1"}], np.zeros((2, 2)), "ratings.csv"])
def test_other_inputs_are_refused(bad: Any) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame or an Arrow table"):
        RatingFrame.from_long(bad, unit="item", rater="judge", value="score")
    with pytest.raises(TypeError, match="pandas DataFrame or an Arrow table"):
        TraceSidecar.from_frame(bad, condition="prompt")


class StreamOnly:
    """Exports an Arrow stream, as a table from a library this test does not import."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        raise AssertionError("not reached without pyarrow")


def test_arrow_input_names_the_extra_when_pyarrow_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"reading a StreamOnly needs pyarrow.*\[parquet\]"):
        RatingFrame.from_long(StreamOnly(), unit="item", rater="judge", value="score")
