import json
import math

import numpy as np
import pytest
from fixtures.krippendorff2011 import FOUR_CODERS_MISSING
from helpers import human_study
from reference import reference_alpha
from scipy import stats

from reliability_lab import BootstrapWarning, DesignError

F4 = FOUR_CODERS_MISSING

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")


def test_jackknife_matches_a_hand_computation() -> None:
    result = human_study(F4["matrix"], "interval", F4["values"]).alpha()
    jk = result.coder_jackknife()
    coders = list(F4["matrix"])
    without = np.array(
        [
            reference_alpha(
                {c: v for c, v in F4["matrix"].items() if c != left_out}, "interval", F4["values"]
            )
            for left_out in coders
        ]
    )
    m = len(coders)
    se = math.sqrt((m - 1) / m * np.sum((without - without.mean()) ** 2))
    assert jk.standard_errors["as_missing"] == pytest.approx(se)
    t = stats.t.ppf(0.975, m - 1)
    interval = jk["as_missing"]
    assert interval.low == pytest.approx(result.alpha - t * se)
    assert interval.high == pytest.approx(min(result.alpha + t * se, 1.0))
    table = jk.influence[jk.influence["estimate"] == "as_missing"].set_index("coder")
    for coder, expected in zip(coders, without, strict=True):
        assert table.loc[coder, "alpha_without"] == pytest.approx(expected)
        assert table.loc[coder, "influence"] == pytest.approx(result.alpha - expected)
    assert table.loc["A", "units_rated"] == 9


def test_influence_flags_an_outlier_coder() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 4, 40)
    matrix = {
        f"c{i}": [int(t) if rng.random() < 0.9 else int(rng.integers(1, 4)) for t in truth]
        for i in range(4)
    }
    matrix["outlier"] = [int(x) for x in rng.integers(1, 4, 40)]
    jk = human_study(matrix, "nominal", [1, 2, 3]).alpha().coder_jackknife()
    table = jk.influence[jk.influence["estimate"] == "as_missing"].set_index("coder")
    assert table["influence"].idxmin() == "outlier"
    assert table.loc["outlier", "alpha_without"] > table.drop("outlier")["alpha_without"].max()
    assert "outlier" in jk.summary()


def test_jackknife_needs_three_coders() -> None:
    result = human_study({"a": [1, 2, 1], "b": [1, 2, 2]}, "nominal", [1, 2]).alpha()
    with pytest.raises(DesignError, match="at least 3"):
        result.coder_jackknife()


@pytest.mark.filterwarnings("ignore:no interval for as_category")
def test_undefined_leave_one_out_alpha_is_reported() -> None:
    # Only coder c ever uses value 2: without c, every value is 1 and alpha is undefined.
    matrix = {"a": [1, 1, 1, 1], "b": [1, 1, 1, 1], "c": [1, 2, 1, 2]}
    result = human_study(matrix, "nominal", [1, 2]).alpha()
    with pytest.warns(BootstrapWarning, match="no interval for as_missing"):
        jk = result.coder_jackknife()
    interval = jk["as_missing"]
    assert not interval.defined
    assert "without coders ['c']" in (interval.undefined_reason or "")
    assert math.isnan(jk.standard_errors["as_missing"])


def test_jackknife_serialises(tmp_path: object) -> None:
    jk = human_study(F4["matrix"], "ordinal", F4["values"]).alpha().coder_jackknife()
    payload = json.loads(jk.to_json())
    assert payload["schema"] == "reliability-lab/jackknife-result"
    assert payload["n_coders"] == 4
    assert payload["intervals"]["as_missing"]["standard_error"] > 0
    assert len(payload["influence"]) == 4 * len(jk.intervals)
    assert set(jk.tables()) == {"intervals", "influence"}
    assert "standard_error" in jk.to_frame().columns
