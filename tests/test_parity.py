"""Parity with independent alpha implementations (test-only dependencies).

``irrCAC`` is not in the locked environment because it pins scipy==1.12.0; its
test is skipped unless the package is importable (CI runs it in an isolated job).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fixtures.krippendorff2011 import EXAMPLES
from helpers import human_study, random_matrix

pytestmark = [
    pytest.mark.parity,
    pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning"),
]

LEVELS = ["nominal", "ordinal", "interval", "ratio"]
VALUES = [1, 2, 3, 4, 5, 6]


def random_matrices(seed: int, count: int = 20) -> list[dict[Any, list[Any]]]:
    rng = np.random.default_rng(seed)
    return [
        random_matrix(rng, int(rng.integers(2, 7)), int(rng.integers(5, 60)), VALUES, 0.3)
        for _ in range(count)
    ]


def as_array(matrix: Mapping[Any, Sequence[Any]]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in matrix.values()], float)


def ours(matrix: Mapping[Any, Sequence[Any]], level: str, values: Sequence[Any]) -> float:
    return human_study(matrix, level, values).alpha().alpha


@pytest.mark.parametrize("level", LEVELS)
def test_krippendorff_package(level: str) -> None:
    krippendorff = pytest.importorskip("krippendorff")
    for matrix in random_matrices(11):
        expected = krippendorff.alpha(
            reliability_data=as_array(matrix), level_of_measurement=level, value_domain=VALUES
        )
        assert ours(matrix, level, VALUES) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("level", LEVELS)
def test_krippendorff_package_on_numeric_fixture(level: str) -> None:
    krippendorff = pytest.importorskip("krippendorff")
    example = EXAMPLES["four_coders_missing"]
    expected = krippendorff.alpha(
        reliability_data=as_array(example["matrix"]),
        level_of_measurement=level,
        value_domain=example["values"],
    )
    assert ours(example["matrix"], level, example["values"]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("level", ["nominal", "interval"])
def test_nltk_annotation_task(level: str) -> None:
    agreement = pytest.importorskip("nltk.metrics.agreement")
    distance = pytest.importorskip("nltk.metrics.distance")
    fn = distance.binary_distance if level == "nominal" else distance.interval_distance
    for matrix in random_matrices(12, count=10):
        triples = [
            (str(coder), str(u), v)
            for coder, row in matrix.items()
            for u, v in enumerate(row)
            if v is not None
        ]
        expected = agreement.AnnotationTask(data=triples, distance=fn).alpha()
        assert ours(matrix, level, VALUES) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("level", LEVELS)
def test_evalica(level: str) -> None:
    evalica = pytest.importorskip("evalica")
    for matrix in random_matrices(13):
        expected = evalica.alpha(pd.DataFrame(as_array(matrix)), distance=level).alpha
        assert ours(matrix, level, VALUES) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(("level", "weights"), [("nominal", "identity"), ("interval", "quadratic")])
def test_irrcac(level: str, weights: str) -> None:
    raw = pytest.importorskip("irrCAC.raw")
    for matrix in random_matrices(14, count=10):
        units_by_raters = pd.DataFrame(as_array(matrix).T)
        cac = raw.CAC(units_by_raters, weights=weights, categories=VALUES)
        expected = cac.krippendorff()["est"]["coefficient_value"]
        # irrCAC rounds its estimates to five decimals.
        assert ours(matrix, level, VALUES) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("level", "weights"),
    [
        ("nominal", "identity"),
        ("interval", "quadratic"),
        ("ordinal", "ordinal"),
        ("interval", "linear"),
        ("interval", "radical"),
        ("ratio", "ratio"),
        ("bipolar", "bipolar"),
        ("interval", "circular"),
    ],
)
def test_irrcac_gwet(level: str, weights: str) -> None:
    raw = pytest.importorskip("irrCAC.raw")
    for matrix in random_matrices(15, count=10):
        units_by_raters = pd.DataFrame(as_array(matrix).T)
        cac = raw.CAC(units_by_raters, weights=weights, categories=VALUES)
        est = cac.gwet()["est"]
        study = human_study(matrix, level, VALUES)
        res = study.gwet(weights=weights)
        primary = res["as_missing"]
        assert res.coefficient_name == est["coefficient_name"]
        assert res.ac == pytest.approx(est["coefficient_value"], abs=1e-5)
        assert primary.pa == pytest.approx(est["pa"], abs=1e-5)
        assert primary.pe == pytest.approx(est["pe"], abs=1e-5)
        assert primary.se == pytest.approx(est["se"], abs=1e-5)
        assert primary.z == pytest.approx(est["z"], abs=1e-5)
        assert primary.p_value == pytest.approx(est["p_value"], abs=1e-5)
        assert primary.low == pytest.approx(est["confidence_interval"][0], abs=1e-5)
        assert primary.high == pytest.approx(est["confidence_interval"][1], abs=1e-5)
        if (level, weights) in (
            ("nominal", "identity"),
            ("interval", "quadratic"),
            ("ratio", "ratio"),
            ("bipolar", "bipolar"),
        ):
            res_default = study.gwet()
            assert res_default.ac == pytest.approx(est["coefficient_value"], abs=1e-5)
