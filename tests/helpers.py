"""Builders shared by the tests."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from reliability_lab import Distance, RatingFrame, ReliabilityStudy


def long_from_matrix(matrix: Mapping[Any, Sequence[Any]]) -> pd.DataFrame:
    """Coders x units matrix (``None`` = missing) to long form, dropping missing cells."""
    rows = [
        {"unit": u, "coder": coder, "value": v}
        for coder, values in matrix.items()
        for u, v in enumerate(values)
        if v is not None
    ]
    return pd.DataFrame(rows, columns=["unit", "coder", "value"])


def human_study(
    matrix: Mapping[Any, Sequence[Any]],
    level: str | Distance,
    values: Sequence[Any],
    **kwargs: Any,
) -> ReliabilityStudy:
    ratings = RatingFrame.from_long(
        long_from_matrix(matrix), unit="unit", rater="coder", value="value"
    )
    return ReliabilityStudy(
        ratings,
        measurement=level,
        values=values,
        coders="rater_id",
        design="human_panel",
        **kwargs,
    )


def random_matrix(
    rng: np.random.Generator,
    n_coders: int,
    n_units: int,
    values: Sequence[Any],
    p_missing: float,
) -> dict[Any, list[Any]]:
    """Random ratings with missing cells; each unit leans towards one value."""
    matrix: dict[Any, list[Any]] = {f"c{i}": [] for i in range(n_coders)}
    for _ in range(n_units):
        anchor = int(rng.integers(len(values)))
        for coder in matrix:
            if rng.random() < p_missing:
                matrix[coder].append(None)
            elif rng.random() < 0.6:
                matrix[coder].append(values[anchor])
            else:
                matrix[coder].append(values[int(rng.integers(len(values)))])
    return matrix
