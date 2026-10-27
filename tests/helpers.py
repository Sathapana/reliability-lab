"""Builders shared by the tests."""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from reliability_lab import RatingFrame, ReliabilityStudy


def long_from_matrix(matrix: dict[Hashable, Sequence[Any]]) -> pd.DataFrame:
    """Coders x units matrix (``None`` = missing) to long form, dropping missing cells."""
    rows = [
        {"unit": u, "coder": coder, "value": v}
        for coder, values in matrix.items()
        for u, v in enumerate(values)
        if v is not None
    ]
    return pd.DataFrame(rows, columns=["unit", "coder", "value"])


def human_study(
    matrix: dict[Hashable, Sequence[Any]], level: str, values: Sequence[Any], **kwargs: Any
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
) -> dict[Hashable, list[Any]]:
    """Random ratings with missing cells; each unit leans towards one value."""
    matrix: dict[Hashable, list[Any]] = {f"c{i}": [] for i in range(n_coders)}
    for _ in range(n_units):
        anchor = rng.integers(len(values))
        for coder in matrix:
            if rng.random() < p_missing:
                matrix[coder].append(None)
            elif rng.random() < 0.6:
                matrix[coder].append(values[anchor])
            else:
                matrix[coder].append(values[rng.integers(len(values))])
    return matrix
