"""Reproduce Krippendorff's (2011) four-observer example through the public API.

Run with ``uv run python examples/quickstart.py``. The published values are
nominal 0.743, ordinal 0.815, interval 0.849 and ratio 0.797.
"""

import pandas as pd

from reliability_lab import CoderSpec, RatingFrame, ReliabilityStudy

# Observers x units; None marks a missing rating.
MATRIX = {
    "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
}

# One row per rating attempt. A missing cell is an attempt that was never made,
# so it has no row; failed attempts would get a row with an error status.
long = pd.DataFrame(
    [
        {"unit": f"u{u + 1}", "observer": observer, "score": value}
        for observer, row in MATRIX.items()
        for u, value in enumerate(row)
        if value is not None
    ]
)
ratings = RatingFrame.from_long(long, unit="unit", rater="observer", value="score")

for level in ("nominal", "ordinal", "interval", "ratio"):
    study = ReliabilityStudy(
        ratings,
        measurement=level,
        values=[1, 2, 3, 4, 5],
        coders=CoderSpec(by="rater_id"),
        design="human_panel",
    )
    print(f"{level:<9} alpha = {study.alpha().alpha:.3f}")

print()
result = ReliabilityStudy(
    ratings,
    measurement="ordinal",
    values=[1, 2, 3, 4, 5],
    coders=CoderSpec(by="rater_id"),
    design="human_panel",
).alpha()
print(result.summary())
print("\nCoincidence matrix (ordinal):")
print(result["as_missing"].coincidence.round(3))
print("\nUnits ranked by disagreement:")
print(result["as_missing"].per_unit.sort_values("disagreement", ascending=False).head(5))

# Two bootstrap schemes, always named. With 12 units the intervals are wide.
for scheme in ("units", "kalpha"):
    print()
    print(result.bootstrap(scheme=scheme, n_resamples=2000, random_state=7).summary())
