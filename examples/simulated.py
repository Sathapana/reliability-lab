"""Simulated coders for every built-in measurement level.

Run with ``uv run python examples/simulated.py``. Each level gets 300 units rated by
4 coders at a target alpha of 0.8, once with random errors and once with near
misses. The alphas printed vary around 0.8 from draw to draw.
"""

from reliability_lab import ReliabilityStudy, circular, simulate_ratings

LEVELS = [
    ("nominal", ["cat", "dog", "bird", "fish"]),
    ("ordinal", [1, 2, 3, 4, 5]),
    ("interval", [1, 2, 3, 4, 5, 6, 7]),
    ("ratio", [0, 1, 2, 5, 10, 20]),
    ("bipolar", [-3, -2, -1, 0, 1, 2, 3]),
    (circular(24), list(range(24))),  # hours of the day
]

print(f"{'measurement':<14}{'random':>8}{'near':>8}")
for measurement, values in LEVELS:
    alphas = []
    for errors in ("random", "near"):
        ratings = simulate_ratings(
            measurement,
            values,
            n_units=300,
            n_coders=4,
            alpha=0.8,
            errors=errors,
            random_state=2026,
        )
        study = ReliabilityStudy(
            ratings, measurement=measurement, values=values, coders="rater_id", design="human_panel"
        )
        alphas.append(study.alpha().alpha)
    name = measurement if isinstance(measurement, str) else measurement.name
    print(f"{name:<14}{alphas[0]:>8.3f}{alphas[1]:>8.3f}")

print()
ratings = simulate_ratings("ordinal", [1, 2, 3, 4, 5], n_units=5, alpha=0.8, random_state=1)
print("The first rows of a simulated RatingFrame:")
print(ratings.to_frame()[["unit_id", "rater_id", "value", "status"]].head(9))
