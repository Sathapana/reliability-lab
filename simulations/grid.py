"""The simulation grid: a baseline swept over sample size, then one factor at a time."""

from __future__ import annotations

from simulations.coverage import Scenario

ORDINAL_PREVALENCE = (0.15, 0.2, 0.3, 0.2, 0.15)

#: Baseline: nominal, 3 balanced categories, 3 coders, accuracy 0.8 (30% hard units).
BASELINE: dict[str, object] = {}

#: Factor variations, each at two sample sizes.
VARIATIONS: dict[str, dict[str, object]] = {
    "skewed": {"prevalence": (0.8, 0.1, 0.1)},
    "coders2": {"n_coders": 2},
    "coders5": {"n_coders": 5},
    "accuracy60": {"accuracy": 0.6},
    "accuracy95": {"accuracy": 0.95},
    "mcar": {"failures": "mcar"},
    "informative": {"failures": "informative"},
    "correlated": {"rho": 0.5},
}


def scenarios() -> list[Scenario]:
    cells = [Scenario("baseline", n) for n in (20, 50, 100, 200)]
    cells += [
        Scenario("ordinal", n, measurement="ordinal", prevalence=ORDINAL_PREVALENCE)
        for n in (20, 50, 100, 200)
    ]
    for name, factor in VARIATIONS.items():
        cells += [Scenario(name, n, **factor) for n in (30, 100)]  # type: ignore[arg-type]
    cells += [Scenario("clustered", n, cluster_size=5, cohesion=0.7) for n in (100, 200)]
    return cells
