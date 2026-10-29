"""Coverage when coders differ: the units, coders and two-way bootstraps, and the jackknife.

Coders are drawn from a pool whose accuracies spread around the scenario's
accuracy (``coder_spread``, tau). When a coder guesses, it leans towards a
coder-specific favourite. With tau = 0 every coder is alike, as in P5.

Each scheme answers its own question, so each is scored against its own estimand:

* **coders fixed** (units scheme): this replication's coders on new questions.
  Computed on 20,000 fresh questions rated by the same coders.
* **questions fixed** (coders scheme, jackknife): these questions with new coders
  from the pool. Computed on the same questions rated by 2,000 fresh coders.
* **both random** (two-way scheme): new coders on new questions. One value per
  scenario, from 200,000 questions each rated by its own fresh coders, so the
  value is the population's and not that of one finite pool.

Every scheme is also scored against the other two estimands. That shows what an
interval misses when it is read as a claim it was not built for.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from reliability_lab import UndefinedAlphaError
from simulations.coverage import (
    CoderPanel,
    Scenario,
    alpha_of,
    draw_panel,
    rate,
    simulate,
    study_of,
)

FRESH_CODERS = 2_000
FIXED_CODERS_UNITS = 20_000
BOTH_RANDOM_UNITS = 200_000
SCHEMES = ("units", "coders", "two_way", "jackknife")
ESTIMANDS = ("coders_fixed", "questions_fixed", "both_random")
#: Which estimand each scheme is built for.
OWN_ESTIMAND = {
    "units": "coders_fixed",
    "coders": "questions_fixed",
    "two_way": "both_random",
    "jackknife": "questions_fixed",
}


@dataclass(frozen=True)
class CoderScenario:
    """A P5 scenario plus the spread of coder accuracy in the pool."""

    base: Scenario
    coder_spread: float

    @property
    def cell(self) -> str:
        return (
            f"{self.base.name}-tau{self.coder_spread:g}-m{self.base.n_coders}-n{self.base.n_units}"
        )


def both_random_truth(cs: CoderScenario, seed: int) -> dict[str, float]:
    """Population alpha over coders and questions, with its Monte Carlo error.

    Every question gets its own coders, drawn afresh, so no finite pool's quirks
    leak into the value. It does not depend on the number of coders per question.
    """
    s = cs.base
    rng = np.random.default_rng([seed, 3])
    halves = []
    for _ in range(2):
        n = BOTH_RANDOM_UNITS // 2
        fresh = draw_panel(s, n * s.n_coders, cs.coder_spread, rng)
        panel = CoderPanel(
            fresh.accuracy.reshape(n, s.n_coders),
            fresh.favourite.reshape(n, s.n_coders),
            fresh.tilt,
        )
        halves.append(alpha_of(simulate(s, n, rng, panel).codes, s))
    return {
        "both_random": float(np.mean(halves)),
        "both_random_mc_error": abs(halves[0] - halves[1]) / 2,
    }


def replicate(
    cs: CoderScenario, seed: Sequence[int], truth: dict[str, float], n_resamples: int
) -> list[dict[str, Any]]:
    """One panel of coders rating one set of questions: one record per scheme."""
    s = cs.base
    seed = list(seed)
    panel = draw_panel(s, s.n_coders, cs.coder_spread, np.random.default_rng([*seed, 2]))
    sim = simulate(s, s.n_units, np.random.default_rng(seed), panel)
    targets = {
        "coders_fixed": alpha_of(
            simulate(s, FIXED_CODERS_UNITS, np.random.default_rng([*seed, 4]), panel).codes, s
        ),
        "questions_fixed": alpha_of(
            rate(
                s,
                sim.truth,
                sim.hard,
                sim.cluster,
                np.random.default_rng([*seed, 6]),
                draw_panel(s, FRESH_CODERS, cs.coder_spread, np.random.default_rng([*seed, 5])),
            ).codes,
            s,
        ),
        "both_random": truth["both_random"],
    }
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = study_of(sim, s).alpha()
        except UndefinedAlphaError:
            return [{"scheme": scheme, "interval_defined": False, **targets} for scheme in SCHEMES]
        boot_seed = int(np.random.default_rng([*seed, 7]).integers(0, 2**63 - 1))
        intervals = {
            scheme: result.bootstrap(
                scheme=scheme,  # type: ignore[arg-type]
                level="unit" if scheme == "units" else None,  # type: ignore[arg-type]
                n_resamples=n_resamples,
                random_state=boot_seed,
            )["as_missing"]
            for scheme in ("units", "coders", "two_way")
        }
        intervals["jackknife"] = result.coder_jackknife()["as_missing"]
    for scheme, interval in intervals.items():
        defined = interval.defined and math.isfinite(interval.low)
        records.append(
            {
                "scheme": scheme,
                "interval_defined": defined,
                "point": result.alpha,
                "low": interval.low,
                "high": interval.high,
                "failed_replicates": interval.n_failed,
                **targets,
            }
        )
    return records


def run_replications(
    cs: CoderScenario,
    truth: dict[str, float],
    replications: Sequence[int] | range,
    n_resamples: int,
    base_seed: int,
    cell_index: int,
) -> pd.DataFrame:
    rows = []
    for r in replications:
        for record in replicate(cs, (base_seed, cell_index, r), truth, n_resamples):
            rows.append({"cell": cs.cell, "replication": r, **record})
    return pd.DataFrame(rows)


@dataclass
class CoderCellResult:
    scenario: CoderScenario
    truth: dict[str, float]
    records: pd.DataFrame = field(repr=False)

    def summary(self) -> pd.DataFrame:
        rows = []
        for scheme, group in self.records.groupby("scheme", sort=False):
            defined = group[group["interval_defined"].astype(bool)]
            k, n = len(defined), len(group)
            row: dict[str, Any] = {
                "cell": self.scenario.cell,
                "coder_spread": self.scenario.coder_spread,
                **{f"scenario_{key}": v for key, v in asdict(self.scenario.base).items()},
                "scheme": scheme,
                "own_estimand": OWN_ESTIMAND[str(scheme)],
                "n_sim": n,
                "n_defined": k,
                "undefined_rate": 1 - k / n if n else math.nan,
                "mean_width": float((defined["high"] - defined["low"]).mean()),
                "median_width": float((defined["high"] - defined["low"]).median()),
                **self.truth,
            }
            for estimand in ESTIMANDS:
                target = defined[estimand]
                covered = (defined["low"] <= target) & (target <= defined["high"])
                row[f"coverage_{estimand}"] = float(covered.mean()) if k else math.nan
                row[f"mean_{estimand}"] = float(group[estimand].mean())
            own = row[f"coverage_{OWN_ESTIMAND[str(scheme)]}"]
            row["coverage_own"] = own
            row["coverage_own_mc_se"] = math.sqrt(own * (1 - own) / k) if k else math.nan
            rows.append(row)
        return pd.DataFrame(rows)


def run_cell(
    cs: CoderScenario, n_sim: int, n_resamples: int, base_seed: int, index: int
) -> CoderCellResult:
    """Truth plus ``n_sim`` replications of one scenario (single process)."""
    truth = both_random_truth(cs, seed=base_seed * 1000 + index)
    records = run_replications(cs, truth, range(n_sim), n_resamples, base_seed, index)
    return CoderCellResult(cs, truth, records)


def scenarios() -> list[CoderScenario]:
    """tau x coders x questions on the nominal baseline, plus one ordinal row."""
    cells = [
        CoderScenario(Scenario("nominal", n, n_coders=m), tau)
        for tau in (0.0, 0.1, 0.2)
        for m in (5, 10, 20)
        for n in (50, 100)
    ]
    ordinal = Scenario(
        "ordinal", 100, measurement="ordinal", prevalence=(0.15, 0.2, 0.3, 0.2, 0.15), n_coders=10
    )
    cells.append(CoderScenario(ordinal, 0.2))
    return cells


def with_units(cs: CoderScenario, n_units: int) -> CoderScenario:
    """The same scenario with a different number of questions (used in tests)."""
    return CoderScenario(replace(cs.base, n_units=n_units), cs.coder_spread)
