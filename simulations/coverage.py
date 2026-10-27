"""How often does each bootstrap scheme's interval contain the true alpha?

A scenario fixes a data-generating process. Ratings follow a latent-class
"know or guess" model: each unit has a true category, and each coder reports it
with probability ``accuracy``. Otherwise the coder guesses: a draw from the
prevalences (nominal), or a shift of one or two scale points (ordinal). Factors:

* about 30% of units are hard, with accuracy lowered by 0.25. With clusters,
  hardness is shared by every unit of a cluster, and each unit takes its
  cluster's topic (category) with probability ``cohesion``, so units of one
  cluster are dependent, as turns of one conversation are;
* with probability ``rho``, a coder who guesses copies the unit's shared wrong
  value: correlated errors, as between models of one family;
* failed attempts are missing completely at random, or informative (much more
  likely on hard units). They enter as ``parse_error`` rows.

The true alpha is the estimator's population value: alpha on 200,000 units from
the same process, with failures treated as missing (``as_missing``). The
complete-data alpha (before failures) is computed too; the gap between the two is
the bias from dropping failed attempts, which no interval can repair.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from reliability_lab import RatingFrame, ReliabilityStudy, UndefinedAlphaError
from reliability_lab.alpha import NOMINAL, ORDINAL, compute_alpha

NOMINAL_VALUES = ("a", "b", "c")
ORDINAL_VALUES = (1, 2, 3, 4, 5)
#: Ordinal guesses shift the true value by -2, -1, +1 or +2 with these probabilities.
ORDINAL_SHIFTS = (np.array([-2, -1, 1, 2]), np.array([0.15, 0.35, 0.35, 0.15]))
HARD_SHARE = 0.3
HARD_PENALTY = 0.25
FAILURE_RATES = {"none": (0.0, 0.0), "mcar": (0.2, 0.2), "informative": (0.05, 0.35)}
TRUTH_UNITS = 200_000
#: (label, scheme, level, method); the group level is used only with clusters.
SCHEMES: tuple[tuple[str, str, str | None, str], ...] = (
    ("units", "units", "unit", "percentile"),
    ("units_group", "units", "group", "percentile"),
    ("kalpha", "kalpha", None, "percentile"),
    ("bca", "units", None, "bca"),
)


@dataclass(frozen=True)
class Scenario:
    """One cell of the simulation grid."""

    name: str
    n_units: int
    measurement: str = "nominal"
    prevalence: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    n_coders: int = 3
    accuracy: float = 0.8
    rho: float = 0.0
    failures: str = "none"
    cluster_size: int = 1
    cohesion: float = 0.0

    @property
    def cell(self) -> str:
        return f"{self.name}-n{self.n_units}"

    @property
    def values(self) -> tuple[Any, ...]:
        return NOMINAL_VALUES if self.measurement == "nominal" else ORDINAL_VALUES

    @property
    def clustered(self) -> bool:
        return self.cluster_size > 1

    def schemes(self) -> list[tuple[str, str, str | None, str]]:
        return [s for s in SCHEMES if self.clustered or s[0] != "units_group"]


@dataclass
class Simulated:
    """Integer-coded ratings: units x coders, with -1 where an attempt failed."""

    codes: np.ndarray
    complete: np.ndarray
    cluster: np.ndarray
    hard: np.ndarray


def simulate(s: Scenario, n_units: int, rng: np.random.Generator) -> Simulated:
    m, n_values = s.n_coders, len(s.values)
    cluster = np.arange(n_units) // s.cluster_size
    n_clusters = int(cluster[-1]) + 1
    hard = (rng.random(n_clusters) < HARD_SHARE)[cluster]
    prevalence = np.asarray(s.prevalence)
    topic = rng.choice(n_values, size=n_clusters, p=prevalence)[cluster]
    own_truth = rng.choice(n_values, size=n_units, p=prevalence)
    truth = np.where(rng.random(n_units) < s.cohesion, topic, own_truth)
    accuracy = np.where(hard, s.accuracy - HARD_PENALTY, s.accuracy)

    def guesses(size: tuple[int, ...]) -> np.ndarray:
        if s.measurement == "nominal":
            return rng.choice(n_values, size=size, p=np.asarray(s.prevalence))
        shifts, p = ORDINAL_SHIFTS
        base = np.broadcast_to(truth[:, None] if len(size) == 2 else truth, size)
        return np.clip(base + rng.choice(shifts, size=size, p=p), 0, n_values - 1)

    knows = rng.random((n_units, m)) < accuracy[:, None]
    own = guesses((n_units, m))
    shared = guesses((n_units,))
    copy = rng.random((n_units, m)) < s.rho
    complete = np.where(knows, truth[:, None], np.where(copy, shared[:, None], own))

    easy_rate, hard_rate = FAILURE_RATES[s.failures]
    failed = rng.random((n_units, m)) < np.where(hard, hard_rate, easy_rate)[:, None]
    codes = np.where(failed, -1, complete)
    return Simulated(codes, complete, cluster, hard)


def alpha_of(codes: np.ndarray, s: Scenario) -> float:
    """Alpha of integer codes (units x coders, -1 = missing), straight from the engine."""
    units, coders = np.nonzero(codes >= 0)
    distance = NOMINAL if s.measurement == "nominal" else ORDINAL
    result = compute_alpha(units, codes[units, coders], len(codes), s.values, distance)
    return float(result.components.alpha)


def true_alpha(s: Scenario, seed: int) -> dict[str, float]:
    """Population alpha (as_missing) and complete-data alpha, with Monte Carlo errors.

    Each is computed on two independent halves of TRUTH_UNITS units; the truth is
    their mean and its error half their difference (the standard error of a mean
    of two).
    """
    rng = np.random.default_rng([seed, 1])
    halves = [simulate(s, TRUTH_UNITS // 2, rng) for _ in range(2)]
    missing = [alpha_of(h.codes, s) for h in halves]
    complete = [alpha_of(h.complete, s) for h in halves]
    return {
        "true_alpha": float(np.mean(missing)),
        "true_alpha_mc_error": abs(missing[0] - missing[1]) / 2,
        "complete_alpha": float(np.mean(complete)),
        "complete_alpha_mc_error": abs(complete[0] - complete[1]) / 2,
    }


def study_of(sim: Simulated, s: Scenario) -> ReliabilityStudy:
    """The simulated ratings as a declared reliability-lab study."""
    n_units, m = sim.codes.shape
    unit = np.repeat(np.arange(n_units), m)
    coder = np.tile(np.arange(m), n_units)
    flat = sim.codes.ravel()
    values = np.asarray(s.values, dtype=object)
    frame = pd.DataFrame(
        {
            "unit": unit,
            "coder": coder,
            "value": np.where(flat >= 0, values[np.maximum(flat, 0)], None),
            "status": np.where(flat >= 0, "observed", "parse_error"),
            "cluster": sim.cluster[unit],
        }
    )
    kwargs = {"group": "cluster"} if s.clustered else {}
    ratings = RatingFrame.from_long(
        frame, unit="unit", rater="coder", value="value", status="status", **kwargs
    )
    if s.failures == "none":
        policies = ["as_missing"]
    elif s.measurement == "nominal":
        policies = ["as_missing", "as_category"]
    else:
        policies = ["as_missing", "sensitivity"]
    return ReliabilityStudy(
        ratings,
        measurement=s.measurement,
        values=list(s.values),
        coders="rater_id",
        design="human_panel",
        missing_policies=policies,  # type: ignore[arg-type]
    )


def replicate(
    s: Scenario, seed: Sequence[int], truth: dict[str, float], n_resamples: int
) -> list[dict[str, Any]]:
    """One simulated data set, bootstrapped by every scheme: one record per scheme."""
    rng = np.random.default_rng(list(seed))
    sim = simulate(s, s.n_units, rng)
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = study_of(sim, s).alpha()
        except UndefinedAlphaError:
            return [
                {"scheme": label, "point_defined": False, "interval_defined": False}
                for label, *_ in s.schemes()
            ]
        points = [e.alpha for e in result.estimates.values() if e.defined]
        sensitivity_covers = (
            min(points) <= truth["complete_alpha"] <= max(points) if len(points) > 1 else None
        )
        boot_seed = int(rng.integers(0, 2**63 - 1))
        for label, scheme, level, method in s.schemes():
            boot = result.bootstrap(
                scheme=scheme,  # type: ignore[arg-type]
                level=level,  # type: ignore[arg-type]
                method=method,  # type: ignore[arg-type]
                n_resamples=n_resamples,
                random_state=boot_seed,
            )
            interval = boot["as_missing"]
            defined = interval.defined and math.isfinite(interval.low)
            records.append(
                {
                    "scheme": label,
                    "point_defined": True,
                    "interval_defined": defined,
                    "point": result.alpha,
                    "low": interval.low,
                    "high": interval.high,
                    "failed_replicates": interval.n_failed,
                    "sensitivity_covers_complete": sensitivity_covers,
                }
            )
    return records


@dataclass
class CellResult:
    scenario: Scenario
    truth: dict[str, float]
    records: pd.DataFrame = field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Coverage and interval metrics per scheme."""
        rows = []
        truth = self.truth["true_alpha"]
        for label, group in self.records.groupby("scheme", sort=False):
            n = len(group)
            defined = group[group["interval_defined"].astype(bool)]
            k = len(defined)
            covered = (defined["low"] <= truth) & (truth <= defined["high"])
            coverage = float(covered.mean()) if k else math.nan
            sens = group["sensitivity_covers_complete"].dropna()
            rows.append(
                {
                    "cell": self.scenario.cell,
                    **{f"scenario_{k_}": v for k_, v in asdict(self.scenario).items()},
                    "scheme": label,
                    "n_sim": n,
                    "n_defined": k,
                    "undefined_rate": 1 - k / n if n else math.nan,
                    "coverage": coverage,
                    "coverage_mc_se": math.sqrt(coverage * (1 - coverage) / k) if k else math.nan,
                    "miss_below": float((defined["high"] < truth).mean()) if k else math.nan,
                    "miss_above": float((defined["low"] > truth).mean()) if k else math.nan,
                    "mean_width": float((defined["high"] - defined["low"]).mean()),
                    "median_width": float((defined["high"] - defined["low"]).median()),
                    "bias": float((group["point"].dropna() - truth).mean()),
                    "mean_failed_replicates": float(defined["failed_replicates"].mean()),
                    "sensitivity_covers_complete": float(sens.astype(float).mean())
                    if len(sens)
                    else math.nan,
                    **self.truth,
                }
            )
        return pd.DataFrame(rows)


def run_cell(
    s: Scenario, n_sim: int, n_resamples: int, base_seed: int, cell_index: int
) -> CellResult:
    """Truth plus ``n_sim`` replications of one scenario (single process)."""
    truth = true_alpha(s, seed=base_seed * 1000 + cell_index)
    records = run_replications(s, truth, range(n_sim), n_resamples, base_seed, cell_index)
    return CellResult(s, truth, records)


def run_replications(
    s: Scenario,
    truth: dict[str, float],
    replications: Sequence[int] | range,
    n_resamples: int,
    base_seed: int,
    cell_index: int,
) -> pd.DataFrame:
    """Records for the given replication indices; seeds depend only on the indices."""
    rows = []
    for r in replications:
        for record in replicate(s, (base_seed, cell_index, r), truth, n_resamples):
            rows.append({"cell": s.cell, "replication": r, **record})
    return pd.DataFrame(rows)
