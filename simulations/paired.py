"""Does a non-inferiority test from ``compare()`` reject at its stated level?

Two configurations rate the same units: arm a with the scenario's accuracy, arm b
with ``accuracy_b``. Both arms see the same true categories, hard units and
clusters, as two judge configurations scoring one item set do, so their errors on
a unit are related through its difficulty.

* **Margin cells** lower arm b's accuracy, so the true difference
  Delta = alpha_b - alpha_a is negative. The margin is set to -Delta, which puts
  the truth exactly on the boundary of the null hypothesis. A level-2.5% test
  (the lower end of the 95% interval above -margin) should then reject in 2.5% of
  replications. Each replication is also compared at 96% (a level-2% test).
* **Equal cells** give both arms the same accuracy, so Delta = 0. The share of
  replications whose lower end clears -margin is the test's power for that margin.

The true alphas and Delta come from 2,000,000 units rated by both arms, in ten
independent chunks whose spread gives the Monte Carlo error, with failures
treated as missing (``as_missing``). In equal cells Delta is 0 by symmetry; the
simulated value is recorded as a check.
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
    TRUTH_UNITS,
    Scenario,
    Simulated,
    alpha_of,
    rate,
    simulate,
    study_of,
)
from simulations.grid import ORDINAL_PREVALENCE

METHODS = ("percentile", "bca")
#: Margins at which the equal cells report power. ("null" would read back as NaN in pandas.)
POWER_MARGINS = (0.02, 0.05, 0.10)
CONFIDENCES = (0.95, 0.96)
#: The true difference is a mean over this many chunks of TRUTH_UNITS units.
TRUTH_CHUNKS = 10


@dataclass(frozen=True)
class PairedScenario:
    """Arm a is ``base``; arm b is ``base`` with ``accuracy_b``."""

    base: Scenario
    accuracy_b: float

    @property
    def cell(self) -> str:
        return f"{self.base.name}-{self.kind}-n{self.base.n_units}"

    @property
    def kind(self) -> str:
        return "equal" if self.accuracy_b == self.base.accuracy else "margin"

    @property
    def arm_b(self) -> Scenario:
        return replace(self.base, accuracy=self.accuracy_b)


def rate_pair(
    ps: PairedScenario, n_units: int, rng: np.random.Generator
) -> tuple[Simulated, Simulated]:
    """Both arms' ratings of the same simulated units."""
    a = simulate(ps.base, n_units, rng)
    b = rate(ps.arm_b, a.truth, a.hard, a.cluster, rng)
    return a, b


def paired_truth(ps: PairedScenario, seed: int) -> dict[str, float]:
    """Population alphas of both arms and their difference, with Monte Carlo errors.

    Each chunk is an independent draw of TRUTH_UNITS units; the error is the
    standard error of the mean over the chunks.
    """
    alpha_a, alpha_b = [], []
    for chunk in range(TRUTH_CHUNKS):
        a, b = rate_pair(ps, TRUTH_UNITS, np.random.default_rng([seed, 1, chunk]))
        alpha_a.append(alpha_of(a.codes, ps.base))
        alpha_b.append(alpha_of(b.codes, ps.arm_b))
    delta = np.asarray(alpha_b) - np.asarray(alpha_a)
    simulated = float(delta.mean())
    return {
        "true_alpha_a": float(np.mean(alpha_a)),
        "true_alpha_b": float(np.mean(alpha_b)),
        "true_delta": 0.0 if ps.kind == "equal" else simulated,
        "simulated_delta": simulated,
        "true_delta_mc_error": float(delta.std(ddof=1) / np.sqrt(TRUTH_CHUNKS)),
    }


def replicate(
    ps: PairedScenario, seed: Sequence[int], truth: dict[str, float], n_resamples: int
) -> list[dict[str, Any]]:
    """One pair of rated data sets, compared by each method at each confidence.

    Every comparison of one replication uses the same resamples.
    """
    rng = np.random.default_rng(list(seed))
    a, b = rate_pair(ps, ps.base.n_units, rng)
    m = a.codes.shape[1]
    both = Simulated(
        np.hstack([a.codes, b.codes]),
        np.hstack([a.complete, b.complete]),
        a.cluster,
        a.hard,
        a.truth,
    )
    study = study_of(both, ps.base)
    arms = {"a": {"rater_id": list(range(m))}, "b": {"rater_id": list(range(m, 2 * m))}}
    margin = -truth["true_delta"] if ps.kind == "margin" else None
    test = {"margin": margin, "test": "non_inferiority"} if margin is not None else {}
    boot_seed = int(rng.integers(0, 2**63 - 1))
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in METHODS:
            for confidence in CONFIDENCES:
                records.append(
                    _compared(study, arms, method, confidence, n_resamples, boot_seed, test)
                )
    return records


def _compared(
    study: Any,
    arms: dict[str, Any],
    method: str,
    confidence: float,
    n_resamples: int,
    boot_seed: int,
    test: dict[str, Any],
) -> dict[str, Any]:
    head = {"method": method, "confidence": confidence}
    try:
        diff = study.compare(
            **arms,
            method=method,
            n_resamples=n_resamples,
            confidence=confidence,
            random_state=boot_seed,
            **test,
        )
    except UndefinedAlphaError:
        return {**head, "point_defined": False, "interval_defined": False}
    interval = diff["as_missing"]
    return {
        **head,
        "point_defined": True,
        "interval_defined": interval.defined and math.isfinite(interval.low),
        "point": interval.point,
        "low": interval.low,
        "high": interval.high,
        "failed_replicates": interval.n_failed,
        "verdict": diff.verdicts.get("as_missing"),
        "p_value": diff.p_values.get("as_missing"),
    }


@dataclass
class PairedCellResult:
    scenario: PairedScenario
    truth: dict[str, float]
    records: pd.DataFrame = field(repr=False)

    def summary(self) -> pd.DataFrame:
        """Size at the margin (margin cells) or power (equal cells), and coverage of Delta."""
        rows = []
        delta = self.truth["true_delta"]
        for (method, confidence), group in self.records.groupby(
            ["method", "confidence"], sort=False
        ):
            n = len(group)
            defined = group[group["interval_defined"].astype(bool)]
            k = len(defined)
            # An undefined interval gives an inconclusive verdict: it never rejects.
            low = group["low"].where(group["interval_defined"].astype(bool))
            covered = (defined["low"] <= delta) & (delta <= defined["high"])
            coverage = float(covered.mean()) if k else math.nan
            row: dict[str, Any] = {
                "cell": self.scenario.cell,
                "kind": self.scenario.kind,
                **{f"scenario_{key}": v for key, v in asdict(self.scenario.base).items()},
                "accuracy_b": self.scenario.accuracy_b,
                "method": method,
                "confidence": confidence,
                "n_sim": n,
                "n_defined": k,
                "undefined_rate": 1 - k / n if n else math.nan,
                "coverage": coverage,
                "coverage_mc_se": math.sqrt(coverage * (1 - coverage) / k) if k else math.nan,
                "miss_below": float((defined["high"] < delta).mean()) if k else math.nan,
                "miss_above": float((defined["low"] > delta).mean()) if k else math.nan,
                "mean_width": float((defined["high"] - defined["low"]).mean()),
                "bias": float((group["point"].dropna() - delta).mean()),
                "nominal_level": (1 - confidence) / 2,
                **self.truth,
            }
            if self.scenario.kind == "margin":
                rejected = (low > delta).fillna(False)
                rate_ = float(rejected.mean())
                row["margin"] = -delta
                row["rejection_rate"] = rate_
                row["rejection_mc_se"] = math.sqrt(rate_ * (1 - rate_) / n)
                verdicts = group["verdict"].where(group["interval_defined"].astype(bool))
                row["verdict_agrees"] = bool(((verdicts == "non_inferior") == rejected).all())
            for margin in POWER_MARGINS:
                key = f"power_{margin:g}"
                row[key] = (
                    float((low > -margin).fillna(False).mean())
                    if self.scenario.kind == "equal"
                    else math.nan
                )
            rows.append(row)
        return pd.DataFrame(rows)


def run_cell(
    ps: PairedScenario, n_sim: int, n_resamples: int, base_seed: int, index: int
) -> PairedCellResult:
    """Truth plus ``n_sim`` replications of one scenario (single process)."""
    truth = paired_truth(ps, seed=base_seed * 1000 + index)
    records = run_replications(ps, truth, range(n_sim), n_resamples, base_seed, index)
    return PairedCellResult(ps, truth, records)


def run_replications(
    ps: PairedScenario,
    truth: dict[str, float],
    replications: Sequence[int] | range,
    n_resamples: int,
    base_seed: int,
    cell_index: int,
) -> pd.DataFrame:
    rows = []
    for r in replications:
        for record in replicate(ps, (base_seed, cell_index, r), truth, n_resamples):
            rows.append({"cell": ps.cell, "replication": r, **record})
    return pd.DataFrame(rows)


#: Arm b's accuracy in the margin cells: about 0.05 below arm a's alpha.
ACCURACY_B = 0.765
SIZES = (50, 100, 200, 500)


def scenarios() -> list[PairedScenario]:
    """Nominal and ordinal at four sizes, at the margin and at Delta = 0, plus two factors."""
    bases = [Scenario("nominal", n) for n in SIZES]
    bases += [
        Scenario("ordinal", n, measurement="ordinal", prevalence=ORDINAL_PREVALENCE) for n in SIZES
    ]
    cells = [
        PairedScenario(s, accuracy_b) for s in bases for accuracy_b in (ACCURACY_B, s.accuracy)
    ]
    cells.append(
        PairedScenario(Scenario("clustered", 200, cluster_size=5, cohesion=0.7), ACCURACY_B)
    )
    cells.append(PairedScenario(Scenario("informative", 200, failures="informative"), ACCURACY_B))
    return cells
