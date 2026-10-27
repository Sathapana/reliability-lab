"""Split before tuning, freeze, then report reliability on the untouched test units.

Run with ``uv run python examples/splits.py``. It builds a synthetic LLM judge log:
40 conversations of 3 responses each, rated by three models, with every fourth
conversation in a "hard" task family. Then it:

1. makes a 25% stratified group holdout and saves its manifest;
2. estimates alpha on the test partition with a group-level bootstrap;
3. writes JSON and Parquet reports;
4. reapplies the saved split to a later log (prompt v2) and checks few-shot leakage.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from reliability_lab import CoderSpec, LeakageError, RatingFrame, ReliabilityStudy


def judge_log(prompt: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(40):
        task = "hard" if g % 4 == 0 else "easy"
        for k in range(3):
            truth = int(rng.integers(1, 6))
            for model, family in (("judge-a", "fam-1"), ("judge-b", "fam-2"), ("judge-c", "fam-3")):
                accuracy = 0.6 if task == "hard" else 0.85
                score = truth if rng.random() < accuracy else int(rng.integers(1, 6))
                rows.append(
                    {
                        "response_id": f"conv{g}-resp{k}",
                        "conversation_id": f"conv{g}",
                        "task_family": task,
                        "model_id": model,
                        "model_family": family,
                        "condition_id": prompt,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


def study_of(df: pd.DataFrame) -> ReliabilityStudy:
    ratings = RatingFrame.from_long(
        df,
        unit="response_id",
        rater="model_id",
        value="score",
        group="conversation_id",
        stratum="task_family",
        model_id="model_id",
        model_family="model_family",
        condition_id="condition_id",
    )
    return ReliabilityStudy(
        ratings,
        measurement="ordinal",
        values=[1, 2, 3, 4, 5],
        coders=CoderSpec(by=["model_id"]),
        design="inter_model",
    )


out = Path(tempfile.mkdtemp(prefix="reliability-lab-"))
study = study_of(judge_log("prompt_v1", seed=1))

# 1. One fold of StratifiedGroupKFold(4) is an approximate 25% stratified group holdout.
dev, test = study.holdout(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7))
test.save_manifest(out / "split.json")
assert test.manifest is not None
print("realised split:", test.manifest.realised)

# ... tune the judge on `dev` only, then freeze it ...

# 2. Alpha on the test partition; the bootstrap resamples whole conversations.
result = test.alpha()
boot = result.bootstrap(scheme="units", n_resamples=2000, random_state=7)
print()
print(boot.summary())

# 3. Reports: strict JSON with provenance, and tidy Parquet tables.
result.to_json(out / "alpha_test.json")
boot.to_parquet(out / "bootstrap_test")
print("\nprovenance gaps:", *result.provenance.gaps if result.provenance else [], sep="\n  - ")

# 4. The same split, by its saved ids, on a later log with a new prompt.
later = study_of(judge_log("prompt_v2", seed=2))
_, later_test = later.apply_manifest(out / "split.json")
print(f"\nprompt_v2 alpha on the same test units: {later_test.alpha().alpha:.3f}")

few_shot = sorted(set(dev.ratings.to_frame()["unit_id"]))[:3]
test.manifest.assert_no_leakage(few_shot)
try:
    test.manifest.assert_no_leakage([*few_shot, sorted(set(test.ratings.to_frame()["unit_id"]))[0]])
except LeakageError as err:
    print(f"\nleakage caught: {err}")
print(f"\nfiles written to {out}")
