"""LLM judge workflows on synthetic logs: repeatability with retries, and an order swap.

Run with ``uv run python examples/llm_judge.py``. No provider is called: the logs
are generated here, with effects planted so you can see the library recover them.
"""

import warnings

import numpy as np
import pandas as pd

from reliability_lab import RatingFrame, ReliabilityStudy, TraceSidecar

rng = np.random.default_rng(0)

# -- 1. Repeatability: one configuration, three runs, retried parse errors ------------

rows = []
for u in range(60):
    truth = int(rng.integers(1, 6))
    for run in (1, 2, 3):
        call = {"unit": f"u{u}", "model": "judge-x", "condition": "cfg-1", "run": run}
        score = truth if rng.random() < 0.8 else int(np.clip(truth + rng.choice([-1, 1]), 1, 5))
        if rng.random() < 0.15:  # the first attempt fails to parse, and the retry succeeds
            rows.append({**call, "attempt": 1, "status": "parse_error", "score": None})
            rows.append({**call, "attempt": 2, "status": "observed", "score": score})
        else:
            rows.append({**call, "attempt": 1, "status": "observed", "score": score})
log = pd.DataFrame(rows)
configs = pd.DataFrame(
    {"condition": ["cfg-1"], "prompt": ["h-111"], "temperature": [0.7], "parser": ["1.2"]}
)

trace = TraceSidecar.from_frame(
    configs, condition="condition", prompt_hash="prompt", temperature="temperature",
    parser_version="parser",
)  # fmt: skip
ratings = RatingFrame.from_long(
    log, unit="unit", rater="model", value="score", status="status", model_id="model",
    condition_id="condition", run_id="run", attempt="attempt",
)  # fmt: skip
study = ReliabilityStudy(
    ratings,
    measurement="ordinal",
    values=[1, 2, 3, 4, 5],
    coders="run_id",
    design="repeatability",
    trace=trace,
    retries="last",  # retried calls must be declared: which attempt counts
)
result = study.alpha()
print(result.summary())
print(result.coverage.per_coder[["attempts", "ratings", "attempt_error_rate", "error_rate"]])

# -- 2. Order robustness: the same pairwise judge, candidates shown in both orders -----

rows = []
for i in range(200):
    winner = "A" if rng.random() < 0.5 else "B"
    for condition, order in (("ab", ("A", "B")), ("ba", ("B", "A"))):
        # Planted position bias: 40% of the time the judge picks whichever came first.
        label = order[0] if rng.random() < 0.4 else winner
        rows.append({"item": f"i{i}", "model": "judge-x", "condition": condition, "label": label})
pairs = pd.DataFrame(rows)
orders = pd.DataFrame(
    {"condition": ["ab", "ba"], "order": ["A,B", "B,A"], "prompt": ["h-pair", "h-pair"]}
)

trace = TraceSidecar.from_frame(
    orders, condition="condition", candidate_order="order", prompt_hash="prompt"
)
ratings = RatingFrame.from_long(
    pairs, unit="item", rater="model", value="label", model_id="model", condition_id="condition"
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # a skew warning is expected: few ties
    result = ReliabilityStudy(
        ratings,
        measurement="nominal",
        values=["A", "tie", "B"],  # labels name the winning candidate, not its position
        coders="condition_id",
        design="order_robustness",
        trace=trace,
    ).alpha()
print()
print(result.summary())
print(result.diagnostics["order_swap"].table)
