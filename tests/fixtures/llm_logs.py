"""Seeded, provider-neutral LLM judge logs with trace sidecars, for end-to-end tests.

Each generator returns ``(log, conditions)``: a long-form ratings table and a
table of judge configurations keyed by ``condition``. Effects are planted, so the
tests can check that the library recovers them.
"""

import numpy as np
import pandas as pd

SCORES = [1, 2, 3, 4, 5]


def _noisy(rng: np.random.Generator, truth: int, accuracy: float) -> int:
    """The true score with probability ``accuracy``, otherwise a neighbouring score."""
    if rng.random() < accuracy:
        return truth
    return int(np.clip(truth + rng.choice([-1, 1]), 1, 5))


def repeatability_log(
    n_units: int = 60,
    runs: int = 3,
    seed: int = 0,
    configs: tuple[tuple[str, float, float], ...] = (("cfg-1", 0.7, 0.8),),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One judge called ``runs`` times per unit under each configuration.

    ``configs`` lists (condition, temperature, accuracy). The default is a single
    configuration at temperature 0.7. About 15% of calls fail to parse on the first
    attempt and succeed on a retry; about 3% fail on both attempts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        truth = int(rng.integers(1, 6))
        for condition, _, accuracy in configs:
            for run in range(1, runs + 1):
                base = {
                    "unit": f"u{u}",
                    "conversation": f"c{u // 3}",
                    "model": "judge-x",
                    "family": "fam-x",
                    "condition": condition,
                    "run": run,
                }
                draw = rng.random()
                if draw < 0.03:
                    rows.append({**base, "attempt": 1, "status": "parse_error", "score": None})
                    rows.append({**base, "attempt": 2, "status": "parse_error", "score": None})
                elif draw < 0.18:
                    rows.append({**base, "attempt": 1, "status": "parse_error", "score": None})
                    score = _noisy(rng, truth, accuracy)
                    rows.append({**base, "attempt": 2, "status": "observed", "score": score})
                else:
                    score = _noisy(rng, truth, accuracy)
                    rows.append({**base, "attempt": 1, "status": "observed", "score": score})
    conditions = pd.DataFrame(
        [
            {
                "condition": condition,
                "provider": "acme",
                "prompt_hash": "h-111",
                "rubric_version": "r1",
                "temperature": temperature,
                "parser_version": "1.2",
                "retry_policy": "retry once on parse error",
            }
            for condition, temperature, _ in configs
        ]
    )
    return pd.DataFrame(rows), conditions


def inter_model_log(n_units: int = 80, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three models. m-b and m-c share a family, and m-c ran with a different prompt."""
    rng = np.random.default_rng(seed)
    models = [
        ("m-a", "fam-1", "cfg-a", 0.85),
        ("m-b", "fam-2", "cfg-b", 0.75),
        ("m-c", "fam-2", "cfg-c", 0.7),
    ]
    rows = []
    for u in range(n_units):
        truth = int(rng.integers(1, 6))
        for model, family, condition, accuracy in models:
            rows.append(
                {
                    "unit": f"u{u}",
                    "model": model,
                    "family": family,
                    "condition": condition,
                    "status": "observed",
                    "score": _noisy(rng, truth, accuracy),
                }
            )
    conditions = pd.DataFrame(
        {
            "condition": ["cfg-a", "cfg-b", "cfg-c"],
            "prompt_hash": ["h-1", "h-1", "h-2"],
            "temperature": [0.0, 0.0, 0.0],
            "parser_version": ["1.2", "1.2", "1.2"],
        }
    )
    return pd.DataFrame(rows), conditions


def configuration_change_log(n_units: int = 80, seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three models judged with prompt v1, then with a better prompt v2, on the same units."""
    rng = np.random.default_rng(seed)
    models = [("m-a", "fam-1"), ("m-b", "fam-2"), ("m-c", "fam-3")]
    accuracy = {"v1": 0.55, "v2": 0.9}
    rows = []
    for u in range(n_units):
        truth = int(rng.integers(1, 6))
        for prompt in ("v1", "v2"):
            for model, family in models:
                rows.append(
                    {
                        "unit": f"u{u}",
                        "model": model,
                        "family": family,
                        "condition": prompt,
                        "status": "observed",
                        "score": _noisy(rng, truth, accuracy[prompt]),
                    }
                )
    conditions = pd.DataFrame(
        {
            "condition": ["v1", "v2"],
            "prompt_hash": ["h-v1", "h-v2"],
            "temperature": [0.0, 0.0],
            "parser_version": ["1.2", "1.2"],
        }
    )
    return pd.DataFrame(rows), conditions


def order_swap_log(
    n_items: int = 200, seed: int = 3, judges: tuple[tuple[str, float], ...] = (("judge-x", 0.4),)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise judges shown each item in both orders, with planted position biases.

    ``judges`` lists (model, bias). A judge picks the candidate shown first with
    probability ``bias`` regardless of merit, and otherwise the true winner; 5% of
    its answers are ties. With the default bias of 0.4, the candidate shown first
    wins 0.4 + 0.6 * 0.5 = 0.7 of the decisive answers.
    """
    rng = np.random.default_rng(seed)
    orders = {"ab": ("A", "B"), "ba": ("B", "A")}
    rows = []
    for i in range(n_items):
        winner = "A" if rng.random() < 0.5 else "B"
        for model, bias in judges:
            for condition, order in orders.items():
                if rng.random() < 0.05:
                    label = "tie"
                elif rng.random() < bias:
                    label = order[0]
                else:
                    label = winner
                rows.append(
                    {
                        "item": f"i{i}",
                        "model": model,
                        "family": f"fam-{model[-1]}",
                        "condition": condition,
                        "status": "observed",
                        "label": label,
                    }
                )
    conditions = pd.DataFrame(
        {
            "condition": ["ab", "ba"],
            "prompt_hash": ["h-pair", "h-pair"],
            "temperature": [0.0, 0.0],
            "parser_version": ["2.0", "2.0"],
            "order": ["A,B", "B,A"],
        }
    )
    return pd.DataFrame(rows), conditions
