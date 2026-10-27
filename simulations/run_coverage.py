"""Run the coverage grid in parallel and write the results.

    uv run python -m simulations.run_coverage                # full grid
    uv run python -m simulations.run_coverage --quick        # 50 replications, a smoke run
    uv run python -m simulations.run_coverage --cells clustered --n-sim 500

Writes ``coverage.csv`` (one row per scenario and scheme) and ``run_info.json``
to the output directory, plus ``replications.parquet`` (one row per replication
and scheme; not committed).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

import reliability_lab
from simulations.coverage import CellResult, Scenario, run_replications, true_alpha
from simulations.grid import scenarios

DEFAULT_SEED = 20261027
RESULTS = Path(__file__).parent / "results"


def _truth(args: tuple[Scenario, int, int]) -> dict[str, float]:
    s, seed, index = args
    return true_alpha(s, seed=seed * 1000 + index)


def _chunk(args: tuple[Scenario, dict[str, float], range, int, int, int]) -> pd.DataFrame:
    s, truth, reps, n_resamples, seed, index = args
    return run_replications(s, truth, reps, n_resamples, seed, index)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-sim", type=int, default=2000, help="replications per scenario")
    parser.add_argument("--n-resamples", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cells", default="", help="only scenarios whose cell id contains this")
    parser.add_argument("--chunk", type=int, default=250, help="replications per task")
    parser.add_argument("--quick", action="store_true", help="50 replications, 500 resamples")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.quick:
        args.n_sim, args.n_resamples = 50, 500
    out = args.out or (RESULTS / "quick" if args.quick else RESULTS)
    out.mkdir(parents=True, exist_ok=True)

    indexed = [(i, s) for i, s in enumerate(scenarios()) if args.cells in s.cell]
    started = time.perf_counter()
    print(
        f"{len(indexed)} scenarios x {args.n_sim} replications, {args.workers} workers", flush=True
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        truths = dict(
            zip(
                [i for i, _ in indexed],
                pool.map(_truth, [(s, args.seed, i) for i, s in indexed]),
                strict=True,
            )
        )
        print(f"truths done in {time.perf_counter() - started:.0f}s", flush=True)
        tasks = [
            (s, truths[i], range(start, min(start + args.chunk, args.n_sim)), args.n_resamples,
             args.seed, i)
            for i, s in indexed
            for start in range(0, args.n_sim, args.chunk)
        ]  # fmt: skip
        futures = [pool.submit(_chunk, task) for task in tasks]
        parts = []
        for done, future in enumerate(as_completed(futures), 1):
            parts.append(future.result())
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"{done}/{len(tasks)} tasks, {elapsed:.0f}s", flush=True)

    records = pd.concat(parts, ignore_index=True)
    summary = pd.concat(
        [
            CellResult(s, truths[i], records[records["cell"] == s.cell]).summary()
            for i, s in indexed
        ],
        ignore_index=True,
    )
    summary.to_csv(out / "coverage.csv", index=False, float_format="%.6g")
    records.to_parquet(out / "replications.parquet", index=False)
    info: dict[str, Any] = {
        "seed": args.seed,
        "n_sim": args.n_sim,
        "n_resamples": args.n_resamples,
        "confidence": 0.95,
        "workers": args.workers,
        "scenarios": len(indexed),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "versions": {
            "reliability_lab": reliability_lab.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {out} in {info['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
