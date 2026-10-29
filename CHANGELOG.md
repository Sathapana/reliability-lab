# Changelog

## Unreleased

## 0.1.5

Simulated ratings.

- `simulate_ratings` makes synthetic coder ratings for any measurement level (nominal, ordinal, interval, ratio, bipolar, `circular(U)` or a custom `Distance`) at a chosen alpha. Errors are random draws from the population or near misses by the metric's own distance, with optional missing cells and `parse_error` failures. The ratings come as a `RatingFrame`, ready for `ReliabilityStudy`.
- A [roadmap](docs/design/roadmap.md) for 0.2 to 1.0. Mixed human-LLM panels move from 0.2 to 1.0, and the messages refusing them say so.

## 0.1.4

All six of Krippendorff's difference functions.

- Two more built-in distances from Krippendorff (2011):
  - `measurement="bipolar"` is for scales with two poles, such as −3…+3. Its poles are the smallest and largest declared values.
  - `measurement=circular(U)` is for values on a circle of circumference U: hours (24), degrees (360), months (12). U must be declared, so the string `"circular"` is refused.

## 0.1.3

Faster bootstrap.

- Every scheme is faster. With 2,000 resamples of 5,000 units and 5 coders (ordinal, 3 estimates):
  - units, percentile: 1.4 s → 0.5 s;
  - units, BCa: 6.4 s → 0.5 s, and `compare(method="bca")`: 18 s → 0.5 s;
  - kalpha: 5.1 s → 2.8 s;
  - coders: 8.2 s → 1.4 s, and two_way: 10 s → 2.1 s.
- BCa intervals no longer go through `scipy.stats.bootstrap`. The jackknife leaves each cluster out in closed form, instead of recomputing alpha from k × k resample weights, so BCa now runs on large studies: 20,000 units take 3.7 s, where they previously ran out of memory. The resamples and the interval are still the ones `scipy.stats.bootstrap` gives, and the tests check this.
- The random draws are unchanged: a given `random_state` gives the same intervals as in 0.1.2, up to floating-point rounding.
- The README links to the docs and examples on GitHub, so the links work on PyPI. The project links on PyPI include the changelog.

## 0.1.2

Generalising over coders.

- Two new bootstrap schemes:
  - `scheme="coders"` resamples coders and holds the units fixed;
  - `scheme="two_way"` resamples units (or groups, within strata) and coders independently.
- When coders are resampled, the copies of one coder are never paired with each other, so a duplicated coder does not agree with itself. With no coder drawn twice, the coincidences are Krippendorff's.
- Both schemes need at least 3 coders, give percentile intervals only (BCa raises a `DesignError`), and are not available in `compare()`.
- `AlphaResult.coder_jackknife()` gives a `JackknifeResult`:
  - alpha without each coder, and each coder's influence;
  - jackknife standard errors;
  - t intervals with m − 1 degrees of freedom;
  - the same JSON and Parquet reports as the other results.
- Every bootstrap result says which dimension it generalises over: `describe()` names what was resampled ("coders fixed" for the units scheme), and a note states the claim. `BootstrapResult.n_coders` records the number of coders resampled.
- Coverage simulations with coders who differ (`python -m simulations.run_coverage --grid coders`). Each scheme is scored against the claim it is built for: the same coders on new questions, the same questions with new coders, or both new. The guidance gains a section on generalising over coders.

## 0.1.1

Bootstrap coverage guidance.

- Bootstrap coverage simulations (`simulations/`):
  - a seeded latent-class data process with difficulty, correlated errors, random or informative failures, and clustered units;
  - true alphas computed from 200,000 units;
  - coverage, width and failure rates of every scheme across 26 scenarios.
- Guidance from those simulations: [docs/guidance/bootstrap-coverage.md](docs/guidance/bootstrap-coverage.md).
- BCa intervals are no longer flagged experimental. They covered 93.8–97.0% in 25 of 26 scenarios (92.9% with 20 clusters). `BootstrapResult.experimental` stays in the result format and is now always false.
- A `BootstrapWarning` now names each estimate left without an interval and, for BCa, points to `method="percentile"`. Undefined replicates are still counted and warned about.

## 0.1.0

LLM judge workflows.

- `TraceSidecar`: one row per judge configuration (`condition_id`). It records provider, prompt hash, rubric and parser versions, temperature, seed, determinism, candidate order, request time and retry policy.
- Each study checks its trace against its design. Configurations that differ in more than the design compares raise a `TraceWarning`; missing conditions or fields become provenance gaps. Comparisons report the model and trace fields that changed between the arms (`configuration_diff`).
- The single-configuration designs (repeatability, prompt robustness, order robustness) can hold several judge configurations for `compare()`. Each configuration is checked on its own, and `alpha()` refuses to pool them.
- `order_robustness` design for pairwise judges whose candidates are shown in swapped orders.
- Diagnostics on every result (`AlphaResult.diagnostics`):
  - `coder_agreement` (pairwise exact agreement);
  - `run_agreement` (repeatability);
  - `order_swap` (reversal rate, and position bias with a Wilson interval).
- Retries: a new `attempt` field in `RatingFrame.from_long`. Retried calls must be declared with `retries="last"` or `retries="first_observed"`. Attempts are grouped by call (unit, rater, model, condition, run), so repeated runs are never treated as retries. Coverage reports attempt-level and final error rates.
- Provenance records the trace rows of the conditions in use and the retries declaration.

## 0.1.0a1

First public release.

- Project scaffold: packaging, BSD-3-Clause licence, CI on Linux, Windows and macOS.
- Core engine: `RatingFrame` long-form schema with rating status, `CoderSpec` and design labels, Krippendorff's alpha (nominal, ordinal, interval, ratio, custom distances) with white-box diagnostics, and missing-data policies (`as_missing`, `as_category`, `sensitivity`).
- Bootstrap inference: `AlphaResult.bootstrap(scheme=...)` with a required scheme.
  - `units` resamples units, or groups, optionally within strata, and recomputes D_e per replicate.
  - `kalpha` follows Krippendorff's algorithm, resampling pairs of values with D_e fixed.
  - Percentile intervals are the default; BCa is opt-in and flagged experimental.
  - Failed replicates are counted and reported, the seed is always recorded, and an optional `alpha_min` reports how many replicates fall below it.
- Paired comparisons: `ReliabilityStudy.compare(a=..., b=...)` places the interval on Δα = α_b − α_a, using the same resampled units for both configurations.
- Group-safe splits:
  - `ReliabilityStudy.holdout()` and `cv()` take any scikit-learn style splitter. The splitter sees one sample per group, so groups and repeated LLM calls never cross partitions.
  - When a group's units span several strata, `group_stratum` must be given.
  - `SplitManifest` saves partition ids, the splitter, realised sizes and fingerprints; `apply_manifest()` reapplies them to a later log.
  - `assert_no_leakage()` checks that few-shot examples come from the dev partition.
- Reports:
  - versioned JSON (`to_json`) with provenance for every result type: software versions, ratings fingerprint, declarations, split and known gaps;
  - tidy tables (`tables`) and Parquet output (`to_parquet`, with the `parquet` extra).
- Tag-triggered release workflow using PyPI trusted publishing.
- New exceptions and warnings: `BootstrapWarning`, `LeakageError`. The scipy floor is now 1.15.
