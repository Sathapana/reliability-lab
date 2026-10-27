# Changelog

## Unreleased

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
