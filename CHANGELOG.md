# Changelog

## Unreleased

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
