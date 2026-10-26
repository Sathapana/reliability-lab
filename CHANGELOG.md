# Changelog

## Unreleased

- Project scaffold: packaging, BSD-3-Clause licence, CI on Linux, Windows and macOS.
- Core engine: `RatingFrame` long-form schema with rating status, `CoderSpec` and design labels, Krippendorff's alpha (nominal, ordinal, interval, ratio, custom distances) with white-box diagnostics, and missing-data policies (`as_missing`, `as_category`, `sensitivity`).
- Bootstrap inference: `AlphaResult.bootstrap(scheme=...)` with a required scheme.
  - `units` resamples units, or groups, optionally within strata, and recomputes D_e per replicate.
  - `kalpha` follows Krippendorff's algorithm, resampling pairs of values with D_e fixed.
  - Percentile intervals are the default; BCa is opt-in and flagged experimental.
  - Failed replicates are counted and reported, the seed is always recorded, and an optional `alpha_min` reports how many replicates fall below it.
- Paired comparisons: `ReliabilityStudy.compare(a=..., b=...)` places the interval on Δα = α_b − α_a, using the same resampled units for both configurations.
- `BootstrapWarning`; the scipy floor is now 1.15.
