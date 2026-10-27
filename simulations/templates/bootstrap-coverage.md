# Bootstrap coverage: which interval to trust

reliability-lab requires you to name a bootstrap scheme; there is no default. This page measures how often each scheme's 95% interval actually contains the true alpha, so you can choose one knowing what it does. The tables, charts and every number quoted in the text come from the committed run in `simulations/results/`.

## Summary

- **Use `scheme="units"`.** It resamples whole units, or whole groups when a `group` is declared. Percentile intervals cover 93–95% across the scenarios with independent units. The exceptions are small samples with a dominant category (about 90%) and clustered data resampled unit by unit (89–91%).
- **`method="bca"` is the most accurate interval measured.** It covers between 93.8% and 97.0% in every scenario except 20 clusters (92.9%), and its misses are balanced on both sides. On this evidence it is no longer flagged experimental. It fails to produce an interval in some small, skewed samples: 18% of runs at n = 30 with an 80% category. The library warns when that happens, and `method="percentile"` still gives an interval from the defined replicates.
- **KALPHA is too narrow whenever a unit holds more than one pair of values.** It covers 85–86% with three coders on nominal data, 77–80% on a 5-point ordinal scale, and 70–71% with five coders. With two coders, where each unit holds one pair, it covers 92–94%. Use it to reproduce published KALPHA intervals, not for new claims.
- **Declare groups when units share a source.** Resampling units of clustered data one by one covers 89–91%. Resampling groups covers 91.8% with 20 clusters and 94.5% with 40.
- **No interval repairs informative failures.** When failed attempts concentrate on hard units, dropping them raises alpha by about 0.03. The intervals cover that inflated value correctly, but not the complete-data alpha. Report the missing-data policies side by side, and read their spread as sensitivity to the policy, not as bounds.

## Setup

**Data.** Each scenario generates ratings from a latent-class "know or guess" model:
- Every unit has a true category. Each coder reports it with probability `accuracy`, otherwise guesses. A guess is a draw from the prevalences (nominal), or a shift of one or two scale points (ordinal, 5 points).
- About 30% of units are hard, with accuracy lowered by 0.25.

Variations:
- skewed prevalence;
- 2 or 5 coders instead of 3;
- lower or higher accuracy;
- errors shared between coders (a coder who guesses copies the unit's shared wrong value with probability 0.5, as models of one family might);
- 20% failed attempts, at random or concentrated on hard units;
- clusters of 5 units, which share hardness and, with probability 0.7, a topic.

**Truth.** The true alpha of a scenario is the estimator's population value: alpha on 200,000 units from the same process, with failures treated as missing. The complete-data alpha (before failures) is computed the same way.

**Schemes.** Every replication is bootstrapped with:
- `units` percentile at unit level;
- `units` percentile at group level (clustered scenarios);
- `kalpha`;
- `units` with BCa at the default level (groups when declared).

All use 2,000 resamples and 95% intervals.

**Run.** {{run_info}}

**How to read.** Coverage is the share of replications whose interval contains the true alpha; 95% is the target. Below 93% means the intervals are too narrow to support a 95% claim. "Below / above" splits the misses: an interval entirely below the truth, or entirely above it. Width is the price of coverage: a narrow interval that misses is not a better interval.

## Sample size and level of measurement

{{figure:coverage_by_n}}

{{table:sample_size}}

- **Units, percentile and BCa:** both stay within about two points of 95% from 20 to 200 units, on nominal and ordinal data.
- **Small samples:** at n = 20–50, the percentile intervals miss more often below the truth than above (4–5% against 1–2% on nominal data). Small-sample alpha is biased low (−0.012 at n = 20), and BCa corrects for that.
- **KALPHA's widths:** its intervals are about a quarter narrower on nominal data and about 40% narrower on ordinal data, and coverage falls accordingly. Ordinal distances depend on the category marginals, which KALPHA holds fixed along with D_e.

## One factor at a time

{{figure:coverage_by_factor}}

{{table:factors}}

- **Number of coders** decides KALPHA's fate. With 2 coders it covers 92–94%; with 5 coders, 70–71%. KALPHA resamples pairs of values as if they were independent. A unit rated by m coders contributes m(m − 1)/2 pairs, and those pairs share the unit's difficulty and true value. More coders mean more dependent pairs and a falsely narrow interval. Resampling units keeps each unit's pairs together, and its coverage does not depend on the number of coders.
- **Skewed prevalence** is the hardest case for small samples. Percentile covers 90.5% at n = 30, missing mostly below. BCa covers 94.9% when it produces an interval, but fails in 18% of runs: a resample can contain only the dominant category, which leaves alpha undefined. The library counts those replicates and raises a `BootstrapWarning` naming the estimate left without an interval; nothing is dropped silently.
- **Accuracy, random failures and correlated errors** move the alpha level, from 0.29 to 0.78. They leave the units scheme's coverage at 93–95%.

## Clustered units

{{table:clusters}}

When units share a source (turns of one conversation, responses to one prompt), resampling units one by one treats dependent units as independent. Coverage falls to 89.5% with 20 clusters and 91.3% with 40. Resampling whole groups, the default whenever a `group` is declared, brings it to 91.8% and 94.5%. With few clusters, even group resampling under-covers somewhat, and BCa at group level (92.9% with 20 clusters) does slightly better. KALPHA cannot see clusters at all, and covers about 80%.

## Failed attempts

{{table:failures}}

- **Failures at random** leave alpha where it was, within 0.001 of the complete-data value. Every scheme behaves as it does without failures.
- **Informative failures** are the LLM case: parse and provider errors concentrated on hard items.
  - Dropping them removes exactly the units where coders disagree, and alpha rises from 0.539 to 0.570–0.573.
  - The units intervals still cover the dropped-failures value, at 93.7–95.0%. But that value is not the complete-data reliability, and the 0.03 gap does not shrink with more units.
- **The spread of missing-data policies is not a bound.** Treating failures as their own category (`as_category`) gives a second, different estimate. The interval between the policies contained the complete-data alpha in only 48–66% of runs. Report the policies and the error rates side by side, and read their spread as how much the policy choice matters.

## Which scheme to use

| Situation | Use | Why |
| --- | --- | --- |
| Units independent, n of about 50 or more | `scheme="units"` (percentile) | 93–95% coverage; simplest |
| Small sample or a dominant category | `scheme="units", method="bca"` | corrects the low-alpha bias; if BCa warns that it has no interval, use percentile |
| Units share a source | declare `group`; `scheme="units"` resamples groups | unit resampling under-covers (89–91%) |
| Fewer than about 30 clusters | group resampling, read as approximate | 92% coverage with 20 clusters |
| Reproducing a published KALPHA interval | `scheme="kalpha"` | matches Krippendorff's algorithm; too narrow for new claims with 3 or more coders |
| Comparing two configurations | `study.compare()` | units scheme on identical resamples (KALPHA has no paired form) |

## Limitations

- **One family of processes.** These scenarios use one family of data-generating processes (know or guess, with difficulty), up to 5 coders, up to 200 units, nominal and 5-point ordinal data, and 95% intervals. Interval and ratio metrics were not simulated.
- **Truth uncertainty.** The true alphas carry a Monte Carlo error of at most 0.0025, small next to interval widths of 0.07 and more.
- **BCa's undefined cases are real.** At n = 30 with an 80% category, one run in six produces no BCa interval.
- **KALPHA's second pick.** KALPHA's result depends on reading its algorithm note as evalica does: a unit's second pick is redrawn when it repeats the first. Its coverage problem does not come from that detail. It comes from treating dependent pairs as independent.

## Reproduce

```bash
uv run python -m simulations.run_coverage --workers 8     # about 20 minutes on 8 cores
uv run --group sim python -m simulations.report           # tables, figures and this page
```

The seed and versions are recorded in `simulations/results/run_info.json`. Each replication's seed depends only on its scenario and index, so a rerun reproduces every number.
