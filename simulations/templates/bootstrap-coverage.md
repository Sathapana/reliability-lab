# Bootstrap coverage: which interval to trust

reliability-lab requires you to name a bootstrap scheme; there is no default. This page measures how often each scheme's 95% interval actually contains the true alpha, so you can choose one knowing what it does. It also measures how often the non-inferiority verdict of `compare()` is wrong. The tables, charts and every number quoted in the text come from the committed runs in `simulations/results/`, `simulations/results/coders/` and `simulations/results/paired/`.

## Summary

- **Decide whose reliability you are claiming.** The units interval is about the coders in the study.
  - When coders differ persistently, it covers new coders on new questions in as few as 48% of runs.
  - For a claim about coders like these, use `scheme="two_way"` with 10 or more coders: it covers 92–98% when coders differ, and errs on the wide side when they do not.
  - For these questions with other coders, use `coder_jackknife()`: it covers 94–96%.
  - The `coders` bootstrap is too narrow (79–92%).
  - Repeated runs of one judge configuration do not differ persistently, so the units interval covers them. See [Generalising over coders](#generalising-over-coders).
- **Use `scheme="units"` for the coders you have.** It resamples whole units, or whole groups when a `group` is declared. Percentile intervals cover 93–95% across the scenarios with independent units. The exceptions are small samples with a dominant category (about 90%) and clustered data resampled unit by unit (89–91%).
- **`method="bca"` is the most accurate interval measured.** It covers between 93.8% and 97.0% in every scenario except 20 clusters (92.9%), and its misses are balanced on both sides. On this evidence it is no longer flagged experimental. It fails to produce an interval in some small, skewed samples: 18% of runs at n = 30 with an 80% category. The library warns when that happens, and `method="percentile"` still gives an interval from the defined replicates.
- **KALPHA is too narrow whenever a unit holds more than one pair of values.** It covers 85–86% with three coders on nominal data, 77–80% on a 5-point ordinal scale, and 70–71% with five coders. With two coders, where each unit holds one pair, it covers 92–94%. Use it to reproduce published KALPHA intervals, not for new claims.
- **Declare groups when units share a source.** Resampling units of clustered data one by one covers 89–91%. Resampling groups covers 91.8% with 20 clusters and 94.5% with 40.
- **No interval repairs informative failures.** When failed attempts concentrate on hard units, dropping them raises alpha by about 0.03. The intervals cover that inflated value correctly, but not the complete-data alpha. Report the missing-data policies side by side, and read their spread as sensitivity to the policy, not as bounds.
- **For a non-inferiority test that holds 2.5%, use `confidence=0.96`.** At the default 95%, `compare()` wrongly declares non-inferiority in 2.9% of runs at the margin, and up to 3.75% in one scenario. At 96% it does so in 2.4%, and at most 3.15%. Fix the margin first: at a margin of 0.05, even 500 units give 31–67% power when the configurations are equally reliable. See [Comparing two configurations](#comparing-two-configurations).

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

## Generalising over coders

In every scenario above, the coders are alike: they share one accuracy, so the only randomness is in which questions were drawn and in each rating. Real coders often differ persistently: one annotator is stricter, one model leans towards a label, one prompt is more careful. Then one study supports three different claims, and each needs its own interval.

| Claim | Interval | Resampled |
| --- | --- | --- |
| These coders, on new questions | `scheme="units"` | questions (or groups) |
| These questions, with new coders like these | `result.coder_jackknife()`, or `scheme="coders"` | coders |
| New coders like these, on new questions | `scheme="two_way"` | questions and coders |

Repeated runs of one judge configuration are the exception. Run 3 and run 7 differ only by chance, with no persistent difference between them, so they behave like the alike coders above. The units interval then already covers new runs.

**Setup.**
- **Coders differ by a spread τ.** Each coder's accuracy is 0.8 + τ·Z, with Z standard normal, clipped to [0.05, 1].
- **Each coder has a favourite.** When a coder guesses, with probability min(0.5, 2τ) it picks its own favourite category (nominal) or leans one point in its own direction (ordinal).
- **Grid.** τ = 0 reproduces the alike coders above; at τ = 0.2, the middle 95% of accuracies runs from about 0.4 to 1. The grid crosses τ ∈ {0, 0.1, 0.2} with 5, 10 or 20 coders and 50 or 100 questions on the nominal baseline. It adds one 5-point ordinal scenario with τ = 0.2, 10 coders and 100 questions.

Each replication draws a panel of coders and a set of questions. Every interval is scored against the claim it is built for, computed fresh for that replication:
- **Same coders, new questions.** Alpha of this panel on 20,000 new questions.
- **Same questions, new coders.** Alpha of these questions rated by 2,000 new coders.
- **New coders, new questions.** The population alpha. It is computed once per scenario, from 200,000 questions each rated by its own new coders.

**Run.** {{run_info:coders}}

{{figure:coverage_by_spread}}

{{table:coders}}

Each cell gives the coverage of the interval's own claim, with the mean width in brackets. The "units: new coders" column scores the units interval against a claim it is not built for.

- **The units interval is about these coders.** It covers its own claim at 93–95% in every scenario.
  - Read as a claim about new coders, it holds only when coders are alike: 94–95% at τ = 0.
  - Once coders differ, it covers 70–83% at τ = 0.1 and 48–68% at τ = 0.2.
  - More questions make this worse, not better. The interval narrows around this panel's alpha, but the panel's own distance from the population does not shrink. More coders help only slowly.
- **`two_way` covers new coders on new questions.**
  - With 10 or 20 coders it covers 92–98%. With 5 coders at τ = 0.2 it covers 88–91%.
  - When coders are alike, it is conservative (99.3–99.8%) and about 1.5 times as wide as the units interval. It counts the rating noise twice, once through the questions and once through the coders, as pigeonhole bootstraps of crossed designs do (Owen 2007).
- **The coder jackknife covers these questions with new coders.**
  - It covers 93.7–96.2% in every scenario, including 5 coders.
  - With 5 coders its t interval, with 4 degrees of freedom, is wide: 0.61–0.65 at τ = 0.2, against 0.41–0.48 for `two_way`.
  - When coders differ strongly, the coders are most of the uncertainty, so it also covers new coders on new questions (93–95% at τ = 0.2). When coders are alike, it leaves out the questions' share and covers that claim at 87–96%.
- **The `coders` bootstrap is too narrow.** It covers 79–85% with 5 coders, 88–91% with 10 and 90–92% with 20. A percentile interval from m coders behaves like a bootstrap of m observations, which understates the spread when m is small. For this claim, use the jackknife.
- **Ordinal data follow the same pattern.** With τ = 0.2, 10 coders and 100 questions:
  - the units interval covers new coders in 75.8% of runs;
  - `two_way` covers them in 93.8%;
  - the jackknife covers its own claim at 94.4%.

**How many coders.** A claim about coders needs enough of them. With 5 coders, `two_way` under-covers when coders differ, and the jackknife covers but is wide. With 10 coders, `two_way` covers 92–98% when coders differ, and the jackknife covers 94–96%.

**Which coders drive alpha.** `result.coder_jackknife().influence` lists alpha without each coder, and each coder's influence. It shows whether one annotator, model or prompt drives the result. It describes the data; it is not a test of whether coders differ. Choose the claim, and so the interval, from the design, before looking at the results.

## Comparing two configurations

`study.compare()` resamples the units once and computes both configurations' alphas on every resample. The interval of the difference Δ = α_b − α_a therefore accounts for the units the two share. With a `margin` δ and `test="non_inferiority"`, it also gives a verdict: `non_inferior` when the lower end of the interval is above −δ. Reading one end of a two-sided interval at `confidence` is a one-sided test at level (1 − `confidence`) / 2, so the default 95% states a 2.5% test. This section measures how often the verdict is wrongly `non_inferior`, and how often it is rightly so.

**Setup.**
- **Paired arms.** Two configurations rate the same units, with the same true categories, hard units and clusters, as two judge configurations scoring one item set do. Arm a is the scenario above, with accuracy 0.8.
- **Margin cells.** Arm b's accuracy is 0.765. The margin is set to −Δ, so the truth sits exactly on the boundary of the null hypothesis, and every `non_inferior` verdict is a false claim. That is 0.049–0.051 on nominal data and 0.024–0.025 on the ordinal scale, where guesses are near misses and alpha falls less.
- **Equal cells.** Both arms have accuracy 0.8, so Δ = 0. The share of `non_inferior` verdicts at a margin is the test's power at that margin.
- **Grid.** Nominal and 5-point ordinal data at n = 50, 100, 200 and 500, in both kinds of cell. Two more margin cells at n = 200: clusters of 5 units (groups declared), and failures concentrated on hard units.
- **Truth.** Δ is the difference of population alphas, from 2,000,000 units rated by both arms, with failures treated as missing.

Every replication is compared with percentile and BCa intervals at 95% and 96%, on the same 2,000 resamples.

**Run.** {{run_info:paired}}

{{table:paired_size}}

- **At the default 95%, the test is slightly liberal.** Over the ten margin cells, BCa rejects 2.89% of the time against the stated 2.5%, and 3.75% in the worst cell (nominal, n = 500). Percentile intervals do the same: 2.88%, and 3.80% at worst.
- **The excess is not a defect in the verdict.** The bootstrap standard error of the difference matches its true spread to within 1% on average, and the estimate is unbiased to within 0.08 standard errors in every cell. The difference is slightly heavy-tailed, and the bootstrap interval ends add a little more. The intervals miss about 0.25 points too often per tail, the slight undercoverage the single-alpha intervals also show above. It does not shrink consistently with more units in this grid.
- **For a test that holds 2.5%, pass `confidence=0.96`.** That states a 2% test. BCa then rejects 2.4% of the time on average, and at most 3.15% in any cell, within Monte Carlo error of 2.5%. The reported interval is then a 96% interval, and the report records the confidence beside the verdict.
- **Clusters and failures behave alike.** With groups declared, the clustered cell rejects 2.50% at 95% and 2.00% at 96%. With failures on hard units it rejects 3.10% and 2.25%. As for one alpha, the comparison is between the dropped-failure alphas of both arms, not their complete-data alphas.
- **Verdicts and p-values agree with their intervals** in all 80,000 comparisons of the margin cells.
- **Equivalence was not simulated.** Two one-sided tests at level γ read both ends of the interval at `confidence` = 1 − 2γ, so a 5% equivalence test uses `confidence=0.90`. Expect the same small per-tail excess, unmeasured at that level.

{{figure:power_by_n}}

{{table:paired_power}}

- **Power depends on the margin far more than on the method.** When the configurations are equally reliable, BCa at 96% declares non-inferiority at a margin of 0.05 in 31% of runs at n = 500 on nominal data, and 67% on the ordinal scale. At a margin of 0.1 that rises to 86% and nearly 100%. A margin of 0.02 is out of reach in this grid: at most 15%.
- **Ordinal alphas are easier to compare.** They are higher here (0.78 against 0.54) and less variable, so the 96% interval at n = 500 is 0.08 wide against 0.13.
- **A rough guide.** Power at Δ = 0 reaches about 80% when the margin is about 0.7 times the width of the 96% interval. Run a pilot, read the width, and choose n accordingly.
- **Fix the margin before looking at the results.** A margin chosen after seeing the interval cannot be wrong, and so says nothing. `compare()` records the margin and the test in the declarations, the provenance and the JSON report.

## Which scheme to use

| Situation | Use | Why |
| --- | --- | --- |
| Units independent, n of about 50 or more | `scheme="units"` (percentile) | 93–95% coverage; simplest |
| Small sample or a dominant category | `scheme="units", method="bca"` | corrects the low-alpha bias; if BCa warns that it has no interval, use percentile |
| Units share a source | declare `group`; `scheme="units"` resamples groups | unit resampling under-covers (89–91%) |
| Fewer than about 30 clusters | group resampling, read as approximate | 92% coverage with 20 clusters |
| Reproducing a published KALPHA interval | `scheme="kalpha"` | matches Krippendorff's algorithm; too narrow for new claims with 3 or more coders |
| Comparing two configurations | `study.compare()` | units scheme on identical resamples (KALPHA has no paired form) |
| A non-inferiority verdict on a configuration change | `study.compare(margin=..., test="non_inferiority", confidence=0.96)`, margin fixed in advance | at 95% the 2.5% test rejects 2.9% at the margin; at 96%, 2.4% |
| Repeated runs of one judge configuration | `scheme="units"` | runs differ only by chance, so the interval already covers new runs |
| Coders differ (annotators, models, prompts), and the claim is about coders like them | `scheme="two_way"`, with 10 coders or more | units covers 48–83% of that claim; `two_way` covers 92–98% |
| These questions with other coders, or which coders drive alpha | `result.coder_jackknife()` | 94–96% coverage, and an influence table per coder; the `coders` bootstrap is too narrow |

## Limitations

- **One family of processes.** These scenarios use one family of data-generating processes (know or guess, with difficulty), up to 20 coders, up to 200 units (500 when comparing), nominal and 5-point ordinal data, and 95% intervals (also 96% when comparing). Interval and ratio metrics were not simulated.
- **Comparisons differ in one way.** In the comparison grid, the two configurations differ only in accuracy, and the true difference at the margin is about −0.05 (nominal) or −0.025 (ordinal). Equivalence tests were not simulated.
- **Coders differ in one way.** In the coder grid, coders differ in accuracy and in a favourite label. Two other kinds of difference were not simulated beyond the one ordinal scenario: coders who use a scale differently, and coders who are good on some kinds of question and poor on others. Every coder rates every question, and no attempts fail.
- **Truth uncertainty.** The true alphas carry a Monte Carlo error of at most 0.0025, small next to interval widths of 0.07 and more. The true differences carry at most 0.0007.
- **BCa's undefined cases are real.** At n = 30 with an 80% category, one run in six produces no BCa interval.
- **KALPHA's second pick.** KALPHA's result depends on reading its algorithm note as evalica does: a unit's second pick is redrawn when it repeats the first. Its coverage problem does not come from that detail. It comes from treating dependent pairs as independent.

## Reproduce

```bash
uv run python -m simulations.run_coverage --workers 8                 # about 20 minutes on 8 cores
uv run python -m simulations.run_coverage --grid coders --workers 8  # about an hour on 8 cores
uv run python -m simulations.run_coverage --grid paired --workers 8  # about 70 minutes on 8 cores
uv run --group sim python -m simulations.report                       # tables, figures and this page
```

The seeds and versions are recorded in `run_info.json` beside each `coverage.csv`. Each replication's seed depends only on its scenario and index, so a rerun reproduces every number.
