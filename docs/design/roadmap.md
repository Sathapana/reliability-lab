# reliability-lab: Roadmap, 0.2 to 1.0

Oct 29, 2026 · @Sathapana

The phases of the [framing](framing-v2.md) (P0 to P5) shipped in 0.1.0a1 to 0.1.4. This plan sets the releases from 0.2 to 1.0. It assigns the proposals of the "Development Roadmap (v0.2.0 → v0.5.0)" draft, after checking each one against the code at 0.1.4. Proposal numbers (1.1 to 4.3) refer to that draft.

Two decisions shape the order:

- **Human-LLM alignment moves to 1.0.** The framing committed mixed panels to v0.2, and the code refused mixed rows with "planned for v0.2". On 29 Oct 2026 they were moved to 1.0, and the refusals now say so. The draft's `study.calibrate()` answers part of the same question, so it is folded into the 1.0 design.
- **About a third of the draft has shipped.** The vectorised bootstrap, the pigeonhole (two-way) bootstrap, crossed coverage simulations, coder influence and per-unit disagreement already exist. They are listed in [Already shipped](#already-shipped) with where they live, and are not rebuilt.

## Summary

| Release | Theme | Contents | Gate |
| --- | --- | --- | --- |
| 0.2.0 | Decisions on a configuration change | Non-inferiority and equivalence in `compare()` (2.2); BCa failure reasons and opt-in fallback (1.2); category-skew diagnostic (1.3); unit triage (1.4) | At `confidence=0.96`, rejection at the margin holds 2.5%; at 0.95 it is measured and reported (revised 30 Oct 2026, see the 0.2.0 gate); a test for each BCa failure cause |
| 0.3.0 | Planning, CI gates and Arrow input | Sample-size planning (2.3); CLI quality gates (4.2); Arrow input (4.1) | Planned sizes reach their stated power on fresh seeds; CLI runs the example fixtures end to end; polars, DuckDB and pyarrow inputs give the same ratings fingerprint as pandas |
| 0.4.0 | Judge bias evidence | Within- and between-family alpha (3.2); unit covariates for length bias and self-enhancement (3.2); call telemetry, reduced (3.3) | Simulated same-family errors detected; fixtures for each diagnostic |
| 0.5.0 | Companion coefficients and plots | `metrics/` with Gwet's AC1 and AC2 (1.3); plotting extra (4.3) | Parity with irrCAC |
| 1.0.0 | Human-LLM alignment and a stable release | Mixed panels; alignment averaged over individual annotators and against a consensus label (3.1); stable API; frozen report schema; coverage simulations for interval, ratio and mixed-panel data | Mixed-panel end-to-end fixture with hand-checked values; paired intervals checked against a naive loop; reports from 1.0 read back by later releases; guidance regenerated from full-size runs |
| Research | Crossed intervals | A less conservative alternative to `two_way` (2.1) | Beats `two_way` on the coders grid |
| Research | A calibrated paired interval | A bootstrap-t or double bootstrap for `compare()` | Holds 2.5% at the margin at 95% on the paired grid |

Each gate is a test, as in the framing: a release ships when its gate passes.

## Review of the proposals

| Proposal | Verdict | Reason | Assigned to |
| --- | --- | --- | --- |
| 1.1 Vectorised multinomial bootstrap | Shipped | 0.1.3 draws one multinomial weight matrix and computes every replicate as `o = W @ P` in batches, shared by all missing-data policies. The draft's formulas also contain two errors (below). | — |
| 1.2 BCa degeneracy fallback | Reduced | Undefined BCa intervals already give no interval, a reason and a warning pointing to percentile. A fallback by default would decide silently. | 0.2.0 |
| 1.3 Prevalence paradox and skew | Split | The skew warning exists. The proposed flag rule holds for nominal data only. AC1/AC2 are coefficients, not diagnostics, and belong in the planned `metrics/` module. | 0.2.0 (entropy), 0.5.0 (AC1/AC2) |
| 1.4 Coder influence and unit triage | Mostly shipped | `coder_jackknife().influence` and `estimate.per_unit` hold the proposed columns. Only a per-unit status breakdown is missing. | 0.2.0 (status breakdown) |
| 2.1 Crossed resampling, G-theory | Shipped / research / deferred | `scheme="two_way"` is the pigeonhole bootstrap, and the coders grid simulates it. A less conservative crossed interval is open research. G-theory variance components are an ICC-family estimand. | Research; `metrics/` backlog |
| 2.2 Non-inferiority and TOST | Accepted | Answers the primary user's question, "did an update change it?", with a declared margin. | 0.2.0 |
| 2.3 Sample-size planning | Accepted, re-scoped | Target alpha, marginals and coder count do not fix a data-generating process; the plan must name one. It belongs in a new module, not `design.py`. | 0.3.0 |
| 3.1 `study.calibrate()` | Merged | Same question as the human-LLM alignment design. One log with the reference as rows, named "alignment", not "validity" or "gold". | 1.0.0 |
| 3.2 Verbosity and family bias | Accepted, moved | Output length is a property of the unit, and the trace has one row per judge configuration, so lengths cannot live there. `model_family` is a ratings field, not a trace field. | 0.4.0 |
| 3.3 Cost, latency, energy | Reduced | Tokens, latency and cost are per call, so they belong in the long-form log. Energy and the "Δα per 1,000 tokens" ratio are dropped. | 0.4.0 |
| 4.1 Zero-copy DataFrame input | Accepted, other protocol | pandas is deprecating the `__dataframe__` interchange protocol in favour of the Arrow PyCapsule interface. Validation ends in pandas, so the input is convenient, not zero-copy. | 0.3.0 |
| 4.2 CLI quality gates | Accepted, with a spec file | A study's declarations do not fit in command-line flags. | 0.3.0 |
| 4.3 Terminal and figure plots | Accepted | Text interval bars in the core; figures in an optional extra. | 0.5.0 |

**Errors in proposal 1.1.** With $d_{o,u} = \frac{1}{m_u - 1}\sum_{j,k} C_{u,j} C_{u,k} D_{j,k}$, a replicate's observed disagreement is $\sum_u w_u d_{o,u} / \sum_u w_u m_u$. The draft multiplies the numerator by $m_u$ a second time. The draft also precomputes $\mathbf{D}$ once, which is wrong for ordinal data: ordinal distances depend on the marginals, so `_batch_delta2` rebuilds them from each replicate's marginals.

**Factual corrections.**

- The baseline is 0.1.4 (bipolar and circular distances), and the package supports Python 3.11 and later (`requires-python`, CI on 3.11 to 3.13), not 3.12 and later.
- There is no `BootstrapConfig`: bootstrap options are keyword arguments of `AlphaResult.bootstrap()`. There is no `StudyReport`: summaries are methods of `AlphaResult`, `BootstrapResult`, `ComparisonResult` and `JackknifeResult`. The trace is built with `TraceSidecar.from_frame()`, not `from_records()`.
- CI runs `uv run ruff check`, `uv run ruff format --check`, `uv run mypy` (strict, on `src` only) and `uv run pytest` on Linux, Windows and macOS, plus an isolated irrCAC parity job. `tests/test_parity.py` holds parity with third-party packages; parity of the vectorised path with the naive loop is in `tests/test_inference_units.py`.
- `python -m simulations.run_coverage --quick` runs 50 replications. At 95% that is a Monte Carlo standard error of about 3 points, so it is a smoke run and cannot show that coverage did not regress. A change to inference reruns the affected cells at full size (`--cells`), commits the results and regenerates the guidance.

## 0.2.0: Decisions on a configuration change

**Non-inferiority and equivalence (2.2).** `compare(a=reference, b=candidate, margin=δ, test="non_inferiority" | "equivalence")` (decided 29 Oct 2026). The difference stays $\Delta = \alpha_b - \alpha_a$, so `a` is the reference.

```python
diff = study.compare(
    a={"condition_id": "frontier"},   # reference
    b={"condition_id": "distilled"},  # candidate
    margin=0.03,
    test="non_inferiority",           # or "equivalence"
    confidence=0.95,                  # one-sided 2.5% test
    n_resamples=2000,
    random_state=7,
)
diff.verdicts["as_missing"]  # "non_inferior" | "inferior" | "inconclusive"
```

- The verdict reads the paired interval that `compare()` already reports at `confidence`. There is no separate significance parameter, so a report holds one interval and the verdict can always be checked against it by eye.
- Reading the lower end of a two-sided interval at `confidence` is a one-sided non-inferiority test at level (1 − `confidence`) / 2. The default 95% gives the conventional 2.5% test. Equivalence (two one-sided tests) at level γ needs `confidence` = 1 − 2γ, so a 5% equivalence test uses `confidence=0.90`; the guidance says so.
- Each test has its own verdicts. The draft's single list mixes them.
  - **Non-inferiority:** `non_inferior` when the lower end is above −δ; `inferior` when the upper end is below −δ; otherwise `inconclusive`.
  - **Equivalence:** `equivalent` when the interval lies inside (−δ, δ); `not_equivalent` when it lies outside [−δ, δ]; otherwise `inconclusive`.
- There is a verdict for every missing-data policy. An undefined interval is `inconclusive`. The margin and the test go into the declarations, the provenance and the JSON. The guidance tells users to fix the margin before looking at results.
- Bootstrap p-values are reported only if they invert the same interval (percentile or BCa), so a p-value can never contradict its verdict.

**Riders from the draft's phase 1.**

- **BCa failure reasons (1.2).** `Interval.undefined_reason` names the cause: undefined replicates (resamples holding one category, the common case in the guidance's skewed scenario), every replicate on one side of the estimate (infinite bias correction), or zero jackknife variance (undefined acceleration). An opt-in `on_undefined="percentile"` (the default `"nan"` keeps today's behaviour) returns the percentile interval over the defined replicates. That interval records `method_used="percentile"` in tables and JSON, and the `BootstrapWarning` says so. There is no fallback by default: in the guidance's skewed scenario at n = 30, the percentile interval covered 90.5%, and a reader of a BCa report should not get it without being told.
- **Category-skew diagnostic (1.3).** A `category_skew` diagnostic gives the normalised entropy of the pairable values over the declared domain, $H / \ln K$, and the effective number of categories, $e^H$. The existing `CategorySkewWarning` (one value holding 90% or more) keeps its share-based rule. The draft's rule, $D_e < 0.15$ and $1 - D_o > 0.85$, holds for nominal data only. Under the other metrics, $D_o$ and $D_e$ are in the scale's squared units, and $1 - D_o$ is not observed agreement.
- **Unit triage (1.4).** `estimate.per_unit` gains each unit's status counts (abstained, inapplicable, failed). Sorted by `d_o_contribution`, it is the adjudication list. No new study methods: coder influence is `coder_jackknife().influence`, and unit disagreement is `estimate.per_unit`.

**Gate.** A simulation grid of paired arms whose true difference sits on the margin (`python -m simulations.run_coverage --grid paired`), with a power curve at Δ = 0. `tests/test_bca.py` gains a case for each BCa failure cause, and the new interval fields survive a JSON round trip.

The gate as first written required the rejection rate at the margin to stay within Monte Carlo error of (1 − `confidence`) / 2. The first full run (30 Oct 2026) missed it. At 95%, BCa rejected 2.89% on average over the margin cells, against 2.5%, and 3.70% in the worst cell. The investigation found no defect in `compare()`, the verdicts or the p-values:
- The bootstrap standard error of the difference matched its true spread to within 1% on average.
- The difference is slightly heavy-tailed: an interval with the true spread still missed 2.62% per tail.
- Normal intervals, with the bootstrap or the jackknife standard error, rejected 2.74–2.78%, with the same worst cell.

The excess is the small-sample undercoverage of bootstrap intervals that the single-alpha guidance also shows. It is about 0.25 points at every level tried.

The gate was therefore revised on 30 Oct 2026, before the rerun that is judged against it. The rerun uses a true difference from 2,000,000 units, and BCa intervals at both 95% and 96%:

- **At `confidence=0.96`** (one-sided 2%), the BCa test rejects at most 2.72% on average over the margin cells, which is 2.5% plus two pooled Monte Carlo standard errors at 2,000 replications per cell. No margin cell exceeds 3.2%, 2.5% plus two of that cell's standard errors. The guidance recommends this setting when the test must hold 2.5%.
- **At `confidence=0.95`**, no margin cell exceeds 4.0%. The guidance reports the measured average and worst cell beside the verdicts' stated level.
- **At both levels**, every verdict agrees with its interval.

**Result (30 Oct 2026): the revised gate passes.**
- At 96%, BCa rejected 2.37% on average over the margin cells, and 3.15% in the worst cell (nominal, n = 500).
- At 95%, it rejected 2.89%, and 3.75% in the same cell.
- Every verdict and p-value agreed with its interval.

The rerun reused the first run's seed, so its 95% figures repeat the first run's. The 96% arm was therefore judged on the replications that suggested it.

A confirmation run of the ten margin cells on a fresh seed also passes. Its results are not committed; it is reproduced by `python -m simulations.run_coverage --grid paired --cells margin --seed 20261030 --out <dir>`.
- At 96%, BCa rejected 2.29% on average, and 2.70% at worst.
- At 95%, it rejected 2.88%, and 3.15% at worst.
- The average excess at 95% repeats, at about 0.4 points. The worst cell does not: nominal at n = 500 rejected 2.85% at 95%, so its 3.75% in the committed run was chance.

A calibrated interval (a bootstrap-t, or a double bootstrap that sets the level from the data) would remove the excess at about a hundred times the computation. It is listed under research.

## 0.3.0: Planning, CI gates and Arrow input

**Sample-size planning (2.3).** A new `planning.py`; `design.py` holds coder definitions and design labels. The draft's inputs (target alpha, marginals, number of coders) do not determine how ratings are generated, so each plan names its generator:

1. For nominal data, the minimum-sample-size rule in Krippendorff's *Content Analysis*: the number of values needed for a given minimum alpha, smallest category probability and significance level. The exact table and edition are to be verified before building.
2. For every metric, simulation with the latent-class "know or guess" generator of `simulations/`. That generator moves into the package, because the wheel ships `src/reliability_lab` only. The search over N uses common random numbers and a monotone fit, because a plain binary search on noisy power estimates can stop on noise.
3. The same machinery plans the non-inferiority test of 0.2.0: the number of units needed to show non-inferiority at margin δ when the true difference is zero.

The result records the generator and its parameters, and the documentation says the answer holds for that generator only.

**CLI quality gates (4.2).** `[project.scripts] reliability-lab = "reliability_lab.cli:main"`, written with `argparse` and `tomllib`, both in the standard library from 3.11, so there are no new core dependencies.

- Declarations come from a study file in TOML (decided 29 Oct 2026), not from flags. A study needs a measurement, values in scale order, coders, a design, missing-data policies, `abstain_as`, `inapplicable_as` and retries. These should be versioned with the judge, and comments in the file can record why a threshold or margin was chosen. The standard library reads TOML but cannot write it, so the CLI only reads study files.

  ```toml
  [study]
  measurement = "ordinal"
  values = [1, 2, 3, 4, 5]
  coders = ["model_id", "condition_id"]
  design = "prompt_robustness"
  missing_policies = ["as_missing", "sensitivity"]
  retries = "last"

  [columns]
  unit = "response_id"
  rater = "rater_id"
  value = "score"
  status = "rating_status"

  [gate]
  min_alpha = 0.70
  min_low = 0.60   # lower end of the 95% interval
  ```

- `reliability-lab evaluate study.toml --min-alpha 0.70 --min-low 0.60 --output report.json`
- `reliability-lab compare study.toml --a condition_id=prompt_v1 --b condition_id=prompt_v2 --margin 0.03 --test non_inferiority`. It reads one log, or several concatenated, and selects the arms as the API does.
- Ratings are read from CSV, or from Parquet with the `parquet` extra.
- Exit codes: 0 pass; 1 usage or input error; 2 a threshold or verdict fails; 3 a design or trace finding (`TraceWarning`, `CoderDependenceWarning`), with `--strict` only.
- Thresholds apply to every reported missing-data policy unless `--policy` names one. A judge that passes only when its failures are dropped does not pass silently.

**Arrow input (4.1).** Moved from 0.5.0 on 29 Oct 2026, because the CLI reads Parquet through pyarrow anyway: one path from Arrow to pandas serves both. `RatingFrame.from_long` and `TraceSidecar.from_frame` accept any object with the Arrow PyCapsule stream interface (`__arrow_c_stream__`), such as polars data frames, DuckDB relations and pyarrow tables. The import goes through pyarrow (the `parquet` extra) and ends in pandas, because validation builds pandas columns. The promise is "no manual `to_pandas()`", not zero-copy.

**Gate.** Planned sample sizes reach their stated power when re-simulated with fresh seeds. CLI tests run the example fixtures end to end and check exit codes and the JSON report. The same ratings passed as pandas, polars, DuckDB and pyarrow give the same ratings fingerprint.

## 0.4.0: Judge bias evidence

**Within- and between-family alpha (3.2).** For `inter_model`, pairwise alpha within model families and between them, from `model_family` in the ratings. That field is constant per `model_id`; the trace records `provider`, not family. This turns today's `CoderDependenceWarning` into a number. Intervals use the units scheme, paired with the overall alpha.

**Unit covariates (3.2).** A unit sidecar keyed by `unit_id` (for example `UnitSidecar.from_frame(df, unit=..., length_a=..., length_b=..., output_family=...)`), checked against the ratings as the trace is. It serves two diagnostics:

- **Length bias:** rank correlation between the length difference (or the response length, for pointwise scores) and the judgment. For `order_robustness`, it also checks whether order-consistent wins favour the longer candidate. A correlation with length is not bias on its own, because longer answers can be better. The diagnostic describes the data; a causal claim needs pairs matched on length.
- **Self-enhancement:** the judged output's model family against the judge's family. The framing deferred this because it needs exactly this field.

**Call telemetry, reduced (3.3).** Optional numeric columns in `RatingFrame.from_long`: `prompt_tokens`, `completion_tokens`, `latency_ms`, `cost`. They are one value per attempt, retries included. Alpha and comparison reports show per-condition totals and means per counted rating beside alpha. Two parts of the draft are left out:

- `energy_wh`: providers do not report it, so it would be an estimate with no provenance.
- The "Δα per 1,000 tokens" ratio: a noisy difference over a cost has no interval here, and invites over-reading. The two are reported side by side instead.

The library still only summarises logs and never calls providers (framing, non-goals).

**Gate.** Simulated same-family correlated errors (the existing correlated-errors process) give higher within-family than between-family alpha. Fixtures cover each diagnostic and the unit-sidecar checks.

## 0.5.0: Companion coefficients and plots

**`metrics/` (1.3).** Gwet's AC1 (nominal) and AC2 (weighted) open the module the framing planned for later coefficients. They are reported as separate, labelled coefficients, never as diagnostics of alpha. They take the same declarations (the distance gives the weights) and the same missing-data policies. Kappas and ICC follow in the same module. G-theory variance components (2.1) belong with ICC, since they are the variance decomposition ICC is built from.

**Plots (4.3).**

- Core: text interval bars in the bootstrap and comparison summaries, one row per missing-data policy, with no new dependency.
- Optional `plot` extra (matplotlib, already in the `sim` group): `plot_policy_forest`, `plot_coincidence` (observed against expected coincidences) and `plot_coder_influence`. They reuse the light and dark styles of `simulations/report.py`.

**Gate.** AC1 and AC2 match irrCAC in the isolated parity job CI already runs. Plot tests check figure structure (axes, labels, one mark per estimate), not pixels.

## 1.0.0: Human-LLM alignment and a stable release

Alignment moved from 0.2.0 on 29 Oct 2026. Until then, `design="human_llm_alignment"` and ratings that mix human and LLM rows are refused with a message naming 1.0. Besides alignment, 1.0 requires a stable API, a frozen report schema and wider coverage simulations ([below](#release-requirements)).

**Mixed panels.** `check_design` accepts human and LLM rows together under `design="human_llm_alignment"`. Coders are defined as now: a human coder by `rater_id`, an LLM coder by model and condition.

**The alignment report.** Each quantity is labelled and none is pooled with another:

1. **Human alpha**: the human coders among themselves. It is the reference's own reliability, and it limits how much can be read into alignment. Beside it, the average pairwise alpha over pairs of human coders, the counterpart of the first alignment coefficient.
2. **LLM alpha**: the LLM coders among themselves.
3. **Alignment**, as two coefficients (decided 29 Oct 2026):
   - **Averaged over annotators.** For each LLM configuration, alpha of the two-coder panel it forms with each human coder, averaged over the human coders. It is compared with the humans' average pairwise alpha, computed the same way, because a two-coder alpha and an alpha of the whole panel are different quantities.
   - **Against a consensus label.** For each LLM configuration, alpha of the two-coder panel it forms with a declared consensus. The consensus enters as rows of the same `RatingFrame` (for example `rater_id="consensus"`) and is declared as the reference, so one log carries one fingerprint and one split manifest. It is derived from the human coders, so it is left out of human alpha and of the averaged coefficient.

   For each coefficient, proposal 3.1 adds the mean shift (ordinal, interval and ratio data) and a status-aware confusion table: LLM value or failure status against the reference value.

Reports call this "alignment with the human reference". The framing holds that human agreement is not automatic ground truth, so "criterion validity" and "gold" are not used.

**Intervals.** The units scheme resamples the same units for every quantity, as `compare()` does for two arms, and recomputes each average in every replicate. The gap between an LLM configuration's average and the humans' average then gets a paired interval. It answers whether the judge agrees with each annotator about as well as annotators agree with each other. Against a consensus, the interval is on the coefficient alone. No human counterpart is reported, because an annotator compared with a consensus built partly from their own labels would be favoured. Whether an LLM can replace a given annotator is a different test (`alttest`, Calderon et al. 2025) and stays out of scope.

### Release requirements

Decided 29 Oct 2026. A hosted API reference was considered and is not required for 1.0.

- **Stable API.** Semantic versioning from 1.0: a name exported by `reliability_lab` (its `__all__`) is removed or changed only in a major release, after a deprecation warning for at least one minor release. Leftovers are removed before the freeze, such as `BootstrapResult.experimental`, which has always been false since 0.1.1. The README drops its "alpha" status, and the classifier moves to "Development Status :: 5 - Production/Stable".
- **Frozen report schema.** JSON and Parquet reports already carry `schema_version` 1. From 1.0, 1.x releases may add fields but never remove or rename one; a change that would break a reader raises `schema_version`. Fields added before 1.0 (non-inferiority verdicts, BCa failure reasons, telemetry) are part of the frozen schema.
- **Wider coverage simulations.** The guidance simulates nominal and 5-point ordinal data only. Before 1.0 it covers interval and ratio data, and mixed panels: the intervals on both alignment coefficients and on the gap between the LLM's average and the humans' average. Each recommendation in the guidance's "Which scheme to use" table holds or is revised.

**Gate.**

- A provider-neutral end-to-end fixture of a mixed-panel study, with and without a consensus (validation plan, item 6). Hand-checked values cover human alpha, the averaged coefficient and the consensus coefficient. The paired intervals are checked against a naive loop, as the units scheme is.
- Golden report files written by 1.0 are committed, and every later 1.x release reads them in its tests.
- The guidance is regenerated from committed full-size runs of the new grids.

## Research: a less conservative crossed interval

`scheme="two_way"` is the pigeonhole bootstrap. When coders are alike, it covered 99.3–99.8% and was about 1.5 times as wide as the units interval, because it counts the rating noise twice (Owen 2007). Candidates are Owen's variance correction for the pigeonhole bootstrap and a two-way jackknife. A replacement ships only if, on the coders grid, it covers 93% or more in every scenario and is narrower than `two_way` when τ = 0. There is no release target.

## Research: a calibrated paired interval

At 95%, the paired intervals of `compare()` miss about 0.25 points too often per tail, so a 2.5% non-inferiority test rejects about 2.9% of the time at the margin (the 0.2.0 gate). Candidates are a bootstrap-t, with a jackknife standard error inside each resample, and a double bootstrap that calibrates the level. Either one ships only if, on the paired grid at 95%, every margin cell rejects within two Monte Carlo standard errors of 2.5%, at a run time usable on studies of a few thousand units. There is no release target.

## Already shipped

| Proposal | Where it is |
| --- | --- |
| Vectorised multinomial bootstrap (1.1) | `draw_weights` and `replicate_alphas` in `inference.py` (0.1.3); parity with the naive loop in `tests/test_inference_units.py` |
| Pigeonhole bootstrap (2.1) | `scheme="two_way"` (0.1.2) |
| Crossed coverage simulations (2.1) | `python -m simulations.run_coverage --grid coders`; "Generalising over coders" in the [coverage guidance](../guidance/bootstrap-coverage.md) |
| `study.coder_influence()` (1.4) | `result.coder_jackknife().influence`: `alpha_without`, `influence` (alpha minus alpha without the coder), `units_rated`; per-coder error rates in `result.coverage.per_coder` |
| `study.unit_disagreement()` (1.4) | `estimate.per_unit`: `pairable_values`, `disagreement`, `d_o_contribution` |
| Skew warning (1.3) | `CategorySkewWarning`, `estimate.category_distribution`, `estimate.observed_agreement` |
| BCa degeneracy handling (1.2, part) | An undefined BCa interval gives a reason and a `BootstrapWarning` pointing to `method="percentile"`; parity with `scipy.stats.bootstrap` on degenerate data in `tests/test_bca.py` |

## Decisions

- [x] Human-LLM alignment moves from 0.2 to 1.0 (29 Oct 2026).
- [x] Alignment coefficients: both, averaged over individual annotators and against a consensus label (29 Oct 2026).
- [x] Non-inferiority API: one `margin` with a `test` name, read from the `confidence` interval; no separate significance parameter (29 Oct 2026).
- [x] Study file format for the CLI: TOML, read with the standard library (29 Oct 2026).
- [x] Arrow input moves from 0.5.0 to 0.3.0, beside the CLI (29 Oct 2026).
- [x] 1.0 also requires a stable API, a frozen report schema and coverage simulations for interval, ratio and mixed-panel data; a hosted API reference is not required (29 Oct 2026).
- [x] The 0.2.0 gate is revised after its first run missed it by about 0.4 points. The test's size at 95% is measured and reported, and `confidence=0.96` must hold 2.5% (30 Oct 2026).

## References

1. Krippendorff, K. *Content Analysis: An Introduction to Its Methodology.* Sage. Minimum-sample-size rule; edition and table to be verified.
2. Owen, A. B. (2007). [The pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122). Annals of Applied Statistics 1(2).
3. Calderon, N., Reichart, R., & Dror, R. (2025). [The Alternative Annotator Test for LLM-as-a-Judge](https://arxiv.org/pdf/2501.10970). ACL 2025.
4. Gwet, K. L. (2008). [Computing inter-rater reliability and its variance in the presence of high agreement](https://doi.org/10.1348/000711006X126600). British Journal of Mathematical and Statistical Psychology 61(1).
5. pandas: [`DataFrame.__dataframe__`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.__dataframe__.html), which recommends the Arrow PyCapsule interface for new development.

Verified on 29 Oct 2026: every "shipped" claim against the code at 0.1.4, the pandas interchange deprecation, and the Owen (2007) and Gwet (2008) citations. The Krippendorff sample-size table was not re-opened.
