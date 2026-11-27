# reliability-lab

**Status: alpha (0.6.0).** The API may still change between releases.

reliability-lab is a study-design library for measuring the reliability of LLM and human judgments. It is built on Krippendorff's alpha. Alpha implementations already exist in Python. This package adds the workflow around the coefficient:

- a validated long-form ratings schema in which every rating has an explicit status (`observed`, `abstained`, `inapplicable`, `parse_error`, `provider_error`);
- declared measurement semantics: level, value domain, coder definition and design label;
- white-box output: observed and expected disagreement, coincidence matrices and per-unit contributions;
- explicit missing-data policies, because LLM failures are rarely unrelated to the item;
- bootstrap intervals by a scheme you name: whole units or groups resampled (`scheme="units"`), coders resampled (`scheme="coders"`), both at once (`scheme="two_way"`), or Krippendorff's KALPHA algorithm (`scheme="kalpha"`);
- a leave-one-coder-out jackknife that shows which coders drive alpha;
- paired comparisons of two configurations on the same units, with the interval on the difference;
- group-safe train/test splits that work with any scikit-learn splitter, with a saved manifest that reproduces the split;
- versioned JSON and Parquet reports with provenance;
- LLM judge workflows: a trace sidecar of judge configurations, declared retry handling, and evidence specific to each design (run agreement, order swaps and position bias).

Mixed human-LLM panels are planned for 1.0 ([roadmap](https://github.com/Sathapana/reliability-lab/blob/main/docs/design/roadmap.md), [design document](https://github.com/Sathapana/reliability-lab/blob/main/docs/design/framing-v2.md)).

> **Reliability is not validity.** Coders can agree consistently and still be wrong in the same way, for example because of a flawed rubric or a bias they share. When results come from different designs (repeatability, inter-model agreement, human panels), they are reported as different quantities and are never pooled.

## Install

```bash
pip install "reliability-lab[sklearn,parquet]"
```

The extras are optional. `sklearn` provides the splitters, although any object with a scikit-learn style `split` method works; `parquet` enables `to_parquet` and ratings from polars, DuckDB and pyarrow tables. For development, run `uv sync` and then `uv run pytest`.

## Example

```python
import pandas as pd
from reliability_lab import CoderSpec, RatingFrame, ReliabilityStudy

df = pd.DataFrame(
    {
        "item": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "annotator": ["ann", "bob", "ann", "bob", "ann", "bob"],
        "label": ["pos", "pos", "neg", "pos", "neg", "neg"],
    }
)

ratings = RatingFrame.from_long(df, unit="item", rater="annotator", value="label")
study = ReliabilityStudy(
    ratings,
    measurement="nominal",
    values=["neg", "pos"],
    coders=CoderSpec(by="rater_id"),
    design="human_panel",
)
result = study.alpha()
print(result.summary())
```

Each estimate lists its units with their share of the observed disagreement and their abstained, inapplicable and failed ratings. Sorted, that is the list of units to adjudicate first:

```python
per_unit = result["as_missing"].per_unit
per_unit.sort_values("d_o_contribution", ascending=False).head(10)
```

The `category_skew` diagnostic shows how evenly the ratings use the declared scale: the normalised entropy (1 when every value is used equally, 0 when one value holds them all) and the effective number of values. Alpha can be low while observed agreement is high when one value dominates.

### Polars, DuckDB and Arrow tables

`RatingFrame.from_long` and `TraceSidecar.from_frame` also accept any table that exports an Arrow stream (`__arrow_c_stream__`): a polars DataFrame, a DuckDB relation, a pyarrow Table or a RecordBatchReader. There is no need to call `to_pandas()` first.

```python
import duckdb

log = duckdb.sql("select * from 'ratings.parquet' where split = 'test'")
ratings = RatingFrame.from_long(log, unit="item", rater="annotator", value="label")
```

The table is converted through pyarrow (the `parquet` extra) and copied into pandas, so the conversion is convenient, not zero-copy. The same ratings give the same fingerprint whatever library they come from. Two exceptions: fractional `float32` values, and timestamps or decimals, can be written differently from their pandas forms.

## Simulated ratings

`simulate_ratings` makes synthetic coders for any measurement level, including `"bipolar"`, `circular(U)` and custom distances, at the alpha you choose. Use it to try the API, to see what a given alpha looks like in raw ratings, or to plan how many units a study needs.

```python
from reliability_lab import circular, simulate_ratings

ratings = simulate_ratings(
    circular(24),
    list(range(24)),
    n_units=200,
    n_coders=4,
    alpha=0.7,
    errors="near",
    missing=0.1,
    failures=0.05,
    random_state=1,
)
```

Each coder reports the true value or makes an error. With `errors="random"` an error is a draw from the population; with `errors="near"` it lands close to the truth, as the metric defines closeness. Alpha computed from the ratings is the target on average, and varies around it from draw to draw. `missing` leaves cells unrated and `failures` records cells as `parse_error`. [examples/simulated.py](https://github.com/Sathapana/reliability-lab/blob/main/examples/simulated.py) runs every built-in metric.

## Intervals and comparisons

Every interval names its scheme, because the schemes answer different questions. `scheme` is a required argument.

```python
boot = result.bootstrap(scheme="units", n_resamples=2000, random_state=7)
print(boot.summary())  # 95% percentile intervals for every missing-data policy
```

- `scheme="units"` resamples whole units, and whole groups when a `group` column is declared. It can also resample within declared strata (`strata=True`), and it recomputes the expected disagreement in every replicate. `method="bca"` gave the most accurate coverage in the simulations; on small samples dominated by one category it can fail. The warning and the interval's `undefined_reason` say why, and `on_undefined="percentile"` reports the percentile interval over the same resamples instead, marked `method_used="percentile"`.
- `scheme="kalpha"` reproduces Krippendorff's KALPHA algorithm. It resamples pairs of values and holds the expected disagreement fixed.

The units scheme holds the coders fixed: its interval is about these coders on other questions. When the claim is about coders like these, such as 10 runs of an LLM judge standing for the judge or a few annotators standing for trained annotators, the coders are a sample too:

- `scheme="two_way"` resamples questions (or groups) and coders independently. Its interval is about new coders on new questions; in the simulations it covered 92–98% with 10 coders or more when the coders differed, and erred on the wide side when they did not.
- `result.coder_jackknife()` recomputes alpha without each coder in turn. It gives an interval for these questions with new coders, and an influence table showing how much each coder moves alpha.
- `scheme="coders"` resamples coders and holds the questions fixed. It answers the jackknife's question, but its percentile interval was too narrow in the simulations; prefer the jackknife.

Repeated runs of one judge configuration differ only by chance, so the units interval already covers new runs.

```python
boot = result.bootstrap(scheme="two_way", n_resamples=2000, random_state=7)
jk = result.coder_jackknife()
print(jk.summary())  # interval, standard error, and the most influential coders
```

When coders are resampled, the copies of one coder are never paired with each other, since a coder always agrees with itself. These schemes need at least 3 coders and give percentile intervals.

Which scheme to trust, and when, is measured by simulation in the [bootstrap coverage guidance](https://github.com/Sathapana/reliability-lab/blob/main/docs/guidance/bootstrap-coverage.md).

To compare two configurations of a judge, for example two prompts, select each one's rows. Both must cover the same units. Every replicate resamples the same units for both, and the interval is on the difference:

```python
diff = study.compare(
    a={"condition_id": "prompt_v1"},
    b={"condition_id": "prompt_v2"},
    n_resamples=2000,
    random_state=7,
)
print(diff.summary())  # alpha_b - alpha_a with a paired interval
```

To decide whether an update changed the judge, declare a margin: the largest drop in alpha that does not matter. `a` is the reference.

```python
diff = study.compare(
    a={"condition_id": "frontier"},  # reference
    b={"condition_id": "distilled"},  # candidate
    margin=0.03,
    test="non_inferiority",  # or "equivalence"
    confidence=0.96,  # a test that holds 2.5% at the margin
    n_resamples=2000,
    random_state=7,
)
diff.verdicts["as_missing"]  # "non_inferior", "inferior" or "inconclusive"
diff.p_values["as_missing"]
```

- **Non-inferiority** reads the lower end of the interval: `non_inferior` when it is above −margin. At the default 95% this states a one-sided test at 2.5%. In simulation it rejected 2.9% of the time when the true difference sat on the margin, and up to 3.75% in one scenario. `confidence=0.96` held 2.5%. See [Comparing two configurations](https://github.com/Sathapana/reliability-lab/blob/main/docs/guidance/bootstrap-coverage.md#comparing-two-configurations).
- **Equivalence** reads both ends: `equivalent` when the interval lies inside (−margin, margin). These are two one-sided tests, so a 5% equivalence test uses `confidence=0.90`.
- Every missing-data policy gets its own verdict, and an undefined interval is `inconclusive`. Each p-value inverts the interval its verdict reads, so the two never disagree.
- Fix the margin before looking at the results. Small margins need many units: at a margin of 0.05, 500 units gave 31–67% power when the two configurations were equally reliable.

## Splits and reports

Teams tune the judge (prompt, few-shot examples, rubric) and should report reliability on units the tuning never saw. So split before tuning, freeze the judge, then estimate on the test partition.

When groups are declared, the splitter sees one sample per group (a conversation, say), so groups and repeated LLM calls never cross partitions:

```python
from sklearn.model_selection import StratifiedGroupKFold

dev, test = study.holdout(
    StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7),
    group_stratum="majority",  # needed only when a group's units span several strata
)
test.save_manifest("split.json")  # the ids of every partition, not just the seed

# ... tune and freeze the judge on dev ...

result = test.alpha()
result.to_json("alpha_test.json")  # strict JSON with provenance
result.to_parquet("alpha_test/")  # tidy tables

dev2, test2 = later_study.apply_manifest("split.json")  # same units, new judge runs
test.manifest.assert_no_leakage(few_shot_units)  # raises LeakageError
```

See [`examples/splits.py`](https://github.com/Sathapana/reliability-lab/blob/main/examples/splits.py) for a full run. Releases are described in [docs/releasing.md](https://github.com/Sathapana/reliability-lab/blob/main/docs/releasing.md).

## LLM judge workflows

A `condition_id` names a frozen judge configuration. A trace sidecar records what each configuration was: provider, prompt hash, rubric and parser versions, temperature, seed, and candidate order for pairwise judges. With it, a study checks that its coders differ only in what the design compares. For example, if prompt-robustness conditions also differ in temperature, it raises a `TraceWarning`. Whatever the trace leaves out is listed among the provenance gaps.

```python
from reliability_lab import TraceSidecar

trace = TraceSidecar.from_frame(
    configs,
    condition="condition_id",
    prompt_hash="prompt_sha",
    temperature="temperature",
    candidate_order="order",  # "A,B" / "B,A"
)
study = ReliabilityStudy(
    ratings,  # RatingFrame.from_long(..., attempt="attempt")
    measurement="nominal",
    values=["A", "tie", "B"],  # labels name the winning candidate, not its position
    coders="condition_id",
    design="order_robustness",
    trace=trace,
    retries="last",  # required when calls were retried
)
result = study.alpha()
result.diagnostics["order_swap"].values  # reversal rate, P(first shown wins) with a Wilson interval
result.trace_findings  # confounds and caveats from the trace
```

| Design | Coders are | Evidence beside alpha |
| --- | --- | --- |
| `repeatability` | repeated runs of one configuration (`run_id`) | run agreement: units where every run agrees |
| `prompt_robustness` | prompt conditions of one model | pairwise agreement between conditions |
| `inter_model` | different models | pairwise agreement; family and configuration checks |
| `order_robustness` | candidate orders of one pairwise judge | reversal rate and position bias |
| configuration change | two configurations, via `study.compare(a=..., b=...)` | paired interval on the difference; the model and trace fields that changed |

The single-configuration designs (repeatability, prompt robustness, order robustness) can hold several configurations to compare, such as an old and a new judge. Each configuration is checked on its own. `alpha()` refuses to pool them; `study.compare(a={"condition_id": "old"}, b={"condition_id": "new"})` reports each one's alpha and diagnostics, with a paired interval on the difference.

Retried calls are never resolved silently. Number the attempts with an `attempt` column, then declare `retries="last"` or `retries="first_observed"`. Coverage then reports error rates both per attempt and for the attempts that count. See [`examples/llm_judge.py`](https://github.com/Sathapana/reliability-lab/blob/main/examples/llm_judge.py).

## Sample-size planning

Plan sample sizes either from Krippendorff's analytical two-coder nominal rule or by simulation under the package's latent-class generator (with common random numbers and a monotone probit power curve):

```python
from reliability_lab import plan_alpha, plan_krippendorff, plan_non_inferiority

# Analytical nominal rule (Krippendorff, 2011, Table 1; Bloch & Kraemer, 1989)
rule = plan_krippendorff(alpha_min=0.80, smallest_prevalence=0.25, significance=0.05)

# Simulation planning for arbitrary metrics, multi-coder dependence, clusters, or missingness
study_plan = plan_alpha(
    "ordinal",
    [1, 2, 3, 4, 5],
    alpha=0.80,
    alpha_min=0.667,
    power=0.80,
    n_coders=3,
    random_state=42,
)

# Paired non-inferiority sample size (BCA at 96% confidence holds ~2.5% one-sided size)
ni_plan = plan_non_inferiority(
    "nominal",
    ["a", "b", "c"],
    alpha=0.80,
    margin=0.05,
    power=0.80,
    random_state=42,
)
```

## CLI quality gates

Run CI reliability checks from the command line (`reliability-lab` or `python -m reliability_lab`) with a TOML study specification and CSV, TSV, or Parquet logs:

```bash
reliability-lab evaluate study.toml ratings.csv --min-alpha 0.70 --min-low 0.65 --strict --output report.json
reliability-lab compare study.toml ratings.parquet --a condition_id=baseline --b condition_id=candidate --margin 0.05 --test non_inferiority --output comparison.json
```

Thresholds and comparison verdicts apply across every reported missing-data policy unless `--policy` selects one (`as_missing`, `as_category`, `sensitivity_pessimistic`, `sensitivity_optimistic`, or `sensitivity`). Exit codes are structured for CI: `0` pass, `1` usage/configuration/schema error, `2` threshold or verdict failure, and `3` passing thresholds with design or trace findings when `--strict` is set.

## Judge bias and telemetry

Declare a `UnitSidecar` alongside `RatingFrame` and `TraceSidecar` to check for shared-family judge correlation, length bias, self-enhancement, and API resource usage:

```python
units = rl.UnitSidecar.from_frame(
    unit_df,
    unit="item_id",
    length_a="len_a",
    length_b="len_b",
    output_family_a="gen_family_a",
    output_family_b="gen_family_b",
)
study = rl.ReliabilityStudy(
    rf,
    traces=traces,
    units=units,
    design="inter_model",
    level="nominal",
)
res = study.alpha(bootstrap=1000, scheme="units")
print(res.bootstrap.family_intervals)
print(res.telemetry)
```

When `prompt_tokens`, `completion_tokens`, `latency_ms`, or `cost` are supplied on `RatingFrame.from_long(...)`, `AlphaResult.telemetry` summarizes totals and per-counted-rating averages per coder across all recorded attempts (including retries).

## Companion coefficients and plots

Compute Gwet's AC1/AC2, Conger's/Fleiss's/Cohen's kappa, Intraclass Correlation Coefficients (`ICC(1,1)`..`ICC(3,k)`), or Generalizability Theory (G-study variance components and D-study projections) alongside Krippendorff's alpha under the same declarations and missing-data policies:

```python
gwet_res = study.gwet()  # or study.gwet(weights="ordinal")
kappa_res = study.kappa(variant="conger")  # or "fleiss", "cohen"
icc_res = study.icc(model="two_way_random", agreement="agreement", unit="single")
gt_res = study.g_theory(k=[1, 2, 3, 5, 10])
print(gt_res.summary())
```

With the optional `plot` extra (`pip install "reliability-lab[plot]"`), render policy forest plots, observed-vs-expected coincidence heatmaps, and coder jackknife influence charts:

```python
rl.plot_policy_forest(boot, mode="light", path="policy_forest.png")
rl.plot_coincidence(result, view="side_by_side", path="coincidence.png")
rl.plot_coder_influence(result, path="coder_influence.png")
```

## Licence

BSD-3-Clause. Alpha is implemented independently from Krippendorff's papers. The GPL-licensed `krippendorff` package is used only in the test suite, as a parity check.
