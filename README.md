# reliability-lab

**Status: alpha (0.1.4).** The API may still change between releases.

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

The extras are optional. `sklearn` provides the splitters, although any object with a scikit-learn style `split` method works; `parquet` enables `to_parquet`. For development, run `uv sync` and then `uv run pytest`.

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

- `scheme="units"` resamples whole units, and whole groups when a `group` column is declared. It can also resample within declared strata (`strata=True`), and it recomputes the expected disagreement in every replicate. `method="bca"` gave the most accurate coverage in the simulations; on small samples dominated by one category it can fail, and that is warned about.
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

## Licence

BSD-3-Clause. Alpha is implemented independently from Krippendorff's papers. The GPL-licensed `krippendorff` package is used only in the test suite, as a parity check.
