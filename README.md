# reliability-lab

**Status: pre-alpha.** The API is not stable and nothing is published to PyPI yet.

reliability-lab is a study-design library for measuring the reliability of LLM and human judgments. It is built on Krippendorff's alpha. Alpha implementations already exist in Python. This package adds the workflow around the coefficient:

- a validated long-form ratings schema in which every rating has an explicit status (`observed`, `abstained`, `inapplicable`, `parse_error`, `provider_error`);
- declared measurement semantics: level, value domain, coder definition and design label;
- white-box output: observed and expected disagreement, coincidence matrices and per-unit contributions;
- explicit missing-data policies, because LLM failures are rarely unrelated to the item;
- bootstrap intervals by a scheme you name: whole units or groups resampled (`scheme="units"`), or Krippendorff's KALPHA algorithm (`scheme="kalpha"`);
- paired comparisons of two configurations on the same units, with the interval on the difference.

Group-safe train/test splits that work with any scikit-learn splitter, and provenance reports, are on the roadmap ([design document](docs/design/framing-v2.md)).

> **Reliability is not validity.** Coders can agree consistently and still be wrong in the same way, for example because of a flawed rubric or a bias they share. When results come from different designs (repeatability, inter-model agreement, human panels), they are reported as different quantities and are never pooled.

## Install (development)

```bash
uv sync
uv run pytest
```

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

## Intervals and comparisons

Every interval names its scheme, because the two schemes answer different questions. `scheme` is a required argument.

```python
boot = result.bootstrap(scheme="units", n_resamples=2000, random_state=7)
print(boot.summary())  # 95% percentile intervals for every missing-data policy
```

- `scheme="units"` resamples whole units, and whole groups when a `group` column is declared. It can also resample within declared strata (`strata=True`), and it recomputes the expected disagreement in every replicate. `method="bca"` is available as an experimental option.
- `scheme="kalpha"` reproduces Krippendorff's KALPHA algorithm. It resamples pairs of values and holds the expected disagreement fixed.

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

## Licence

BSD-3-Clause. Alpha is implemented independently from Krippendorff's papers. The GPL-licensed `krippendorff` package is used only in the test suite, as a parity check.
