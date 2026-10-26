# reliability-lab: Project Framing (v2)

Oct 26, 2026 · @Sathapana

## Recommendation

Build reliability-lab as a study-design library for reliability of LLM and human judgments, not as another Krippendorff's alpha implementation. Alpha, bootstrap intervals and several kappas already exist in Python (`krippendorff`, `evalica`, `iaa-kit`, `irrCAC`); what is missing is one workflow that declares the measurement design, resamples correctly, prevents train/test leakage, and records provenance.

The first release combines:

- A validated long-form ratings schema with explicit rating status.
- Krippendorff's alpha with declared measurement semantics and white-box output (observed and expected disagreement D\_o and D\_e, coincidence matrix, per-unit contributions).
- Two bootstrap schemes for confidence intervals, chosen by the user: unit/cluster resampling, and Krippendorff's own KALPHA algorithm.
- Group-safe train/test splitting that accepts any scikit-learn splitter.
- Design-labelled results: repeatability, inter-model, human-LLM alignment are reported as different quantities, never pooled.
- Machine-readable results and provenance.

Reliability stays distinct from validity throughout. Consistent judgments can still reflect a flawed rubric or a bias shared by coders.

## Problem

Annotation studies and LLM-as-a-judge pipelines usually end with one agreement number. That number does not say whether the protocol is stable, whether it holds on unseen items, or whether a judge changed after a model or prompt update.

Krippendorff's alpha is the right starting coefficient: it handles any number of coders, missing ratings, several levels of measurement and custom distance functions. But its meaning depends on choices that are rarely reported: the distance function, the category distribution, how missing or failed ratings are treated, and the sampling design.

LLM coders add measurement variables of their own: model version and drift, prompt sensitivity, parser failures, non-determinism, position bias, and errors correlated across models from the same family. These belong in the data, not in a footnote.

## Target users

The primary user is the LLM evaluation engineer; the others shape requirements but are not the v0.1 focus.

| User | Decision | Evidence needed |
| --- | --- | --- |
| LLM evaluation engineer (primary) | Is this judge configuration reliable enough to use, and did an update change it? | Repeatability alpha, inter-model alpha, paired difference intervals, error rates |
| ML or NLP researcher | Does an evaluator generalise to unseen items? | Group-safe holdout, held-out alpha, provenance |
| Annotation lead | Is the rubric ready for a dataset release? | Human alpha, coverage, category distribution, missingness, interval |
| Domain expert | Which cases need adjudication? | Per-unit disagreement, slice analysis |
| Reviewer | Can the study be reproduced? | Unit definition, distance, split manifest, seed, package version |

## Competitive landscape

Bootstrap intervals for alpha are already available in Python, so the differentiator is integration and guardrails, argued against the packages below, not a new formula.

| Resource | Capability | Remaining gap |
| --- | --- | --- |
| [`krippendorff`](https://pypi.org/project/krippendorff/) (GPL-3.0-or-later, v0.9.0, Oct 2026) | Fast alpha from a coders x units matrix | No inference, splits, status model or diagnostics; V x V matrix limits large value domains |
| [`evalica`](https://evalica.readthedocs.io/latest/alpha.html) | Alpha with D\_o and D\_e exposed, KALPHA-style bootstrap | No unit/cluster bootstrap choice, split workflow or LLM trace model |
| [`iaa-kit`](https://pypi.org/project/iaa-kit/) | Alpha, Cohen/weighted/Fleiss kappa, Gwet AC1/AC2, bootstrap CIs, missing data | Small and young; no design declarations, splits or provenance |
| [`irrCAC`](https://pypi.org/project/irrcac) | Gwet's framework, many coefficients including alpha, weights | Coefficient-centred; no resampling-design or LLM workflow |
| NLTK `AnnotationTask` | Alpha, kappa, pi, custom distance | No study-level reporting or inference |
| `statsmodels`, `pingouin` | Cohen/Fleiss kappa; ICC variants | Not alpha-first; ICC has different assumptions |
| [`promptstability`](https://promptstability.readthedocs.io/en/latest/api_reference.html) | Repeated-run and paraphrase stability of LLM annotation with bootstrapped alpha | Tied to its own annotation runner; one design family |
| [`alttest`](https://github.com/nitaytech/alttest) | Leave-one-annotator-out test of whether an LLM can replace a human annotator | Answers the replacement question only; not alpha-based |
| scikit-learn, SciPy, `arch` | Splitters, `scipy.stats.bootstrap` (percentile, BCa), generic bootstraps | No notion of raters, units, clusters or rating status |

scikit-learn and SciPy are interoperability targets, not competitors: reliability-lab should accept their splitters and delegate interval arithmetic where possible.

## Goals and non-goals

**Goals**

- Compute alpha correctly from validated long-form ratings, for nominal, ordinal, interval, ratio and declared custom distances.
- Return diagnostics, uncertainty and provenance with every result, including the white-box decomposition.
- Make the resampling design explicit: unit, cluster, stratified, paired, or KALPHA-compatible.
- Prevent train/test leakage with group-aware splitting that accepts any scikit-learn splitter object.
- Support LLM-only and human-only studies in v0.1, each result labelled by its design; mixed human-LLM panels follow in v0.2.
- Export stable JSON, pandas-friendly tables and short reports.

**Non-goals**

- A hosted labelling platform or a benchmark leaderboard.
- Calling LLM providers from the statistical core (users pass judgment logs).
- Treating reliability as proof of validity, safety or fairness.
- Every reliability coefficient in the first release (kappas, Gwet AC1/AC2 and ICC come later).

## Statistical methodology

Every analysis declares its design up front, and the library refuses to compute alpha on an undeclared or undefined design.

**Declarations**

- **Unit of analysis:** response, document, conversation, case or other rated object.
- **Coder definition:** which columns of the log define a coder (human id, model, prompt, run). This decides the design.
- **Design label:** repeatability (runs of one configuration), prompt robustness, inter-model, human panel, or human-LLM alignment.
- **Measurement level and value domain:** nominal, ordinal, interval, ratio or custom distance, with legal values and their order.
- **Status policy:** how abstentions, inapplicable items, parser errors and provider errors are treated (below).
- **Sampling design:** independent units, clusters, strata, matched comparison or holdout.
- **Claim:** reliability, repeatability, alignment, configuration difference or validity.

**Coder independence**

Alpha's chance model assumes coders work independently. Repeated runs of one model, and models of the same family, violate this, so their errors correlate and alpha is likely inflated. The library therefore never pools designs into one number, and every result carries its design label and a warning when coders share a model family.

**Missing and failed ratings**

Treating parser or provider errors as missing assumes they are unrelated to the item. For LLMs this is doubtful: failures tend to concentrate on hard or borderline items, and dropping them removes exactly the units where coders would disagree. Alpha is therefore reported under at least two policies:

1. Errors as missing (the classic treatment).
2. Errors as their own category (nominal designs) or as a sensitivity bound (ordered designs).

Abstention is declared as either a value in the domain or missing; it is never decided silently. Per-coder error and abstention rates are always reported.

**Undefined and fragile cases**

The engine refuses a design with zero expected disagreement; units with fewer than two pairable values contribute nothing and are counted as such. Each result reports: number of units, ratings, coders and pairable values; coding density and per-coder coverage; status counts; category distribution with a skew warning (high agreement can still give low alpha when one category dominates); and observed agreement beside alpha.

**Reliability is not validity**

High agreement among LLM coders can coexist with poor agreement against experts. When human labels exist, alignment is reported as a separate quantity, never folded into reliability.

## Bootstrap design

Two schemes are offered side by side and the user chooses. In v0.1 the scheme is a required argument, so no interval is reported without naming its method. Simulations compare their coverage and are published as guidance, not used to impose a default.

|  | Unit/cluster bootstrap (scheme="units") | KALPHA (scheme="kalpha") |
| --- | --- | --- |
| What is resampled | Whole units, or whole groups | Pairs of values from the coincidence data |
| Expected disagreement D\_e | Recomputed each replicate | Held fixed from the observed data |
| Rationale | Pairs within a unit are not independent; letting D\_e vary propagates its uncertainty | Reproduces published KALPHA intervals |
| Status | Coverage measured by simulation, published as guidance | Reference algorithm ([Krippendorff](https://asc.upenn.edu/sites/default/files/2021-03/Algorithm%20for%20Bootstrapping%20a%20Distribution%20of%20Alpha.pdf)) |

Rules:

- **Resampling level follows the declaration:** if `group` is declared, the default level is the group, not the unit.
- **Paired comparisons:** two configurations (prompt A vs B, model X vs Y) are evaluated on the same resampled units in each replicate, and the interval is placed on the difference Δα. Different item sets answer a different question and are refused for paired claims.
- **Strata:** resample within declared strata when requested.
- **Defaults:** 2,000 replicates, 95% percentile interval. Failed or undefined replicates (e.g. D\_e = 0) are counted and reported, never silently dropped.
- **BCa:** bias-corrected and accelerated intervals are opt-in, only with scheme="units", through `scipy.stats.bootstrap`, but are marked experimental until coverage simulations validate them for alpha.
- **Implementation:** the coincidence matrix is a sum of per-unit contributions, so precomputing a units x V x V tensor makes each replicate a weighted sum plus a V x V computation. Verify this holds for every distance before relying on it.

## Train/test split design

Alpha itself is never fitted; a split exists because teams tune the judge (prompt, few-shot examples, rubric wording, score mapping) and must report reliability on units the tuning never saw.

- **Split before tuning**, and freeze the judge configuration before estimating alpha on the test partition.
- **Few-shot examples** drawn from the labelled set must belong to the training partition.
- **Atomic units:** every rating and every repeated LLM call for one unit stays in one partition.
- **Groups:** when units share a source (conversation, document, prompt template), split by group.
- **sklearn interoperability:** `split` accepts any object with sklearn's `split(X, y, groups)` interface and applies it to units or groups, so `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold` and custom splitters all work.
- **Size + stratification + groups together:** `GroupShuffleSplit` cannot stratify, and `StratifiedGroupKFold` stratifies only as k-fold. A 25% stratified group holdout is therefore one fold of `StratifiedGroupKFold(n_splits=4)`, and its size is approximate. The library reports the realised size.
- **Group-level strata:** a group whose units belong to several strata needs a rule (majority stratum, or a declared group-level stratum column). The library requires the rule rather than guessing.
- **Manifest:** persist the split indices, not only the random seed.
- **Small data:** repeated group-aware cross-validation for exploratory work, with an untouched final test set reserved for deployment claims.

## Data model

One long-form row per rating attempt. The coder definition (above) pivots this log into a coders x units matrix, so one log supports every design without re-querying any LLM.

| Field | Requirement | Purpose |
| --- | --- | --- |
| `unit_id` | Required | Rated-object identifier |
| `rater_id` | Required | Human annotator, or LLM configuration id |
| `value` | Required when status = observed | Categorical or numerical rating |
| `status` | Required | observed, abstained, inapplicable, parse\_error, provider\_error |
| `group_id` | Recommended | Conversation, document, prompt template or source-case cluster |
| `stratum` | Optional | Task family, language, class, severity |
| `model_id` | Required for LLM rows | Model identifier and snapshot |
| `model_family` | Recommended for LLM rows | Used for the coder-independence warning |
| `condition_id` | Required for LLM rows | Frozen prompt, rubric, parser, candidate order, generation settings |
| `run_id` | Required for repeated calls | Repeat index of the same configuration |

An LLM trace sidecar, keyed by `condition_id`, keeps provider, prompt hash, rubric version, temperature, seed or determinism setting, parser version, request time, candidate order and retry status.

## Architecture and API

Declarations are made once on the study and inherited by everything derived from it: splits, resamples and comparisons can never silently fall back to defaults.

&#91;embedded content: data flow · 7 steps, 2 external libraries\]

Dashed boxes are external libraries the package delegates to rather than reimplements.

```text
reliability_lab/
  data.py       RatingFrame: schema validation, status handling, value domains
  design.py     CoderSpec (which columns define a coder), design labels, warnings
  alpha.py      coincidence matrix, distances, D_o / D_e, per-unit contributions
  inference.py  unit / cluster / paired bootstrap, KALPHA-compatible mode
  split.py      holdout and CV over units or groups; sklearn splitter adapter
  llm.py        trace sidecar, repeatability and order-swap workflows
  report.py     typed results, JSON / Parquet, provenance manifest
  metrics/      later: kappas, Gwet AC1/AC2, ICC
```

Corrected API sketch (illustrative):

```python
from reliability_lab import RatingFrame, ReliabilityStudy, CoderSpec
from sklearn.model_selection import StratifiedGroupKFold

ratings = RatingFrame.from_long(
    df,
    unit="response_id",
    rater="rater_id",
    value="score",
    status="rating_status",
    group="conversation_id",
    stratum="task_family",
)

study = ReliabilityStudy(
    ratings,
    measurement="ordinal",
    values=[1, 2, 3, 4, 5],
    coders=CoderSpec(by=["model_id", "condition_id"]),
    design="inter_model",
    missing_policies=["as_missing", "as_category"],
)

# Any sklearn splitter; groups and strata come from the RatingFrame.
# The returned studies inherit measurement, values, coders and design.
dev, test = study.holdout(
    StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7),
    group_stratum="majority",
)
test.save_manifest("split.json")

# ... tune and freeze the judge on `dev` only ...

result = test.alpha()
ci = result.bootstrap(
    n_resamples=2000,
    confidence=0.95,
    scheme="units",         # or "kalpha"; required, no silent default
    level="group",          # default when a group is declared
    method="percentile",
    random_state=7,
)

# Paired comparison on identical resampled groups -> interval on the difference
diff = test.compare(
    a={"condition_id": "prompt_v1"},
    b={"condition_id": "prompt_v2"},
    n_resamples=2000,
    random_state=7,
)
result.to_json("alpha_test.json")
```

The three fixes against v1: `holdout` returns studies that keep their declarations; the bootstrap resamples groups because a group is declared; and the split is delegated to an sklearn splitter whose behaviour (approximate fold size) is documented.

## LLM workflows

Each workflow is one design label; a study names one primary claim and versions the complete judge configuration.

| Workflow | Coders are | Evidence | Caveat |
| --- | --- | --- | --- |
| Repeatability | Repeated runs of one configuration | Alpha across runs, parse/provider error rates | Deterministic output can still be wrong |
| Prompt robustness | Paraphrased prompts, same model | Alpha across conditions, paired intervals | Paraphrases must keep the rubric's meaning |
| Inter-model | Different models | Alpha across models, family warning | Same-family models share errors |
| Configuration change | Old vs new prompt or model | Paired bootstrap interval on Δα | Must use the same units |
| Order robustness (pairwise only) | Same judge, candidate order swapped | Agreement between swapped orders | Applies only to A-vs-B judgments, not pointwise scores |
| Human-LLM alignment (v0.2, not in v0.1) | Humans + LLM judge | Mixed-panel alpha, per-slice disagreement | Human agreement is not automatic ground truth |

Self-enhancement bias (a judge favouring outputs of its own family) is out of scope for v0.1; it would need the model family of the judged outputs as an extra field.

**Judge output types in v0.1.** Both scores and pairwise preferences are supported; the data decides which, through the declared measurement level.

- **Scores** (e.g. 1 to 5): ordinal distance by default, interval when the scale is declared equally spaced.
- **Pairwise** (A, B, tie): nominal by default; ordinal (A < tie < B) only when declared. Labels must name the winning candidate, not its position, so that order-swapped runs are comparable. Order-swap checks therefore ship in v0.1.

## Validation plan

1. Hand-worked fixtures from Krippendorff's published examples: nominal, ordinal, interval, ratio, missing data, perfect agreement, degeneracy.
2. Parity tests against `krippendorff` (test-only dependency), NLTK and `irrCAC`; KALPHA-mode parity against `evalica`.
3. Property tests: invariance to row and coder order, nominal relabelling, serialisation round-trips, distance validation.
4. Coverage simulations comparing unit/cluster bootstrap, KALPHA mode and BCa, varying prevalence, missingness (including informative LLM failures), coder quality, coder correlation and cluster dependence.
5. Leakage tests: no unit, group or few-shot example crosses partitions; manifests reproduce splits.
6. Provider-neutral end-to-end fixtures for repeatability, inter-model, configuration change and mixed-panel studies.

## Roadmap

A first public release needs phases 1 to 3; LLM workflows ride on the same engine in phase 4.

The original roadmap diagram did not survive export. The phasing below was adopted on 26 Oct 2026, when the build started.

| Phase | Scope | Gate |
| --- | --- | --- |
| P0 Scaffold | Packaging, licence, CI | CI green on Linux, Windows, macOS |
| P1 Core engine | `RatingFrame`, `CoderSpec`, design labels, distances, D\_o / D\_e, status policies | Krippendorff's published examples reproduced; parity with `krippendorff`, NLTK, `irrCAC` |
| P2 Inference | Unit / cluster / stratified / paired bootstrap, KALPHA mode, BCa (experimental) | Vectorised path equals naive loop; KALPHA parity with `evalica` |
| P3 Splits and reports | sklearn splitter adapter, split manifest, JSON / Parquet, provenance | Leakage tests pass; manifests reproduce splits. First public release (0.1.0a1) |
| P4 LLM workflows | Trace sidecar, repeatability, prompt robustness, configuration change, order swap | Provider-neutral end-to-end fixtures. Release 0.1.0 |
| P5 Coverage simulations | Unit/cluster vs KALPHA vs BCa coverage | Guidance report published |

Each gate is a test: the next phase starts only when it passes.

## Risks

| Risk | Mitigation |
| --- | --- |
| Reliability read as validity | Design labels on every result; alignment reported separately |
| Correlated LLM coders inflate alpha | Model-family field, independence warning, no pooled designs |
| Informative LLM failures inflate alpha | Multiple missing-data policies; error rates always reported |
| Wrong resampling level gives overconfident intervals | Group-level default when declared; coverage simulations |
| Two bootstrap schemes confuse readers | Scheme is a required argument and printed in every report |
| GPL licence of `krippendorff` | Independent implementation; `krippendorff` used only in tests (not legal advice) |
| Provider drift and missing version metadata | Trace sidecar; provenance gaps shown in reports |
| Large value domains make V x V matrices costly | Domain-size validation and warnings |
| Scope creep into generic LLM evaluation | Non-goals; provider-neutral core |

## Decisions

All framing questions are settled; the decisions are recorded here.

- [x] Bootstrap scheme: both unit/cluster and KALPHA offered; `scheme=` is a required argument in v0.1, and simulations are published as guidance.
- [x] BCa intervals: opt-in only with `scheme="units"`, flagged experimental in results and reports until coverage simulations validate them; percentile is the default.
- [x] Licence: BSD-3-Clause; implemented from Krippendorff's papers, with `krippendorff` used only in tests.
- [x] Judge output types for v0.1: scores and pairwise preferences, chosen by the data.
- [x] Human-LLM alignment: not in v0.1; planned for v0.2.

## References

1. Krippendorff, K. (2011). [Computing Krippendorff's alpha-reliability](https://repository.upenn.edu/entities/publication/80b129fd-0721-4899-beba-50ed6729e884). University of Pennsylvania ScholarlyCommons.
2. Krippendorff, K. (2016 revision). [Bootstrapping distributions for Krippendorff's alpha](https://asc.upenn.edu/sites/default/files/2021-03/Algorithm%20for%20Bootstrapping%20a%20Distribution%20of%20Alpha.pdf).
3. Hayes, A. F., & Krippendorff, K. (2007). [Answering the call for a standard reliability measure for coding data](https://doi.org/10.1080/19312450709336664).
4. Zapf, A. et al. (2016). [Measuring inter-rater reliability for nominal data: which coefficients and confidence intervals are appropriate?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4974794/) BMC Medical Research Methodology 16:93.
5. Artstein, R., & Poesio, M. (2008). [Inter-coder agreement for computational linguistics](https://aclanthology.org/J08-4004/).
6. Calderon, N., Reichart, R., & Dror, R. (2025). [The Alternative Annotator Test for LLM-as-a-Judge](https://arxiv.org/pdf/2501.10970). ACL 2025.
7. Zheng, L. et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://doi.org/10.48550/arXiv.2306.05685).
8. Gu, J. et al. [A Survey on LLM-as-a-Judge](https://doi.org/10.48550/arXiv.2411.15594).
9. Packages: [`krippendorff`](https://pypi.org/project/krippendorff/), [`evalica`](https://evalica.readthedocs.io/latest/alpha.html), [`iaa-kit`](https://pypi.org/project/iaa-kit/), [`irrCAC`](https://pypi.org/project/irrcac), [`promptstability`](https://promptstability.readthedocs.io/en/latest/api_reference.html), [`alttest`](https://github.com/nitaytech/alttest), [scikit-learn model\_selection](https://scikit-learn.org/stable/api/sklearn.model_selection.html).

Verified on 26 Oct 2026: the `krippendorff` licence and release, Zapf et al., Krippendorff's bootstrap note, the sklearn splitters, and the competitor capabilities listed. References 1, 3, 5, 7 and 8 were carried over from v1 and not re-opened.
