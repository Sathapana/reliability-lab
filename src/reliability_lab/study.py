"""A declared reliability study: ratings plus every choice that gives alpha its meaning.

The study refuses to compute alpha until the measurement level, value domain,
coder definition, design label and status policy are all declared and consistent
with the data. Phases 2 and 3 derive bootstraps, comparisons and splits from a
study, and those inherit its declarations.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from scipy import sparse

from reliability_lab._version import __version__
from reliability_lab.alpha import (
    BUILTIN_DISTANCES,
    NOMINAL,
    RATIO,
    AlphaComputation,
    Distance,
    check_domain_size,
    compute_alpha,
)
from reliability_lab.data import ERROR_STATUSES, RatingFrame, Status, _plain, preview
from reliability_lab.design import CoderSpec, Design, check_design
from reliability_lab.exceptions import (
    CategorySkewWarning,
    DesignError,
    DomainSizeWarning,
    ReliabilityWarning,
    TraceWarning,
    UndefinedAlphaError,
)
from reliability_lab.inference import EstimateInput, StudySample
from reliability_lab.inference import compare as compare_arms
from reliability_lab.llm import (
    CodedRatings,
    TraceSidecar,
    check_trace,
    coder_agreement,
    configuration_diff,
    order_swap,
    run_agreement,
)
from reliability_lab.report import (
    AlphaEstimate,
    AlphaResult,
    ComparisonResult,
    Coverage,
    Diagnostic,
    Level,
    Method,
    Provenance,
    RecordedWarning,
    emit,
)
from reliability_lab.split import TEST, SplitManifest, UnitLayout, assign_units, make_manifests

MissingPolicy = Literal["as_missing", "as_category", "sensitivity"]
Retries = Literal["last", "first_observed"]

MISSING_POLICIES: tuple[str, ...] = ("as_missing", "as_category", "sensitivity")
#: Category that failed attempts become under the ``as_category`` policy.
ERROR_CATEGORY = "<error>"
#: Share of pairable values in one category above which a CategorySkewWarning is raised.
SKEW_THRESHOLD = 0.9
#: Declares that abstentions or inapplicable items are treated as missing.
MISSING = "missing"
#: Fields that may select the arms of a paired comparison.
SELECTOR_FIELDS: tuple[str, ...] = (
    "rater_id",
    "model_id",
    "model_family",
    "condition_id",
    "run_id",
)

IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True, eq=False)
class Fold:
    """One fold of a cross-validation: dev and test studies and their manifest."""

    index: int
    dev: ReliabilityStudy
    test: ReliabilityStudy
    manifest: SplitManifest


class ReliabilityStudy:
    """Ratings plus the declarations that fix what alpha measures.

    Parameters
    ----------
    ratings:
        Validated long-form ratings.
    measurement:
        ``"nominal"``, ``"ordinal"``, ``"interval"``, ``"ratio"``, or a
        :class:`~reliability_lab.alpha.Distance`.
    values:
        The legal values. For ordinal data their order is the scale order. The
        domain is always declared: pass ``ratings.observed_values()`` to declare the
        observed values explicitly.
    coders:
        Which fields identify a coder, as a :class:`CoderSpec` or field names.
    design:
        A :class:`Design` label: what the coders are, and so what alpha measures.
    missing_policies:
        How parse and provider errors are treated. ``"as_missing"`` (always
        required) drops them. ``"as_category"`` (nominal only) makes them one extra
        category. ``"sensitivity"`` imputes each one pessimistically and
        optimistically from its unit's other values. When failures exist, at least
        two policies are required, because LLM failures tend to concentrate on hard
        items. Defaults: ``as_missing`` + ``as_category`` for nominal data,
        ``as_missing`` + ``sensitivity`` otherwise.
    abstain_as, inapplicable_as:
        ``"missing"`` or a declared value. They are required whenever such rows
        exist, so abstentions are never treated silently.
    trace:
        A :class:`~reliability_lab.llm.TraceSidecar` describing each LLM
        configuration (``condition_id``). The study checks that coders differ only
        in what the design compares, and reports what the trace does not record.
    retries:
        Which attempt counts when a coder rated a unit several times (an
        ``attempt`` column numbers them): ``"last"``, or ``"first_observed"`` (the
        first observed attempt, else the last). Required whenever retries exist.
        Error rates are reported per attempt and for the counted attempts.
    """

    def __init__(
        self,
        ratings: RatingFrame,
        *,
        measurement: str | Distance,
        values: Sequence[Any],
        coders: CoderSpec | str | Sequence[str],
        design: Design | str,
        missing_policies: Sequence[MissingPolicy] | None = None,
        abstain_as: Any = None,
        inapplicable_as: Any = None,
        trace: TraceSidecar | None = None,
        retries: Retries | None = None,
    ) -> None:
        if not isinstance(ratings, RatingFrame):
            raise TypeError(f"ratings must be a RatingFrame, got {type(ratings).__name__}")
        if trace is not None and not isinstance(trace, TraceSidecar):
            raise TypeError(f"trace must be a TraceSidecar, got {type(trace).__name__}")
        if retries not in (None, "last", "first_observed"):
            raise DesignError(f"retries must be None, 'last' or 'first_observed'; got {retries!r}")
        self._ratings = ratings
        self._trace = trace
        self._retries = retries
        self._warnings: list[RecordedWarning] = []
        #: The manifest and partition this study was split into, if any.
        self._split: tuple[SplitManifest, str] | None = None
        self._split_notes: tuple[str, ...] = ()

        self._distance = _resolve_distance(measurement)
        self._values = _resolve_domain(values, self._distance)
        # Validate the distance against the domain now rather than at alpha().
        self._distance.matrix(self._values, np.ones(len(self._values)))
        size_message = check_domain_size(len(self._values))
        if size_message:
            self._warnings.append(emit(DomainSizeWarning, size_message, stacklevel=2))
        self._coders = coders if isinstance(coders, CoderSpec) else CoderSpec(coders)
        self._design = _resolve_design(design)

        # Every attempt stays in self._ratings; estimation uses the counted attempt
        # of each (unit, coder), selected by the retries declaration.
        attempt_labels = self._coders.labels(ratings)
        kept, self._n_retried = _resolve_retries(ratings, attempt_labels, retries)
        resolved = ratings if kept is None else ratings._subset(kept)
        coder_labels = attempt_labels if kept is None else attempt_labels[kept]
        check = check_design(self._design, resolved, self._coders, coder_labels)
        for category, message in check.warnings:
            self._warnings.append(emit(category, message, stacklevel=2))
        self._notes = check.notes

        data = resolved._data
        self._trace_check = check_trace(self._design, trace, data, self._values)
        for message in self._trace_check.confounds:
            self._warnings.append(emit(TraceWarning, message, stacklevel=2))
        self._attempt_status = ratings._data["status"].to_numpy(dtype=object)
        self._attempt_coders = attempt_labels
        self._status = data["status"].to_numpy(dtype=object)
        self._condition_ids = data["condition_id"].to_numpy(dtype=object)
        self._abstain_as = _resolve_status_value(
            "abstain_as", abstain_as, Status.ABSTAINED, self._status, self._values
        )
        self._inapplicable_as = _resolve_status_value(
            "inapplicable_as", inapplicable_as, Status.INAPPLICABLE, self._status, self._values
        )
        self._is_error = pd.Series(self._status).isin(ERROR_STATUSES).to_numpy()
        self._policies = _resolve_policies(
            missing_policies, self._distance, int(self._is_error.sum())
        )

        # Unit codes are taken over every attempt, so splits can select whole units.
        unit_codes, unit_labels = pd.factorize(ratings._data["unit_id"], sort=False)
        coder_codes, coder_names = pd.factorize(coder_labels, sort=False)
        self._attempt_unit_codes: IntArray = np.asarray(unit_codes, dtype=np.intp)
        self._unit_codes: IntArray = (
            self._attempt_unit_codes if kept is None else self._attempt_unit_codes[kept]
        )
        self._unit_labels = np.asarray(unit_labels, dtype=object)
        self._coder_codes: IntArray = np.asarray(coder_codes, dtype=np.intp)
        self._coder_names = np.asarray(coder_names, dtype=object)
        self._value_idx = self._code_values(data)
        # Group and stratum are constant within a unit, so the first row tells.
        first_rows = np.unique(self._unit_codes, return_index=True)[1]
        self._unit_groups, self._unit_strata = (
            data[field].to_numpy(dtype=object)[first_rows] if field in ratings.declared else None
            for field in ("group_id", "stratum")
        )

    # -- declarations -------------------------------------------------------------

    @property
    def ratings(self) -> RatingFrame:
        return self._ratings

    @property
    def measurement(self) -> Distance:
        return self._distance

    @property
    def values(self) -> tuple[Any, ...]:
        return self._values

    @property
    def coders(self) -> CoderSpec:
        return self._coders

    @property
    def design(self) -> Design:
        return self._design

    @property
    def missing_policies(self) -> tuple[str, ...]:
        return self._policies

    @property
    def abstain_as(self) -> Any:
        return self._abstain_as

    @property
    def inapplicable_as(self) -> Any:
        return self._inapplicable_as

    @property
    def trace(self) -> TraceSidecar | None:
        return self._trace

    @property
    def retries(self) -> str | None:
        return self._retries

    @property
    def warnings(self) -> tuple[RecordedWarning, ...]:
        """Warnings raised while checking the declarations."""
        return tuple(self._warnings)

    @property
    def manifest(self) -> SplitManifest | None:
        """The split this study is a partition of, or None."""
        return self._split[0] if self._split else None

    @property
    def partition(self) -> str | None:
        """``"dev"`` or ``"test"`` when this study is a partition of a split."""
        return self._split[1] if self._split else None

    def __repr__(self) -> str:
        part = f", partition={self.partition!r}" if self._split else ""
        return (
            f"ReliabilityStudy(design={self._design.value!r}, "
            f"measurement={self._distance.name!r}, values={list(self._values)!r}, "
            f"coders={list(self._coders.by)!r}, units={len(self._unit_labels)}, "
            f"coders_found={len(self._coder_names)}, "
            f"missing_policies={list(self._policies)!r}{part})"
        )

    # -- splits -------------------------------------------------------------------

    def holdout(
        self, splitter: Any, *, fold: int = 0, group_stratum: Any = None
    ) -> tuple[ReliabilityStudy, ReliabilityStudy]:
        """Split into dev and test studies with any scikit-learn style splitter.

        The splitter sees one sample per group when ``group_id`` is declared
        (otherwise one per unit), so groups, units and repeated LLM calls never
        cross partitions. Declared strata are passed as ``y``. When a group's
        units span several strata, ``group_stratum`` must say how to label the
        group: ``"majority"`` or a mapping from group to stratum. ``fold`` picks
        which of the splitter's folds becomes test; ``StratifiedGroupKFold(4)``
        fold 0 is an approximate 25% stratified group holdout, and the realised
        size is recorded in the manifest.

        Split before tuning the judge, freeze it, then estimate alpha on test.
        Both studies inherit every declaration and carry the manifest, which
        ``save_manifest`` persists.
        """
        (manifest,) = make_manifests(self._layout(), splitter, group_stratum, folds=[fold])
        return self._partitions(manifest)

    def cv(self, splitter: Any, *, group_stratum: Any = None) -> list[Fold]:
        """Every fold of ``splitter`` as dev and test studies (see :meth:`holdout`).

        For exploratory work on small data; keep an untouched final test set for
        deployment claims.
        """
        manifests = make_manifests(self._layout(), splitter, group_stratum)
        return [Fold(m.fold, *self._partitions(m), m) for m in manifests]

    def apply_manifest(
        self, manifest: SplitManifest | str | os.PathLike[str]
    ) -> tuple[ReliabilityStudy, ReliabilityStudy]:
        """Reapply a saved split, by its ids, to these ratings.

        Use it for a later log, such as new judge runs on the same units. Every unit
        must be in the manifest and, at group level, in the same group as before.
        """
        if not isinstance(manifest, SplitManifest):
            manifest = SplitManifest.load(manifest)
        part_of_unit, notes = assign_units(self._layout(), manifest)
        return self._partitions(manifest, part_of_unit, tuple(notes))

    def save_manifest(self, path: str | os.PathLike[str]) -> None:
        """Write the manifest of the split this study is a partition of."""
        if self._split is None:
            raise DesignError("this study is not a partition of a split; use holdout() first")
        self._split[0].save(path)

    # -- estimation ---------------------------------------------------------------

    def alpha(self) -> AlphaResult:
        """Alpha under every declared missing-data policy, with diagnostics.

        Raises :class:`UndefinedAlphaError` when the reference (``as_missing``)
        estimate is undefined.
        """
        base = self._value_idx >= 0
        reference = self._compute(self._unit_codes[base], self._value_idx[base], self._values)
        estimates: dict[str, AlphaEstimate] = {}
        inputs: dict[str, EstimateInput] = {}
        for policy in self._policies:
            if policy == "as_missing":
                estimates["as_missing"] = self._estimate("as_missing", reference, self._values)
                inputs["as_missing"] = EstimateInput(
                    self._unit_codes[base], self._value_idx[base], self._values
                )
            elif policy == "as_category":
                values = (*self._values, ERROR_CATEGORY)
                codes = self._value_idx.copy()
                codes[self._is_error] = len(self._values)
                keep = codes >= 0
                computed = self._compute(self._unit_codes[keep], codes[keep], values)
                estimates["as_category"] = self._estimate("as_category", computed, values)
                inputs["as_category"] = EstimateInput(self._unit_codes[keep], codes[keep], values)
            else:
                for variant in ("pessimistic", "optimistic"):
                    codes, n_imputed, n_unanchored = self._impute(reference, variant)
                    keep = codes >= 0
                    computed = self._compute(self._unit_codes[keep], codes[keep], self._values)
                    inputs[f"sensitivity_{variant}"] = EstimateInput(
                        self._unit_codes[keep], codes[keep], self._values
                    )
                    estimates[f"sensitivity_{variant}"] = self._estimate(
                        "sensitivity",
                        computed,
                        self._values,
                        name=f"sensitivity_{variant}",
                        n_imputed=n_imputed,
                        n_unanchored=n_unanchored,
                    )

        primary = estimates["as_missing"]
        if not primary.defined:
            raise UndefinedAlphaError(
                f"alpha is undefined for these data: {primary.undefined_reason}"
            )

        recorded = list(self._warnings)
        skew = _skew_message(primary)
        if skew:
            recorded.append(emit(CategorySkewWarning, skew, stacklevel=2))
        notes = list(self._notes)
        if not self._is_error.any() and len(self._policies) > 1:
            notes.append(
                "No counted rating is a failed attempt, so every missing-data policy coincides."
            )
        if self._split is not None:
            manifest, partition = self._split
            notes.append(
                f"These are the {partition} units of a {manifest.level}-level split "
                f"(fold {manifest.fold} of {manifest.n_folds}, {manifest.splitter_class})."
            )
            notes.extend(self._split_notes)
        if self._retries is not None:
            notes.append(
                f"Retries: {self._n_retried} calls had several attempts; the "
                f"{'last' if self._retries == 'last' else 'first observed'} attempt is counted."
            )
        diagnostics = self._diagnostics(base)
        findings = [*self._trace_check.confounds, *self._trace_check.notes]
        for diagnostic in diagnostics.values():
            findings.extend(n for n in diagnostic.notes if "declared deterministic" in n)

        return AlphaResult(
            design=self._design.value,
            measurement=self._distance.name,
            values=self._values,
            coders=self._coders.by,
            missing_policies=self._policies,
            abstain_as=self._abstain_as,
            inapplicable_as=self._inapplicable_as,
            estimates=estimates,
            coverage=self._coverage(),
            warnings=tuple(recorded),
            notes=tuple(notes),
            provenance=self._provenance(),
            diagnostics=diagnostics,
            trace_findings=tuple(findings),
            _sample=StudySample(
                unit_labels=self._unit_labels,
                unit_groups=self._unit_groups,
                unit_strata=self._unit_strata,
                distance=self._distance,
                inputs=inputs,
            ),
        )

    def compare(
        self,
        a: Mapping[str, Any],
        b: Mapping[str, Any],
        *,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        level: Level | None = None,
        strata: bool = False,
        method: Method = "percentile",
        random_state: int | np.random.Generator | None = None,
        scheme: str = "units",
    ) -> ComparisonResult:
        """Paired comparison of two configurations on the same units.

        ``a`` and ``b`` select rows by field, for example
        ``{"condition_id": "prompt_v1"}``; a value may also be a list. Each arm is
        a study on its rows that inherits every declaration of this one. Both arms
        must cover exactly the same units, because different item sets answer a
        different question. Every replicate resamples the same units (or groups)
        for both arms, and the interval is placed on alpha_b - alpha_a. Only the
        units scheme has a paired form.
        """
        if scheme != "units":
            raise DesignError(
                "paired comparisons use scheme='units' only: KALPHA resamples pairs of "
                "values, which has no paired form"
            )
        mask_a, mask_b = self._select(a, "a"), self._select(b, "b")
        overlap = self._ratings._data.index[mask_a & mask_b]
        if len(overlap):
            raise DesignError(
                f"{len(overlap)} rows are selected by both a and b {preview(overlap)}; the "
                "configurations must be disjoint"
            )
        arm_a = self._derive(mask_a, self._split, self._split_notes)
        arm_b = self._derive(mask_b, self._split, self._split_notes)
        units_a, units_b = set(arm_a._unit_labels), set(arm_b._unit_labels)
        if units_a != units_b:
            only_a, only_b = sorted(units_a - units_b, key=str), sorted(units_b - units_a, key=str)
            raise DesignError(
                "a paired comparison needs the same units in both configurations; "
                f"{len(only_a)} units appear only in a {preview(only_a)} and {len(only_b)} only "
                f"in b {preview(only_b)}. Filter the ratings to the shared units first."
            )
        result = compare_arms(
            arm_a.alpha(),
            arm_b.alpha(),
            a=dict(a),
            b=dict(b),
            n_resamples=n_resamples,
            confidence=confidence,
            level=level,
            strata=strata,
            method=method,
            random_state=random_state,
        )
        if self._trace is None:
            return dataclasses.replace(
                result,
                notes=(
                    *result.notes,
                    "No trace sidecar: what changed between a and b is not recorded.",
                ),
            )
        diff = configuration_diff(self._trace, arm_a._conditions(), arm_b._conditions())
        note = (
            f"The configurations differ in {', '.join(diff)}."
            if diff
            else "The trace records no difference between the configurations."
        )
        return dataclasses.replace(result, configuration_diff=diff, notes=(*result.notes, note))

    # -- internals ----------------------------------------------------------------

    def _select(self, selector: Mapping[str, Any], name: str) -> npt.NDArray[np.bool_]:
        """Rows matching every field of a comparison selector."""
        if not isinstance(selector, Mapping) or not selector:
            raise DesignError(f"{name} must be a non-empty mapping of field to value(s)")
        data = self._ratings._data
        mask = np.ones(len(data), dtype=bool)
        for field, wanted in selector.items():
            if field not in SELECTOR_FIELDS:
                raise DesignError(
                    f"{name} selects on {field!r}; selectors may use {list(SELECTOR_FIELDS)}"
                )
            if field not in self._ratings.declared:
                raise DesignError(f"{name} selects on {field!r}, which was not mapped")
            options = (
                list(wanted) if isinstance(wanted, list | tuple | set | frozenset) else [wanted]
            )
            mask &= data[field].isin(options).to_numpy()
        if not mask.any():
            raise DesignError(f"{name}={dict(selector)} selects no rows")
        return mask

    def _derive(
        self,
        mask: npt.NDArray[np.bool_],
        split: tuple[SplitManifest, str] | None = None,
        split_notes: tuple[str, ...] = (),
    ) -> ReliabilityStudy:
        """A study on a subset of rows that inherits every declaration of this one.

        Its design warnings are recorded but not raised again: this study already
        raised them, and ten folds should not repeat each one ten times.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ReliabilityWarning)
            derived = ReliabilityStudy(
                self._ratings._subset(mask),
                measurement=self._distance,
                values=self._values,
                coders=self._coders,
                design=self._design,
                missing_policies=self._policies,  # type: ignore[arg-type]
                abstain_as=self._abstain_as,
                inapplicable_as=self._inapplicable_as,
                trace=self._trace,
                retries=self._retries,
            )
        derived._split = split
        derived._split_notes = split_notes
        return derived

    def _conditions(self) -> list[Any]:
        """The LLM configurations (condition ids) this study's counted ratings use."""
        return [c for c in pd.unique(self._condition_ids) if not pd.isna(c)]

    def _diagnostics(self, base: npt.NDArray[np.bool_]) -> dict[str, Diagnostic]:
        """Design-specific evidence, computed on the counted, pairable ratings."""
        coded = CodedRatings.build(
            self._unit_codes[base],
            self._coder_codes[base],
            self._value_idx[base],
            self._unit_labels,
            self._coder_names,
            self._values,
        )
        diagnostics = {"coder_agreement": coder_agreement(coded)}
        if self._design is Design.REPEATABILITY:
            conditions = self._conditions()
            deterministic = (
                self._trace.value(conditions[0], "deterministic")
                if self._trace is not None and conditions
                else None
            )
            diagnostics["run_agreement"] = run_agreement(coded, deterministic)
        elif self._design is Design.ORDER_ROBUSTNESS and self._trace is not None:
            condition_of_coder = dict(
                zip(self._coder_names[self._coder_codes], self._condition_ids, strict=True)
            )
            orders = {
                coder: self._trace.value(condition, "candidate_order")
                for coder, condition in condition_of_coder.items()
            }
            diagnostics["order_swap"] = order_swap(coded, orders)
        return diagnostics

    def _layout(self) -> UnitLayout:
        return UnitLayout(
            labels=self._unit_labels,
            groups=self._unit_groups,
            strata=self._unit_strata,
            ratings_fingerprint=self._ratings.fingerprint(),
        )

    def _partitions(
        self,
        manifest: SplitManifest,
        part_of_unit: npt.NDArray[np.object_] | None = None,
        notes: tuple[str, ...] = (),
    ) -> tuple[ReliabilityStudy, ReliabilityStudy]:
        """The dev and test studies of a manifest (rows of units outside both are dropped)."""
        if part_of_unit is None:
            where = manifest.partition_of()
            part_of_unit = np.array([where[_plain(u)] for u in self._unit_labels], dtype=object)
        # Masks run over every attempt row, so a unit's retries stay together.
        part_of_row = part_of_unit[self._attempt_unit_codes]
        studies = []
        for name in ("dev", TEST):
            mask = part_of_row == name
            if not mask.any():
                raise DesignError(f"the split leaves the {name} partition empty")
            studies.append(self._derive(mask, (manifest, name), notes))
        return studies[0], studies[1]

    def _provenance(self) -> Provenance:
        ratings = self._ratings
        gaps = []
        if ratings.implicit_status:
            gaps.append(
                "No status column: every row was taken as observed, so failed attempts "
                "cannot be told apart from ratings."
            )
        if "group_id" not in ratings.declared:
            gaps.append("No group_id: units are treated as independent in resampling and splits.")
        if ratings.is_llm.any():
            family = ratings._data["model_family"]
            if "model_family" not in ratings.declared or family.isna().any():
                gaps.append(
                    "model_family is missing for some LLM rows, so coder independence "
                    "cannot be fully checked."
                )
        gaps.extend(self._trace_check.gaps)
        split = None
        if self._split is not None:
            manifest, partition = self._split
            split = {
                "partition": partition,
                "level": manifest.level,
                "fold": manifest.fold,
                "n_folds": manifest.n_folds,
                "splitter": manifest.splitter,
                "units_fingerprint": manifest.units_fingerprint,
                "manifest_created_at": manifest.created_at,
            }
        return Provenance(
            versions={
                "reliability_lab": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pandas": pd.__version__,
            },
            platform=platform.platform(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ratings_fingerprint=ratings.fingerprint(),
            n_rows=len(ratings),
            declarations={
                "measurement": self._distance.name,
                "values": list(self._values),
                "coders": list(self._coders.by),
                "design": self._design.value,
                "missing_policies": list(self._policies),
                "abstain_as": self._abstain_as,
                "inapplicable_as": self._inapplicable_as,
                "retries": self._retries,
            },
            split=split,
            gaps=tuple(gaps),
            trace=None if self._trace is None else self._trace.to_dict(self._conditions()),
        )

    def _code_values(self, data: pd.DataFrame) -> IntArray:
        """Domain index of each row's value, or -1 when the row carries no value."""
        index = {v: i for i, v in enumerate(self._values)}
        codes = np.full(len(self._status), -1, dtype=np.intp)
        raw = data["value"].to_numpy(dtype=object)
        unknown: list[int] = []
        for i in np.flatnonzero(self._status == Status.OBSERVED.value):
            code = index.get(raw[i], -1)
            if code < 0:
                unknown.append(int(i))
            codes[i] = code
        if unknown:
            raise DesignError(
                f"{len(unknown)} observed values are outside the declared domain: values "
                f"{preview(pd.unique(raw[unknown]))} in rows {preview(data.index[unknown])}"
            )
        for kind, declared in (
            (Status.ABSTAINED, self._abstain_as),
            (Status.INAPPLICABLE, self._inapplicable_as),
        ):
            if declared is not None and declared != MISSING:
                codes[self._status == kind.value] = index[declared]
        return codes

    def _compute(
        self, unit_idx: IntArray, value_idx: IntArray, values: Sequence[Any]
    ) -> AlphaComputation:
        return compute_alpha(unit_idx, value_idx, len(self._unit_labels), values, self._distance)

    def _impute(self, reference: AlphaComputation, variant: str) -> tuple[IntArray, int, int]:
        """Impute each failed attempt from the other values of its unit.

        Pessimistic imputation picks the domain value with the largest summed
        squared distance to the unit's other values; optimistic picks the smallest.
        Distances are taken at the reference (``as_missing``) marginals, and ties go
        to the first value in domain order. Failed attempts in units without any
        other value stay missing. This is a sensitivity range, not a strict bound:
        the expected disagreement moves with the imputation too.
        """
        codes = self._value_idx.copy()
        error_rows = np.flatnonzero(self._is_error)
        if not len(error_rows):
            return codes, 0, 0
        base = self._value_idx >= 0
        counts = sparse.csr_array(
            (np.ones(int(base.sum())), (self._unit_codes[base], self._value_idx[base])),
            shape=(len(self._unit_labels), len(self._values)),
        )
        unit_counts = counts[self._unit_codes[error_rows], :]
        scores = np.asarray(unit_counts @ reference.components.delta2, dtype=float)
        anchored = np.asarray(unit_counts.sum(axis=1), dtype=float).ravel() > 0
        choice = scores.argmax(axis=1) if variant == "pessimistic" else scores.argmin(axis=1)
        codes[error_rows[anchored]] = choice[anchored]
        return codes, int(anchored.sum()), int((~anchored).sum())

    def _estimate(
        self,
        policy: str,
        computed: AlphaComputation,
        values: Sequence[Any],
        *,
        name: str | None = None,
        n_imputed: int = 0,
        n_unanchored: int = 0,
    ) -> AlphaEstimate:
        c = computed.components
        labels = pd.Index(list(values), name="value", dtype=object)
        n = c.n
        pairable = computed.pairable
        is_pairable = pairable >= 2
        mean_disagreement = np.full(len(pairable), np.nan)
        mean_disagreement[is_pairable] = (
            computed.unit_disagreement[is_pairable] / pairable[is_pairable]
        )
        contribution = computed.unit_disagreement / n if n > 0 else np.zeros(len(pairable))
        per_unit = pd.DataFrame(
            {
                "unit_id": self._unit_labels,
                "pairable_values": pairable.astype(int),
                "disagreement": mean_disagreement,
                "d_o_contribution": contribution,
            }
        )
        share = c.marginals / n if n > 0 else np.full(len(values), np.nan)
        return AlphaEstimate(
            name=name or policy,
            policy=policy,
            alpha=c.alpha,
            d_o=c.d_o,
            d_e=c.d_e,
            observed_agreement=float(np.trace(computed.coincidence) / n) if n > 0 else math.nan,
            n_pairable_values=round(n),
            n_pairable_units=int(is_pairable.sum()),
            n_unpairable_units=int((~is_pairable).sum()),
            coincidence=pd.DataFrame(computed.coincidence, index=labels, columns=labels),
            expected=pd.DataFrame(c.expected, index=labels, columns=labels),
            category_distribution=pd.Series(share, index=labels, name="share"),
            per_unit=per_unit,
            undefined_reason=c.undefined_reason,
            n_imputed=n_imputed,
            n_unanchored=n_unanchored,
        )

    def _coverage(self) -> Coverage:
        statuses = [s.value for s in Status]
        coder = pd.Series(self._coder_names[self._coder_codes], name="coder")
        status = pd.Series(self._status, name="status")
        table = pd.crosstab(coder, status).reindex(
            index=list(self._coder_names), columns=statuses, fill_value=0
        )
        table.columns.name = None
        n_units = len(self._unit_labels)
        per_coder = table.copy()
        per_coder.insert(0, "ratings", table.sum(axis=1))
        per_coder["coverage"] = table[Status.OBSERVED.value] / n_units
        per_coder["abstention_rate"] = table[Status.ABSTAINED.value] / per_coder["ratings"]
        per_coder["error_rate"] = (
            table[Status.PARSE_ERROR.value] + table[Status.PROVIDER_ERROR.value]
        ) / per_coder["ratings"]
        # Attempt-level counts, before retries are resolved.
        attempts = pd.DataFrame(
            {
                "coder": self._attempt_coders,
                "error": pd.Series(self._attempt_status).isin(ERROR_STATUSES).to_numpy(),
            }
        ).groupby("coder", sort=False)["error"]
        per_coder["attempts"] = attempts.size().reindex(per_coder.index).astype(int)
        per_coder["retries"] = per_coder["attempts"] - per_coder["ratings"]
        per_coder["attempt_error_rate"] = (
            attempts.sum().reindex(per_coder.index) / per_coder["attempts"]
        )
        n_observed = int(table[Status.OBSERVED.value].sum())
        n_coders = len(self._coder_names)
        return Coverage(
            n_units=n_units,
            n_ratings=len(self._status),
            n_coders=n_coders,
            n_observed=n_observed,
            coding_density=n_observed / (n_units * n_coders),
            status_counts={s: int(table[s].sum()) for s in statuses},
            per_coder=per_coder,
        )


#: Fields that identify one call to a judge; attempts of the same call are retries.
CALL_FIELDS: tuple[str, ...] = ("unit_id", "rater_id", "model_id", "condition_id", "run_id")


def _resolve_retries(
    ratings: RatingFrame, coder_labels: npt.NDArray[np.object_], retries: str | None
) -> tuple[npt.NDArray[np.bool_] | None, int]:
    """Which rows count under the retries declaration, and how many calls were retried.

    Attempts are grouped by call (unit, rater, model, condition and run), so
    repeated runs are never mistaken for retries. Returns ``(None, 0)`` when no
    declaration was made; duplicate rows are then refused by the design check.
    """
    if retries is None:
        return None, 0
    if "attempt" not in ratings.declared:
        raise DesignError(
            "retries= needs an attempt column: RatingFrame.from_long(..., attempt=...)"
        )
    data = ratings._data
    calls = [f for f in CALL_FIELDS if f in ratings.declared or f == "unit_id"]
    frame = pd.DataFrame({f: data[f].map(str).to_numpy() for f in calls})
    frame["coder"] = coder_labels
    frame["attempt"] = data["attempt"].to_numpy(dtype=float)
    frame["observed"] = (data["status"] == Status.OBSERVED.value).to_numpy()
    frame["row"] = np.arange(len(frame))
    keys = [*calls, "coder"]
    repeated = frame.duplicated([*keys, "attempt"], keep=False)
    if repeated.any():
        raise DesignError(
            f"{int(repeated.sum())} rows repeat an attempt number of the same call "
            f"{preview(data.index[repeated.to_numpy()])}"
        )
    ordered = frame.sort_values("attempt", kind="stable")
    grouped = ordered.groupby(keys, sort=False)
    chosen = grouped["row"].last()
    if retries == "first_observed":
        first_observed = ordered[ordered["observed"]].groupby(keys, sort=False)["row"].first()
        chosen = first_observed.combine_first(chosen)
    kept = np.zeros(len(frame), dtype=bool)
    kept[chosen.to_numpy(dtype=np.intp)] = True
    return kept, int((grouped.size() > 1).sum())


def _skew_message(estimate: AlphaEstimate) -> str | None:
    share = estimate.category_distribution
    if share.empty or not share.max() >= SKEW_THRESHOLD:
        return None
    return (
        f"value {_plain(share.idxmax())!r} holds {share.max():.0%} of pairable values; alpha "
        "can be low even when observed agreement is high (here "
        f"{estimate.observed_agreement:.3f})"
    )


def _resolve_distance(measurement: str | Distance) -> Distance:
    if isinstance(measurement, Distance):
        return measurement
    if isinstance(measurement, str) and measurement in BUILTIN_DISTANCES:
        return BUILTIN_DISTANCES[measurement]
    raise DesignError(
        f"measurement must be one of {list(BUILTIN_DISTANCES)} or a Distance; got {measurement!r}"
    )


def _resolve_domain(values: Sequence[Any], distance: Distance) -> tuple[Any, ...]:
    if values is None or isinstance(values, str):
        raise DesignError(
            "values must be a sequence of the legal values (in scale order for ordinal data); "
            "use ratings.observed_values() to declare the observed values explicitly"
        )
    domain = tuple(_plain(v) for v in values)
    if len(domain) < 2:
        raise DesignError("the value domain needs at least two values")
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in domain):
        raise DesignError("values cannot contain missing values; use the status column instead")
    try:
        distinct = len(set(domain))
    except TypeError:
        raise DesignError("values must be hashable") from None
    if distinct != len(domain):
        raise DesignError(f"values repeat: {list(domain)}")
    if ERROR_CATEGORY in domain:
        raise DesignError(f"{ERROR_CATEGORY!r} is reserved for the as_category policy")
    if distance.numeric:
        try:
            numeric = np.asarray(domain, dtype=float)
        except (TypeError, ValueError):
            raise DesignError(f"{distance.name} data need numeric values") from None
        if not np.isfinite(numeric).all():
            raise DesignError(f"{distance.name} values must be finite")
        if distance is RATIO and (numeric < 0).any():
            raise DesignError("ratio data need non-negative values")
    return domain


def _resolve_design(design: Design | str) -> Design:
    try:
        return Design(design)
    except ValueError:
        raise DesignError(
            f"unknown design {design!r}; expected one of {[d.value for d in Design]}"
        ) from None


def _resolve_status_value(
    name: str,
    declared: Any,
    kind: Status,
    status: npt.NDArray[np.object_],
    values: tuple[Any, ...],
) -> Any:
    present = int((status == kind.value).sum())
    if declared is None:
        if present:
            raise DesignError(
                f"{present} rows have status {kind.value!r}; declare {name}={MISSING!r} or a "
                "value from the domain. They are never treated silently."
            )
        return None
    if isinstance(declared, str) and declared == MISSING:
        return MISSING
    if declared not in values:
        raise DesignError(
            f"{name}={declared!r} is neither {MISSING!r} nor a declared value {list(values)}"
        )
    return declared


def _resolve_policies(
    policies: Sequence[str] | None, distance: Distance, n_errors: int
) -> tuple[str, ...]:
    if policies is None:
        resolved: tuple[str, ...] = (
            ("as_missing", "as_category") if distance is NOMINAL else ("as_missing", "sensitivity")
        )
    elif isinstance(policies, str):
        resolved = (policies,)
    else:
        resolved = tuple(policies)
    unknown = [p for p in resolved if p not in MISSING_POLICIES]
    if unknown:
        raise DesignError(
            f"unknown missing policies {preview(unknown)}; expected among {list(MISSING_POLICIES)}"
        )
    if len(set(resolved)) != len(resolved):
        raise DesignError(f"missing policies repeat: {list(resolved)}")
    if "as_missing" not in resolved:
        raise DesignError(
            "'as_missing' (the classic treatment) is always reported as the reference policy"
        )
    if "as_category" in resolved and distance is not NOMINAL:
        raise DesignError(
            "'as_category' is defined for nominal data only: an error category has no distance "
            "to ordered or custom values. Use 'sensitivity' instead."
        )
    if n_errors and len(resolved) < 2:
        raise DesignError(
            f"{n_errors} rows are parse or provider errors; report alpha under at least two "
            "policies (for example ['as_missing', 'sensitivity']), because LLM failures tend "
            "to concentrate on hard items"
        )
    return resolved
