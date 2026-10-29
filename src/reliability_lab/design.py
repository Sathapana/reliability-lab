"""Coder definitions and design labels.

The coder definition decides which columns of the long-form log identify a coder:
a human id, a model, a prompt condition, a run. Together with the design label it
fixes which quantity alpha measures. Different designs answer different questions,
so the library never pools them into one number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab.data import RatingFrame, preview
from reliability_lab.exceptions import (
    CoderDependenceWarning,
    DesignError,
    ReliabilityWarning,
)


class Design(StrEnum):
    """What the coders are, and therefore what alpha measures."""

    HUMAN_PANEL = "human_panel"
    REPEATABILITY = "repeatability"
    PROMPT_ROBUSTNESS = "prompt_robustness"
    INTER_MODEL = "inter_model"
    CONFIGURATION_CHANGE = "configuration_change"
    ORDER_ROBUSTNESS = "order_robustness"
    HUMAN_LLM_ALIGNMENT = "human_llm_alignment"


#: Fields a CoderSpec may combine.
CODER_FIELDS: tuple[str, ...] = ("rater_id", "model_id", "model_family", "condition_id", "run_id")

LLM_DESIGNS = frozenset(
    {
        Design.REPEATABILITY,
        Design.PROMPT_ROBUSTNESS,
        Design.INTER_MODEL,
        Design.ORDER_ROBUSTNESS,
    }
)

#: What each implemented design measures, attached to every result.
DESIGN_NOTES: dict[Design, str] = {
    Design.HUMAN_PANEL: "Inter-coder reliability of a human panel.",
    Design.REPEATABILITY: (
        "Repeatability of one judge configuration across repeated runs. Runs of one "
        "configuration are not independent coders, so this is not inter-coder reliability, "
        "and a repeatable judge can still be consistently wrong."
    ),
    Design.PROMPT_ROBUSTNESS: (
        "Agreement of one model across prompt conditions. All coders share one model, so "
        "their errors are correlated; the result holds only if the prompts keep the "
        "rubric's meaning."
    ),
    Design.INTER_MODEL: (
        "Agreement among different models. It is not agreement with human experts or "
        "with ground truth."
    ),
    Design.ORDER_ROBUSTNESS: (
        "Agreement of one judge with itself when the candidates' presentation order is "
        "swapped. It applies to pairwise judgments whose labels name the winning candidate, "
        "not its position."
    ),
}

_NOT_YET: dict[Design, str] = {
    Design.HUMAN_LLM_ALIGNMENT: "human-LLM alignment (mixed panels) is planned for v1.0",
    Design.CONFIGURATION_CHANGE: (
        "configuration_change is a paired comparison, not a single-alpha design: declare the "
        "design each configuration is measured under and call study.compare(a=..., b=...)"
    ),
}


@dataclass(frozen=True, init=False)
class CoderSpec:
    """Which canonical RatingFrame fields together identify a coder.

    ``CoderSpec(by="rater_id")`` treats each human annotator as a coder;
    ``CoderSpec(by=["model_id", "condition_id"])`` treats each model and prompt
    combination as a coder.
    """

    by: tuple[str, ...]

    def __init__(self, by: str | Sequence[str]) -> None:
        fields = (by,) if isinstance(by, str) else tuple(by)
        if not fields:
            raise DesignError("CoderSpec needs at least one field")
        unknown = [f for f in fields if f not in CODER_FIELDS]
        if unknown:
            raise DesignError(
                f"CoderSpec fields must be among {list(CODER_FIELDS)}; got {preview(unknown)}"
            )
        if len(set(fields)) != len(fields):
            raise DesignError(f"CoderSpec fields repeat: {list(fields)}")
        object.__setattr__(self, "by", fields)

    def labels(self, ratings: RatingFrame) -> npt.NDArray[np.object_]:
        """One coder label per row of ``ratings``."""
        data = ratings._data
        for field in self.by:
            if field not in ratings.declared:
                raise DesignError(
                    f"CoderSpec uses {field!r}, which was not mapped in RatingFrame.from_long"
                )
            missing = data.index[data[field].isna()]
            if len(missing):
                raise DesignError(
                    f"coder field {field!r} is missing in {len(missing)} rows "
                    f"{preview(missing)}; every row needs a value for each coder field"
                )
        parts = [data[f].map(_label).to_numpy(dtype=object) for f in self.by]
        if len(parts) == 1:
            return parts[0]
        return np.array([" | ".join(row) for row in zip(*parts, strict=True)], dtype=object)


def _label(x: Any) -> str:
    """Readable label: integral floats (from columns with gaps) print as integers."""
    if isinstance(x, float | np.floating) and float(x).is_integer():
        return str(int(x))
    return str(x)


@dataclass(frozen=True)
class DesignCheck:
    """Outcome of checking a design against the data.

    ``configurations`` lists the judge configurations of a single-configuration
    design (one tuple of key values each). A study holding several can be
    compared configuration by configuration, but has no alpha of its own.
    """

    notes: tuple[str, ...]
    warnings: tuple[tuple[type[ReliabilityWarning], str], ...]
    configurations: tuple[tuple[Any, ...], ...] = ()


#: Designs measured on one judge configuration, and the fields that identify it.
SINGLE_CONFIGURATION: dict[Design, tuple[str, ...]] = {
    Design.REPEATABILITY: ("model_id", "condition_id"),
    Design.PROMPT_ROBUSTNESS: ("model_id",),
    Design.ORDER_ROBUSTNESS: ("model_id",),
}


def configuration_masks(
    design: Design, data: pd.DataFrame
) -> list[tuple[tuple[Any, ...], npt.NDArray[np.bool_]]]:
    """Each judge configuration of a single-configuration design, with its row mask.

    Other designs, and data with one configuration, give a single all-rows entry.
    """
    keys = SINGLE_CONFIGURATION.get(design)
    if keys is None:
        return [((), np.ones(len(data), dtype=bool))]
    frame = data[list(keys)].astype(object)
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)
    return [(tuple(_label(v) for v in key), codes == i) for i, key in enumerate(uniques)]


def check_design(
    design: Design,
    ratings: RatingFrame,
    coders: CoderSpec,
    coder_labels: npt.NDArray[np.object_],
) -> DesignCheck:
    """Refuse designs the data cannot support, and collect the caveats of the rest.

    A single-configuration design (repeatability, prompt or order robustness)
    holding several configurations is checked configuration by configuration.
    """
    if design in _NOT_YET:
        raise NotImplementedError(_NOT_YET[design])
    data = ratings._data
    is_llm = ratings.is_llm
    if is_llm.any() and not is_llm.all():
        raise DesignError(
            f"the ratings mix {int((~is_llm).sum())} human rows and {int(is_llm.sum())} LLM "
            "rows; mixed human-LLM panels are planned for v1.0. Filter the ratings to one kind."
        )
    if design is Design.HUMAN_PANEL and is_llm.any():
        raise DesignError("human_panel was declared but the rows carry a model_id (LLM rows)")
    if design in LLM_DESIGNS and not is_llm.all():
        raise DesignError(f"{design.value} needs LLM rows (rows with a model_id)")

    configurations = configuration_masks(design, data)
    if len(configurations) > 1:
        checks = [
            check_design(design, ratings._subset(mask), coders, coder_labels[mask])
            for _, mask in configurations
        ]
        return DesignCheck(
            notes=checks[0].notes,
            warnings=tuple(dict.fromkeys(w for c in checks for w in c.warnings)),
            configurations=tuple(key for key, _ in configurations),
        )

    warnings: list[tuple[type[ReliabilityWarning], str]] = []
    if design is Design.REPEATABILITY:
        if "run_id" not in coders.by:
            raise DesignError(
                "repeatability coders are repeated runs: include 'run_id' in CoderSpec(by=...)"
            )
    elif design is Design.PROMPT_ROBUSTNESS:
        if "condition_id" not in coders.by:
            raise DesignError(
                "prompt_robustness coders are prompt conditions: include 'condition_id' in "
                "CoderSpec(by=...)"
            )
    elif design is Design.INTER_MODEL:
        if "model_id" not in coders.by:
            raise DesignError(
                "inter_model coders are models: include 'model_id' in CoderSpec(by=...)"
            )
        if data["model_id"].nunique() < 2:
            raise DesignError("inter_model needs at least two distinct model_id values")
    elif design is Design.ORDER_ROBUSTNESS and "condition_id" not in coders.by:
        raise DesignError(
            "order_robustness coders are candidate-order conditions: include "
            "'condition_id' in CoderSpec(by=...)"
        )

    coders_present = pd.unique(coder_labels)
    if len(coders_present) < 2:
        raise DesignError(
            f"alpha needs at least two coders; CoderSpec(by={list(coders.by)}) yields "
            f"{len(coders_present)}"
        )
    if design is Design.INTER_MODEL:
        warnings.extend(_dependence_warnings(ratings, coder_labels))

    _require_one_rating_per_coder(data["unit_id"].to_numpy(), coder_labels)
    notes = (DESIGN_NOTES[design],) if design in DESIGN_NOTES else ()
    return DesignCheck(
        notes=notes,
        warnings=tuple(warnings),
        configurations=tuple(key for key, _ in configurations),
    )


def _dependence_warnings(
    ratings: RatingFrame, coder_labels: npt.NDArray[np.object_]
) -> list[tuple[type[ReliabilityWarning], str]]:
    data = ratings._data
    found: list[tuple[type[ReliabilityWarning], str]] = []
    coder_models = pd.DataFrame({"coder": coder_labels, "model": data["model_id"].to_numpy()})
    coder_models = coder_models.drop_duplicates()
    shared_model = coder_models.groupby("model", sort=False)["coder"].agg(list)
    for model, members in shared_model.items():
        if len(members) > 1:
            found.append(
                (
                    CoderDependenceWarning,
                    f"coders {preview(members)} use the same model {model!r}; their errors are "
                    "likely correlated and alpha may be inflated",
                )
            )

    if "model_family" not in ratings.declared or data["model_family"].isna().any():
        unknown = pd.unique(data.loc[data["model_family"].isna(), "model_id"])
        found.append(
            (
                CoderDependenceWarning,
                f"model_family is missing for models {preview(unknown)}, so coder independence "
                "across model families cannot be checked",
            )
        )
        return found

    families = data[["model_id", "model_family"]].drop_duplicates()
    per_family = families.groupby("model_family", sort=False)["model_id"].agg(list)
    for family, models in per_family.items():
        if len(models) > 1:
            found.append(
                (
                    CoderDependenceWarning,
                    f"models {preview(models)} share model family {family!r}; same-family "
                    "models share errors, so alpha may be inflated",
                )
            )
    return found


def _require_one_rating_per_coder(
    units: npt.NDArray[Any], coder_labels: npt.NDArray[np.object_]
) -> None:
    keys = pd.DataFrame({"unit": units, "coder": coder_labels})
    duplicated = keys[keys.duplicated(keep="first")].drop_duplicates()
    if len(duplicated):
        pairs = list(zip(duplicated["unit"], duplicated["coder"], strict=True))
        raise DesignError(
            f"{len(pairs)} (unit, coder) pairs have more than one row {preview(pairs)}. Each "
            "coder may rate a unit once: for retried calls, map an attempt column and declare "
            "retries='last' or retries='first_observed'; for repeated runs, add run_id to the "
            "CoderSpec."
        )
