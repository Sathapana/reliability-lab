"""Group-safe train/test splits that delegate to any scikit-learn-style splitter.

Alpha is never fitted, but teams tune the judge (prompt, few-shot examples, rubric,
score mapping) and must report reliability on units the tuning never saw. A split
assigns whole units to partitions, so every rating and every repeated LLM call of a
unit stays together. When groups are declared it assigns whole groups: the splitter
sees one sample per group, so no splitter, group-aware or not, can break a group.

A :class:`SplitManifest` persists the ids of every partition, not only the seed, so
the split can be reapplied to a later log, such as new judge runs on the same units.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_lab._version import __version__
from reliability_lab.data import _plain, fingerprint_rows, preview
from reliability_lab.exceptions import DesignError, LeakageError

SCHEMA = "reliability-lab/split-manifest"
SCHEMA_VERSION = 1
#: Partition names: what the splitter calls train is ``dev`` here.
DEV, TEST, UNUSED = "dev", "test", "unused"

IntArray = npt.NDArray[np.intp]
ObjArray = npt.NDArray[np.object_]


@dataclass(frozen=True, eq=False)
class SplitManifest:
    """The assignment of units (or groups) to partitions, and how it was made.

    ``partitions`` holds the ids the splitter assigned (groups at group level,
    units otherwise), and ``units`` the units each partition holds.
    ``unit_groups`` maps each unit to its group at group level. A splitter that
    leaves some samples out (for example with both train and test sizes set)
    produces an ``unused`` partition.
    """

    level: str
    partitions: Mapping[str, tuple[Any, ...]]
    units: Mapping[str, tuple[Any, ...]]
    unit_groups: Mapping[Any, Any] | None
    splitter: str
    splitter_class: str
    random_state: Any
    fold: int
    n_folds: int | None
    group_stratum: Any
    realised: Mapping[str, Mapping[str, float]]
    ratings_fingerprint: str
    units_fingerprint: str
    reliability_lab_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION

    def partition_of(self) -> dict[Any, str]:
        """Each unit's partition."""
        return {u: p for p, units in self.units.items() for u in units}

    def assert_no_leakage(self, few_shot_units: Iterable[Any], partition: str = TEST) -> None:
        """Raise :class:`LeakageError` if a few-shot example sits in ``partition``.

        Few-shot examples drawn from the labelled set must come from the dev
        partition. Groups are never split, so a unit's partition is also its group's:
        a few-shot unit from the same conversation as a test unit is itself in test.
        """
        where = self.partition_of()
        units = [_plain(u) for u in few_shot_units]
        unknown = [u for u in units if u not in where]
        if unknown:
            raise DesignError(
                f"{len(unknown)} few-shot units are not in this split {preview(unknown)}, so "
                "their partition cannot be checked"
            )
        leaked = [u for u in units if where[u] == partition]
        if leaked:
            raise LeakageError(
                f"{len(leaked)} few-shot examples are in the {partition} partition "
                f"{preview(leaked)}; few-shot examples must come from the dev partition"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "reliability_lab_version": self.reliability_lab_version,
            "created_at": self.created_at,
            "level": self.level,
            "splitter": self.splitter,
            "splitter_class": self.splitter_class,
            "random_state": self.random_state,
            "fold": self.fold,
            "n_folds": self.n_folds,
            "group_stratum": self.group_stratum,
            "realised": {p: dict(r) for p, r in self.realised.items()},
            "ratings_fingerprint": self.ratings_fingerprint,
            "units_fingerprint": self.units_fingerprint,
            "partitions": {p: list(ids) for p, ids in self.partitions.items()},
            "units": {p: list(ids) for p, ids in self.units.items()},
            "unit_groups": None
            if self.unit_groups is None
            else [[u, g] for u, g in self.unit_groups.items()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitManifest:
        if payload.get("schema") != SCHEMA:
            raise DesignError(f"not a split manifest (schema {payload.get('schema')!r})")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise DesignError(
                f"split manifest schema version {payload.get('schema_version')} is not "
                f"supported (expected {SCHEMA_VERSION})"
            )
        pairs = payload["unit_groups"]
        return cls(
            level=payload["level"],
            partitions={p: tuple(ids) for p, ids in payload["partitions"].items()},
            units={p: tuple(ids) for p, ids in payload["units"].items()},
            unit_groups=None if pairs is None else {u: g for u, g in pairs},
            splitter=payload["splitter"],
            splitter_class=payload["splitter_class"],
            random_state=payload["random_state"],
            fold=payload["fold"],
            n_folds=payload["n_folds"],
            group_stratum=payload["group_stratum"],
            realised=payload["realised"],
            ratings_fingerprint=payload["ratings_fingerprint"],
            units_fingerprint=payload["units_fingerprint"],
            reliability_lab_version=payload["reliability_lab_version"],
            created_at=payload["created_at"],
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SplitManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class UnitLayout:
    """The units of a study with their group and stratum labels (None if undeclared)."""

    labels: ObjArray
    groups: ObjArray | None
    strata: ObjArray | None
    ratings_fingerprint: str

    def fingerprint(self) -> str:
        n = len(self.labels)
        return fingerprint_rows(
            {
                "unit_id": self.labels,
                "group_id": self.groups if self.groups is not None else [None] * n,
                "stratum": self.strata if self.strata is not None else [None] * n,
            }
        )


def make_manifests(
    layout: UnitLayout, splitter: Any, group_stratum: Any, folds: Sequence[int] | None = None
) -> list[SplitManifest]:
    """Run ``splitter`` over the samples (groups or units) and record each fold."""
    split = getattr(splitter, "split", None)
    if not callable(split):
        raise TypeError(
            f"splitter must have a scikit-learn style split(X, y, groups) method; got "
            f"{type(splitter).__name__}"
        )
    level, sample_ids, unit_sample = _samples(layout)
    y, rule = _sample_strata(layout, level, sample_ids, unit_sample, group_stratum)
    n = len(sample_ids)
    x = np.zeros((n, 1))
    groups = np.arange(n)
    with warnings.catch_warnings():
        # Each group is one sample, so a splitter that ignores groups is still safe.
        warnings.filterwarnings("ignore", message="The groups parameter is ignored")
        assignments = [(np.asarray(tr), np.asarray(te)) for tr, te in split(x, y, groups)]
        try:
            n_folds: int | None = int(splitter.get_n_splits(x, y, groups))
        except Exception:
            n_folds = len(assignments)
    if folds is not None:
        for k in folds:
            if not 0 <= k < len(assignments):
                raise DesignError(f"fold {k} does not exist; the splitter made {len(assignments)}")
        assignments = [assignments[k] for k in folds]
        indices = list(folds)
    else:
        indices = list(range(len(assignments)))
    random_state = getattr(splitter, "random_state", None)
    if not (random_state is None or isinstance(random_state, int | np.integer)):
        random_state = repr(random_state)
    units_fp = layout.fingerprint()

    manifests = []
    for k, (train, test) in zip(indices, assignments, strict=True):
        if np.intersect1d(train, test).size:
            raise DesignError(f"the splitter put the same samples in train and test (fold {k})")
        part_of_sample = np.full(n, UNUSED, dtype=object)
        part_of_sample[train] = DEV
        part_of_sample[test] = TEST
        part_of_unit = part_of_sample[unit_sample]
        names = [p for p in (DEV, TEST, UNUSED) if (part_of_sample == p).any()]
        realised: dict[str, dict[str, float]] = {}
        for p in names:
            n_units = int((part_of_unit == p).sum())
            realised[p] = {"units": n_units, "share": n_units / len(layout.labels)}
            if level == "group":
                realised[p]["groups"] = int((part_of_sample == p).sum())
        manifests.append(
            SplitManifest(
                level=level,
                partitions={p: _ids(sample_ids[part_of_sample == p]) for p in names},
                units={p: _ids(layout.labels[part_of_unit == p]) for p in names},
                unit_groups=None
                if layout.groups is None
                else dict(zip(_ids(layout.labels), _ids(layout.groups), strict=True)),
                splitter=repr(splitter),
                splitter_class=f"{type(splitter).__module__}.{type(splitter).__qualname__}",
                random_state=_plain(random_state),
                fold=k,
                n_folds=n_folds,
                group_stratum=rule,
                realised=realised,
                ratings_fingerprint=layout.ratings_fingerprint,
                units_fingerprint=units_fp,
            )
        )
    return manifests


def assign_units(layout: UnitLayout, manifest: SplitManifest) -> tuple[ObjArray, list[str]]:
    """Each unit's partition under a saved manifest, plus notes on what changed.

    Every unit must be in the manifest, and at group level in the same group as
    when the manifest was made; otherwise the split could leak.
    """
    where = manifest.partition_of()
    labels = [_plain(u) for u in layout.labels]
    unknown = [u for u in labels if u not in where]
    if unknown:
        raise DesignError(
            f"{len(unknown)} units are not in the manifest {preview(unknown)}; a saved split "
            "can only be applied to the units it assigned. Filter the ratings first."
        )
    if manifest.level == "group":
        if layout.groups is None or manifest.unit_groups is None:
            raise DesignError("the manifest splits by group, but no group_id is declared")
        moved = [
            (u, g)
            for u, g in zip(labels, (_plain(g) for g in layout.groups), strict=True)
            if manifest.unit_groups.get(u) != g
        ]
        if moved:
            raise DesignError(
                f"{len(moved)} units changed group since the manifest was made "
                f"{preview(moved)}; reapplying the split could leak"
            )
    notes = []
    absent = len(where) - len(set(labels))
    if absent:
        notes.append(f"{absent} units of the manifest have no ratings in these data.")
    if layout.fingerprint() != manifest.units_fingerprint:
        notes.append("The units, groups or strata differ from those the manifest was made on.")
    return np.array([where[u] for u in labels], dtype=object), notes


def _samples(layout: UnitLayout) -> tuple[str, ObjArray, IntArray]:
    """The splitter's samples: groups when declared, otherwise units."""
    if layout.groups is None:
        return "unit", layout.labels, np.arange(len(layout.labels))
    codes, uniques = pd.factorize(pd.Series(layout.groups, dtype=object), sort=False)
    return "group", np.asarray(uniques, dtype=object), np.asarray(codes, dtype=np.intp)


def _sample_strata(
    layout: UnitLayout, level: str, sample_ids: ObjArray, unit_sample: IntArray, rule: Any
) -> tuple[npt.NDArray[np.str_] | None, Any]:
    """Stratum labels per sample (as text, for the splitter's ``y``), and the rule used."""
    if layout.strata is None:
        if rule is not None:
            raise DesignError("group_stratum needs a stratum column declared")
        return None, None
    if level == "unit":
        if rule is not None:
            raise DesignError("group_stratum applies only when groups are declared")
        return np.array([str(s) for s in layout.strata]), None

    frame = pd.DataFrame({"sample": unit_sample, "stratum": layout.strata})
    if rule is None:
        spread = frame.groupby("sample")["stratum"].nunique()
        mixed = spread.index[spread > 1]
        if len(mixed):
            raise DesignError(
                f"{len(mixed)} groups hold units from several strata {preview(sample_ids[mixed])}; "
                "pass group_stratum='majority' or a mapping from group to stratum"
            )
        strata = frame.drop_duplicates("sample").sort_values("sample")["stratum"].tolist()
        return np.array([str(s) for s in strata]), None
    if isinstance(rule, str) and rule == "majority":
        strata = []
        for _, members in frame.groupby("sample", sort=True)["stratum"]:
            counts = members.value_counts()
            top = counts.max()
            strata.append(min((s for s, c in counts.items() if c == top), key=str))
        return np.array([str(s) for s in strata]), "majority"
    if isinstance(rule, Mapping):
        missing = [g for g in sample_ids if g not in rule]
        if missing:
            raise DesignError(f"group_stratum is missing {len(missing)} groups {preview(missing)}")
        return np.array([str(rule[g]) for g in sample_ids]), [
            [_plain(g), _plain(rule[g])] for g in sample_ids
        ]
    raise DesignError(
        f"group_stratum must be None, 'majority' or a mapping from group to stratum; got {rule!r}"
    )


def _ids(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(_plain(v) for v in values)
