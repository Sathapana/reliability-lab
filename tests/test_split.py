import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold

from reliability_lab import (
    CoderDependenceWarning,
    CoderSpec,
    DesignError,
    LeakageError,
    RatingFrame,
    ReliabilityStudy,
    SplitManifest,
)
from reliability_lab.split import UnitLayout, _sample_strata

pytestmark = pytest.mark.filterwarnings("ignore::reliability_lab.CategorySkewWarning")

MODELS = (("m1", "f1"), ("m2", "f2"), ("m3", "f3"))


def llm_log(
    n_groups: int = 20,
    per_group: int = 3,
    models: tuple[tuple[str, str], ...] = MODELS,
    runs: tuple[int, ...] = (1,),
    mixed_strata: bool = False,
    prompt: str = "p1",
    seed: int = 0,
) -> pd.DataFrame:
    """Conversations of a few responses each; every fourth conversation is 'hard'."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        stratum = "hard" if g % 4 == 0 else "easy"
        for k in range(per_group):
            unit_stratum = stratum
            if mixed_strata and k == 0:
                unit_stratum = "easy" if stratum == "hard" else "hard"
            truth = int(rng.integers(1, 6))
            for model, family in models:
                for run in runs:
                    score = truth if rng.random() < 0.8 else int(rng.integers(1, 6))
                    rows.append(
                        {
                            "unit": f"c{g}-r{k}",
                            "conversation": f"c{g}",
                            "task": unit_stratum,
                            "model": model,
                            "family": family,
                            "prompt": prompt,
                            "run": run,
                            "score": score,
                        }
                    )
    return pd.DataFrame(rows)


def make_study(
    df: pd.DataFrame,
    groups: bool = True,
    strata: bool = True,
    design: str = "inter_model",
    by: tuple[str, ...] = ("model_id",),
) -> ReliabilityStudy:
    kwargs = {"model_id": "model", "model_family": "family", "condition_id": "prompt"}
    kwargs["run_id"] = "run"
    if groups:
        kwargs["group"] = "conversation"
    if strata:
        kwargs["stratum"] = "task"
    ratings = RatingFrame.from_long(df, unit="unit", rater="model", value="score", **kwargs)
    return ReliabilityStudy(
        ratings,
        measurement="ordinal",
        values=[1, 2, 3, 4, 5],
        coders=CoderSpec(by=list(by)),
        design=design,
    )


def units_of(study: ReliabilityStudy) -> set[Any]:
    return set(study.ratings.to_frame()["unit_id"])


def groups_of(study: ReliabilityStudy) -> set[Any]:
    return set(study.ratings.to_frame()["group_id"])


class HalfSplitter:
    """A minimal custom splitter: the first half is train, the rest test."""

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[Any, Any]]:
        n = len(X)
        yield np.arange(n // 2), np.arange(n // 2, n)


class LeaveOutSplitter:
    """Uses only part of the samples: the rest end up in no partition."""

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[Any, Any]]:
        n = len(X)
        yield np.arange(0, n // 2), np.arange(n // 2, n - 2)


SPLITTERS = [
    GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0),
    StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7),
    KFold(n_splits=4, shuffle=True, random_state=1),
    HalfSplitter(),
]


# -- leakage ----------------------------------------------------------------------


@pytest.mark.parametrize("splitter", SPLITTERS, ids=lambda s: type(s).__name__)
def test_holdout_keeps_groups_and_units_whole(splitter: Any) -> None:
    study = make_study(llm_log())
    dev, test = study.holdout(splitter)
    assert units_of(dev).isdisjoint(units_of(test))
    assert units_of(dev) | units_of(test) == units_of(study)
    assert groups_of(dev).isdisjoint(groups_of(test))
    assert len(dev.ratings) + len(test.ratings) == len(study.ratings)
    assert (dev.partition, test.partition) == ("dev", "test")
    assert dev.manifest is test.manifest
    assert test.manifest is not None and test.manifest.level == "group"


def test_repeated_llm_calls_stay_with_their_unit() -> None:
    df = llm_log(models=(("m1", "f1"),), runs=(1, 2, 3))
    study = make_study(df, groups=False, strata=False, design="repeatability", by=("run_id",))
    dev, test = study.holdout(KFold(n_splits=4, shuffle=True, random_state=3))
    test_rows = test.ratings.to_frame()
    assert (test_rows.groupby("unit_id").size() == 3).all()
    assert units_of(dev).isdisjoint(units_of(test))
    assert test.manifest is not None and test.manifest.level == "unit"


def test_partitions_inherit_every_declaration() -> None:
    study = make_study(llm_log())
    _, test = study.holdout(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0))
    assert test.measurement is study.measurement
    assert test.values == study.values
    assert test.coders == study.coders
    assert test.design == study.design
    assert test.missing_policies == study.missing_policies
    assert "test" in repr(test)


def test_unused_samples_get_their_own_partition() -> None:
    study = make_study(llm_log())
    dev, test = study.holdout(LeaveOutSplitter())
    assert test.manifest is not None
    assert set(test.manifest.units) == {"dev", "test", "unused"}
    assert test.manifest.realised["unused"]["groups"] == 2
    assert len(units_of(dev) | units_of(test)) == len(units_of(study)) - 6


# -- size and strata ------------------------------------------------------------------


def test_realised_size_is_reported() -> None:
    study = make_study(llm_log(n_groups=22))
    _, test = study.holdout(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7))
    assert test.manifest is not None
    realised = test.manifest.realised["test"]
    assert realised["groups"] == len(groups_of(test))
    assert realised["units"] == len(units_of(test))
    assert realised["share"] == pytest.approx(len(units_of(test)) / 66)
    assert test.manifest.n_folds == 4


def test_strata_are_balanced_by_a_stratified_splitter() -> None:
    study = make_study(llm_log(n_groups=40))
    _, test = study.holdout(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7))
    hard = test.ratings.to_frame().drop_duplicates("group_id")["stratum"].eq("hard").mean()
    assert hard == pytest.approx(0.25, abs=0.1)


def test_groups_spanning_strata_need_a_rule() -> None:
    study = make_study(llm_log(mixed_strata=True))
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7)
    with pytest.raises(DesignError, match="several strata"):
        study.holdout(splitter)
    _, test = study.holdout(splitter, group_stratum="majority")
    assert test.manifest is not None and test.manifest.group_stratum == "majority"
    mapping = {f"c{g}": "hard" if g % 2 else "easy" for g in range(20)}
    _, test = study.holdout(splitter, group_stratum=mapping)
    assert test.manifest is not None
    assert dict(map(tuple, test.manifest.group_stratum)) == mapping
    with pytest.raises(DesignError, match="missing 1 groups"):
        study.holdout(splitter, group_stratum={k: v for k, v in mapping.items() if k != "c3"})
    with pytest.raises(DesignError, match="must be None, 'majority'"):
        study.holdout(splitter, group_stratum="minority")


def test_group_stratum_rule_needs_groups_and_strata() -> None:
    no_strata = make_study(llm_log(), strata=False)
    with pytest.raises(DesignError, match="needs a stratum"):
        no_strata.holdout(GroupKFold(4), group_stratum="majority")
    no_groups = make_study(llm_log(), groups=False)
    with pytest.raises(DesignError, match="only when groups"):
        no_groups.holdout(KFold(4), group_stratum="majority")


def test_majority_breaks_ties_by_stratum_text() -> None:
    layout = UnitLayout(
        labels=np.array(["u1", "u2", "u3", "u4", "u5"], dtype=object),
        groups=np.array(["g1", "g1", "g1", "g2", "g2"], dtype=object),
        strata=np.array(["b", "a", "b", "b", "a"], dtype=object),
        ratings_fingerprint="",
    )
    y, rule = _sample_strata(
        layout, "group", np.array(["g1", "g2"], dtype=object), np.array([0, 0, 0, 1, 1]), "majority"
    )
    assert list(y) == ["b", "a"]
    assert rule == "majority"


# -- cross-validation -----------------------------------------------------------------


def test_cv_folds_partition_the_units() -> None:
    study = make_study(llm_log())
    folds = study.cv(GroupKFold(n_splits=5))
    assert [f.index for f in folds] == list(range(5))
    test_sets = [units_of(f.test) for f in folds]
    assert set().union(*test_sets) == units_of(study)
    assert sum(len(s) for s in test_sets) == len(units_of(study))
    assert all(f.manifest.n_folds == 5 and f.manifest.fold == f.index for f in folds)


def test_derived_studies_record_but_do_not_repeat_warnings() -> None:
    models = (("m1", "f1"), ("m2", "f2"), ("m3", "f2"))
    with pytest.warns(CoderDependenceWarning):
        study = make_study(llm_log(models=models))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = study.cv(GroupKFold(n_splits=3))
    assert not [w for w in caught if issubclass(w.category, CoderDependenceWarning)]
    recorded = [w.category for w in folds[0].test.alpha().warnings]
    assert "CoderDependenceWarning" in recorded


def test_bad_splitters_and_folds() -> None:
    study = make_study(llm_log())
    with pytest.raises(TypeError, match="split"):
        study.holdout(object())
    with pytest.raises(DesignError, match="fold 4 does not exist"):
        study.holdout(GroupKFold(n_splits=4), fold=4)
    with pytest.raises(DesignError, match="not a partition"):
        study.save_manifest("unused.json")


# -- manifests ------------------------------------------------------------------------


def test_manifest_round_trip_reproduces_the_split(tmp_path: Path) -> None:
    study = make_study(llm_log())
    dev, test = study.holdout(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=7))
    path = tmp_path / "split.json"
    test.save_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "reliability-lab/split-manifest"
    assert payload["schema_version"] == 1
    assert test.manifest is not None
    loaded = SplitManifest.load(path)
    assert loaded.to_dict() == test.manifest.to_dict()
    assert SplitManifest.from_dict(loaded.to_dict()).to_dict() == loaded.to_dict()

    dev2, test2 = study.apply_manifest(path)
    assert units_of(dev2) == units_of(dev) and units_of(test2) == units_of(test)
    assert test2.alpha().alpha == pytest.approx(test.alpha().alpha)


def test_manifest_applies_to_new_runs_on_the_same_units(tmp_path: Path) -> None:
    study = make_study(llm_log())
    _, test = study.holdout(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=2))
    path = tmp_path / "split.json"
    test.save_manifest(path)
    later = make_study(llm_log(prompt="p2", seed=99))
    _, later_test = later.apply_manifest(path)
    assert units_of(later_test) == units_of(test)
    assert later.ratings.fingerprint() != study.ratings.fingerprint()
    notes = later_test.alpha().notes
    assert not any("units, groups or strata differ" in n for n in notes)


def test_manifest_refuses_moved_and_unknown_units(tmp_path: Path) -> None:
    df = llm_log()
    _, test = make_study(df).holdout(GroupKFold(n_splits=4))
    path = tmp_path / "split.json"
    test.save_manifest(path)

    moved = df.copy()
    moved.loc[moved["unit"] == "c0-r0", "conversation"] = "c1"
    with pytest.raises(DesignError, match="changed group"):
        make_study(moved).apply_manifest(path)

    extra = pd.concat([df, df[df["unit"] == "c0-r0"].assign(unit="new-unit")])
    with pytest.raises(DesignError, match="not in the manifest"):
        make_study(extra).apply_manifest(path)

    with pytest.raises(DesignError, match="no group_id"):
        make_study(df, groups=False).apply_manifest(path)


def test_manifest_notes_units_without_ratings(tmp_path: Path) -> None:
    df = llm_log()
    _, test = make_study(df).holdout(GroupKFold(n_splits=4))
    path = tmp_path / "split.json"
    test.save_manifest(path)
    fewer = df[~df["conversation"].isin(["c0", "c1"])]
    dev, _ = make_study(fewer).apply_manifest(path)
    notes = dev.alpha().notes
    assert any("6 units of the manifest have no ratings" in n for n in notes)
    assert any("units, groups or strata differ" in n for n in notes)


def test_manifest_schema_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(DesignError, match="not a split manifest"):
        SplitManifest.load(path)


def test_assert_no_leakage() -> None:
    study = make_study(llm_log())
    dev, test = study.holdout(GroupKFold(n_splits=4))
    assert test.manifest is not None
    manifest = test.manifest
    dev_unit = sorted(units_of(dev))[0]
    test_unit = sorted(units_of(test))[0]
    manifest.assert_no_leakage([dev_unit])
    with pytest.raises(LeakageError, match="1 few-shot examples are in the test partition"):
        manifest.assert_no_leakage([dev_unit, test_unit])
    # Groups are never split: a unit from a test unit's conversation is itself in test.
    conversation = test_unit.split("-")[0]
    sibling = f"{conversation}-r1" if not test_unit.endswith("-r1") else f"{conversation}-r0"
    with pytest.raises(LeakageError):
        manifest.assert_no_leakage([sibling])
    with pytest.raises(DesignError, match="not in this split") as info:
        manifest.assert_no_leakage(["outside"])
    assert not isinstance(info.value, LeakageError)


def test_split_studies_report_their_partition() -> None:
    _, test = make_study(llm_log()).holdout(GroupKFold(n_splits=4))
    result = test.alpha()
    assert result.provenance is not None and result.provenance.split is not None
    assert result.provenance.split["partition"] == "test"
    assert any("test units of a group-level split" in n for n in result.notes)
    boot = result.bootstrap(scheme="units", n_resamples=50, random_state=0)
    assert boot.level == "group"
