"""reliability-lab: study design for the reliability of LLM and human judgments."""

from reliability_lab._version import __version__
from reliability_lab.alpha import BIPOLAR, INTERVAL, NOMINAL, ORDINAL, RATIO, Distance, circular
from reliability_lab.data import RatingFrame, Status
from reliability_lab.design import CoderSpec, Design
from reliability_lab.exceptions import (
    BootstrapWarning,
    CategorySkewWarning,
    CoderDependenceWarning,
    DesignError,
    DomainSizeWarning,
    LeakageError,
    ReliabilityLabError,
    ReliabilityWarning,
    SchemaError,
    TraceWarning,
    UndefinedAlphaError,
)
from reliability_lab.llm import TraceSidecar, UnitSidecar
from reliability_lab.planning import (
    SampleSizePlan,
    plan_alpha,
    plan_krippendorff,
    plan_non_inferiority,
)
from reliability_lab.report import (
    AlphaEstimate,
    AlphaResult,
    BootstrapResult,
    ComparisonResult,
    Coverage,
    Diagnostic,
    Interval,
    JackknifeResult,
    Provenance,
)
from reliability_lab.simulate import simulate_ratings
from reliability_lab.split import SplitManifest
from reliability_lab.study import Fold, ReliabilityStudy

__all__ = [
    "BIPOLAR",
    "INTERVAL",
    "NOMINAL",
    "ORDINAL",
    "RATIO",
    "AlphaEstimate",
    "AlphaResult",
    "BootstrapResult",
    "BootstrapWarning",
    "CategorySkewWarning",
    "CoderDependenceWarning",
    "CoderSpec",
    "ComparisonResult",
    "Coverage",
    "Design",
    "DesignError",
    "Diagnostic",
    "Distance",
    "DomainSizeWarning",
    "Fold",
    "Interval",
    "JackknifeResult",
    "LeakageError",
    "Provenance",
    "RatingFrame",
    "ReliabilityLabError",
    "ReliabilityStudy",
    "ReliabilityWarning",
    "SampleSizePlan",
    "SchemaError",
    "SplitManifest",
    "Status",
    "TraceSidecar",
    "TraceWarning",
    "UndefinedAlphaError",
    "UnitSidecar",
    "__version__",
    "circular",
    "plan_alpha",
    "plan_krippendorff",
    "plan_non_inferiority",
    "simulate_ratings",
]
