"""reliability-lab: study design for the reliability of LLM and human judgments."""

from reliability_lab._version import __version__
from reliability_lab.alpha import INTERVAL, NOMINAL, ORDINAL, RATIO, Distance
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
    UndefinedAlphaError,
)
from reliability_lab.report import (
    AlphaEstimate,
    AlphaResult,
    BootstrapResult,
    ComparisonResult,
    Coverage,
    Interval,
    Provenance,
)
from reliability_lab.split import SplitManifest
from reliability_lab.study import Fold, ReliabilityStudy

__all__ = [
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
    "Distance",
    "DomainSizeWarning",
    "Fold",
    "Interval",
    "LeakageError",
    "Provenance",
    "RatingFrame",
    "ReliabilityLabError",
    "ReliabilityStudy",
    "ReliabilityWarning",
    "SchemaError",
    "SplitManifest",
    "Status",
    "UndefinedAlphaError",
    "__version__",
]
