"""reliability-lab: study design for the reliability of LLM and human judgments."""

from reliability_lab._version import __version__
from reliability_lab.alpha import INTERVAL, NOMINAL, ORDINAL, RATIO, Distance
from reliability_lab.data import RatingFrame, Status
from reliability_lab.design import CoderSpec, Design
from reliability_lab.exceptions import (
    CategorySkewWarning,
    CoderDependenceWarning,
    DesignError,
    DomainSizeWarning,
    ReliabilityLabError,
    ReliabilityWarning,
    SchemaError,
    UndefinedAlphaError,
)
from reliability_lab.report import AlphaEstimate, AlphaResult, Coverage
from reliability_lab.study import ReliabilityStudy

__all__ = [
    "INTERVAL",
    "NOMINAL",
    "ORDINAL",
    "RATIO",
    "AlphaEstimate",
    "AlphaResult",
    "CategorySkewWarning",
    "CoderDependenceWarning",
    "CoderSpec",
    "Coverage",
    "Design",
    "DesignError",
    "Distance",
    "DomainSizeWarning",
    "RatingFrame",
    "ReliabilityLabError",
    "ReliabilityStudy",
    "ReliabilityWarning",
    "SchemaError",
    "Status",
    "UndefinedAlphaError",
    "__version__",
]
