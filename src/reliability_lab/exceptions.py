"""Errors and warnings raised by reliability-lab."""


class ReliabilityLabError(Exception):
    """Base class for every error raised by reliability-lab."""


class SchemaError(ReliabilityLabError, ValueError):
    """The ratings table does not satisfy the long-form schema."""


class DesignError(ReliabilityLabError, ValueError):
    """The declared study design is incomplete, inconsistent with the data, or unsupported."""


class LeakageError(DesignError):
    """Data meant for tuning or few-shot examples reaches the test partition."""


class UndefinedAlphaError(ReliabilityLabError, ValueError):
    """Alpha is undefined: no pairable values, or zero expected disagreement."""


class ReliabilityWarning(UserWarning):
    """Base class for warnings about how a result should be read."""


class CoderDependenceWarning(ReliabilityWarning):
    """Coders may not be independent, so alpha's chance model may not hold."""


class CategorySkewWarning(ReliabilityWarning):
    """One category dominates, so alpha can be low even when agreement is high."""


class DomainSizeWarning(ReliabilityWarning):
    """The value domain is large enough to make V x V matrices costly."""


class BootstrapWarning(ReliabilityWarning):
    """An interval needs care: failed replicates, a resampling level that ignores
    declared groups, or an experimental method."""
