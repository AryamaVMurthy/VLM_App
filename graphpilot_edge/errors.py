class ValidationError(ValueError):
    """Raised when a workflow or plan violates explicit structural constraints."""


class EnumerationError(ValueError):
    """Raised when candidate-plan enumeration cannot proceed explicitly."""

