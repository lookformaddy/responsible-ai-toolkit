"""Typed errors for the responsible-ai toolkit."""


class ResponsibleAIError(Exception):
    """Base error for the toolkit."""


class ValidationError(ResponsibleAIError):
    """Raised when inputs are malformed or inconsistent."""


class EmptyGroupError(ValidationError):
    """Raised when a group or dataset has no rows to measure."""
