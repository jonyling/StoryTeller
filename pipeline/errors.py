class PipelineError(Exception):
    """Base class for pipeline errors meant to be shown to the user."""


class ValidationError(PipelineError):
    """Raised when user-supplied input fails a local validation check."""
