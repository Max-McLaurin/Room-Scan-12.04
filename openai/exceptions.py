"""
Custom exceptions for OpenAI API interactions.
"""


class OpenAIError(Exception):
    """Base exception for OpenAI-related errors."""
    pass


class OpenAIRateLimitError(OpenAIError):
    """Raised when OpenAI API rate limit is exceeded."""
    pass


class OpenAIAPIConnectionError(OpenAIError):
    """Raised when there's a connection error with OpenAI API."""
    pass


class OpenAIAPIResponseError(OpenAIError):
    """Raised when OpenAI API returns an unexpected response."""
    pass

