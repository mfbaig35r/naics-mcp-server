"""
Custom exceptions and error handling for NAICS MCP Server.

Provides a clear exception hierarchy with:
- Categorized errors (transient vs permanent)
- Retry guidance
- Structured error responses
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


# --- Error Categories ---


class ErrorCategory(str, Enum):
    """Categories of errors for handling decisions."""

    VALIDATION = "validation"  # User input error (400)
    NOT_FOUND = "not_found"  # Resource not found (404)
    TRANSIENT = "transient"  # Temporary failure, retry (503)
    PERMANENT = "permanent"  # Permanent failure, don't retry (500)
    TIMEOUT = "timeout"  # Operation timed out (504)
    CONFIGURATION = "configuration"  # Config error, fail fast (500)


# --- Base Exception ---


class NAICSException(Exception):
    """
    Base exception for all NAICS MCP Server errors.

    Attributes:
        message: Human-readable error message
        category: Error category for handling decisions
        retryable: Whether the operation can be retried
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PERMANENT,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.retryable = retryable
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "error": self.message,
            "category": self.category.value,
            "retryable": self.retryable,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# --- Specific Exceptions ---


class DatabaseError(NAICSException):
    """Database connection or query errors."""

    def __init__(
        self,
        message: str,
        retryable: bool = True,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TRANSIENT if retryable else ErrorCategory.PERMANENT,
            retryable=retryable,
            details=details,
            cause=cause,
        )


class ConnectionError(DatabaseError):
    """Database connection failure - typically transient."""

    def __init__(
        self,
        message: str = "Failed to connect to database",
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message=message, retryable=True, details=details, cause=cause)


class QueryError(DatabaseError):
    """Database query execution error."""

    def __init__(
        self,
        message: str,
        query: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        details = details or {}
        if query:
            # Truncate query for safety
            details["query_preview"] = query[:100] + "..." if len(query) > 100 else query
        super().__init__(
            message=message,
            retryable=False,  # Query errors typically aren't retryable
            details=details,
            cause=cause,
        )


class ValidationError(NAICSException):
    """Input validation error - user fixable."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        constraints: dict[str, Any] | None = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            # Sanitize value for logging
            details["value_preview"] = str(value)[:50]
        if constraints:
            details["constraints"] = constraints

        super().__init__(
            message=message, category=ErrorCategory.VALIDATION, retryable=False, details=details
        )


class NotFoundError(NAICSException):
    """Resource not found error."""

    def __init__(self, resource_type: str, identifier: str, message: str | None = None):
        msg = message or f"{resource_type} not found: {identifier}"
        super().__init__(
            message=msg,
            category=ErrorCategory.NOT_FOUND,
            retryable=False,
            details={"resource_type": resource_type, "identifier": identifier},
        )


class ConfigurationError(NAICSException):
    """Configuration or setup error - requires intervention."""

    def __init__(
        self, message: str, config_key: str | None = None, details: dict[str, Any] | None = None
    ):
        details = details or {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message, category=ErrorCategory.CONFIGURATION, retryable=False, details=details
        )


class EmbeddingError(NAICSException):
    """Embedding model or generation error."""

    def __init__(
        self,
        message: str,
        retryable: bool = True,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TRANSIENT if retryable else ErrorCategory.PERMANENT,
            retryable=retryable,
            details=details,
            cause=cause,
        )


class TimeoutError(NAICSException):
    """Operation timed out."""

    def __init__(
        self, operation: str, timeout_seconds: float, details: dict[str, Any] | None = None
    ):
        details = details or {}
        details["operation"] = operation
        details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            category=ErrorCategory.TIMEOUT,
            retryable=True,
            details=details,
        )


class SearchError(NAICSException):
    """Search operation error."""

    def __init__(
        self,
        message: str,
        query: str | None = None,
        strategy: str | None = None,
        retryable: bool = True,
        cause: Exception | None = None,
    ):
        details = {}
        if query:
            details["query_preview"] = query[:50]
        if strategy:
            details["strategy"] = strategy

        super().__init__(
            message=message,
            category=ErrorCategory.TRANSIENT if retryable else ErrorCategory.PERMANENT,
            retryable=retryable,
            details=details,
            cause=cause,
        )


# --- Retry Configuration ---


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd

    # Which exception types to retry
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


# Default retry configurations
DEFAULT_RETRY = RetryConfig()
DATABASE_RETRY = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)
EMBEDDING_RETRY = RetryConfig(max_attempts=2, base_delay=5.0, max_delay=15.0)


# --- Retry Decorators ---

T = TypeVar("T")


def retry_sync(
    config: RetryConfig = DEFAULT_RETRY, on_retry: Callable[[Exception, int], None] | None = None
):
    """
    Decorator for synchronous functions with retry logic.

    Args:
        config: Retry configuration
        on_retry: Optional callback called before each retry
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )

                except Exception:
                    # Non-retryable exception, raise immediately
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


def retry_async(
    config: RetryConfig = DEFAULT_RETRY, on_retry: Callable[[Exception, int], None] | None = None
):
    """
    Decorator for async functions with retry logic.

    Args:
        config: Retry configuration
        on_retry: Optional callback called before each retry
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}: {e}"
                        )

                except Exception:
                    # Non-retryable exception, raise immediately
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


# --- Error Response Builder ---


@dataclass
class ErrorResponse:
    """Standardized error response for MCP tools."""

    error: str
    category: str
    retryable: bool
    details: dict[str, Any] = field(default_factory=dict)

    # Optional fields for specific contexts
    fallback_used: str | None = None
    partial_results: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "error": self.error,
            "error_category": self.category,
            "retryable": self.retryable,
        }

        if self.details:
            result["error_details"] = self.details

        if self.fallback_used:
            result["fallback_used"] = self.fallback_used

        if self.partial_results is not None:
            result["partial_results"] = self.partial_results

        return result

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        fallback_used: str | None = None,
        partial_results: list[Any] | None = None,
    ) -> "ErrorResponse":
        """Create ErrorResponse from an exception."""
        if isinstance(exc, NAICSException):
            return cls(
                error=exc.message,
                category=exc.category.value,
                retryable=exc.retryable,
                details=exc.details,
                fallback_used=fallback_used,
                partial_results=partial_results,
            )
        else:
            # Generic exception
            return cls(
                error=str(exc)[:200],  # Truncate for safety
                category=ErrorCategory.PERMANENT.value,
                retryable=False,
                fallback_used=fallback_used,
                partial_results=partial_results,
            )


# --- Graceful Degradation Helper ---


async def with_fallback(
    primary: Callable, fallback: Callable, fallback_name: str, *args, **kwargs
) -> tuple[Any, str | None]:
    """
    Execute primary function with fallback on failure.

    Returns:
        Tuple of (result, fallback_used) where fallback_used is None if primary succeeded
    """
    try:
        result = await primary(*args, **kwargs)
        return result, None
    except Exception as e:
        logger.warning(f"Primary operation failed, using fallback '{fallback_name}': {e}")
        try:
            result = await fallback(*args, **kwargs)
            return result, fallback_name
        except Exception as fallback_error:
            logger.error(f"Fallback '{fallback_name}' also failed: {fallback_error}")
            raise


def handle_tool_error(
    exc: Exception, operation: str, fallback_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Standard error handler for MCP tools.

    Returns a dictionary suitable for tool response.
    """
    error_response = ErrorResponse.from_exception(exc)

    logger.error(f"Tool error in {operation}: {exc}", exc_info=True)

    result = error_response.to_dict()

    # Merge with fallback result if provided
    if fallback_result:
        result.update(fallback_result)

    return result
