"""
Unit tests for error handling module.

Tests custom exceptions, retry logic, and error response handling.
"""

import pytest

from naics_mcp_server.core.errors import (
    DATABASE_RETRY,
    EMBEDDING_RETRY,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    EmbeddingError,
    ErrorCategory,
    ErrorResponse,
    NAICSException,
    NotFoundError,
    QueryError,
    RetryConfig,
    SearchError,
    TimeoutError,
    ValidationError,
    handle_tool_error,
    retry_async,
    retry_sync,
    with_fallback,
)


class TestErrorCategory:
    """Tests for error category enumeration."""

    def test_all_categories_exist(self):
        """All expected categories should exist."""
        assert ErrorCategory.VALIDATION == "validation"
        assert ErrorCategory.NOT_FOUND == "not_found"
        assert ErrorCategory.TRANSIENT == "transient"
        assert ErrorCategory.PERMANENT == "permanent"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.CONFIGURATION == "configuration"


class TestNAICSException:
    """Tests for base exception class."""

    def test_exception_creation(self):
        """Exception should be created with all attributes."""
        exc = NAICSException(
            message="Test error",
            category=ErrorCategory.TRANSIENT,
            retryable=True,
            details={"key": "value"},
        )

        assert exc.message == "Test error"
        assert exc.category == ErrorCategory.TRANSIENT
        assert exc.retryable is True
        assert exc.details == {"key": "value"}

    def test_exception_defaults(self):
        """Exception should have sensible defaults."""
        exc = NAICSException("Test error")

        assert exc.category == ErrorCategory.PERMANENT
        assert exc.retryable is False
        assert exc.details == {}
        assert exc.cause is None

    def test_exception_with_cause(self):
        """Exception should chain to underlying cause."""
        original = ValueError("Original error")
        exc = NAICSException("Wrapper", cause=original)

        assert exc.cause is original
        assert "Original error" in str(exc)

    def test_to_dict(self):
        """Exception should convert to dictionary."""
        exc = NAICSException(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            retryable=False,
            details={"field": "code"},
        )

        result = exc.to_dict()

        assert result["error"] == "Test error"
        assert result["category"] == "validation"
        assert result["retryable"] is False
        assert result["details"] == {"field": "code"}


class TestDatabaseError:
    """Tests for database-related errors."""

    def test_default_is_retryable(self):
        """Database errors are retryable by default."""
        exc = DatabaseError("Connection failed")

        assert exc.retryable is True
        assert exc.category == ErrorCategory.TRANSIENT

    def test_non_retryable_database_error(self):
        """Database errors can be marked as non-retryable."""
        exc = DatabaseError("Invalid query", retryable=False)

        assert exc.retryable is False
        assert exc.category == ErrorCategory.PERMANENT


class TestConnectionError:
    """Tests for connection errors."""

    def test_default_message(self):
        """Connection error has a default message."""
        exc = ConnectionError()

        assert "connect to database" in exc.message
        assert exc.retryable is True


class TestQueryError:
    """Tests for query errors."""

    def test_includes_query_preview(self):
        """Query error should include truncated query."""
        long_query = "SELECT * FROM " + "a" * 200
        exc = QueryError("Syntax error", query=long_query)

        assert "query_preview" in exc.details
        assert len(exc.details["query_preview"]) <= 103  # 100 + "..."

    def test_not_retryable(self):
        """Query errors are not retryable by default."""
        exc = QueryError("Invalid SQL")

        assert exc.retryable is False


class TestValidationError:
    """Tests for validation errors."""

    def test_includes_field_info(self):
        """Validation error should include field information."""
        exc = ValidationError(message="Invalid format", field="naics_code", value="abc123")

        assert exc.details["field"] == "naics_code"
        assert "abc123" in exc.details["value_preview"]
        assert exc.category == ErrorCategory.VALIDATION

    def test_value_truncation(self):
        """Long values should be truncated."""
        long_value = "a" * 100
        exc = ValidationError("Too long", value=long_value)

        assert len(exc.details["value_preview"]) <= 50


class TestNotFoundError:
    """Tests for not found errors."""

    def test_includes_resource_info(self):
        """Not found error should include resource details."""
        exc = NotFoundError(resource_type="NAICS code", identifier="999999")

        assert "NAICS code" in exc.message
        assert "999999" in exc.message
        assert exc.details["resource_type"] == "NAICS code"
        assert exc.details["identifier"] == "999999"
        assert exc.category == ErrorCategory.NOT_FOUND


class TestConfigurationError:
    """Tests for configuration errors."""

    def test_includes_config_key(self):
        """Configuration error should include config key."""
        exc = ConfigurationError(message="Missing setting", config_key="database_path")

        assert exc.details["config_key"] == "database_path"
        assert exc.retryable is False


class TestEmbeddingError:
    """Tests for embedding errors."""

    def test_default_is_retryable(self):
        """Embedding errors are retryable by default."""
        exc = EmbeddingError("Model failed")

        assert exc.retryable is True
        assert exc.category == ErrorCategory.TRANSIENT

    def test_non_retryable_embedding_error(self):
        """Embedding errors can be non-retryable."""
        exc = EmbeddingError("Model not found", retryable=False)

        assert exc.retryable is False
        assert exc.category == ErrorCategory.PERMANENT


class TestTimeoutError:
    """Tests for timeout errors."""

    def test_includes_operation_info(self):
        """Timeout error should include operation details."""
        exc = TimeoutError(operation="embedding", timeout_seconds=30.0)

        assert "embedding" in exc.message
        assert "30" in exc.message
        assert exc.details["operation"] == "embedding"
        assert exc.details["timeout_seconds"] == 30.0
        assert exc.category == ErrorCategory.TIMEOUT


class TestSearchError:
    """Tests for search errors."""

    def test_includes_query_info(self):
        """Search error should include query details."""
        exc = SearchError(message="Search failed", query="dog food", strategy="hybrid")

        assert exc.details["query_preview"] == "dog food"
        assert exc.details["strategy"] == "hybrid"


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.jitter is True

    def test_get_delay_exponential(self):
        """Delay should increase exponentially."""
        config = RetryConfig(jitter=False)

        delay_0 = config.get_delay(0)
        delay_1 = config.get_delay(1)
        delay_2 = config.get_delay(2)

        assert delay_1 > delay_0
        assert delay_2 > delay_1

    def test_get_delay_max_cap(self):
        """Delay should not exceed max_delay."""
        config = RetryConfig(max_delay=5.0, jitter=False)

        delay = config.get_delay(10)

        assert delay == 5.0

    def test_get_delay_with_jitter(self):
        """Jitter should add randomness to delay."""
        config = RetryConfig(jitter=True)

        delays = [config.get_delay(1) for _ in range(10)]

        # With jitter, delays should vary
        assert len(set(delays)) > 1


class TestDatabaseRetryConfig:
    """Tests for pre-configured database retry."""

    def test_database_retry_config(self):
        """Database retry config should be properly configured."""
        assert DATABASE_RETRY.max_attempts == 3
        assert DATABASE_RETRY.base_delay == 1.0
        assert DATABASE_RETRY.max_delay == 10.0


class TestEmbeddingRetryConfig:
    """Tests for pre-configured embedding retry."""

    def test_embedding_retry_config(self):
        """Embedding retry config should be properly configured."""
        assert EMBEDDING_RETRY.max_attempts == 2
        assert EMBEDDING_RETRY.base_delay == 5.0
        assert EMBEDDING_RETRY.max_delay == 15.0


class TestRetrySyncDecorator:
    """Tests for synchronous retry decorator."""

    def test_retry_on_success(self):
        """Should return result on success."""

        @retry_sync(RetryConfig(max_attempts=3))
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_on_transient_failure(self):
        """Should retry on transient failures."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable(self):
        """Should not retry non-retryable exceptions."""
        call_count = 0

        @retry_sync(RetryConfig(max_attempts=3))
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            failing_func()

        assert call_count == 1  # No retries


class TestRetryAsyncDecorator:
    """Tests for asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_on_success(self):
        """Should return result on success."""

        @retry_async(RetryConfig(max_attempts=3))
        async def successful_func():
            return "success"

        result = await successful_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry_on_transient_failure(self):
        """Should retry on transient failures."""
        call_count = 0

        @retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3


class TestErrorResponse:
    """Tests for error response builder."""

    def test_error_response_creation(self):
        """Error response should be created with all fields."""
        response = ErrorResponse(
            error="Test error", category="transient", retryable=True, details={"key": "value"}
        )

        assert response.error == "Test error"
        assert response.category == "transient"
        assert response.retryable is True
        assert response.details == {"key": "value"}

    def test_to_dict(self):
        """Error response should convert to dictionary."""
        response = ErrorResponse(
            error="Test error", category="validation", retryable=False, fallback_used="lexical"
        )

        result = response.to_dict()

        assert result["error"] == "Test error"
        assert result["error_category"] == "validation"
        assert result["retryable"] is False
        assert result["fallback_used"] == "lexical"

    def test_from_naics_exception(self):
        """Should create from NAICS exception."""
        exc = ValidationError("Invalid code", field="naics_code")
        response = ErrorResponse.from_exception(exc)

        assert response.error == "Invalid code"
        assert response.category == "validation"
        assert response.retryable is False

    def test_from_generic_exception(self):
        """Should create from generic exception."""
        exc = RuntimeError("Unknown error")
        response = ErrorResponse.from_exception(exc)

        assert "Unknown error" in response.error
        assert response.category == "permanent"
        assert response.retryable is False


class TestWithFallback:
    """Tests for fallback helper."""

    @pytest.mark.asyncio
    async def test_primary_success(self):
        """Should return primary result when successful."""

        async def primary():
            return "primary result"

        async def fallback():
            return "fallback result"

        result, used = await with_fallback(primary, fallback, "fallback")

        assert result == "primary result"
        assert used is None

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Should use fallback when primary fails."""

        async def primary():
            raise ValueError("Primary failed")

        async def fallback():
            return "fallback result"

        result, used = await with_fallback(primary, fallback, "fallback")

        assert result == "fallback result"
        assert used == "fallback"

    @pytest.mark.asyncio
    async def test_raises_when_both_fail(self):
        """Should raise when both primary and fallback fail."""

        async def primary():
            raise ValueError("Primary failed")

        async def fallback():
            raise RuntimeError("Fallback failed")

        with pytest.raises(RuntimeError):
            await with_fallback(primary, fallback, "fallback")


class TestHandleToolError:
    """Tests for tool error handler."""

    def test_handle_naics_exception(self):
        """Should handle NAICS exceptions."""
        exc = SearchError("Search failed", query="test")
        result = handle_tool_error(exc, "search_tool")

        assert "error" in result
        assert "error_category" in result
        assert "retryable" in result

    def test_handle_generic_exception(self):
        """Should handle generic exceptions."""
        exc = RuntimeError("Unknown error")
        result = handle_tool_error(exc, "some_tool")

        assert "error" in result
        assert result["retryable"] is False

    def test_merge_fallback_result(self):
        """Should merge with fallback result."""
        exc = SearchError("Search failed")
        fallback = {"results": [], "partial": True}
        result = handle_tool_error(exc, "search_tool", fallback)

        assert "error" in result
        assert "results" in result
        assert result["partial"] is True
