"""
Tests for structured logging module.
"""

import json
import logging

import pytest

from naics_mcp_server.config import LoggingConfig
from naics_mcp_server.observability.logging import (
    JSONFormatter,
    LogConfig,
    StructuredLogger,
    TextFormatter,
    clear_request_context,
    generate_correlation_id,
    generate_request_id,
    get_logger,
    get_request_context,
    sanitize_dict,
    sanitize_text,
    set_request_context,
    set_service_metadata,
)


class TestSanitization:
    """Tests for sensitive data sanitization."""

    def test_sanitize_text_truncates_long_strings(self):
        """Test that long strings are truncated."""
        long_text = "a" * 1000
        result = sanitize_text(long_text, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_sanitize_text_redacts_ssn(self):
        """Test SSN redaction."""
        text = "My SSN is 123-45-6789"
        result = sanitize_text(text)
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_sanitize_text_redacts_email(self):
        """Test email redaction."""
        text = "Contact me at user@example.com"
        result = sanitize_text(text)
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_sanitize_text_handles_empty(self):
        """Test empty string handling."""
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""  # type: ignore

    def test_sanitize_dict_redacts_sensitive_keys(self):
        """Test sensitive key redaction."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "key123",
            "token": "token456",
        }
        result = sanitize_dict(data)
        assert result["username"] == "john"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"

    def test_sanitize_dict_handles_nested(self):
        """Test nested dictionary sanitization."""
        data = {
            "user": {
                "name": "john",
                "credentials": {
                    "password": "secret",
                },
            },
        }
        result = sanitize_dict(data)
        assert result["user"]["name"] == "john"
        assert result["user"]["credentials"]["password"] == "[REDACTED]"

    def test_sanitize_dict_limits_list_items(self):
        """Test list truncation."""
        data = {"items": list(range(20))}
        result = sanitize_dict(data)
        assert len(result["items"]) == 11  # 10 items + "... and 10 more"
        assert "... and 10 more" in result["items"]

    def test_sanitize_dict_custom_sensitive_keys(self):
        """Test custom sensitive key patterns."""
        data = {"custom_secret": "value", "normal": "data"}
        result = sanitize_dict(data, sensitive_keys={"custom_secret"})
        assert result["custom_secret"] == "[REDACTED]"
        assert result["normal"] == "data"


class TestRequestContext:
    """Tests for request context management."""

    def setup_method(self):
        """Clear context before each test."""
        clear_request_context()

    def teardown_method(self):
        """Clear context after each test."""
        clear_request_context()

    def test_set_and_get_request_context(self):
        """Test setting and getting request context."""
        set_request_context(
            request_id="req_123",
            session_id="sess_456",
            tool_name="search_naics_codes",
            correlation_id="corr_789",
        )

        context = get_request_context()
        assert context["request_id"] == "req_123"
        assert context["session_id"] == "sess_456"
        assert context["tool"] == "search_naics_codes"
        assert context["correlation_id"] == "corr_789"

    def test_clear_request_context(self):
        """Test clearing request context."""
        set_request_context(request_id="req_123")
        clear_request_context()

        context = get_request_context()
        assert context["request_id"] is None
        assert context["session_id"] is None
        assert context["tool"] is None

    def test_partial_context_update(self):
        """Test partial context updates."""
        set_request_context(request_id="req_123")
        set_request_context(tool_name="classify_business")

        context = get_request_context()
        assert context["request_id"] == "req_123"
        assert context["tool"] == "classify_business"


class TestIDGeneration:
    """Tests for ID generation functions."""

    def test_generate_request_id_format(self):
        """Test request ID format."""
        req_id = generate_request_id()
        assert req_id.startswith("req_")
        assert len(req_id) == 16  # "req_" + 12 hex chars

    def test_generate_request_id_unique(self):
        """Test request ID uniqueness."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_correlation_id_format(self):
        """Test correlation ID format."""
        corr_id = generate_correlation_id()
        assert corr_id.startswith("corr_")
        assert len(corr_id) == 21  # "corr_" + 16 hex chars

    def test_generate_correlation_id_unique(self):
        """Test correlation ID uniqueness."""
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100


class TestServiceMetadata:
    """Tests for service metadata."""

    def test_set_service_metadata(self):
        """Test setting service metadata."""
        set_service_metadata(
            name="test-service",
            version="1.2.3",
            environment="testing",
        )
        # Metadata is used by formatters, tested via JSONFormatter tests


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_request_context()
        set_service_metadata("test-service", "1.0.0", "test")

    def teardown_method(self):
        """Clean up."""
        clear_request_context()

    def test_format_basic_record(self):
        """Test basic log record formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_includes_service_info(self):
        """Test service info is included."""
        formatter = JSONFormatter(include_service_info=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "service" in data
        assert data["service"]["name"] == "test-service"
        assert data["service"]["version"] == "1.0.0"
        assert data["service"]["environment"] == "test"

    def test_format_includes_context(self):
        """Test request context is included."""
        set_request_context(
            request_id="req_abc123",
            tool_name="test_tool",
        )

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "context" in data
        assert data["context"]["request_id"] == "req_abc123"
        assert data["context"]["tool"] == "test_tool"

    def test_format_includes_data(self):
        """Test structured data is included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.data = {"name": "value", "count": 42}

        result = formatter.format(record)
        data = json.loads(result)

        assert "data" in data
        assert data["data"]["name"] == "value"
        assert data["data"]["count"] == 42

    def test_format_includes_source_for_warnings(self):
        """Test source location is included for warnings and above."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/project/naics_mcp_server/module.py",
            lineno=42,
            msg="Warning",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "source" in data
        assert data["source"]["line"] == 42
        assert "naics_mcp_server" in data["source"]["file"]

    def test_format_excludes_source_for_info(self):
        """Test source location is excluded for info level."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Info",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "source" not in data

    def test_format_handles_exception(self):
        """Test exception formatting."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "error" in data
        assert data["error"]["type"] == "ValueError"
        assert "Test error" in data["error"]["message"]
        assert "stack" in data["error"]

    def test_format_truncates_long_messages(self):
        """Test message truncation."""
        formatter = JSONFormatter()
        long_msg = "x" * 2000

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg=long_msg,
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert len(data["message"]) < 2000
        assert data["message"].endswith("...")

    def test_format_without_timestamp(self):
        """Test timestamp exclusion."""
        formatter = JSONFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data


class TestTextFormatter:
    """Tests for text log formatter."""

    def setup_method(self):
        """Clear context before each test."""
        clear_request_context()

    def teardown_method(self):
        """Clear context after each test."""
        clear_request_context()

    def test_format_basic_record(self):
        """Test basic text formatting."""
        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test.logger" in result
        assert "Test message" in result

    def test_format_includes_abbreviated_context(self):
        """Test context is abbreviated in text format."""
        set_request_context(
            request_id="req_abc123def456",
            tool_name="search_naics_codes",
        )

        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should show last 8 chars of request_id
        assert "def456" in result
        assert "tool=search_naics_codes" in result

    def test_format_includes_data(self):
        """Test data is included in text format."""
        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.data = {"key": "value"}

        result = formatter.format(record)

        assert "key=value" in result


class TestStructuredLogger:
    """Tests for StructuredLogger wrapper."""

    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_logger_with_data(self):
        """Test logging with structured data."""
        # This test verifies the API works without errors
        logger = get_logger("test.module")
        logger.info("Test message", data={"key": "value"})
        logger.warning("Warning", data={"count": 42})
        logger.error("Error", data={"error": "details"})


class TestLogConfig:
    """Tests for LogConfig (legacy)."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file_path is None


class TestLoggingConfig:
    """Tests for LoggingConfig (Pydantic)."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.log_file is None
        assert config.service_name == "naics-mcp-server"
        assert config.environment == "development"

    def test_validation_log_level(self):
        """Test log level validation."""
        config = LoggingConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

        with pytest.raises(ValueError):
            LoggingConfig(log_level="INVALID")

    def test_validation_log_format(self):
        """Test log format validation."""
        config = LoggingConfig(log_format="text")
        assert config.log_format == "text"

        with pytest.raises(ValueError):
            LoggingConfig(log_format="invalid")

    def test_to_dict(self):
        """Test configuration serialization."""
        config = LoggingConfig()
        config_dict = config.to_dict()

        assert "log_level" in config_dict
        assert "log_format" in config_dict
        assert "service_name" in config_dict
        assert "environment" in config_dict
