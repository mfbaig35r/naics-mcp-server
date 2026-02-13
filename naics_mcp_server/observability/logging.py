"""
Structured logging for NAICS MCP Server.

Provides JSON-formatted logs with request context, performance metrics,
and sensitive data handling for production deployments.

Features:
- JSON and text formatters for different environments
- Request context tracking (request_id, session_id, correlation_id)
- Sensitive data redaction (SSN, email, phone, secrets)
- Service metadata (name, version, environment)
- Exception logging with stack frame details
- Rotating file handler support
"""

import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar

# Context variables for request tracking
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_tool_name: ContextVar[str | None] = ContextVar("tool_name", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Service metadata (set at startup)
_service_name: str = "naics-mcp-server"
_service_version: str = "0.1.0"
_environment: str = "development"

# Sensitive data patterns for redaction
REDACT_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
    r"\b\d{9}\b",  # SSN without dashes
    r"\b\d{10,16}\b",  # Long numbers (phone, credit card, account)
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
]


# --- Configuration ---


class LogConfig:
    """
    Logging configuration from environment variables.

    Note: For production, prefer using LoggingConfig from config.py which
    provides Pydantic validation. This class is maintained for backwards
    compatibility with setup_logging().
    """

    def __init__(self):
        self.level = os.getenv("NAICS_LOG_LEVEL", "INFO").upper()
        self.format = os.getenv("NAICS_LOG_FORMAT", "json")  # json or text
        self.file_path = os.getenv("NAICS_LOG_FILE")
        self.max_size_mb = int(os.getenv("NAICS_LOG_MAX_SIZE_MB", "100"))
        self.retention_count = int(os.getenv("NAICS_LOG_RETENTION_COUNT", "5"))
        self.include_timestamp = os.getenv("NAICS_LOG_TIMESTAMP", "true").lower() == "true"

        # Service metadata
        self.service_name = os.getenv("NAICS_SERVICE_NAME", "naics-mcp-server")
        self.environment = os.getenv("NAICS_ENVIRONMENT", "development")

        # Sensitive data handling
        self.max_message_length = int(os.getenv("NAICS_LOG_MAX_MSG_LENGTH", "1000"))
        self.max_data_length = int(os.getenv("NAICS_LOG_MAX_DATA_LENGTH", "500"))
        self.redact_patterns = REDACT_PATTERNS


# Global config instance
_config = LogConfig()


# --- Context Management ---


def set_request_context(
    request_id: str | None = None,
    session_id: str | None = None,
    tool_name: str | None = None,
    correlation_id: str | None = None,
) -> None:
    """
    Set the current request context for logging.

    Call this at the start of each request/tool invocation.

    Args:
        request_id: Unique ID for this specific request
        session_id: MCP session ID (if available)
        tool_name: Name of the tool being invoked
        correlation_id: ID for tracing across services (propagated from upstream)
    """
    if request_id:
        _request_id.set(request_id)
    if session_id:
        _session_id.set(session_id)
    if tool_name:
        _tool_name.set(tool_name)
    if correlation_id:
        _correlation_id.set(correlation_id)


def clear_request_context() -> None:
    """Clear the current request context."""
    _request_id.set(None)
    _session_id.set(None)
    _tool_name.set(None)
    _correlation_id.set(None)


def get_request_context() -> dict[str, str | None]:
    """Get the current request context."""
    return {
        "request_id": _request_id.get(),
        "session_id": _session_id.get(),
        "tool": _tool_name.get(),
        "correlation_id": _correlation_id.get(),
    }


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for distributed tracing."""
    return f"corr_{uuid.uuid4().hex[:16]}"


def set_service_metadata(name: str, version: str, environment: str) -> None:
    """
    Set service metadata for log records.

    Call this once at application startup.

    Args:
        name: Service name (e.g., "naics-mcp-server")
        version: Service version (e.g., "0.1.0")
        environment: Environment name (e.g., "production")
    """
    global _service_name, _service_version, _environment
    _service_name = name
    _service_version = version
    _environment = environment


# --- Sensitive Data Handling ---


def sanitize_text(text: str, max_length: int | None = None) -> str:
    """
    Sanitize text for logging by truncating and redacting sensitive data.

    Args:
        text: The text to sanitize
        max_length: Maximum length (uses config default if None)

    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return ""

    max_len = max_length or _config.max_data_length

    # Redact sensitive patterns
    sanitized = text
    for pattern in REDACT_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)

    # Truncate if needed
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "..."

    return sanitized


def sanitize_dict(data: dict[str, Any], sensitive_keys: set | None = None) -> dict[str, Any]:
    """
    Sanitize a dictionary for logging.

    Args:
        data: Dictionary to sanitize
        sensitive_keys: Keys to fully redact (default: password, token, secret, key)

    Returns:
        Sanitized dictionary safe for logging
    """
    if sensitive_keys is None:
        sensitive_keys = {"password", "token", "secret", "key", "api_key", "auth"}

    result = {}
    for k, v in data.items():
        k_lower = k.lower()
        if any(sensitive in k_lower for sensitive in sensitive_keys):
            result[k] = "[REDACTED]"
        elif isinstance(v, str):
            result[k] = sanitize_text(v)
        elif isinstance(v, dict):
            result[k] = sanitize_dict(v, sensitive_keys)
        elif isinstance(v, list):
            result[k] = [
                sanitize_text(item) if isinstance(item, str) else item
                for item in v[:10]  # Limit list items
            ]
            if len(v) > 10:
                result[k].append(f"... and {len(v) - 10} more")
        else:
            result[k] = v

    return result


# --- JSON Formatter ---


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.

    Produces logs compatible with ELK stack, Datadog, and other log aggregators.
    Each log line is a single JSON object with consistent field names.
    """

    def __init__(self, include_timestamp: bool = True, include_service_info: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_service_info = include_service_info

    def format(self, record: logging.LogRecord) -> str:
        # Core log fields
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._truncate_message(record.getMessage()),
        }

        # ISO 8601 timestamp with timezone
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(UTC).isoformat()

        # Service metadata for log aggregation
        if self.include_service_info:
            log_data["service"] = {
                "name": _service_name,
                "version": _service_version,
                "environment": _environment,
            }

        # Request context for tracing
        context = get_request_context()
        context_values = {k: v for k, v in context.items() if v is not None}
        if context_values:
            log_data["context"] = context_values

        # Structured data attached to log
        if hasattr(record, "data") and record.data:
            log_data["data"] = self._sanitize_data(record.data)

        # Exception handling with stack frames
        if record.exc_info:
            log_data["error"] = self._format_exception(record.exc_info)

        # Source location for warnings and errors
        if record.levelno >= logging.WARNING:
            log_data["source"] = {
                "file": self._shorten_path(record.pathname),
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str, ensure_ascii=False)

    def _truncate_message(self, message: str) -> str:
        """Truncate message if too long."""
        max_len = _config.max_message_length
        if len(message) > max_len:
            return message[:max_len] + "..."
        return message

    def _sanitize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize data dictionary for logging."""
        return sanitize_dict(data)

    def _shorten_path(self, pathname: str) -> str:
        """Shorten file path for readability."""
        # Extract relative path from naics_mcp_server
        if "naics_mcp_server" in pathname:
            idx = pathname.find("naics_mcp_server")
            return pathname[idx:]
        return pathname

    def _format_exception(self, exc_info) -> dict[str, Any]:
        """Format exception with structured stack trace."""
        exc_type, exc_value, exc_tb = exc_info

        error_data: dict[str, Any] = {
            "type": exc_type.__name__ if exc_type else "Unknown",
            "message": str(exc_value)[:500] if exc_value else "",
        }

        # Extract stack frames (limit to avoid huge logs)
        if exc_tb:
            frames = []
            for frame_info in traceback.extract_tb(exc_tb)[-5:]:  # Last 5 frames
                frames.append(
                    {
                        "file": self._shorten_path(frame_info.filename),
                        "line": frame_info.lineno,
                        "function": frame_info.name,
                        "code": frame_info.line[:100] if frame_info.line else None,
                    }
                )
            error_data["stack"] = frames

        return error_data


class TextFormatter(logging.Formatter):
    """
    Formats log records as human-readable text for development.

    Uses colors for different log levels when outputting to a terminal.
    """

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Level with optional color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"

        # Base format
        base = f"{timestamp} [{level_str}] {record.name}: {record.getMessage()}"

        # Add context if present (abbreviated for readability)
        context = get_request_context()
        context_parts = []
        if context.get("request_id"):
            context_parts.append(f"req={context['request_id'][-8:]}")  # Last 8 chars
        if context.get("tool"):
            context_parts.append(f"tool={context['tool']}")
        if context.get("correlation_id"):
            context_parts.append(f"corr={context['correlation_id'][-8:]}")

        if context_parts:
            base = f"{base} ({', '.join(context_parts)})"

        # Add data if present
        if hasattr(record, "data") and record.data:
            data_items = []
            for k, v in list(record.data.items())[:5]:  # Limit items
                v_str = str(v)[:50]  # Truncate values
                data_items.append(f"{k}={v_str}")
            data_str = " ".join(data_items)
            base = f"{base} | {data_str}"

        # Add exception info if present
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


# --- Logger Wrapper ---


class StructuredLogger:
    """
    A logger wrapper that supports structured data logging.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, message: str, data: dict[str, Any] | None = None, **kwargs):
        """Internal logging method that attaches structured data."""
        extra = {"data": data} if data else {}
        self.logger.log(level, message, extra=extra, **kwargs)

    def debug(self, message: str, data: dict[str, Any] | None = None, **kwargs):
        self._log(logging.DEBUG, message, data, **kwargs)

    def info(self, message: str, data: dict[str, Any] | None = None, **kwargs):
        self._log(logging.INFO, message, data, **kwargs)

    def warning(self, message: str, data: dict[str, Any] | None = None, **kwargs):
        self._log(logging.WARNING, message, data, **kwargs)

    def error(self, message: str, data: dict[str, Any] | None = None, **kwargs):
        self._log(logging.ERROR, message, data, **kwargs)

    def exception(self, message: str, data: dict[str, Any] | None = None, **kwargs):
        self._log(logging.ERROR, message, data, exc_info=True, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


# --- Setup Functions ---


def setup_logging(
    level: str | None = None,
    format: str | None = None,
    log_file: str | None = None,
    service_name: str | None = None,
    service_version: str | None = None,
    environment: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Log format (json or text)
        log_file: Optional file path for log output
        service_name: Service name for log records
        service_version: Service version for log records
        environment: Environment name (development, staging, production)

    Call this once at application startup.
    """
    config = _config

    # Override config with parameters
    if level:
        config.level = level.upper()
    if format:
        config.format = format
    if log_file:
        config.file_path = log_file

    # Set service metadata
    set_service_metadata(
        name=service_name or config.service_name,
        version=service_version or "0.1.0",
        environment=environment or config.environment,
    )

    # Get root logger for naics_mcp_server
    root_logger = logging.getLogger("naics_mcp_server")
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter based on format type
    if config.format == "json":
        formatter = JSONFormatter(include_timestamp=config.include_timestamp)
    else:
        formatter = TextFormatter(use_colors=True)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if configured (always JSON for file logs)
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path, maxBytes=config.max_size_mb * 1024 * 1024, backupCount=config.retention_count
        )
        # Always use JSON for file logs (easier to parse)
        file_handler.setFormatter(JSONFormatter(include_timestamp=True))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def setup_logging_from_config() -> None:
    """
    Configure logging from LoggingConfig (Pydantic).

    This is the preferred method for production deployments.
    """
    try:
        from ..config import get_logging_config, get_server_config

        logging_config = get_logging_config()
        server_config = get_server_config()

        setup_logging(
            level=logging_config.log_level,
            format=logging_config.log_format,
            log_file=logging_config.log_file,
            service_name=logging_config.service_name,
            service_version=server_config.version,
            environment=logging_config.environment,
        )
    except Exception:
        # Fall back to environment variables if config not available
        setup_logging()


# --- Decorators ---

F = TypeVar("F", bound=Callable[..., Any])


def log_tool_call(func: F) -> F:
    """
    Decorator to log MCP tool calls with timing and context.

    Usage:
        @log_tool_call
        async def my_tool(request: MyRequest, ctx: Context) -> Dict:
            ...
    """
    logger = get_logger(func.__module__)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Generate request ID and set context
        request_id = generate_request_id()
        tool_name = func.__name__

        # Try to extract session_id from Context if available
        session_id = None
        for arg in args:
            if hasattr(arg, "request_context"):
                # MCP Context object
                session_id = getattr(arg.request_context, "session_id", None)
                break

        set_request_context(request_id=request_id, session_id=session_id, tool_name=tool_name)

        start_time = time.time()

        # Log request (sanitize input)
        input_summary = {}
        for arg in args:
            if hasattr(arg, "__dict__"):
                input_summary = sanitize_dict(vars(arg))
                break

        logger.info(f"Tool invoked: {tool_name}", data={"input_preview": input_summary})

        try:
            result = await func(*args, **kwargs)

            latency_ms = int((time.time() - start_time) * 1000)

            # Log success
            logger.info(
                f"Tool completed: {tool_name}", data={"latency_ms": latency_ms, "success": True}
            )

            # Log slow requests
            if latency_ms > 200:
                logger.warning(
                    f"Slow tool execution: {tool_name}",
                    data={"latency_ms": latency_ms, "threshold_ms": 200},
                )

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            logger.exception(
                f"Tool failed: {tool_name}",
                data={
                    "latency_ms": latency_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                },
            )
            raise

        finally:
            clear_request_context()

    return wrapper  # type: ignore


def log_operation(operation_name: str):
    """
    Decorator to log any operation with timing.

    Usage:
        @log_operation("embedding_generation")
        async def generate_embeddings():
            ...
    """

    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            logger.debug(f"Starting: {operation_name}")

            try:
                result = await func(*args, **kwargs)
                latency_ms = int((time.time() - start_time) * 1000)

                logger.info(f"Completed: {operation_name}", data={"latency_ms": latency_ms})

                return result

            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                logger.error(
                    f"Failed: {operation_name}",
                    data={"latency_ms": latency_ms, "error": str(e)[:200]},
                )
                raise

        return wrapper  # type: ignore

    return decorator


# --- Startup/Shutdown Logging ---


def log_server_start(config: dict[str, Any]) -> None:
    """Log server startup with configuration summary."""
    logger = get_logger("naics_mcp_server.server")

    # Sanitize config before logging
    safe_config = sanitize_dict(config)

    logger.info(
        "NAICS MCP Server starting",
        data={"config": safe_config, "log_level": _config.level, "log_format": _config.format},
    )


def log_server_ready(stats: dict[str, Any]) -> None:
    """Log server ready with data statistics."""
    logger = get_logger("naics_mcp_server.server")

    logger.info(
        "NAICS MCP Server ready",
        data={
            "naics_codes": stats.get("total_codes", 0),
            "cross_references": stats.get("total_cross_references", 0),
            "embeddings": stats.get("embeddings_count", 0),
        },
    )


def log_server_shutdown() -> None:
    """Log server shutdown."""
    logger = get_logger("naics_mcp_server.server")
    logger.info("NAICS MCP Server shutting down")
