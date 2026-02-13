"""
Structured logging for NAICS MCP Server.

Provides JSON-formatted logs with request context, performance metrics,
and sensitive data handling for production deployments.
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar

# Context variables for request tracking
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_tool_name: ContextVar[str | None] = ContextVar("tool_name", default=None)


# --- Configuration ---


class LogConfig:
    """Logging configuration from environment variables."""

    def __init__(self):
        self.level = os.getenv("NAICS_LOG_LEVEL", "INFO").upper()
        self.format = os.getenv("NAICS_LOG_FORMAT", "json")  # json or text
        self.file_path = os.getenv("NAICS_LOG_FILE")
        self.max_size_mb = int(os.getenv("NAICS_LOG_MAX_SIZE_MB", "100"))
        self.retention_count = int(os.getenv("NAICS_LOG_RETENTION_COUNT", "5"))
        self.include_timestamp = os.getenv("NAICS_LOG_TIMESTAMP", "true").lower() == "true"

        # Sensitive data handling
        self.max_description_length = int(os.getenv("NAICS_LOG_MAX_DESC_LENGTH", "100"))
        self.redact_patterns = [
            r"\b\d{9}\b",  # SSN-like
            r"\b\d{10,}\b",  # Long numbers (phone, account)
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]


# Global config instance
_config = LogConfig()


# --- Context Management ---


def set_request_context(
    request_id: str | None = None, session_id: str | None = None, tool_name: str | None = None
) -> None:
    """
    Set the current request context for logging.

    Call this at the start of each request/tool invocation.
    """
    if request_id:
        _request_id.set(request_id)
    if session_id:
        _session_id.set(session_id)
    if tool_name:
        _tool_name.set(tool_name)


def clear_request_context() -> None:
    """Clear the current request context."""
    _request_id.set(None)
    _session_id.set(None)
    _tool_name.set(None)


def get_request_context() -> dict[str, str | None]:
    """Get the current request context."""
    return {
        "request_id": _request_id.get(),
        "session_id": _session_id.get(),
        "tool": _tool_name.get(),
    }


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


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

    max_len = max_length or _config.max_description_length

    # Redact sensitive patterns
    sanitized = text
    for pattern in _config.redact_patterns:
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
    """

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add request context
        context = get_request_context()
        if any(v is not None for v in context.values()):
            log_data["context"] = {k: v for k, v in context.items() if v is not None}

        # Add extra data if present
        if hasattr(record, "data") and record.data:
            log_data["data"] = record.data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """
    Formats log records as human-readable text for development.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base format
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        base = f"{timestamp} [{record.levelname:8}] {record.name}: {record.getMessage()}"

        # Add context if present
        context = get_request_context()
        context_parts = [f"{k}={v}" for k, v in context.items() if v is not None]
        if context_parts:
            base = f"{base} ({', '.join(context_parts)})"

        # Add data if present
        if hasattr(record, "data") and record.data:
            data_str = " ".join(f"{k}={v}" for k, v in record.data.items())
            base = f"{base} | {data_str}"

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
    level: str | None = None, format: str | None = None, log_file: str | None = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Log format (json or text)
        log_file: Optional file path for log output

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

    # Get root logger for naics_mcp_server
    root_logger = logging.getLogger("naics_mcp_server")
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if config.format == "json":
        formatter = JSONFormatter(include_timestamp=config.include_timestamp)
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if configured
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path, maxBytes=config.max_size_mb * 1024 * 1024, backupCount=config.retention_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


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
