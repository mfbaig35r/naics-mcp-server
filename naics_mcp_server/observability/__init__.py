"""
Observability module for NAICS MCP Server.
"""

from .audit import SearchAuditLog, SearchEvent
from .logging import (
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    generate_request_id,
    sanitize_text,
    sanitize_dict,
    log_tool_call,
    log_operation,
    log_server_start,
    log_server_ready,
    log_server_shutdown,
    LogConfig,
)

__all__ = [
    # Audit
    "SearchAuditLog",
    "SearchEvent",
    # Logging
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "generate_request_id",
    "sanitize_text",
    "sanitize_dict",
    "log_tool_call",
    "log_operation",
    "log_server_start",
    "log_server_ready",
    "log_server_shutdown",
    "LogConfig",
]
