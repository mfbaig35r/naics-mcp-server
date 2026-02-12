"""
Observability module for NAICS MCP Server.
"""

from .audit import SearchAuditLog, SearchEvent

__all__ = ["SearchAuditLog", "SearchEvent"]
