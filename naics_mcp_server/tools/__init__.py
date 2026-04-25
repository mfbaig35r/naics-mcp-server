"""
Tool modules for NAICS MCP Server.
"""

from .analytics import register_tools as register_analytics
from .classification import register_tools as register_classification
from .diagnostics import register_tools as register_diagnostics
from .hierarchy import register_tools as register_hierarchy
from .relationships import register_tools as register_relationships
from .search import register_tools as register_search

# Backward compatibility for workbook request models
from .workbook import WorkbookSearchRequest as WorkbookSearchRequest
from .workbook import WorkbookTemplateRequest as WorkbookTemplateRequest
from .workbook import WorkbookWriteRequest as WorkbookWriteRequest
from .workbook import register_tools as register_workbook


def register_all_tools(mcp):
    register_search(mcp)
    register_relationships(mcp)
    register_classification(mcp)
    register_hierarchy(mcp)
    register_analytics(mcp)
    register_workbook(mcp)
    register_diagnostics(mcp)
