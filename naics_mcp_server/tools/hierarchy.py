"""
Hierarchy tools for NAICS MCP Server.

Provides: get_code_hierarchy, get_children, get_siblings
"""

from typing import Any

from mcp.server.fastmcp import Context

from ..app_context import get_app_context
from ..core.errors import NAICSException, handle_tool_error
from ..models.response_models import ChildrenResponse, HierarchyResponse, SiblingsResponse
from ..observability.logging import get_logger

logger = get_logger(__name__)


def register_tools(mcp):
    """Register hierarchy tools on the MCP server."""

    @mcp.tool()
    async def get_code_hierarchy(naics_code: str, ctx: Context) -> dict[str, Any]:
        """
        Get the complete hierarchical path for a NAICS code.

        Shows how a code fits into the classification system from
        Sector (2-digit) down to National Industry (6-digit).
        """
        app_ctx = get_app_context(ctx)

        try:
            hierarchy = await app_ctx.database.get_hierarchy(naics_code)

            if not hierarchy:
                return {"error": f"NAICS code {naics_code} not found", "hierarchy": []}

            hierarchy_list = [
                {
                    "level": code.level.value,
                    "code": code.node_code,
                    "title": code.title,
                    "description": code.description[:200] + "..."
                    if code.description and len(code.description) > 200
                    else code.description,
                }
                for code in hierarchy
            ]

            return HierarchyResponse(naics_code=naics_code, hierarchy=hierarchy_list).to_dict()

        except NAICSException as e:
            logger.error(f"Failed to get hierarchy: {e}")
            result = handle_tool_error(e, "get_code_hierarchy")
            result["hierarchy"] = []
            return result
        except Exception as e:
            logger.error(f"Failed to get hierarchy: {e}")
            result = handle_tool_error(e, "get_code_hierarchy")
            result["hierarchy"] = []
            return result

    @mcp.tool()
    async def get_children(naics_code: str, ctx: Context) -> dict[str, Any]:
        """
        Get immediate children of a NAICS code.

        Shows the next level of detail in the classification hierarchy.
        For example, children of sector "31" would be its subsectors.
        """
        app_ctx = get_app_context(ctx)

        try:
            children = await app_ctx.database.get_children(naics_code)

            return ChildrenResponse(
                parent_code=naics_code,
                children=[
                    {"code": child.node_code, "title": child.title, "level": child.level.value}
                    for child in children
                ],
                count=len(children),
            ).to_dict()

        except NAICSException as e:
            logger.error(f"Failed to get children: {e}")
            result = handle_tool_error(e, "get_children")
            result["children"] = []
            return result
        except Exception as e:
            logger.error(f"Failed to get children: {e}")
            result = handle_tool_error(e, "get_children")
            result["children"] = []
            return result

    @mcp.tool()
    async def get_siblings(naics_code: str, limit: int = 10, ctx: Context = None) -> dict[str, Any]:
        """
        Get sibling codes at the same hierarchical level.

        Shows alternative codes that share the same parent.
        Useful for exploring related industries.
        """
        app_ctx = get_app_context(ctx)

        try:
            code = await app_ctx.database.get_by_code(naics_code)
            if not code:
                return {"error": f"NAICS code {naics_code} not found", "siblings": []}

            siblings = await app_ctx.database.get_siblings(naics_code, limit=limit)

            return SiblingsResponse(
                code=naics_code,
                title=code.title,
                level=code.level.value,
                siblings=[{"code": sib.node_code, "title": sib.title} for sib in siblings],
            ).to_dict()

        except NAICSException as e:
            logger.error(f"Failed to get siblings: {e}")
            result = handle_tool_error(e, "get_siblings")
            result["siblings"] = []
            return result
        except Exception as e:
            logger.error(f"Failed to get siblings: {e}")
            result = handle_tool_error(e, "get_siblings")
            result["siblings"] = []
            return result
