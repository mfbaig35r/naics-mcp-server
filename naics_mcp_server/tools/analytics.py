"""
Analytics tools for NAICS MCP Server.

Provides: get_sector_overview, compare_codes
"""

from typing import Any

from mcp.server.fastmcp import Context

from ..app_context import get_app_context
from ..core.errors import ValidationError
from ..core.validation import validate_batch_codes
from ..observability.logging import get_logger

logger = get_logger(__name__)


def register_tools(mcp):
    """Register analytics tools on the MCP server."""

    @mcp.tool()
    async def get_sector_overview(
        sector_code: str | None = None, ctx: Context = None
    ) -> dict[str, Any]:
        """
        Get an overview of NAICS sectors or a specific sector.

        Without a sector_code, returns all 20 sectors.
        With a sector_code (2-digit), returns its subsectors.
        """
        app_ctx = get_app_context(ctx)

        try:
            if sector_code:
                # Get specific sector and its children
                sector = await app_ctx.database.get_by_code(sector_code)
                if not sector:
                    return {"error": f"Sector {sector_code} not found"}

                children = await app_ctx.database.get_children(sector_code)

                return {
                    "sector": {
                        "code": sector.node_code,
                        "title": sector.title,
                        "description": sector.description,
                    },
                    "subsectors": [
                        {"code": child.node_code, "title": child.title} for child in children
                    ],
                }
            else:
                # Get all sectors
                results = app_ctx.database.connection.execute("""
                    SELECT node_code, title, description
                    FROM naics_nodes
                    WHERE level = 'sector'
                    ORDER BY node_code
                """).fetchall()

                return {
                    "sectors": [
                        {
                            "code": row[0],
                            "title": row[1],
                            "description": row[2][:150] + "..."
                            if row[2] and len(row[2]) > 150
                            else row[2],
                        }
                        for row in results
                    ],
                    "total": len(results),
                }

        except Exception as e:
            logger.error(f"Failed to get sector overview: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def compare_codes(codes: list[str], ctx: Context) -> dict[str, Any]:
        """
        Compare multiple NAICS codes side-by-side.

        Useful for understanding the differences between similar codes
        when making a classification decision.

        Constraints:
        - Maximum 20 codes can be compared at once
        - Each code must be 2-6 digits
        """
        app_ctx = get_app_context(ctx)

        # Validate codes
        try:
            codes_result = validate_batch_codes(codes)
            validated_codes = codes_result.value
        except ValidationError as e:
            logger.warning(
                "Code comparison validation failed",
                data={"error": e.message, "field": e.details.get("field")},
            )
            return {
                "error": e.message,
                "error_category": "validation",
                "codes_compared": [],
                "comparisons": [],
            }

        try:
            comparisons = []

            for code_str in validated_codes:
                code = await app_ctx.database.get_by_code(code_str)
                if code:
                    cross_refs = await app_ctx.database.get_cross_references(code_str)
                    index_terms = await app_ctx.database.get_index_terms_for_code(code_str)

                    comparisons.append(
                        {
                            "code": code.node_code,
                            "title": code.title,
                            "level": code.level.value,
                            "description": code.description,
                            "hierarchy": code.get_hierarchy_path(),
                            "index_terms": [t.index_term for t in index_terms[:5]],
                            "exclusions": [
                                {
                                    "activity": cr.excluded_activity,
                                    "classified_under": cr.target_code,
                                }
                                for cr in cross_refs
                                if cr.reference_type == "excludes"
                            ][:3],
                        }
                    )
                else:
                    comparisons.append({"code": code_str, "error": "Code not found"})

            return {"codes_compared": codes, "comparisons": comparisons}

        except Exception as e:
            logger.error(f"Failed to compare codes: {e}")
            return {"error": str(e)}
