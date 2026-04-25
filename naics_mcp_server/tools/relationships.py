"""
Relationship tools for NAICS MCP Server.

Provides: get_similar_codes, get_cross_sector_alternatives, get_relationship_stats
"""

from typing import Any

from mcp.server.fastmcp import Context

from ..app_context import get_app_context
from ..observability.logging import get_logger

logger = get_logger(__name__)


def register_tools(mcp):
    """Register relationship tools on the MCP server."""

    @mcp.tool()
    async def get_similar_codes(
        naics_code: str,
        min_similarity: float = 0.75,
        include_cross_sector: bool = True,
        limit: int = 10,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Get pre-computed similar NAICS codes from the semantic relationship graph.

        This uses pre-computed FAISS similarity relationships (much faster than
        real-time similarity search). Relationships are computed across all codes
        and stored in the database.

        USE CASES:
        - Discover alternative classifications for a business activity
        - Find codes in OTHER sectors that may apply (cross-sector alternatives)
        - Explore related industries within the same sector

        CROSS-SECTOR INSIGHT:
        When include_cross_sector=True, this surfaces codes from different sectors
        that are semantically similar. This is valuable when:
        - A business might fit in multiple sectors
        - Activities span traditional sector boundaries
        - Validating that a cross-sector code doesn't better fit

        Example: A "food truck" (722330) might show similarity to:
        - Same sector: 722511 (Full-Service Restaurants)
        - Cross sector: 445110 (Grocery Stores) - if they also sell prepared food

        Args:
            naics_code: The NAICS code to find similar codes for
            min_similarity: Minimum similarity threshold (0-1, default 0.75)
            include_cross_sector: Include codes from different sectors (default True)
            limit: Maximum results per category (default 10)

        Returns:
            Dict with same_sector_alternatives and cross_sector_alternatives
        """
        app_ctx = get_app_context(ctx)

        if app_ctx.relationship_service is None:
            return {
                "error": "Relationship service not available",
                "same_sector_alternatives": [],
                "cross_sector_alternatives": [],
            }

        try:
            result = await app_ctx.relationship_service.get_similar_codes(
                naics_code,
                min_similarity=min_similarity,
                include_cross_sector=include_cross_sector,
                limit=limit,
            )

            # Log if cross-sector alternatives found (valuable insight)
            if result.get("cross_sector_alternatives"):
                logger.info(
                    "Cross-sector alternatives found",
                    data={
                        "naics_code": naics_code,
                        "cross_sector_count": len(result["cross_sector_alternatives"]),
                    },
                )

            return result

        except Exception as e:
            logger.error(f"Failed to get similar codes: {e}")
            return {
                "error": str(e),
                "same_sector_alternatives": [],
                "cross_sector_alternatives": [],
            }

    @mcp.tool()
    async def get_cross_sector_alternatives(
        naics_code: str,
        min_similarity: float = 0.70,
        top_n: int = 10,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """
        Get codes in DIFFERENT sectors that are semantically similar.

        This is a powerful tool for discovering when a business activity might
        be classified in a completely different sector than initially expected.

        WHY THIS MATTERS:
        - NAICS sectors are broad categories, but real businesses often span boundaries
        - A "bakery" might be Manufacturing (311) or Retail (445) or Food Service (722)
        - Cross-sector alternatives reveal these boundary cases for human evaluation

        INTERPRETATION:
        - High similarity (>0.85): Strong alternative - carefully evaluate which fits better
        - Medium similarity (0.75-0.85): Consider if the business has activities in both areas
        - Lower similarity (0.70-0.75): Tangentially related - may not be a true alternative

        Example insights:
        - 541511 (Custom Software) may show similarity to 518210 (Data Processing)
        - 722511 (Full-Service Restaurants) may show similarity to 311 (Food Manufacturing)

        Args:
            naics_code: The NAICS code to find cross-sector alternatives for
            min_similarity: Minimum similarity threshold (default 0.70)
            top_n: Maximum results to return (default 10)

        Returns:
            Dict with cross-sector alternatives and relationship notes
        """
        app_ctx = get_app_context(ctx)

        if app_ctx.relationship_service is None:
            return {
                "error": "Relationship service not available",
                "naics_code": naics_code,
                "cross_sector_alternatives": [],
            }

        try:
            alternatives = await app_ctx.relationship_service.get_cross_sector_alternatives(
                naics_code,
                min_similarity=min_similarity,
                top_n=top_n,
            )

            # Get the original code info for context
            code = await app_ctx.database.get_by_code(naics_code)

            return {
                "naics_code": naics_code,
                "title": code.title if code else "Unknown",
                "sector_code": code.sector_code if code else None,
                "cross_sector_alternatives": alternatives,
                "total_found": len(alternatives),
                "note": (
                    "Cross-sector alternatives are codes in DIFFERENT sectors that are "
                    "semantically similar. High similarity suggests the business activity "
                    "might legitimately fit in either sector - evaluate based on PRIMARY activity."
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cross-sector alternatives: {e}")
            return {
                "error": str(e),
                "naics_code": naics_code,
                "cross_sector_alternatives": [],
            }

    @mcp.tool()
    async def get_relationship_stats(naics_code: str, ctx: Context = None) -> dict[str, Any]:
        """
        Get statistics about a code's semantic relationships.

        Provides aggregate metrics about how a code relates to others:
        - How many same-sector alternatives exist
        - How many cross-sector alternatives exist
        - Maximum and average similarity scores

        USE CASES:
        - Understand if a code has many vs few alternatives
        - Check if cross-sector relationships exist (might need evaluation)
        - Get a quick summary before diving into detailed alternatives

        INTERPRETATION:
        - High cross_sector_count: Code has broad semantic overlap with other sectors
        - Low cross_sector_count: Code is highly specific to its sector
        - High avg_similarity: Strong alternatives exist (may need careful selection)
        - has_cross_sector=True: Flag for potential classification boundary case

        Args:
            naics_code: The NAICS code to get relationship stats for

        Returns:
            Dict with relationship statistics
        """
        app_ctx = get_app_context(ctx)

        if app_ctx.relationship_service is None:
            return {"error": "Relationship service not available"}

        try:
            return await app_ctx.relationship_service.get_relationship_stats(naics_code)

        except Exception as e:
            logger.error(f"Failed to get relationship stats: {e}")
            return {"error": str(e)}
