"""
Search tools for NAICS MCP Server.

Provides: search_naics_codes, search_index_terms, find_similar_industries
"""

from typing import Any

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from ..app_context import check_rate_limit, get_app_context
from ..core.errors import NAICSException, ValidationError, handle_tool_error
from ..core.search_engine import generate_search_guidance
from ..core.validation import (
    validate_confidence,
    validate_limit,
    validate_search_query,
    validate_strategy,
)
from ..models.response_models import IndexTermSearchResponse, SimilarIndustriesResponse
from ..models.search_models import SearchStrategy
from ..observability.audit import SearchEvent
from ..observability.logging import get_logger, sanitize_text
from ..observability.metrics import record_search_metrics

logger = get_logger(__name__)


class SearchRequest(BaseModel):
    """Parameters for NAICS code search."""

    query: str = Field(description="Natural language description of the business or activity")
    strategy: str = Field(
        default="hybrid",
        description="Search strategy: 'hybrid' (best match), 'semantic' (meaning), or 'lexical' (exact)",
    )
    limit: int = Field(default=10, description="Maximum results to return", ge=1, le=50)
    min_confidence: float = Field(
        default=0.3, description="Minimum confidence threshold (0-1)", ge=0.0, le=1.0
    )
    include_cross_refs: bool = Field(
        default=True, description="Include cross-reference checks for exclusions"
    )


class SimilarityRequest(BaseModel):
    """Parameters for finding similar NAICS codes."""

    naics_code: str = Field(description="NAICS code to find similar codes for")
    limit: int = Field(default=5, description="Maximum results", ge=1, le=20)
    min_similarity: float = Field(default=0.7, description="Minimum similarity", ge=0.0, le=1.0)


def register_tools(mcp):
    """Register search tools on the MCP server."""

    @mcp.tool()
    async def search_naics_codes(request: SearchRequest, ctx: Context) -> dict[str, Any]:
        """
        Search for NAICS codes using natural language.

        NAICS codes classify businesses by industry using a 6-digit hierarchical system:
        Sector (2) → Subsector (3) → Industry Group (4) → NAICS Industry (5) → National Industry (6)

        SEARCH STRATEGIES:
        • hybrid (default): Balances semantic meaning with exact term matching
        • semantic: Focuses on conceptual similarity and related meanings
        • lexical: Prioritizes exact term matching

        OPTIMIZING YOUR SEARCH:
        Include relevant details about:
        - Primary business activity
        - Products or services offered
        - Customer type (business vs consumer)
        - Production process or method

        UNDERSTANDING RESULTS:
        - Confidence scores indicate relative match strength
        - Results include full hierarchy from sector to specific code
        - Cross-reference warnings indicate the activity may belong elsewhere
        - Index term matches show alignment with official NAICS terminology
        """
        app_ctx = get_app_context(ctx)

        # Check rate limit
        await check_rate_limit(ctx, "search_naics_codes")

        # Validate inputs
        try:
            query_result = validate_search_query(request.query)
            validated_query = query_result.value

            strategy_result = validate_strategy(request.strategy)
            validated_strategy = strategy_result.value

            limit_result = validate_limit(request.limit)
            validated_limit = limit_result.value

            confidence_result = validate_confidence(request.min_confidence)
            validated_confidence = confidence_result.value
        except ValidationError as e:
            logger.warning(
                "Search validation failed",
                data={"error": e.message, "field": e.details.get("field")},
            )
            return {
                "query": request.query,
                "results": [],
                "expanded": False,
                "strategy_used": request.strategy,
                "total_found": 0,
                "search_time_ms": 0,
                "guidance": [f"Validation error: {e.message}"],
            }

        # Start audit event
        search_event = SearchEvent.start(validated_query, validated_strategy)

        try:
            # Map strategy
            strategy_map = {
                "hybrid": SearchStrategy.HYBRID,
                "best_match": SearchStrategy.HYBRID,
                "semantic": SearchStrategy.SEMANTIC,
                "meaning": SearchStrategy.SEMANTIC,
                "lexical": SearchStrategy.LEXICAL,
                "exact": SearchStrategy.LEXICAL,
            }
            strategy = strategy_map.get(validated_strategy, SearchStrategy.HYBRID)

            # Perform search
            results = await app_ctx.search_engine.search(
                query=validated_query,
                strategy=strategy,
                limit=validated_limit,
                min_confidence=validated_confidence,
                include_cross_refs=request.include_cross_refs,
            )

            # Log search completion
            search_event.complete(results)
            await app_ctx.audit_log.log_search(search_event)

            # Record metrics
            record_search_metrics(
                strategy=strategy.value,
                duration_seconds=search_event.duration_ms / 1000.0,
                results_count=len(results.matches),
                top_confidence=results.matches[0].confidence.overall if results.matches else None,
            )

            # Generate guidance
            guidance = generate_search_guidance(results)

            # Build response from dataclass
            return results.to_dict(guidance=guidance)

        except Exception as e:
            search_event.fail(str(e))
            await app_ctx.audit_log.log_search(search_event)

            logger.error(
                "Search failed",
                data={
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                    "query_preview": sanitize_text(request.query, 50),
                },
            )
            return {
                "query": request.query,
                "results": [],
                "expanded": False,
                "strategy_used": request.strategy,
                "total_found": 0,
                "search_time_ms": search_event.duration_ms,
            }

    @mcp.tool()
    async def search_index_terms(
        search_text: str, limit: int = 20, ctx: Context = None
    ) -> dict[str, Any]:
        """
        Search the official NAICS index terms.

        NAICS has 20,398 official index terms that map specific activities
        to NAICS codes. This is useful for finding the exact terminology
        used by the Census Bureau.

        Examples:
        - "dog grooming" → 812910
        - "software publishing" → 511210
        - "soybean farming" → 111110
        """
        app_ctx = get_app_context(ctx)

        # Check rate limit
        await check_rate_limit(ctx, "search_index_terms")

        try:
            terms = await app_ctx.database.search_index_terms(search_text, limit=limit)

            return IndexTermSearchResponse(
                search_text=search_text,
                matches=[{"index_term": t.index_term, "naics_code": t.naics_code} for t in terms],
                total_found=len(terms),
            ).to_dict()

        except NAICSException as e:
            logger.error(f"Index term search failed: {e}")
            result = handle_tool_error(e, "search_index_terms")
            result["matches"] = []
            return result
        except Exception as e:
            logger.error(f"Index term search failed: {e}")
            result = handle_tool_error(e, "search_index_terms")
            result["matches"] = []
            return result

    @mcp.tool()
    async def find_similar_industries(request: SimilarityRequest, ctx: Context) -> dict[str, Any]:
        """
        Find NAICS codes similar to a given code.

        This helps explore related classifications and alternatives.
        Uses semantic similarity based on code descriptions.
        """
        app_ctx = get_app_context(ctx)

        # Check rate limit
        await check_rate_limit(ctx, "find_similar_industries")

        try:
            code = await app_ctx.database.get_by_code(request.naics_code)

            if not code:
                return {"error": f"NAICS code {request.naics_code} not found", "similar_codes": []}

            # Use the code's description for similarity search
            search_text = code.raw_embedding_text or f"{code.title} {code.description or ''}"

            results = await app_ctx.search_engine.search(
                query=search_text,
                strategy=SearchStrategy.SEMANTIC,
                limit=request.limit + 1,
                min_confidence=request.min_similarity,
            )

            # Filter out the original code
            similar = [
                {
                    "code": match.code.node_code,
                    "title": match.code.title,
                    "similarity": match.confidence.semantic,
                    "level": match.code.level.value,
                }
                for match in results.matches
                if match.code.node_code != request.naics_code
            ][: request.limit]

            return SimilarIndustriesResponse(
                original_code=request.naics_code,
                original_title=code.title,
                similar_codes=similar,
            ).to_dict()

        except NAICSException as e:
            logger.error(f"Similarity search failed: {e}")
            result = handle_tool_error(e, "find_similar_industries")
            result["similar_codes"] = []
            return result
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            result = handle_tool_error(e, "find_similar_industries")
            result["similar_codes"] = []
            return result
