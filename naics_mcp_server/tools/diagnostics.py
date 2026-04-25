"""
Diagnostics tools for NAICS MCP Server.

Provides: get_workflow_guide, ping, check_readiness, get_server_health, get_metrics, get_shutdown_status
"""

from datetime import UTC
from typing import Any

from mcp.server.fastmcp import Context

from ..app_context import get_app_context
from ..models.response_models import PingResponse, ReadinessResponse
from ..observability.logging import get_logger

logger = get_logger(__name__)


def register_tools(mcp):
    """Register diagnostics tools on the MCP server."""

    @mcp.tool()
    async def get_workflow_guide() -> dict[str, Any]:
        """
        Get the recommended workflows for NAICS classification.

        Call this to understand how to use the NAICS tools effectively.
        Returns workflow guidance for common use cases:
        - Quick classification (simple businesses)
        - Full classification (complex/multi-segment companies)
        - Boundary cases (activity fits multiple codes)
        - Batch classification
        - Hierarchy exploration

        This is especially useful on first use or when unsure
        which tools to use in what order.
        """
        return {
            "workflows": {
                "quick_classification": {
                    "description": "For straightforward single-activity businesses",
                    "steps": [
                        "1. classify_business with a clear description",
                        "2. If confidence > 60%, you're likely done",
                        "3. Optionally verify with search_index_terms for exact terminology",
                    ],
                },
                "full_classification": {
                    "description": "For companies with multiple business lines or ambiguous activities",
                    "steps": [
                        "1. Research the company's segments and primary revenue sources",
                        "2. classify_business for initial assessment",
                        "3. search_naics_codes for each major segment/activity",
                        "4. search_index_terms for specific products/services",
                        "5. compare_codes for top candidates side-by-side",
                        "6. get_cross_references to check for exclusions (CRITICAL)",
                        "7. get_siblings or find_similar_industries for alternatives",
                        "8. write_to_workbook with form_type='classification_analysis' to document",
                    ],
                },
                "boundary_cases": {
                    "description": "When a business sits between two codes",
                    "steps": [
                        "1. Search from multiple angles (product vs service vs customer)",
                        "2. get_cross_references on both candidates - exclusions are authoritative",
                        "3. compare_codes to see descriptions and index terms side-by-side",
                        "4. get_code_hierarchy to understand where each code sits conceptually",
                        "5. Document with write_to_workbook form_type='decision_tree'",
                    ],
                },
                "batch_classification": {
                    "description": "For processing many businesses efficiently",
                    "steps": [
                        "1. classify_batch for efficiency",
                        "2. Triage by confidence: >50% accept, 35-50% spot-check, <35% manual review",
                        "3. Use full_classification workflow for low-confidence items",
                    ],
                },
                "hierarchy_exploration": {
                    "description": "For exploring the NAICS structure itself",
                    "steps": [
                        "1. get_sector_overview to see all 20 sectors",
                        "2. get_children to drill down into subsectors",
                        "3. get_siblings to see what else is at the same level",
                        "4. find_similar_industries for semantically related codes",
                    ],
                },
                "cross_sector_analysis": {
                    "description": "For discovering when a business might fit in a different sector",
                    "steps": [
                        "1. classify_business for initial classification",
                        "2. get_relationship_stats to check if cross-sector alternatives exist",
                        "3. get_cross_sector_alternatives if has_cross_sector=True",
                        "4. compare_codes on the primary code vs top cross-sector alternatives",
                        "5. get_cross_references on both to check for exclusions",
                        "6. Document decision if classification is ambiguous",
                    ],
                    "when_to_use": [
                        "Business activities span traditional sector boundaries",
                        "Initial classification has moderate confidence",
                        "Business description mentions multiple distinct activities",
                        "Validating that a cross-sector code doesn't better fit",
                    ],
                },
                "documented_analysis": {
                    "description": "For complex cases requiring traceable reasoning with linked entries",
                    "steps": [
                        "1. write_to_workbook with form_type='business_profile' - capture initial understanding",
                        "2. Perform research using search tools, cross-references, comparisons",
                        "3. write_to_workbook with form_type='research_notes', parent_entry_id=step1_id",
                        "4. If comparing codes: write_to_workbook form_type='industry_comparison', parent_entry_id=step3_id",
                        "5. Final decision: write_to_workbook form_type='classification_analysis', parent_entry_id=previous_id",
                        "6. Use search_workbook(parent_entry_id=root_id) to retrieve full analysis chain",
                    ],
                    "linking_patterns": {
                        "deep_dive": "business_profile → research_notes → classification_analysis",
                        "comparison": "business_profile → industry_comparison → classification_analysis",
                        "boundary_case": "classification_analysis → cross_reference_notes → revised_classification",
                        "sic_migration": "sic_conversion → research_notes → classification_analysis",
                    },
                },
                "validation": {
                    "description": "For verifying existing or user-provided classifications",
                    "steps": [
                        "1. validate_classification with code and business description",
                        "2. If status='valid', classification is confirmed",
                        "3. If status='questionable', review exclusion_warnings and suggested_alternatives",
                        "4. If status='invalid', use suggested_alternatives or run classify_business",
                        "5. Document decision with write_to_workbook if needed",
                    ],
                },
            },
            "key_principles": [
                "PRIMARY ACTIVITY determines classification (largest revenue source)",
                "6-digit codes are most specific and preferred",
                "Cross-references tell you what activities are EXCLUDED - always check",
                "Index term matches are strong evidence of correct classification",
                "When uncertain, document reasoning with the workbook tools",
            ],
            "tool_categories": {
                "search": [
                    "search_naics_codes",
                    "search_index_terms",
                    "classify_business",
                    "classify_batch",
                ],
                "navigation": [
                    "get_code_hierarchy",
                    "get_children",
                    "get_siblings",
                    "get_sector_overview",
                    "find_similar_industries",
                ],
                "relationships": [
                    "get_similar_codes",
                    "get_cross_sector_alternatives",
                    "get_relationship_stats",
                ],
                "validation": ["get_cross_references", "compare_codes", "validate_classification"],
                "documentation": [
                    "get_workbook_template",
                    "write_to_workbook",
                    "search_workbook",
                    "get_workbook_entry",
                ],
                "diagnostics": ["ping", "check_readiness", "get_server_health", "get_workflow_guide"],
            },
            "workbook_linking": {
                "description": "Use parent_entry_id to create traceable analysis chains",
                "how_it_works": [
                    "1. First entry returns an entry_id (e.g., 'abc123')",
                    "2. Pass that as parent_entry_id when creating follow-up entries",
                    "3. Chain continues: each entry can be parent to the next",
                    "4. Use search_workbook(parent_entry_id='abc123') to find all children",
                    "5. Use get_workbook_entry to read any entry and see its parent_entry_id",
                ],
                "benefits": [
                    "Full audit trail of classification reasoning",
                    "Easy to review multi-step analysis",
                    "Supports complex boundary case documentation",
                    "Enables 'show your work' for compliance",
                ],
            },
        }

    @mcp.tool()
    async def ping() -> dict[str, Any]:
        """
        Simple liveness check.

        Returns immediately to confirm the server process is alive.
        Use this for quick health checks or Kubernetes liveness probes.

        For detailed health information, use get_server_health instead.
        """
        from datetime import datetime

        return PingResponse(
            status="alive",
            timestamp=datetime.now(UTC).isoformat(),
        ).to_dict()

    @mcp.tool()
    async def check_readiness(ctx: Context) -> dict[str, Any]:
        """
        Check if the server is ready to handle requests.

        Returns ready/not_ready status based on critical components:
        - Database connection
        - Embedding model loaded

        Use this for Kubernetes readiness probes or before sending requests.
        For detailed diagnostics, use get_server_health instead.
        """
        app_ctx = get_app_context(ctx)

        is_ready = await app_ctx.health_checker.check_readiness()
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            uptime_seconds=round(app_ctx.health_checker.uptime_seconds, 1),
        ).to_dict()

    @mcp.tool()
    async def get_server_health(ctx: Context) -> dict[str, Any]:
        """
        Comprehensive health check with detailed diagnostics.

        Call this first if you're getting empty results or errors.
        Returns detailed status of all critical components:
        - Database: connection, code counts, index terms
        - Embedder: model loaded, dimension
        - Search engine: cache stats, embeddings ready
        - Embeddings: coverage percentage
        - Cross-references: data availability

        Status levels:
        - healthy: All components ready
        - degraded: Some components not fully ready, but functional
        - unhealthy: Critical components unavailable

        For simple liveness checks, use ping instead.
        """
        app_ctx = get_app_context(ctx)

        # Get comprehensive health check
        result = await app_ctx.health_checker.check_health()

        # Convert to dict and add workbook check (not in core health checker)
        health = result.to_dict()

        # Add workbook status
        try:
            count = app_ctx.database.connection.execute(
                "SELECT COUNT(*) FROM classification_workbook"
            ).fetchone()[0]
            health["components"]["workbook"] = {"status": "ready", "entries": count}
        except Exception as e:
            health["components"]["workbook"] = {"status": "error", "message": str(e)[:100]}
            if "issues" not in health:
                health["issues"] = []
            health["issues"].append(f"Workbook not available: {e}")

        return health

    @mcp.tool()
    async def get_metrics() -> dict[str, Any]:
        """
        Get Prometheus metrics for monitoring.

        Returns metrics in a structured format including:
        - Tool request counts and latencies
        - Search performance metrics
        - Cache hit rates
        - Database query statistics
        - Health status

        For Prometheus scraping, use the metrics HTTP endpoint instead.
        This tool is useful for debugging and inline monitoring.
        """
        from ..observability.metrics import get_metrics_text

        # Return metrics as structured data
        metrics_text = get_metrics_text()

        # Parse key metrics into summary
        summary = {
            "format": "prometheus",
            "metrics_count": metrics_text.count("# HELP"),
            "raw_metrics": metrics_text,
        }

        return summary

    @mcp.tool()
    async def get_shutdown_status(ctx: Context) -> dict[str, Any]:
        """
        Get the current shutdown manager status.

        Returns information about:
        - Server state (running, shutting_down, draining, stopped)
        - In-flight request count
        - Registered cleanup hooks
        - Shutdown configuration

        Use this to monitor shutdown progress or verify server state.
        """
        app_ctx = get_app_context(ctx)

        if app_ctx.shutdown_manager is None:
            return {
                "status": "unavailable",
                "message": "Shutdown manager not configured",
            }

        return await app_ctx.shutdown_manager.get_status()
