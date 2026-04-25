#!/usr/bin/env python3
"""
NAICS MCP Server

An intelligent industry classification service for NAICS 2022,
built with clarity and purpose.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from .app_context import AppContext
from .config import (
    SearchConfig,
    ServerConfig,
    get_http_server_config,
    get_metrics_config,
    get_rate_limit_config,
    get_shutdown_config,
)
from .core.classification_workbook import ClassificationWorkbook
from .core.cross_reference import CrossReferenceService
from .core.database import NAICSDatabase
from .core.embeddings import TextEmbedder
from .core.health import HealthChecker
from .core.relationships import RelationshipService
from .core.search_engine import NAICSSearchEngine
from .core.shutdown import ShutdownConfig, create_shutdown_manager
from .http_server import HTTPServer, HTTPServerConfig
from .observability.audit import SearchAuditLog
from .observability.logging import (
    get_logger,
    log_server_ready,
    log_server_shutdown,
    log_server_start,
    setup_logging,
)
from .observability.metrics import (
    initialize_metrics,
    update_data_stats,
    update_health_status,
)
from .observability.rate_limiting import RateLimiter
from .tools import register_all_tools

# Configure structured logging
setup_logging(
    level=os.getenv("NAICS_LOG_LEVEL", "INFO"),
    format=os.getenv("NAICS_LOG_FORMAT", "text"),  # Use "json" for production
    log_file=os.getenv("NAICS_LOG_FILE"),
)
logger = get_logger(__name__)


# Initialize the MCP server
@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage server lifecycle - initialization and cleanup."""
    # Load configuration
    search_config = SearchConfig.from_env()
    server_config = ServerConfig.from_env()
    shutdown_config = get_shutdown_config()

    # Create shutdown manager with config
    shutdown_mgr_config = ShutdownConfig(
        timeout_seconds=shutdown_config.shutdown_timeout_seconds,
        drain_check_interval=shutdown_config.drain_check_interval,
        grace_period_seconds=shutdown_config.grace_period_seconds,
        force_after_timeout=shutdown_config.force_after_timeout,
        handle_sigterm=shutdown_config.handle_sigterm,
        handle_sigint=shutdown_config.handle_sigint,
    )
    shutdown_manager = create_shutdown_manager(shutdown_mgr_config)

    # Log startup with config
    log_server_start(
        {
            "database_path": str(search_config.database_path),
            "embedding_model": search_config.embedding_model,
            "debug": server_config.debug,
            "shutdown_timeout": shutdown_config.shutdown_timeout_seconds,
        }
    )

    # Initialize database
    database = NAICSDatabase(search_config.database_path)
    database.connect()
    logger.info("Database connected", data={"path": str(search_config.database_path)})

    # Initialize embedder
    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(model_name=search_config.embedding_model, cache_dir=cache_dir)
    embedder.load_model()
    logger.info("Embedding model loaded", data={"model": search_config.embedding_model})

    # Initialize search engine
    search_engine = NAICSSearchEngine(database, embedder, search_config)

    # Initialize embeddings if needed
    init_result = await search_engine.initialize_embeddings()
    logger.info(
        "Embeddings initialized",
        data={
            "action": init_result.get("action"),
            "count": init_result.get("embeddings_count") or init_result.get("embeddings_generated"),
            "time_seconds": init_result.get("time_seconds"),
        },
    )

    # Initialize cross-reference service
    cross_ref_service = CrossReferenceService(database)

    # Initialize relationship service (for pre-computed semantic similarities)
    relationship_service = RelationshipService(database)
    relationships_available = await relationship_service.check_relationships_available()
    if relationships_available:
        logger.info("Relationship service initialized (pre-computed relationships available)")
    else:
        logger.info(
            "Relationship service initialized (no pre-computed relationships - "
            "run notebooks/06_compute_relationships.py to generate)"
        )

    # Initialize classification workbook
    classification_workbook = ClassificationWorkbook(database)
    logger.info("Classification workbook initialized")

    # Initialize audit log
    log_dir = Path.home() / ".cache" / "naics-mcp-server" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    audit_log = SearchAuditLog(
        log_dir=log_dir,
        retention_days=search_config.audit_retention_days,
        enable_file_logging=search_config.enable_audit_log,
    )

    # Create health checker
    health_checker = HealthChecker(
        database=database,
        embedder=embedder,
        search_engine=search_engine,
        version=server_config.version,
    )

    # Initialize rate limiter
    rate_limit_config = get_rate_limit_config()
    rate_limiter = (
        RateLimiter(rate_limit_config) if rate_limit_config.enable_rate_limiting else None
    )
    if rate_limiter:
        logger.info(
            "Rate limiting enabled",
            data={
                "default_rpm": rate_limit_config.default_rpm,
                "search_rpm": rate_limit_config.search_rpm,
                "classify_rpm": rate_limit_config.classify_rpm,
            },
        )

    # Create application context (http_server will be added later)
    app_context = AppContext(
        database,
        embedder,
        search_engine,
        audit_log,
        cross_ref_service,
        classification_workbook,
        health_checker,
        relationship_service,
        rate_limiter,
        shutdown_manager,
        http_server=None,  # Will be set after HTTP server initialization
    )

    # Register shutdown hooks (in order of priority - lower = earlier)
    shutdown_manager.register_hook(
        "audit_log",
        lambda: audit_log.flush() if hasattr(audit_log, "flush") else None,
        priority=10,
        timeout_seconds=2.0,
    )
    shutdown_manager.register_hook(
        "search_cache",
        lambda: search_engine.clear_caches() if hasattr(search_engine, "clear_caches") else None,
        priority=20,
        timeout_seconds=2.0,
    )
    shutdown_manager.register_hook(
        "database",
        database.disconnect,
        priority=100,  # Database last
        timeout_seconds=5.0,
    )
    logger.info(
        "Shutdown hooks registered",
        data={"hooks": ["audit_log", "search_cache", "database"]},
    )

    # Log server ready with stats
    stats = await database.get_statistics()
    stats["embeddings_count"] = init_result.get("embeddings_count") or init_result.get(
        "embeddings_generated", 0
    )
    log_server_ready(stats)

    # Initialize metrics
    metrics_config = get_metrics_config()
    if metrics_config.enable_metrics:
        initialize_metrics(
            version=server_config.version,
            embedding_model=search_config.embedding_model,
            database_path=str(search_config.database_path),
        )
        # Update data statistics
        update_data_stats(
            total_codes=stats.get("total_codes", 0),
            total_embeddings=stats.get("embeddings_count", 0),
            total_index_terms=stats.get("total_index_terms", 0),
            total_cross_references=stats.get("total_cross_references", 0),
        )
        # Set initial health status
        update_health_status("healthy", uptime_seconds=0)
        logger.info("Metrics initialized", data={"port": metrics_config.metrics_port})

    # Initialize HTTP server for health checks and metrics
    http_server_config = get_http_server_config()
    http_server = None

    if http_server_config.http_enabled:
        http_config = HTTPServerConfig(
            enabled=http_server_config.http_enabled,
            host=http_server_config.http_host,
            port=http_server_config.http_port,
            health_path=http_server_config.health_path,
            ready_path=http_server_config.ready_path,
            status_path=http_server_config.status_path,
            metrics_path=http_server_config.metrics_path,
        )
        http_server = HTTPServer(
            config=http_config,
            health_checker=health_checker,
            shutdown_manager=shutdown_manager,
            server_version=server_config.version,
        )

        # Register HTTP server shutdown hook (high priority - stop early)
        shutdown_manager.register_hook(
            "http_server",
            http_server.stop,
            priority=5,  # Stop HTTP server first
            timeout_seconds=5.0,
        )
        logger.info(
            "HTTP server configured",
            data={
                "port": http_server_config.http_port,
                "endpoints": [
                    http_server_config.health_path,
                    http_server_config.ready_path,
                    http_server_config.status_path,
                    http_server_config.metrics_path,
                ],
            },
        )

    # Update app context with HTTP server
    app_context.http_server = http_server

    # Use shutdown manager's lifespan for signal handling
    async with shutdown_manager.lifespan():
        try:
            # Start HTTP server if enabled
            if http_server:
                await http_server.start()

            yield app_context
        finally:
            log_server_shutdown()
            # Shutdown hooks are executed by the shutdown manager
            logger.info("Shutdown complete")


SERVER_INSTRUCTIONS = """
## Quick Classification (Simple Business)
For straightforward single-activity businesses:
1. `classify_business` with a clear description
2. If confidence > 60%, you're likely done
3. Optionally verify with `search_index_terms` for exact terminology

## Full Classification (Complex/Multi-Segment Business)
For companies with multiple business lines or ambiguous activities:
1. Research the company's segments and primary revenue sources
2. `classify_business` for initial assessment
3. `search_naics_codes` for each major segment/activity
4. `search_index_terms` for specific products/services
5. `compare_codes` for top candidates side-by-side
6. `get_cross_references` to check for exclusions (CRITICAL)
7. `get_siblings` or `find_similar_industries` for alternatives
8. `write_to_workbook` with form_type="classification_analysis" to document

## Boundary Cases (Activity Could Fit Multiple Codes)
When a business sits between two codes:
1. Search from multiple angles (product vs service vs customer)
2. `get_cross_sector_alternatives` to discover codes in OTHER sectors that may apply
3. `get_cross_references` on both candidates - exclusions are authoritative
4. `compare_codes` to see descriptions and index terms side-by-side
5. `get_code_hierarchy` to understand where each code sits conceptually
6. Document with `write_to_workbook` form_type="decision_tree"

## Batch Classification
For processing many businesses:
1. `classify_batch` for efficiency
2. Triage by confidence: >50% accept, 35-50% spot-check, <35% manual review
3. Use full classification workflow for low-confidence items

## Exploring the NAICS Hierarchy
1. `get_sector_overview` to see all 20 sectors
2. `get_children` to drill down into subsectors
3. `get_siblings` to see what else is at the same level
4. `find_similar_industries` for semantically related codes

## Key Principles
- PRIMARY ACTIVITY determines classification (largest revenue source)
- 6-digit codes are most specific and preferred
- Cross-references tell you what activities are EXCLUDED - always check
- Index term matches are strong evidence of correct classification
- When uncertain, document reasoning with the workbook tools
"""

# Create the MCP server
mcp = FastMCP(
    name="NAICS Classification Assistant", instructions=SERVER_INSTRUCTIONS, lifespan=lifespan
)

mcp.description = (
    "An intelligent industry classification service for NAICS 2022. "
    "I help you find the right NAICS codes using natural language descriptions, "
    "with support for hierarchical navigation and cross-reference lookup."
)

# Register all tool modules
register_all_tools(mcp)


# === Resources ===


@mcp.resource("naics://statistics")
async def get_statistics(ctx: Context) -> dict[str, Any]:
    """
    Get statistics about the NAICS database.
    """
    from .app_context import get_app_context

    app_ctx = get_app_context(ctx)

    stats = await app_ctx.database.get_statistics()

    # Add search engine stats
    stats["embedding_cache"] = app_ctx.search_engine.embedding_cache.get_stats()
    stats["search_cache"] = app_ctx.search_engine.search_cache.get_stats()

    # Add recent search patterns
    if app_ctx.audit_log:
        patterns = await app_ctx.audit_log.analyze_patterns(timeframe_hours=24)
        stats["recent_searches"] = patterns

    # Add rate limiting status if enabled
    if app_ctx.rate_limiter:
        stats["rate_limiting"] = await app_ctx.rate_limiter.get_status()

    return stats


@mcp.resource("naics://recent_searches")
async def get_recent_searches(ctx: Context) -> list[dict[str, Any]]:
    """
    Get recent search queries for monitoring.
    """
    from .app_context import get_app_context

    app_ctx = get_app_context(ctx)
    return app_ctx.audit_log.get_recent_searches(limit=20)


# === Prompts ===


@mcp.prompt()
def classification_assistant_prompt() -> str:
    """Template for NAICS classification assistance."""
    return """You are a NAICS classification expert. Help the user find the right NAICS codes by:

1. Understanding their business description
2. Asking clarifying questions if needed (primary activity, customers, products)
3. Searching for appropriate codes
4. Checking cross-references for exclusions
5. Explaining why each result matches

Always explain the hierarchy (Sector → Subsector → Industry Group → NAICS Industry → National Industry)
and highlight any important cross-reference exclusions.

Key classification principles:
- Primary activity determines classification (what generates the most revenue)
- 6-digit codes are most specific and preferred
- Cross-references tell you what activities belong elsewhere
- Index terms are the official vocabulary"""


@mcp.prompt()
def classification_methodology() -> str:
    """NAICS classification methodology guide."""
    return """NAICS CLASSIFICATION METHODOLOGY

PRIMARY RULE:
Establishments are classified based on their PRIMARY activity - the activity
that generates the largest share of their revenue.

HIERARCHY (5 levels):
- Sector (2-digit): Broadest category (e.g., 31 = Manufacturing)
- Subsector (3-digit): Industry groupings
- Industry Group (4-digit): More specific groupings
- NAICS Industry (5-digit): Industry level
- National Industry (6-digit): Most specific (US-specific detail)

BUILDING EFFECTIVE SEARCHES:
1. Start with the primary activity or product
2. Include customer type (business vs consumer)
3. Describe the production process if relevant
4. Use specific terminology from your industry

INTERPRETING RESULTS:
- Higher confidence = stronger match
- Check cross-references for exclusions
- Index term matches indicate official terminology alignment
- The most specific applicable code is preferred

CROSS-REFERENCES ARE CRITICAL:
- They tell you what activities are EXCLUDED from a code
- They point to where excluded activities SHOULD be classified
- Always check cross-references when confidence is moderate

COMMON PITFALLS:
- Classifying by product sold rather than primary activity
- Ignoring cross-reference exclusions
- Choosing a broad code when a specific one applies
- Not considering vertically integrated operations"""


def main():
    """Run the MCP server."""
    import sys

    if "--debug" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)

    mcp.run()


if __name__ == "__main__":
    main()
