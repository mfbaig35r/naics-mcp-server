"""
Application context and shared utilities for NAICS MCP Server tools.
"""


from mcp.server.fastmcp import Context

from .core.classification_workbook import ClassificationWorkbook
from .core.cross_reference import CrossReferenceService
from .core.database import NAICSDatabase
from .core.embeddings import TextEmbedder
from .core.errors import RateLimitError
from .core.health import HealthChecker
from .core.relationships import RelationshipService
from .core.search_engine import NAICSSearchEngine
from .core.shutdown import ShutdownManager
from .http_server import HTTPServer
from .observability.audit import SearchAuditLog
from .observability.rate_limiting import RateLimiter


class AppContext:
    """Application context with all initialized services."""

    def __init__(
        self,
        database: NAICSDatabase,
        embedder: TextEmbedder,
        search_engine: NAICSSearchEngine,
        audit_log: SearchAuditLog,
        cross_ref_service: CrossReferenceService,
        classification_workbook: ClassificationWorkbook,
        health_checker: HealthChecker,
        relationship_service: RelationshipService | None = None,
        rate_limiter: RateLimiter | None = None,
        shutdown_manager: ShutdownManager | None = None,
        http_server: HTTPServer | None = None,
    ):
        self.database = database
        self.embedder = embedder
        self.search_engine = search_engine
        self.audit_log = audit_log
        self.cross_ref_service = cross_ref_service
        self.classification_workbook = classification_workbook
        self.health_checker = health_checker
        self.relationship_service = relationship_service
        self.rate_limiter = rate_limiter
        self.shutdown_manager = shutdown_manager
        self.http_server = http_server


def get_app_context(ctx: Context) -> "AppContext":
    """Extract AppContext from MCP Context."""
    return ctx.request_context.lifespan_context


async def check_rate_limit(ctx: Context, tool_name: str) -> None:
    """
    Check rate limit for a tool invocation.

    Raises RateLimitError if limit is exceeded.
    Does nothing if rate limiting is disabled or not configured.
    """
    app_ctx: AppContext = get_app_context(ctx)
    if app_ctx.rate_limiter is None:
        return

    result = await app_ctx.rate_limiter.check_limit(tool_name)
    if not result.allowed:
        raise RateLimitError(
            tool_name=tool_name,
            category=result.category.value,
            retry_after=result.retry_after_seconds,
            message=result.message,
        )
