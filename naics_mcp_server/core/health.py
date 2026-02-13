"""
Health check module for NAICS MCP Server.

Provides health check endpoints for monitoring and container orchestration:
- Liveness: Is the server process alive?
- Readiness: Is the server ready to handle requests?
- Detailed: Full component status with diagnostics

Designed for Kubernetes probes, Docker health checks, and monitoring systems.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Individual component status."""

    READY = "ready"
    PARTIAL = "partial"
    NOT_READY = "not_ready"
    ERROR = "error"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: ComponentStatus
    message: str | None = None
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "status": self.status.value,
        }
        if self.message:
            result["message"] = self.message
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.details:
            result.update(self.details)
        return result


@dataclass
class HealthCheckResult:
    """Complete health check result."""

    status: HealthStatus
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """True if status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """True if server can handle requests (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "components": {
                name: component.to_dict() for name, component in self.components.items()
            },
        }
        if self.issues:
            result["issues"] = self.issues

        # Add summary
        ready_count = sum(1 for c in self.components.values() if c.status == ComponentStatus.READY)
        result["summary"] = f"{ready_count}/{len(self.components)} components ready"

        return result


class HealthChecker:
    """
    Health checker for NAICS MCP Server.

    Performs health checks on all server components and aggregates results.

    Usage:
        checker = HealthChecker(database, embedder, search_engine)

        # Quick liveness check
        if checker.is_alive():
            print("Server is alive")

        # Full health check
        result = await checker.check_health()
        if result.is_ready:
            print("Server is ready")
    """

    def __init__(
        self,
        database=None,
        embedder=None,
        search_engine=None,
        version: str = "0.1.0",
    ):
        """
        Initialize health checker.

        Args:
            database: NAICSDatabase instance
            embedder: TextEmbedder instance
            search_engine: NAICSSearchEngine instance
            version: Server version string
        """
        self.database = database
        self.embedder = embedder
        self.search_engine = search_engine
        self.version = version
        self._start_time = time.monotonic()

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.monotonic() - self._start_time

    def is_alive(self) -> bool:
        """
        Quick liveness check.

        Returns True if the server process is alive and responsive.
        This is a synchronous check for fast response times.

        Used for Kubernetes livenessProbe.
        """
        # If we can execute this code, we're alive
        return True

    async def check_readiness(self) -> bool:
        """
        Check if server is ready to handle requests.

        Returns True if critical components are ready.
        Used for Kubernetes readinessProbe.
        """
        try:
            # Check database connection
            if self.database is None or not self.database.is_connected:
                return False

            # Check embedder is loaded
            if self.embedder is None or self.embedder.model is None:
                return False

            return True
        except Exception:
            return False

    async def check_health(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        Checks all components and returns detailed status.
        Used for monitoring dashboards and detailed diagnostics.
        """
        components = {}
        issues = []

        # Check database
        db_health = await self._check_database()
        components["database"] = db_health
        if db_health.status == ComponentStatus.ERROR:
            issues.append(f"Database: {db_health.message}")
        elif db_health.status == ComponentStatus.NOT_READY:
            issues.append("Database has no NAICS codes")

        # Check embedder
        embedder_health = await self._check_embedder()
        components["embedder"] = embedder_health
        if embedder_health.status == ComponentStatus.ERROR:
            issues.append(f"Embedder: {embedder_health.message}")
        elif embedder_health.status == ComponentStatus.NOT_READY:
            issues.append("Embedding model not loaded")

        # Check search engine
        search_health = await self._check_search_engine()
        components["search_engine"] = search_health
        if search_health.status != ComponentStatus.READY:
            if search_health.message:
                issues.append(f"Search: {search_health.message}")

        # Check embeddings coverage
        embeddings_health = await self._check_embeddings()
        components["embeddings"] = embeddings_health
        if embeddings_health.status == ComponentStatus.PARTIAL:
            issues.append("Embeddings partially loaded - semantic search may be limited")
        elif embeddings_health.status == ComponentStatus.NOT_READY:
            issues.append("Embeddings not initialized")

        # Check cross-references
        xref_health = await self._check_cross_references()
        components["cross_references"] = xref_health
        if xref_health.status == ComponentStatus.NOT_READY:
            issues.append("Cross-reference data not loaded")

        # Determine overall status
        error_count = sum(1 for c in components.values() if c.status == ComponentStatus.ERROR)
        not_ready_count = sum(
            1 for c in components.values() if c.status == ComponentStatus.NOT_READY
        )
        partial_count = sum(1 for c in components.values() if c.status == ComponentStatus.PARTIAL)

        if error_count > 0:
            status = HealthStatus.UNHEALTHY
        elif not_ready_count >= 2:  # Multiple critical components not ready
            status = HealthStatus.UNHEALTHY
        elif not_ready_count > 0 or partial_count > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthCheckResult(
            status=status,
            timestamp=datetime.now(UTC),
            version=self.version,
            uptime_seconds=self.uptime_seconds,
            components=components,
            issues=issues,
        )

    async def _check_database(self) -> ComponentHealth:
        """Check database health."""
        start = time.monotonic()

        if self.database is None:
            return ComponentHealth(
                name="database",
                status=ComponentStatus.NOT_READY,
                message="Database not initialized",
            )

        try:
            if not self.database.is_connected:
                return ComponentHealth(
                    name="database",
                    status=ComponentStatus.NOT_READY,
                    message="Database not connected",
                )

            # Query for stats
            stats = await self.database.get_statistics()
            latency = (time.monotonic() - start) * 1000

            total_codes = stats.get("total_codes", 0)
            if total_codes == 0:
                return ComponentHealth(
                    name="database",
                    status=ComponentStatus.NOT_READY,
                    message="Database empty - run ETL",
                    latency_ms=latency,
                )

            return ComponentHealth(
                name="database",
                status=ComponentStatus.READY,
                latency_ms=latency,
                details={
                    "total_codes": total_codes,
                    "index_terms": stats.get("total_index_terms", 0),
                    "cross_references": stats.get("total_cross_references", 0),
                },
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="database",
                status=ComponentStatus.ERROR,
                message=str(e)[:100],
                latency_ms=latency,
            )

    async def _check_embedder(self) -> ComponentHealth:
        """Check embedder health."""
        if self.embedder is None:
            return ComponentHealth(
                name="embedder",
                status=ComponentStatus.NOT_READY,
                message="Embedder not initialized",
            )

        try:
            if self.embedder.model is None:
                return ComponentHealth(
                    name="embedder",
                    status=ComponentStatus.NOT_READY,
                    message="Model not loaded",
                )

            return ComponentHealth(
                name="embedder",
                status=ComponentStatus.READY,
                details={
                    "model": self.embedder.model_name,
                    "dimension": self.embedder.embedding_dim,
                },
            )

        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
            return ComponentHealth(
                name="embedder",
                status=ComponentStatus.ERROR,
                message=str(e)[:100],
            )

    async def _check_search_engine(self) -> ComponentHealth:
        """Check search engine health."""
        if self.search_engine is None:
            return ComponentHealth(
                name="search_engine",
                status=ComponentStatus.NOT_READY,
                message="Search engine not initialized",
            )

        try:
            # Check if embeddings are ready
            if not self.search_engine.embeddings_ready:
                return ComponentHealth(
                    name="search_engine",
                    status=ComponentStatus.PARTIAL,
                    message="Embeddings not fully loaded",
                )

            # Get cache stats
            embedding_cache = self.search_engine.embedding_cache.get_stats()
            search_cache = self.search_engine.search_cache.get_stats()

            return ComponentHealth(
                name="search_engine",
                status=ComponentStatus.READY,
                details={
                    "embedding_cache_size": embedding_cache.get("size", 0),
                    "search_cache_size": search_cache.get("size", 0),
                    "embedding_cache_hits": embedding_cache.get("hits", 0),
                    "search_cache_hits": search_cache.get("hits", 0),
                },
            )

        except Exception as e:
            logger.error(f"Search engine health check failed: {e}")
            return ComponentHealth(
                name="search_engine",
                status=ComponentStatus.ERROR,
                message=str(e)[:100],
            )

    async def _check_embeddings(self) -> ComponentHealth:
        """Check embeddings coverage."""
        if self.database is None or not self.database.is_connected:
            return ComponentHealth(
                name="embeddings",
                status=ComponentStatus.NOT_READY,
                message="Database not available",
            )

        try:
            stats = await self.database.get_statistics()
            coverage = stats.get("embedding_coverage", {})
            coverage_percent = coverage.get("coverage_percent", 0)

            if coverage_percent >= 99:
                return ComponentHealth(
                    name="embeddings",
                    status=ComponentStatus.READY,
                    details={"coverage_percent": round(coverage_percent, 1)},
                )
            elif coverage_percent > 0:
                return ComponentHealth(
                    name="embeddings",
                    status=ComponentStatus.PARTIAL,
                    message=f"{coverage_percent:.0f}% coverage",
                    details={"coverage_percent": round(coverage_percent, 1)},
                )
            else:
                return ComponentHealth(
                    name="embeddings",
                    status=ComponentStatus.NOT_READY,
                    message="No embeddings generated",
                )

        except Exception as e:
            logger.error(f"Embeddings health check failed: {e}")
            return ComponentHealth(
                name="embeddings",
                status=ComponentStatus.ERROR,
                message=str(e)[:100],
            )

    async def _check_cross_references(self) -> ComponentHealth:
        """Check cross-reference data."""
        if self.database is None or not self.database.is_connected:
            return ComponentHealth(
                name="cross_references",
                status=ComponentStatus.NOT_READY,
                message="Database not available",
            )

        try:
            stats = await self.database.get_statistics()
            xref_count = stats.get("total_cross_references", 0)

            if xref_count > 0:
                # Check if excluded_activity is populated
                result = self.database.connection.execute(
                    "SELECT COUNT(*) FROM naics_cross_references WHERE excluded_activity IS NOT NULL"
                ).fetchone()
                with_activity = result[0] if result else 0

                return ComponentHealth(
                    name="cross_references",
                    status=ComponentStatus.READY if with_activity > 0 else ComponentStatus.PARTIAL,
                    details={
                        "total": xref_count,
                        "with_excluded_activity": with_activity,
                    },
                )
            else:
                return ComponentHealth(
                    name="cross_references",
                    status=ComponentStatus.NOT_READY,
                    message="No cross-reference data",
                )

        except Exception as e:
            logger.error(f"Cross-reference health check failed: {e}")
            return ComponentHealth(
                name="cross_references",
                status=ComponentStatus.ERROR,
                message=str(e)[:100],
            )


# Convenience functions for simple health checks


async def liveness_check() -> dict[str, Any]:
    """
    Simple liveness check.

    Returns minimal response indicating server is alive.
    Suitable for Kubernetes livenessProbe.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def readiness_check(checker: HealthChecker) -> dict[str, Any]:
    """
    Readiness check.

    Returns whether server is ready to handle requests.
    Suitable for Kubernetes readinessProbe.

    Args:
        checker: HealthChecker instance

    Returns:
        Dict with ready status and basic info
    """
    is_ready = await checker.check_readiness()
    return {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": round(checker.uptime_seconds, 1),
    }
