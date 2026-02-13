"""
Unit tests for health check module.

Tests health check functionality for liveness, readiness, and detailed diagnostics.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from naics_mcp_server.core.health import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    ComponentStatus,
    ComponentHealth,
    liveness_check,
    readiness_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_status_values(self):
        """Should have expected status values."""
        assert ComponentStatus.READY.value == "ready"
        assert ComponentStatus.PARTIAL.value == "partial"
        assert ComponentStatus.NOT_READY.value == "not_ready"
        assert ComponentStatus.ERROR.value == "error"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Should create with required fields."""
        health = ComponentHealth(
            name="database",
            status=ComponentStatus.READY
        )
        assert health.name == "database"
        assert health.status == ComponentStatus.READY
        assert health.message is None
        assert health.latency_ms is None
        assert health.details == {}

    def test_with_all_fields(self):
        """Should create with all fields."""
        health = ComponentHealth(
            name="database",
            status=ComponentStatus.READY,
            message="Connected",
            latency_ms=5.5,
            details={"total_codes": 1000}
        )
        assert health.message == "Connected"
        assert health.latency_ms == 5.5
        assert health.details["total_codes"] == 1000

    def test_to_dict_minimal(self):
        """to_dict should include status."""
        health = ComponentHealth(
            name="database",
            status=ComponentStatus.READY
        )
        result = health.to_dict()
        assert result["status"] == "ready"
        assert "message" not in result
        assert "latency_ms" not in result

    def test_to_dict_with_all_fields(self):
        """to_dict should include all set fields."""
        health = ComponentHealth(
            name="database",
            status=ComponentStatus.ERROR,
            message="Connection failed",
            latency_ms=100.5,
            details={"retry_count": 3}
        )
        result = health.to_dict()
        assert result["status"] == "error"
        assert result["message"] == "Connection failed"
        assert result["latency_ms"] == 100.5
        assert result["retry_count"] == 3


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        """Should create with required fields."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=120.5
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.version == "0.1.0"
        assert result.uptime_seconds == 120.5
        assert result.components == {}
        assert result.issues == []

    def test_is_healthy_property(self):
        """is_healthy should return True only for healthy status."""
        healthy = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )
        degraded = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )
        unhealthy = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )

        assert healthy.is_healthy is True
        assert degraded.is_healthy is False
        assert unhealthy.is_healthy is False

    def test_is_ready_property(self):
        """is_ready should return True for healthy or degraded."""
        healthy = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )
        degraded = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )
        unhealthy = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(timezone.utc),
            version="0.1.0",
            uptime_seconds=0
        )

        assert healthy.is_ready is True
        assert degraded.is_ready is True
        assert unhealthy.is_ready is False

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            version="0.1.0",
            uptime_seconds=3600.5,
            components={
                "database": ComponentHealth(
                    name="database",
                    status=ComponentStatus.READY
                )
            },
            issues=["Some warning"]
        )

        d = result.to_dict()

        assert d["status"] == "degraded"
        assert d["version"] == "0.1.0"
        assert d["uptime_seconds"] == 3600.5
        assert "database" in d["components"]
        assert d["issues"] == ["Some warning"]
        assert "summary" in d


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def test_initialization(self):
        """Should initialize with optional components."""
        checker = HealthChecker()
        assert checker.database is None
        assert checker.embedder is None
        assert checker.search_engine is None
        assert checker.version == "0.1.0"

    def test_initialization_with_components(self):
        """Should accept all components."""
        db = Mock()
        embedder = Mock()
        search_engine = Mock()

        checker = HealthChecker(
            database=db,
            embedder=embedder,
            search_engine=search_engine,
            version="1.2.3"
        )

        assert checker.database is db
        assert checker.embedder is embedder
        assert checker.search_engine is search_engine
        assert checker.version == "1.2.3"

    def test_uptime_seconds(self):
        """uptime_seconds should increase over time."""
        checker = HealthChecker()
        uptime1 = checker.uptime_seconds
        assert uptime1 >= 0
        # Uptime should be a small positive number
        assert uptime1 < 1.0

    def test_is_alive(self):
        """is_alive should always return True."""
        checker = HealthChecker()
        assert checker.is_alive() is True

    @pytest.mark.asyncio
    async def test_check_readiness_no_components(self):
        """Should return False when components are None."""
        checker = HealthChecker()
        assert await checker.check_readiness() is False

    @pytest.mark.asyncio
    async def test_check_readiness_database_not_connected(self):
        """Should return False when database not connected."""
        db = Mock()
        db.is_connected = False

        checker = HealthChecker(database=db)
        assert await checker.check_readiness() is False

    @pytest.mark.asyncio
    async def test_check_readiness_embedder_no_model(self):
        """Should return False when embedder model not loaded."""
        db = Mock()
        db.is_connected = True

        embedder = Mock()
        embedder.model = None

        checker = HealthChecker(database=db, embedder=embedder)
        assert await checker.check_readiness() is False

    @pytest.mark.asyncio
    async def test_check_readiness_all_ready(self):
        """Should return True when all components ready."""
        db = Mock()
        db.is_connected = True

        embedder = Mock()
        embedder.model = Mock()

        checker = HealthChecker(database=db, embedder=embedder)
        assert await checker.check_readiness() is True


class TestHealthCheckerDatabaseCheck:
    """Tests for database health check."""

    @pytest.mark.asyncio
    async def test_database_not_initialized(self):
        """Should return NOT_READY when database is None."""
        checker = HealthChecker()
        result = await checker._check_database()

        assert result.status == ComponentStatus.NOT_READY
        assert "not initialized" in result.message.lower()

    @pytest.mark.asyncio
    async def test_database_not_connected(self):
        """Should return NOT_READY when not connected."""
        db = Mock()
        db.is_connected = False

        checker = HealthChecker(database=db)
        result = await checker._check_database()

        assert result.status == ComponentStatus.NOT_READY
        assert "not connected" in result.message.lower()

    @pytest.mark.asyncio
    async def test_database_empty(self):
        """Should return NOT_READY when database is empty."""
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(return_value={"total_codes": 0})

        checker = HealthChecker(database=db)
        result = await checker._check_database()

        assert result.status == ComponentStatus.NOT_READY
        assert "empty" in result.message.lower()

    @pytest.mark.asyncio
    async def test_database_ready(self):
        """Should return READY when database has data."""
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(return_value={
            "total_codes": 2125,
            "total_index_terms": 20000,
            "total_cross_references": 4500
        })

        checker = HealthChecker(database=db)
        result = await checker._check_database()

        assert result.status == ComponentStatus.READY
        assert result.details["total_codes"] == 2125
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_database_error(self):
        """Should return ERROR when exception occurs."""
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(side_effect=Exception("Connection lost"))

        checker = HealthChecker(database=db)
        result = await checker._check_database()

        assert result.status == ComponentStatus.ERROR
        assert "Connection lost" in result.message


class TestHealthCheckerEmbedderCheck:
    """Tests for embedder health check."""

    @pytest.mark.asyncio
    async def test_embedder_not_initialized(self):
        """Should return NOT_READY when embedder is None."""
        checker = HealthChecker()
        result = await checker._check_embedder()

        assert result.status == ComponentStatus.NOT_READY

    @pytest.mark.asyncio
    async def test_embedder_model_not_loaded(self):
        """Should return NOT_READY when model not loaded."""
        embedder = Mock()
        embedder.model = None

        checker = HealthChecker(embedder=embedder)
        result = await checker._check_embedder()

        assert result.status == ComponentStatus.NOT_READY
        assert "not loaded" in result.message.lower()

    @pytest.mark.asyncio
    async def test_embedder_ready(self):
        """Should return READY when model is loaded."""
        embedder = Mock()
        embedder.model = Mock()
        embedder.model_name = "all-MiniLM-L6-v2"
        embedder.embedding_dim = 384

        checker = HealthChecker(embedder=embedder)
        result = await checker._check_embedder()

        assert result.status == ComponentStatus.READY
        assert result.details["model"] == "all-MiniLM-L6-v2"
        assert result.details["dimension"] == 384


class TestHealthCheckerSearchEngineCheck:
    """Tests for search engine health check."""

    @pytest.mark.asyncio
    async def test_search_engine_not_initialized(self):
        """Should return NOT_READY when search engine is None."""
        checker = HealthChecker()
        result = await checker._check_search_engine()

        assert result.status == ComponentStatus.NOT_READY

    @pytest.mark.asyncio
    async def test_search_engine_embeddings_not_ready(self):
        """Should return PARTIAL when embeddings not ready."""
        search_engine = Mock()
        search_engine.embeddings_ready = False

        checker = HealthChecker(search_engine=search_engine)
        result = await checker._check_search_engine()

        assert result.status == ComponentStatus.PARTIAL
        assert "not fully loaded" in result.message.lower()

    @pytest.mark.asyncio
    async def test_search_engine_ready(self):
        """Should return READY when embeddings ready."""
        search_engine = Mock()
        search_engine.embeddings_ready = True
        search_engine.embedding_cache.get_stats.return_value = {"size": 100, "hits": 50}
        search_engine.search_cache.get_stats.return_value = {"size": 20, "hits": 10}

        checker = HealthChecker(search_engine=search_engine)
        result = await checker._check_search_engine()

        assert result.status == ComponentStatus.READY
        assert result.details["embedding_cache_size"] == 100


class TestHealthCheckerFullCheck:
    """Tests for full health check."""

    @pytest.mark.asyncio
    async def test_all_components_ready(self):
        """Should return HEALTHY when all components ready."""
        # Mock database
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(return_value={
            "total_codes": 2125,
            "total_index_terms": 20000,
            "total_cross_references": 4500,
            "embedding_coverage": {"coverage_percent": 100}
        })
        db.connection.execute.return_value.fetchone.return_value = (4000,)

        # Mock embedder
        embedder = Mock()
        embedder.model = Mock()
        embedder.model_name = "all-MiniLM-L6-v2"
        embedder.embedding_dim = 384

        # Mock search engine
        search_engine = Mock()
        search_engine.embeddings_ready = True
        search_engine.embedding_cache.get_stats.return_value = {"size": 100, "hits": 50}
        search_engine.search_cache.get_stats.return_value = {"size": 20, "hits": 10}

        checker = HealthChecker(
            database=db,
            embedder=embedder,
            search_engine=search_engine,
            version="0.1.0"
        )

        result = await checker.check_health()

        assert result.status == HealthStatus.HEALTHY
        assert result.is_healthy is True
        assert result.is_ready is True
        assert len(result.issues) == 0
        assert "database" in result.components
        assert "embedder" in result.components
        assert "search_engine" in result.components

    @pytest.mark.asyncio
    async def test_degraded_when_partial(self):
        """Should return DEGRADED when some components are partial."""
        # Mock database - ready
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(return_value={
            "total_codes": 2125,
            "total_index_terms": 20000,
            "total_cross_references": 0,  # No cross-refs
            "embedding_coverage": {"coverage_percent": 50}  # Partial
        })
        db.connection.execute.return_value.fetchone.return_value = (0,)

        # Mock embedder - ready
        embedder = Mock()
        embedder.model = Mock()
        embedder.model_name = "all-MiniLM-L6-v2"
        embedder.embedding_dim = 384

        # Mock search engine - ready
        search_engine = Mock()
        search_engine.embeddings_ready = True
        search_engine.embedding_cache.get_stats.return_value = {"size": 100, "hits": 50}
        search_engine.search_cache.get_stats.return_value = {"size": 20, "hits": 10}

        checker = HealthChecker(
            database=db,
            embedder=embedder,
            search_engine=search_engine
        )

        result = await checker.check_health()

        assert result.status == HealthStatus.DEGRADED
        assert result.is_ready is True
        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_unhealthy_when_error(self):
        """Should return UNHEALTHY when critical component has error."""
        # Mock database with error
        db = Mock()
        db.is_connected = True
        db.get_statistics = AsyncMock(side_effect=Exception("Database error"))

        checker = HealthChecker(database=db)

        result = await checker.check_health()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.is_ready is False
        assert any("database" in issue.lower() for issue in result.issues)


class TestLivenessCheck:
    """Tests for liveness_check function."""

    @pytest.mark.asyncio
    async def test_returns_alive_status(self):
        """Should return alive status."""
        result = await liveness_check()

        assert result["status"] == "alive"
        assert "timestamp" in result


class TestReadinessCheck:
    """Tests for readiness_check function."""

    @pytest.mark.asyncio
    async def test_ready_when_components_ready(self):
        """Should return ready when all components ready."""
        db = Mock()
        db.is_connected = True
        embedder = Mock()
        embedder.model = Mock()

        checker = HealthChecker(database=db, embedder=embedder)
        result = await readiness_check(checker)

        assert result["status"] == "ready"
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_not_ready_when_components_missing(self):
        """Should return not_ready when components missing."""
        checker = HealthChecker()
        result = await readiness_check(checker)

        assert result["status"] == "not_ready"
