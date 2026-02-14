"""
Tests for HTTP server module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from naics_mcp_server.core.health import (
    ComponentHealth,
    ComponentStatus,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
)
from naics_mcp_server.core.shutdown import ShutdownManager, ShutdownState
from naics_mcp_server.http_server import (
    HTTPServer,
    HTTPServerConfig,
    HTTPServerState,
    create_http_server,
)


class TestHTTPServerConfig:
    """Tests for HTTPServerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HTTPServerConfig()

        assert config.enabled is True
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.health_path == "/health"
        assert config.ready_path == "/ready"
        assert config.status_path == "/status"
        assert config.metrics_path == "/metrics"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HTTPServerConfig(
            enabled=False,
            host="127.0.0.1",
            port=8080,
            health_path="/healthz",
            ready_path="/readyz",
        )

        assert config.enabled is False
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.health_path == "/healthz"
        assert config.ready_path == "/readyz"

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "NAICS_HTTP_ENABLED": "false",
                "NAICS_HTTP_HOST": "localhost",
                "NAICS_HTTP_PORT": "8000",
            },
        ):
            config = HTTPServerConfig.from_env()

            assert config.enabled is False
            assert config.host == "localhost"
            assert config.port == 8000


class TestHTTPServerState:
    """Tests for HTTPServerState."""

    def test_default_state(self):
        """Test default state values."""
        state = HTTPServerState()

        assert state.health_checker is None
        assert state.shutdown_manager is None
        assert state.server_version == "0.1.0"
        assert state.uptime_seconds >= 0

    def test_uptime_calculation(self):
        """Test uptime calculation."""
        start_time = time.monotonic()
        state = HTTPServerState(start_time=start_time)

        # Wait a tiny bit
        time.sleep(0.01)

        assert state.uptime_seconds >= 0.01


class TestHTTPServer:
    """Tests for HTTPServer."""

    def test_initialization(self):
        """Test server initialization."""
        config = HTTPServerConfig(enabled=True, port=9999)
        server = HTTPServer(config=config, server_version="1.0.0")

        assert server.config.port == 9999
        assert server.state.server_version == "1.0.0"
        assert server.is_running is False

    def test_create_app(self):
        """Test Starlette app creation."""
        config = HTTPServerConfig()
        server = HTTPServer(config=config)

        app = server._create_app()

        # Check routes exist
        route_paths = [route.path for route in app.routes]
        assert config.health_path in route_paths
        assert config.ready_path in route_paths
        assert config.status_path in route_paths
        assert config.metrics_path in route_paths

    def test_update_health_checker(self):
        """Test updating health checker reference."""
        server = HTTPServer()
        mock_checker = MagicMock(spec=HealthChecker)

        server.update_health_checker(mock_checker)

        assert server.state.health_checker is mock_checker

    def test_update_shutdown_manager(self):
        """Test updating shutdown manager reference."""
        server = HTTPServer()
        mock_manager = MagicMock(spec=ShutdownManager)

        server.update_shutdown_manager(mock_manager)

        assert server.state.shutdown_manager is mock_manager


class TestHTTPEndpoints:
    """Tests for HTTP endpoints using TestClient."""

    @pytest.fixture
    def http_server(self):
        """Create HTTP server for testing."""
        config = HTTPServerConfig(enabled=True)
        server = HTTPServer(config=config, server_version="0.1.0-test")
        return server

    @pytest.fixture
    def test_client(self, http_server):
        """Create test client."""
        app = http_server._create_app()
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test /health endpoint returns alive status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_health_endpoint_with_shutdown(self, http_server, test_client):
        """Test /health endpoint includes shutdown state."""
        # Create a mock shutdown manager that's shutting down
        mock_manager = MagicMock(spec=ShutdownManager)
        mock_manager.state = ShutdownState.DRAINING
        http_server.state.shutdown_manager = mock_manager

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["shutdown_state"] == "draining"

    def test_ready_endpoint_no_checker(self, test_client):
        """Test /ready endpoint without health checker returns ready."""
        response = test_client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "uptime_seconds" in data

    def test_ready_endpoint_ready(self, http_server, test_client):
        """Test /ready endpoint when server is ready."""
        # Create a mock health checker that returns ready
        mock_checker = MagicMock(spec=HealthChecker)
        mock_checker.check_readiness = AsyncMock(return_value=True)
        http_server.state.health_checker = mock_checker

        response = test_client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_ready_endpoint_not_ready(self, http_server, test_client):
        """Test /ready endpoint when server is not ready."""
        # Create a mock health checker that returns not ready
        mock_checker = MagicMock(spec=HealthChecker)
        mock_checker.check_readiness = AsyncMock(return_value=False)
        http_server.state.health_checker = mock_checker

        response = test_client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["reason"] == "components_not_ready"

    def test_ready_endpoint_shutting_down(self, http_server, test_client):
        """Test /ready endpoint during shutdown."""
        # Create a mock shutdown manager that's shutting down
        mock_manager = MagicMock(spec=ShutdownManager)
        mock_manager.state = ShutdownState.SHUTTING_DOWN
        http_server.state.shutdown_manager = mock_manager

        response = test_client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["reason"] == "shutting_down"

    def test_status_endpoint_basic(self, test_client, http_server):
        """Test /status endpoint basic response."""
        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "0.1.0-test"
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_status_endpoint_with_health(self, http_server, test_client):
        """Test /status endpoint with health checker."""
        from datetime import UTC, datetime

        # Create mock health result
        mock_result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(UTC),
            version="0.1.0",
            uptime_seconds=100.0,
            components={
                "database": ComponentHealth(
                    name="database",
                    status=ComponentStatus.READY,
                )
            },
        )

        mock_checker = MagicMock(spec=HealthChecker)
        mock_checker.check_health = AsyncMock(return_value=mock_result)
        http_server.state.health_checker = mock_checker

        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "health" in data
        assert data["health"]["status"] == "healthy"

    def test_status_endpoint_with_shutdown_manager(self, http_server, test_client):
        """Test /status endpoint with shutdown manager."""
        mock_manager = MagicMock(spec=ShutdownManager)
        mock_manager.get_status = AsyncMock(
            return_value={
                "state": "running",
                "in_flight_requests": 5,
            }
        )
        http_server.state.shutdown_manager = mock_manager

        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "shutdown" in data
        assert data["shutdown"]["state"] == "running"
        assert data["shutdown"]["in_flight_requests"] == 5

    def test_metrics_endpoint(self, test_client):
        """Test /metrics endpoint returns Prometheus format."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Should contain some prometheus metrics
        assert "naics_" in response.text or "# HELP" in response.text or response.text == ""


class TestHTTPServerLifecycle:
    """Tests for HTTP server start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Test start does nothing when disabled."""
        config = HTTPServerConfig(enabled=False)
        server = HTTPServer(config=config)

        await server.start()

        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_start_already_started(self):
        """Test start does nothing when already started."""
        config = HTTPServerConfig(enabled=True)
        server = HTTPServer(config=config)
        server._started = True

        # Should not raise
        await server.start()

    @pytest.mark.asyncio
    async def test_stop_not_started(self):
        """Test stop does nothing when not started."""
        config = HTTPServerConfig(enabled=True)
        server = HTTPServer(config=config)

        # Should not raise
        await server.stop()


class TestCreateHTTPServer:
    """Tests for create_http_server factory function."""

    def test_creates_server_with_defaults(self):
        """Test factory creates server with default config."""
        server = create_http_server()

        assert isinstance(server, HTTPServer)
        assert server.config.enabled is True

    def test_creates_server_with_components(self):
        """Test factory accepts components."""
        mock_checker = MagicMock(spec=HealthChecker)
        mock_manager = MagicMock(spec=ShutdownManager)

        server = create_http_server(
            health_checker=mock_checker,
            shutdown_manager=mock_manager,
            server_version="2.0.0",
        )

        assert server.state.health_checker is mock_checker
        assert server.state.shutdown_manager is mock_manager
        assert server.state.server_version == "2.0.0"


class TestHTTPServerIntegration:
    """Integration tests for HTTP server with real async operations."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full server start/stop lifecycle."""
        config = HTTPServerConfig(enabled=True, port=19090)  # Use non-standard port
        server = HTTPServer(config=config)

        try:
            # Start server
            await server.start()
            assert server.is_running is True

            # Give it a moment to start
            await asyncio.sleep(0.1)

            # Stop server
            await server.stop()
            assert server.is_running is False

        finally:
            # Ensure cleanup
            if server.is_running:
                await server.stop()

    @pytest.mark.asyncio
    async def test_server_responds_while_running(self):
        """Test server responds to HTTP requests while running."""
        import httpx

        config = HTTPServerConfig(enabled=True, port=19091)
        server = HTTPServer(config=config)

        try:
            await server.start()
            await asyncio.sleep(0.2)  # Give server time to start

            # Make HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{config.port}{config.health_path}",
                    timeout=5.0,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "alive"

        finally:
            await server.stop()
