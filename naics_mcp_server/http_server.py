"""
HTTP server for health checks, metrics, and status endpoints.

Provides HTTP endpoints for production deployments:
- /health - Liveness probe (is the process alive?)
- /ready - Readiness probe (is the server ready for traffic?)
- /status - Detailed server status with component health
- /metrics - Prometheus metrics endpoint

The HTTP server runs alongside the MCP server on a separate port,
allowing container orchestration (Kubernetes, Docker) and monitoring
systems (Prometheus, Grafana) to interact with the server.

Usage:
    from naics_mcp_server.http_server import HTTPServer

    http_server = HTTPServer(
        port=9090,
        health_checker=health_checker,
        shutdown_manager=shutdown_manager,
    )

    # Start in background
    await http_server.start()

    # ... run MCP server ...

    # Stop gracefully
    await http_server.stop()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .core.health import HealthChecker
from .core.shutdown import ShutdownManager, ShutdownState
from .observability.metrics import get_metrics_text

logger = logging.getLogger(__name__)


@dataclass
class HTTPServerConfig:
    """Configuration for the HTTP server."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 9090
    health_path: str = "/health"
    ready_path: str = "/ready"
    status_path: str = "/status"
    metrics_path: str = "/metrics"

    @classmethod
    def from_env(cls) -> "HTTPServerConfig":
        """Create config from environment variables."""
        import os

        return cls(
            enabled=os.getenv("NAICS_HTTP_ENABLED", "true").lower() in ("true", "1", "yes"),
            host=os.getenv("NAICS_HTTP_HOST", "0.0.0.0"),
            port=int(os.getenv("NAICS_HTTP_PORT", "9090")),
            health_path=os.getenv("NAICS_HEALTH_PATH", "/health"),
            ready_path=os.getenv("NAICS_READY_PATH", "/ready"),
            status_path=os.getenv("NAICS_STATUS_PATH", "/status"),
            metrics_path=os.getenv("NAICS_METRICS_PATH", "/metrics"),
        )


@dataclass
class HTTPServerState:
    """State shared with HTTP handlers."""

    health_checker: HealthChecker | None = None
    shutdown_manager: ShutdownManager | None = None
    server_version: str = "0.1.0"
    start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.monotonic() - self.start_time


class HTTPServer:
    """
    HTTP server for health checks and metrics.

    Runs alongside the MCP server to provide HTTP endpoints for:
    - Container health probes (Kubernetes liveness/readiness)
    - Prometheus metrics scraping
    - Status dashboards
    """

    def __init__(
        self,
        config: HTTPServerConfig | None = None,
        health_checker: HealthChecker | None = None,
        shutdown_manager: ShutdownManager | None = None,
        server_version: str = "0.1.0",
    ):
        """
        Initialize HTTP server.

        Args:
            config: HTTP server configuration
            health_checker: Health checker instance for health endpoints
            shutdown_manager: Shutdown manager for status info
            server_version: Server version for status endpoint
        """
        self.config = config or HTTPServerConfig()
        self.state = HTTPServerState(
            health_checker=health_checker,
            shutdown_manager=shutdown_manager,
            server_version=server_version,
        )
        self._server: asyncio.Server | None = None
        self._app: Starlette | None = None
        self._started = False

    def _create_app(self) -> Starlette:
        """Create the Starlette application with routes."""
        routes = [
            Route(self.config.health_path, self._health_handler, methods=["GET"]),
            Route(self.config.ready_path, self._ready_handler, methods=["GET"]),
            Route(self.config.status_path, self._status_handler, methods=["GET"]),
            Route(self.config.metrics_path, self._metrics_handler, methods=["GET"]),
        ]

        app = Starlette(routes=routes)
        app.state.http_state = self.state
        return app

    async def _health_handler(self, request: Request) -> Response:
        """
        Liveness probe handler.

        Returns 200 if the process is alive and responsive.
        Used by Kubernetes livenessProbe.

        Response:
            200: Server is alive
            503: Server is not alive (shouldn't happen if this runs)
        """
        state: HTTPServerState = request.app.state.http_state

        # If shutdown is in progress, we're still alive but signaling shutdown
        shutdown_state = None
        if state.shutdown_manager:
            shutdown_state = state.shutdown_manager.state

        response_data = {
            "status": "alive",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Include shutdown state if shutting down
        if shutdown_state and shutdown_state != ShutdownState.RUNNING:
            response_data["shutdown_state"] = shutdown_state.value

        return JSONResponse(response_data, status_code=200)

    async def _ready_handler(self, request: Request) -> Response:
        """
        Readiness probe handler.

        Returns 200 if the server is ready to accept traffic.
        Used by Kubernetes readinessProbe.

        Response:
            200: Server is ready for traffic
            503: Server is not ready (still initializing or shutting down)
        """
        state: HTTPServerState = request.app.state.http_state

        # Check if shutting down
        if state.shutdown_manager:
            if state.shutdown_manager.state != ShutdownState.RUNNING:
                return JSONResponse(
                    {
                        "status": "not_ready",
                        "reason": "shutting_down",
                        "shutdown_state": state.shutdown_manager.state.value,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    status_code=503,
                )

        # Check health if checker available
        if state.health_checker:
            is_ready = await state.health_checker.check_readiness()
            if not is_ready:
                return JSONResponse(
                    {
                        "status": "not_ready",
                        "reason": "components_not_ready",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    status_code=503,
                )

        return JSONResponse(
            {
                "status": "ready",
                "uptime_seconds": round(state.uptime_seconds, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            status_code=200,
        )

    async def _status_handler(self, request: Request) -> Response:
        """
        Detailed status handler.

        Returns comprehensive server status including component health.
        Used for monitoring dashboards and debugging.

        Response:
            200: Status retrieved (regardless of health)
        """
        state: HTTPServerState = request.app.state.http_state

        status_data: dict[str, Any] = {
            "version": state.server_version,
            "uptime_seconds": round(state.uptime_seconds, 1),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add shutdown manager status
        if state.shutdown_manager:
            shutdown_status = await state.shutdown_manager.get_status()
            status_data["shutdown"] = {
                "state": shutdown_status["state"],
                "in_flight_requests": shutdown_status["in_flight_requests"],
            }

        # Add health check results
        if state.health_checker:
            try:
                health_result = await state.health_checker.check_health()
                status_data["health"] = health_result.to_dict()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                status_data["health"] = {
                    "status": "error",
                    "error": str(e)[:100],
                }
        else:
            status_data["health"] = {
                "status": "unknown",
                "message": "Health checker not configured",
            }

        return JSONResponse(status_data, status_code=200)

    async def _metrics_handler(self, request: Request) -> Response:
        """
        Prometheus metrics handler.

        Returns metrics in Prometheus text format.
        Used by Prometheus scraper.
        """
        try:
            metrics_text = get_metrics_text()
            return PlainTextResponse(
                metrics_text,
                status_code=200,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return PlainTextResponse(
                f"# Error generating metrics: {e}\n",
                status_code=500,
            )

    async def start(self) -> None:
        """
        Start the HTTP server.

        Runs the server in the background using asyncio.
        """
        if not self.config.enabled:
            logger.info("HTTP server disabled by configuration")
            return

        if self._started:
            logger.warning("HTTP server already started")
            return

        self._app = self._create_app()

        try:
            # Import uvicorn config for server creation
            import uvicorn

            config = uvicorn.Config(
                app=self._app,
                host=self.config.host,
                port=self.config.port,
                log_level="warning",  # Reduce uvicorn noise
                access_log=False,
            )
            server = uvicorn.Server(config)

            # Start server in background task
            self._server_task = asyncio.create_task(server.serve())
            self._uvicorn_server = server
            self._started = True

            logger.info(
                f"HTTP server started on http://{self.config.host}:{self.config.port}",
                extra={
                    "http_port": self.config.port,
                    "endpoints": [
                        self.config.health_path,
                        self.config.ready_path,
                        self.config.status_path,
                        self.config.metrics_path,
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the HTTP server gracefully."""
        if not self._started:
            return

        logger.info("Stopping HTTP server...")

        try:
            if hasattr(self, "_uvicorn_server") and self._uvicorn_server:
                self._uvicorn_server.should_exit = True

            if hasattr(self, "_server_task") and self._server_task:
                # Wait for server to stop with timeout
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    logger.warning("HTTP server stop timed out, cancelling...")
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass

            self._started = False
            logger.info("HTTP server stopped")

        except Exception as e:
            logger.error(f"Error stopping HTTP server: {e}")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._started

    def update_health_checker(self, health_checker: HealthChecker) -> None:
        """Update the health checker reference."""
        self.state.health_checker = health_checker

    def update_shutdown_manager(self, shutdown_manager: ShutdownManager) -> None:
        """Update the shutdown manager reference."""
        self.state.shutdown_manager = shutdown_manager


# Convenience function for creating HTTP server from config
def create_http_server(
    health_checker: HealthChecker | None = None,
    shutdown_manager: ShutdownManager | None = None,
    server_version: str = "0.1.0",
) -> HTTPServer:
    """
    Create HTTP server with configuration from environment.

    Args:
        health_checker: Health checker instance
        shutdown_manager: Shutdown manager instance
        server_version: Server version string

    Returns:
        Configured HTTPServer instance
    """
    config = HTTPServerConfig.from_env()
    return HTTPServer(
        config=config,
        health_checker=health_checker,
        shutdown_manager=shutdown_manager,
        server_version=server_version,
    )
