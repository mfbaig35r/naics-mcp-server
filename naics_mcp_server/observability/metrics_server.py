"""
Simple HTTP server for Prometheus metrics endpoint.

This provides a standalone HTTP endpoint that Prometheus can scrape.
Run alongside the MCP server to expose metrics.

Usage:
    python -m naics_mcp_server.observability.metrics_server
    # or
    naics-mcp metrics-server --port 9090
"""

import argparse
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .metrics import get_metrics

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Prometheus metrics."""

    def do_GET(self):
        """Handle GET requests for metrics."""
        if self.path == "/metrics" or self.path == "/":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404, "Not Found")

    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        try:
            metrics_data = get_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(metrics_data)))
            self.end_headers()
            self.wfile.write(metrics_data)
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_error(500, str(e))

    def _serve_health(self):
        """Serve simple health check."""
        response = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use Python logging instead of stderr."""
        logger.debug(f"Metrics server: {format % args}")


def run_metrics_server(host: str = "0.0.0.0", port: int = 9090) -> None:
    """
    Run the metrics HTTP server.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 9090)
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, MetricsHandler)
    logger.info(f"Starting metrics server on {host}:{port}")
    print(f"Metrics available at http://{host}:{port}/metrics")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Metrics server shutting down")
        httpd.shutdown()


def main():
    """CLI entry point for metrics server."""
    parser = argparse.ArgumentParser(description="NAICS MCP Server Metrics Endpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_metrics_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
