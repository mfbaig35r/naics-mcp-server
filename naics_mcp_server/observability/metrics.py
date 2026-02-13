"""
Prometheus metrics for NAICS MCP Server.

Provides counters, histograms, and gauges for monitoring:
- Tool invocations and latency
- Search operations and cache performance
- Database queries and connection health
- Embedding generation and cache stats
"""

import time
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from threading import Lock
from typing import Any, TypeVar

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# Context variable for tracking in-flight requests
_active_tool: ContextVar[str | None] = ContextVar("active_tool", default=None)

# Registry lock for thread-safe metric creation
_metrics_lock = Lock()
_initialized = False


# --- Metric Definitions ---

# Tool-level metrics
TOOL_REQUESTS = Counter(
    "naics_tool_requests_total",
    "Total number of tool invocations",
    ["tool", "status"],
)

TOOL_DURATION = Histogram(
    "naics_tool_duration_seconds",
    "Tool execution duration in seconds",
    ["tool"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)

TOOL_ERRORS = Counter(
    "naics_tool_errors_total",
    "Total number of tool errors",
    ["tool", "error_type"],
)

ACTIVE_REQUESTS = Gauge(
    "naics_active_requests",
    "Number of currently active requests",
    ["tool"],
)

# Search metrics
SEARCH_REQUESTS = Counter(
    "naics_search_requests_total",
    "Total search requests",
    ["strategy"],
)

SEARCH_DURATION = Histogram(
    "naics_search_duration_seconds",
    "Search operation duration in seconds",
    ["strategy"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

SEARCH_RESULTS = Histogram(
    "naics_search_results_count",
    "Number of results returned per search",
    ["strategy"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

SEARCH_CONFIDENCE = Histogram(
    "naics_search_confidence_score",
    "Distribution of top result confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

SEARCH_FALLBACKS = Counter(
    "naics_search_fallbacks_total",
    "Number of search strategy fallbacks",
    ["from_strategy", "to_strategy"],
)

# Cache metrics
CACHE_OPERATIONS = Counter(
    "naics_cache_operations_total",
    "Cache operations",
    ["cache_type", "operation"],  # operation: hit, miss, evict
)

CACHE_SIZE = Gauge(
    "naics_cache_size",
    "Current cache size (entries)",
    ["cache_type"],
)

CACHE_HIT_RATE = Gauge(
    "naics_cache_hit_rate",
    "Cache hit rate (0-1)",
    ["cache_type"],
)

# Database metrics
DB_QUERIES = Counter(
    "naics_db_queries_total",
    "Total database queries",
    ["query_type", "status"],
)

DB_QUERY_DURATION = Histogram(
    "naics_db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

DB_CONNECTIONS = Gauge(
    "naics_db_connections_active",
    "Number of active database connections",
)

DB_CONNECTION_ERRORS = Counter(
    "naics_db_connection_errors_total",
    "Database connection errors",
    ["error_type"],
)

# Embedding metrics
EMBEDDING_OPERATIONS = Counter(
    "naics_embedding_operations_total",
    "Embedding generation operations",
    ["operation_type"],  # single, batch
)

EMBEDDING_DURATION = Histogram(
    "naics_embedding_duration_seconds",
    "Embedding generation duration",
    ["operation_type"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

EMBEDDING_BATCH_SIZE = Histogram(
    "naics_embedding_batch_size",
    "Batch sizes for embedding generation",
    buckets=[1, 5, 10, 20, 50, 100, 200],
)

# Cross-reference metrics
CROSSREF_LOOKUPS = Counter(
    "naics_crossref_lookups_total",
    "Cross-reference lookups",
)

CROSSREF_EXCLUSIONS_FOUND = Counter(
    "naics_crossref_exclusions_found_total",
    "Number of exclusion warnings found",
)

# Health metrics
HEALTH_STATUS = Gauge(
    "naics_health_status",
    "Overall health status (1=healthy, 0.5=degraded, 0=unhealthy)",
)

COMPONENT_STATUS = Gauge(
    "naics_component_status",
    "Component health status (1=ready, 0=not ready)",
    ["component"],
)

UPTIME_SECONDS = Gauge(
    "naics_uptime_seconds",
    "Server uptime in seconds",
)

# Data statistics
DATA_STATS = Gauge(
    "naics_data_stats",
    "Data statistics",
    ["stat_type"],  # codes, embeddings, index_terms, cross_references
)

# Server info
SERVER_INFO = Info(
    "naics_server",
    "NAICS MCP Server information",
)


# --- Timer Context Manager ---


class Timer:
    """Context manager for timing operations and recording to histograms."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            if self.labels:
                self.histogram.labels(**self.labels).observe(self.duration)
            else:
                self.histogram.observe(self.duration)

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.duration is not None:
            return int(self.duration * 1000)
        if self.start_time is not None:
            return int((time.perf_counter() - self.start_time) * 1000)
        return 0


# --- Decorators ---

F = TypeVar("F", bound=Callable[..., Any])


def track_tool_metrics(func: F) -> F:
    """
    Decorator to track metrics for MCP tool invocations.

    Records:
    - Request count (success/failure)
    - Duration histogram
    - Error count by type
    - Active request gauge
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        _active_tool.set(tool_name)

        # Increment active requests
        ACTIVE_REQUESTS.labels(tool=tool_name).inc()

        timer = Timer(TOOL_DURATION, {"tool": tool_name})

        try:
            with timer:
                result = await func(*args, **kwargs)

            TOOL_REQUESTS.labels(tool=tool_name, status="success").inc()
            return result

        except Exception as e:
            TOOL_REQUESTS.labels(tool=tool_name, status="failure").inc()
            TOOL_ERRORS.labels(tool=tool_name, error_type=type(e).__name__).inc()
            raise

        finally:
            ACTIVE_REQUESTS.labels(tool=tool_name).dec()
            _active_tool.set(None)

    return wrapper  # type: ignore


def track_db_query(query_type: str):
    """
    Decorator to track database query metrics.

    Args:
        query_type: Type of query (get_by_code, search_text, etc.)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            timer = Timer(DB_QUERY_DURATION, {"query_type": query_type})

            try:
                with timer:
                    result = await func(*args, **kwargs)

                DB_QUERIES.labels(query_type=query_type, status="success").inc()
                return result

            except Exception:
                DB_QUERIES.labels(query_type=query_type, status="failure").inc()
                raise

        return wrapper  # type: ignore

    return decorator


def track_embedding_operation(operation_type: str = "single"):
    """
    Decorator to track embedding generation metrics.

    Args:
        operation_type: Type of operation (single, batch)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            timer = Timer(EMBEDDING_DURATION, {"operation_type": operation_type})

            try:
                with timer:
                    result = await func(*args, **kwargs)

                EMBEDDING_OPERATIONS.labels(operation_type=operation_type).inc()
                return result

            except Exception:
                raise

        return wrapper  # type: ignore

    return decorator


# --- Helper Functions ---


def record_search_metrics(
    strategy: str,
    duration_seconds: float,
    results_count: int,
    top_confidence: float | None = None,
):
    """Record metrics for a completed search operation."""
    SEARCH_REQUESTS.labels(strategy=strategy).inc()
    SEARCH_DURATION.labels(strategy=strategy).observe(duration_seconds)
    SEARCH_RESULTS.labels(strategy=strategy).observe(results_count)

    if top_confidence is not None:
        SEARCH_CONFIDENCE.observe(top_confidence)


def record_cache_hit(cache_type: str):
    """Record a cache hit."""
    CACHE_OPERATIONS.labels(cache_type=cache_type, operation="hit").inc()


def record_cache_miss(cache_type: str):
    """Record a cache miss."""
    CACHE_OPERATIONS.labels(cache_type=cache_type, operation="miss").inc()


def update_cache_stats(cache_type: str, size: int, hit_rate: float):
    """Update cache statistics gauges."""
    CACHE_SIZE.labels(cache_type=cache_type).set(size)
    CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)


def record_search_fallback(from_strategy: str, to_strategy: str):
    """Record a search strategy fallback."""
    SEARCH_FALLBACKS.labels(from_strategy=from_strategy, to_strategy=to_strategy).inc()


def record_crossref_lookup(exclusions_found: int = 0):
    """Record a cross-reference lookup."""
    CROSSREF_LOOKUPS.inc()
    if exclusions_found > 0:
        CROSSREF_EXCLUSIONS_FOUND.inc(exclusions_found)


def update_health_status(
    status: str,
    components: dict[str, bool] | None = None,
    uptime_seconds: float | None = None,
):
    """
    Update health status metrics.

    Args:
        status: Overall status (healthy, degraded, unhealthy)
        components: Dict of component name to ready status
        uptime_seconds: Server uptime
    """
    status_values = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}
    HEALTH_STATUS.set(status_values.get(status, 0.0))

    if components:
        for component, ready in components.items():
            COMPONENT_STATUS.labels(component=component).set(1.0 if ready else 0.0)

    if uptime_seconds is not None:
        UPTIME_SECONDS.set(uptime_seconds)


def update_data_stats(
    total_codes: int = 0,
    total_embeddings: int = 0,
    total_index_terms: int = 0,
    total_cross_references: int = 0,
):
    """Update data statistics gauges."""
    DATA_STATS.labels(stat_type="codes").set(total_codes)
    DATA_STATS.labels(stat_type="embeddings").set(total_embeddings)
    DATA_STATS.labels(stat_type="index_terms").set(total_index_terms)
    DATA_STATS.labels(stat_type="cross_references").set(total_cross_references)


def set_server_info(version: str, model: str, database_path: str):
    """Set server info labels."""
    SERVER_INFO.info(
        {
            "version": version,
            "embedding_model": model,
            "database_path": database_path,
        }
    )


def record_db_connection_error(error_type: str):
    """Record a database connection error."""
    DB_CONNECTION_ERRORS.labels(error_type=error_type).inc()


def update_db_connections(count: int):
    """Update active database connection count."""
    DB_CONNECTIONS.set(count)


def record_embedding_batch(batch_size: int):
    """Record embedding batch size."""
    EMBEDDING_BATCH_SIZE.observe(batch_size)


# --- Metrics Export ---


def get_metrics() -> bytes:
    """
    Generate Prometheus metrics output.

    Returns:
        Prometheus text format metrics
    """
    return generate_latest(REGISTRY)


def get_metrics_text() -> str:
    """
    Generate Prometheus metrics as string.

    Returns:
        Prometheus text format metrics as string
    """
    return generate_latest(REGISTRY).decode("utf-8")


# --- Initialization ---


def initialize_metrics(
    version: str = "0.1.0",
    embedding_model: str = "all-MiniLM-L6-v2",
    database_path: str = "",
) -> None:
    """
    Initialize metrics with server information.

    Call this once at server startup.
    """
    global _initialized

    with _metrics_lock:
        if _initialized:
            return

        # Set server info
        set_server_info(version, embedding_model, database_path)

        # Initialize health status
        HEALTH_STATUS.set(0.0)  # Will be updated when health check runs

        _initialized = True


def reset_metrics() -> None:
    """
    Reset all metrics (for testing).
    """
    global _initialized
    _initialized = False
