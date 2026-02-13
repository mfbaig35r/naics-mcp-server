"""
Configuration for the NAICS MCP Server.

Uses Pydantic for validation and environment variable loading.
Every setting has a clear purpose, sensible default, and validation.

Environment Variables:
    NAICS_DATABASE_PATH: Path to DuckDB database file
    NAICS_EMBEDDING_MODEL: Sentence transformer model name
    NAICS_EMBEDDING_DIMENSION: Model embedding dimension (must match model)
    NAICS_SEMANTIC_WEIGHT: Weight for semantic search (0.0-1.0)
    NAICS_MIN_CONFIDENCE: Minimum confidence threshold (0.0-1.0)
    NAICS_BOOST_INDEX_TERMS: Boost factor for index term matches (>= 1.0)
    NAICS_MAX_CANDIDATES: Maximum candidates to consider
    NAICS_DEFAULT_LIMIT: Default result limit
    NAICS_QUERY_TIMEOUT: Query timeout in seconds
    NAICS_ENABLE_AUDIT: Enable audit logging (true/false)
    NAICS_ENABLE_QUERY_EXPANSION: Enable query expansion (true/false)
    NAICS_ENABLE_CROSS_REFERENCES: Enable cross-reference lookup (true/false)
    NAICS_DEBUG: Enable debug mode (true/false)
    DEBUG: Alternative debug flag

    Rate Limiting:
    NAICS_ENABLE_RATE_LIMITING: Enable rate limiting (true/false)
    NAICS_DEFAULT_RPM: Default requests per minute (default: 60)
    NAICS_SEARCH_RPM: RPM for search tools (default: 30)
    NAICS_CLASSIFY_RPM: RPM for classification tools (default: 20)
    NAICS_BATCH_RPM: RPM for batch operations (default: 10)

    Logging:
    NAICS_LOG_LEVEL: Log level - DEBUG, INFO, WARNING, ERROR (default: INFO)
    NAICS_LOG_FORMAT: Log format - json or text (default: json)
    NAICS_LOG_FILE: Log file path (logs to stderr if not set)
    NAICS_SERVICE_NAME: Service name for log records (default: naics-mcp-server)
    NAICS_ENVIRONMENT: Environment name (default: development)
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class SearchConfig(BaseSettings):
    """
    Configuration for the NAICS search engine.

    All settings are validated and can be overridden via environment variables.
    Environment variables use the NAICS_ prefix (e.g., NAICS_DATABASE_PATH).
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database configuration
    database_path: Path | None = Field(default=None, description="Path to DuckDB database file")
    connection_pool_size: int = Field(
        default=5, ge=1, le=50, description="Number of database connections to pool"
    )
    query_timeout_seconds: int = Field(
        default=5, ge=1, le=300, description="Query timeout in seconds"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", min_length=1, description="Sentence transformer model name"
    )
    embedding_dimension: int = Field(
        default=384, ge=32, le=4096, description="Embedding vector dimension (must match model)"
    )
    batch_size: int = Field(
        default=32, ge=1, le=512, description="Batch size for embedding generation"
    )
    normalize_embeddings: bool = Field(
        default=True, description="Normalize embeddings for cosine similarity"
    )

    # Search behavior - hybrid search weights
    hybrid_weight_semantic: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for semantic search (0.0-1.0)"
    )
    hybrid_weight_lexical: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for lexical search (0.0-1.0)"
    )
    boost_index_terms: float = Field(
        default=1.5, ge=1.0, le=10.0, description="Boost factor for official NAICS index terms"
    )

    # Performance tuning
    max_candidates: int = Field(
        default=500, ge=10, le=5000, description="Maximum candidates to consider in search"
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold for results"
    )
    default_limit: int = Field(
        default=10, ge=1, le=100, description="Default number of results to return"
    )

    # User experience
    explain_results: bool = Field(
        default=True, description="Include explanations in search results"
    )
    include_hierarchy: bool = Field(
        default=True, description="Include hierarchy information in results"
    )

    # Query expansion
    enable_query_expansion: bool = Field(
        default=True, description="Enable synonym and term expansion"
    )
    max_expansion_terms: int = Field(
        default=5, ge=0, le=20, description="Maximum terms to add via expansion"
    )

    # Cross-reference integration
    enable_cross_references: bool = Field(default=True, description="Enable cross-reference lookup")
    cross_ref_penalty: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Score penalty for excluded activities"
    )

    # Observability
    enable_audit_log: bool = Field(default=True, description="Enable search audit logging")
    audit_retention_days: int = Field(
        default=90, ge=1, le=365, description="Days to retain audit logs"
    )
    log_slow_queries: bool = Field(default=True, description="Log queries exceeding threshold")
    slow_query_threshold_ms: int = Field(
        default=200, ge=10, le=10000, description="Slow query threshold in milliseconds"
    )

    @field_validator("database_path", mode="before")
    @classmethod
    def resolve_database_path(cls, v: Any) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "SearchConfig":
        """Ensure hybrid weights sum to approximately 1.0."""
        total = self.hybrid_weight_semantic + self.hybrid_weight_lexical
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Hybrid weights sum to {total:.2f}, expected 1.0. Adjusting lexical weight."
            )
            self.hybrid_weight_lexical = 1.0 - self.hybrid_weight_semantic
        return self

    @model_validator(mode="after")
    def resolve_default_database_path(self) -> "SearchConfig":
        """Set default database path if not provided."""
        if self.database_path is not None:
            return self

        # Try to find the database in standard locations
        try:
            import naics_mcp_server

            package_dir = Path(naics_mcp_server.__file__).parent
            project_root = package_dir.parent

            # Check locations in priority order
            candidates = [
                project_root / "data" / "naics.duckdb",  # Development
                package_dir / "data" / "naics.duckdb",  # Bundled
                Path.cwd() / "data" / "naics.duckdb",  # Current directory
            ]

            for path in candidates:
                if path.exists():
                    self.database_path = path
                    logger.info(f"Using database: {self.database_path}")
                    return self

            # Default: create in project data directory
            self.database_path = candidates[0]
            logger.info(f"Database will be created at: {self.database_path}")

        except ImportError:
            self.database_path = Path("data/naics.duckdb")

        return self

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "database_path": str(self.database_path) if self.database_path else None,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "hybrid_weight_semantic": self.hybrid_weight_semantic,
            "hybrid_weight_lexical": self.hybrid_weight_lexical,
            "min_confidence": self.min_confidence,
            "max_candidates": self.max_candidates,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_cross_references": self.enable_cross_references,
            "enable_audit_log": self.enable_audit_log,
        }


class MetricsConfig(BaseSettings):
    """
    Configuration for Prometheus metrics.

    Environment variables use the NAICS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Enable/disable metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics collection")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Port for metrics HTTP server"
    )
    metrics_path: str = Field(default="/metrics", description="Path for metrics endpoint")

    # Performance settings
    slow_request_threshold_ms: int = Field(
        default=200, ge=10, le=10000, description="Threshold for slow request warnings (ms)"
    )

    # Cache stats update interval
    cache_stats_interval_seconds: int = Field(
        default=60, ge=10, le=600, description="Interval for updating cache statistics"
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "metrics_path": self.metrics_path,
            "slow_request_threshold_ms": self.slow_request_threshold_ms,
        }


class LoggingConfig(BaseSettings):
    """
    Configuration for structured logging.

    Supports JSON and text formats with sensitive data handling.
    Environment variables use the NAICS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Log level and format
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    log_format: str = Field(
        default="json",
        description="Log format: 'json' for production, 'text' for development",
        pattern=r"^(json|text)$",
    )

    # Output configuration
    log_file: str | None = Field(
        default=None, description="Log file path (logs to stderr if not set)"
    )
    log_max_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum log file size in MB"
    )
    log_retention_count: int = Field(
        default=5, ge=1, le=30, description="Number of rotated log files to keep"
    )

    # Timestamps
    include_timestamp: bool = Field(
        default=True, description="Include ISO timestamp in log records"
    )

    # Sensitive data handling
    max_message_length: int = Field(
        default=1000, ge=100, le=10000, description="Maximum length for log messages"
    )
    max_data_length: int = Field(
        default=500, ge=50, le=5000, description="Maximum length for data field values"
    )
    redact_patterns: bool = Field(
        default=True, description="Redact sensitive patterns (SSN, email, etc.)"
    )

    # Service metadata for log records
    service_name: str = Field(
        default="naics-mcp-server", description="Service name for log records"
    )
    environment: str = Field(
        default="development", description="Environment name (development, staging, production)"
    )

    # Performance
    slow_request_threshold_ms: int = Field(
        default=200, ge=10, le=10000, description="Threshold for slow request warnings (ms)"
    )

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file,
            "service_name": self.service_name,
            "environment": self.environment,
            "slow_request_threshold_ms": self.slow_request_threshold_ms,
        }


class RateLimitConfig(BaseSettings):
    """
    Configuration for rate limiting.

    Uses token bucket algorithm with per-tool limits.
    Environment variables use the NAICS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Enable/disable rate limiting
    enable_rate_limiting: bool = Field(default=False, description="Enable rate limiting for tools")

    # Global defaults (requests per minute)
    default_rpm: int = Field(
        default=60, ge=1, le=1000, description="Default requests per minute for all tools"
    )
    burst_multiplier: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Burst allowance multiplier over base rate"
    )

    # Per-category limits (requests per minute)
    # High-impact tools (CPU-intensive search/classification)
    search_rpm: int = Field(
        default=30, ge=1, le=500, description="RPM for search tools (search_naics_codes, etc.)"
    )
    classify_rpm: int = Field(default=20, ge=1, le=500, description="RPM for classification tools")
    batch_rpm: int = Field(
        default=10, ge=1, le=100, description="RPM for batch operations (classify_batch)"
    )

    # Medium-impact tools
    hierarchy_rpm: int = Field(
        default=60, ge=1, le=500, description="RPM for hierarchy navigation tools"
    )
    analytics_rpm: int = Field(default=30, ge=1, le=500, description="RPM for analytics tools")

    # Low-impact tools (health checks, workbook)
    health_rpm: int = Field(default=120, ge=1, le=1000, description="RPM for health check tools")
    workbook_rpm: int = Field(default=60, ge=1, le=500, description="RPM for workbook tools")

    # Behavior settings
    include_retry_after: bool = Field(
        default=True, description="Include Retry-After hint in rate limit errors"
    )
    log_rate_limit_hits: bool = Field(default=True, description="Log when rate limits are hit")

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "enable_rate_limiting": self.enable_rate_limiting,
            "default_rpm": self.default_rpm,
            "burst_multiplier": self.burst_multiplier,
            "search_rpm": self.search_rpm,
            "classify_rpm": self.classify_rpm,
            "batch_rpm": self.batch_rpm,
            "hierarchy_rpm": self.hierarchy_rpm,
            "analytics_rpm": self.analytics_rpm,
            "health_rpm": self.health_rpm,
            "workbook_rpm": self.workbook_rpm,
        }


class ShutdownConfig(BaseSettings):
    """
    Configuration for graceful shutdown.

    Environment variables use the NAICS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Timeouts
    shutdown_timeout_seconds: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Maximum time to wait for shutdown"
    )
    drain_check_interval: float = Field(
        default=0.5, ge=0.1, le=5.0, description="Interval between drain checks"
    )
    grace_period_seconds: float = Field(
        default=1.0, ge=0.0, le=30.0, description="Grace period before draining"
    )

    # Behavior
    force_after_timeout: bool = Field(
        default=True, description="Force shutdown after timeout (vs wait indefinitely)"
    )
    handle_sigterm: bool = Field(default=True, description="Handle SIGTERM for graceful shutdown")
    handle_sigint: bool = Field(default=True, description="Handle SIGINT (Ctrl+C) gracefully")

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "grace_period_seconds": self.grace_period_seconds,
            "force_after_timeout": self.force_after_timeout,
            "handle_sigterm": self.handle_sigterm,
            "handle_sigint": self.handle_sigint,
        }


class ServerConfig(BaseSettings):
    """
    Configuration for the MCP server itself.

    Environment variables use the NAICS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NAICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = Field(
        default="NAICS Classification Assistant", min_length=1, description="Server display name"
    )
    description: str = Field(
        default=(
            "An intelligent industry classification service for NAICS 2022. "
            "I help you find the right NAICS codes using natural language descriptions, "
            "with support for hierarchical navigation and cross-reference lookup."
        ),
        description="Server description",
    )
    version: str = Field(
        default="0.1.0", pattern=r"^\d+\.\d+\.\d+", description="Server version (semver)"
    )

    # MCP server settings
    stateless_http: bool = Field(default=False, description="Run in stateless HTTP mode")
    enable_cors: bool = Field(default=True, description="Enable CORS for browser clients")
    debug: bool = Field(default=False, description="Enable debug mode")

    @model_validator(mode="before")
    @classmethod
    def check_debug_env(cls, data: Any) -> Any:
        """Check both NAICS_DEBUG and DEBUG environment variables."""
        if isinstance(data, dict):
            if data.get("debug") is None:
                # Check both DEBUG and NAICS_DEBUG
                debug_value = os.getenv("DEBUG", os.getenv("NAICS_DEBUG", "false"))
                data["debug"] = debug_value.lower() in ("true", "1", "yes")
        return data

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/debugging."""
        return {
            "name": self.name,
            "version": self.version,
            "debug": self.debug,
            "stateless_http": self.stateless_http,
            "enable_cors": self.enable_cors,
        }


class AppConfig(BaseModel):
    """
    Combined application configuration.

    Provides a single entry point for all configuration with validation.
    """

    search: SearchConfig = Field(default_factory=SearchConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)

    def validate_startup(self) -> list[str]:
        """
        Validate configuration for startup.

        Returns list of warnings (empty if all OK).
        Raises ConfigurationError for fatal issues.
        """
        from .core.errors import ConfigurationError

        warnings = []

        # Check database path
        if self.search.database_path is None:
            raise ConfigurationError(
                "Database path not configured", config_key="NAICS_DATABASE_PATH"
            )

        # Check if database directory exists (for new databases)
        db_dir = self.search.database_path.parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created database directory: {db_dir}")
            except OSError as e:
                raise ConfigurationError(
                    f"Cannot create database directory: {e}", config_key="NAICS_DATABASE_PATH"
                )

        # Warn about debug mode in production
        if self.server.debug:
            warnings.append("Debug mode is enabled - not recommended for production")

        # Warn about disabled audit logging
        if not self.search.enable_audit_log:
            warnings.append("Audit logging is disabled")

        return warnings

    def to_dict(self) -> dict:
        """Convert full config to dictionary."""
        return {
            "search": self.search.to_dict(),
            "server": self.server.to_dict(),
            "metrics": self.metrics.to_dict(),
            "rate_limit": self.rate_limit.to_dict(),
            "logging": self.logging.to_dict(),
            "shutdown": self.shutdown.to_dict(),
        }


# Singleton instance management
_config_instance: AppConfig | None = None


def get_config() -> AppConfig:
    """
    Get the application configuration singleton.

    Creates and validates config on first call.
    Subsequent calls return the cached instance.

    Returns:
        AppConfig: The validated application configuration

    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = AppConfig()
        warnings = _config_instance.validate_startup()
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

    return _config_instance


def reset_config() -> None:
    """
    Reset the configuration singleton.

    Useful for testing or reloading configuration.
    """
    global _config_instance
    _config_instance = None


def get_search_config() -> SearchConfig:
    """Get search configuration (convenience function)."""
    return get_config().search


def get_server_config() -> ServerConfig:
    """Get server configuration (convenience function)."""
    return get_config().server


def get_metrics_config() -> MetricsConfig:
    """Get metrics configuration (convenience function)."""
    return get_config().metrics


def get_rate_limit_config() -> RateLimitConfig:
    """Get rate limit configuration (convenience function)."""
    return get_config().rate_limit


def get_logging_config() -> LoggingConfig:
    """Get logging configuration (convenience function)."""
    return get_config().logging


def get_shutdown_config() -> ShutdownConfig:
    """Get shutdown configuration (convenience function)."""
    return get_config().shutdown


# Backwards compatibility - these are deprecated but maintained for existing code
# TODO: Remove in v1.0
@lru_cache(maxsize=1)
def _create_legacy_search_config() -> SearchConfig:
    """Legacy factory for SearchConfig.from_env() compatibility."""
    return get_search_config()


@lru_cache(maxsize=1)
def _create_legacy_server_config() -> ServerConfig:
    """Legacy factory for ServerConfig.from_env() compatibility."""
    return get_server_config()


# Add from_env() methods for backwards compatibility
SearchConfig.from_env = staticmethod(lambda: _create_legacy_search_config())
ServerConfig.from_env = staticmethod(lambda: _create_legacy_server_config())
