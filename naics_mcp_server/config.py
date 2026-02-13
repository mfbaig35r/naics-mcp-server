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
"""

from pathlib import Path
from typing import Optional, Any
from functools import lru_cache
import logging
import os

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
    database_path: Optional[Path] = Field(
        default=None,
        description="Path to DuckDB database file"
    )
    connection_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of database connections to pool"
    )
    query_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Query timeout in seconds"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        min_length=1,
        description="Sentence transformer model name"
    )
    embedding_dimension: int = Field(
        default=384,
        ge=32,
        le=4096,
        description="Embedding vector dimension (must match model)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding generation"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Normalize embeddings for cosine similarity"
    )

    # Search behavior - hybrid search weights
    hybrid_weight_semantic: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search (0.0-1.0)"
    )
    hybrid_weight_lexical: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for lexical search (0.0-1.0)"
    )
    boost_index_terms: float = Field(
        default=1.5,
        ge=1.0,
        le=10.0,
        description="Boost factor for official NAICS index terms"
    )

    # Performance tuning
    max_candidates: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum candidates to consider in search"
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for results"
    )
    default_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results to return"
    )

    # User experience
    explain_results: bool = Field(
        default=True,
        description="Include explanations in search results"
    )
    include_hierarchy: bool = Field(
        default=True,
        description="Include hierarchy information in results"
    )

    # Query expansion
    enable_query_expansion: bool = Field(
        default=True,
        description="Enable synonym and term expansion"
    )
    max_expansion_terms: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum terms to add via expansion"
    )

    # Cross-reference integration
    enable_cross_references: bool = Field(
        default=True,
        description="Enable cross-reference lookup"
    )
    cross_ref_penalty: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Score penalty for excluded activities"
    )

    # Observability
    enable_audit_log: bool = Field(
        default=True,
        description="Enable search audit logging"
    )
    audit_retention_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days to retain audit logs"
    )
    log_slow_queries: bool = Field(
        default=True,
        description="Log queries exceeding threshold"
    )
    slow_query_threshold_ms: int = Field(
        default=200,
        ge=10,
        le=10000,
        description="Slow query threshold in milliseconds"
    )

    @field_validator("database_path", mode="before")
    @classmethod
    def resolve_database_path(cls, v: Any) -> Optional[Path]:
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
                f"Hybrid weights sum to {total:.2f}, expected 1.0. "
                "Adjusting lexical weight."
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
                package_dir / "data" / "naics.duckdb",   # Bundled
                Path.cwd() / "data" / "naics.duckdb",    # Current directory
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
        default="NAICS Classification Assistant",
        min_length=1,
        description="Server display name"
    )
    description: str = Field(
        default=(
            "An intelligent industry classification service for NAICS 2022. "
            "I help you find the right NAICS codes using natural language descriptions, "
            "with support for hierarchical navigation and cross-reference lookup."
        ),
        description="Server description"
    )
    version: str = Field(
        default="0.1.0",
        pattern=r"^\d+\.\d+\.\d+",
        description="Server version (semver)"
    )

    # MCP server settings
    stateless_http: bool = Field(
        default=False,
        description="Run in stateless HTTP mode"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS for browser clients"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

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
                "Database path not configured",
                config_key="NAICS_DATABASE_PATH"
            )

        # Check if database directory exists (for new databases)
        db_dir = self.search.database_path.parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created database directory: {db_dir}")
            except OSError as e:
                raise ConfigurationError(
                    f"Cannot create database directory: {e}",
                    config_key="NAICS_DATABASE_PATH"
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
        }


# Singleton instance management
_config_instance: Optional[AppConfig] = None


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
