"""
Configuration for the NAICS MCP Server.

Every setting has a clear purpose and sensible default.
These aren't magic numbers—they're deliberate choices.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    Configuration for the NAICS search engine.

    Each setting is documented with its purpose and rationale.
    """

    # Database configuration
    database_path: Path = None  # Will be set dynamically
    connection_pool_size: int = 5  # Balance resources and concurrency
    query_timeout_seconds: int = 5  # Fail fast if something's wrong

    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, accurate enough for our needs
    embedding_dimension: int = 384  # Model-specific, for all-MiniLM-L6-v2
    batch_size: int = 32  # Process embeddings in batches for efficiency
    normalize_embeddings: bool = True  # Normalized vectors for faster cosine similarity

    # Search behavior - NAICS-specific hybrid search weights
    hybrid_weight_semantic: float = 0.7  # Favor meaning over exact match
    hybrid_weight_lexical: float = 0.3   # Still value exact matches
    boost_index_terms: float = 1.5  # Boost for official NAICS index terms

    # Performance tuning - NAICS is smaller than HTS
    max_candidates: int = 500  # NAICS has ~1,012 6-digit codes
    min_confidence: float = 0.3  # Below this, results aren't useful
    default_limit: int = 10  # Reasonable default for most queries

    # User experience
    explain_results: bool = True  # Help users understand why results were returned
    include_hierarchy: bool = True  # Show the full classification path

    # Query expansion
    enable_query_expansion: bool = True  # Use synonyms and related terms
    max_expansion_terms: int = 5  # Limit expansion to maintain precision

    # Cross-reference integration (NAICS-specific)
    enable_cross_references: bool = True  # Surface exclusions/inclusions
    cross_ref_penalty: float = 0.7  # Penalty multiplier for excluded activities

    # Observability
    enable_audit_log: bool = True  # Track searches for improvement
    audit_retention_days: int = 90  # Keep audit logs for analysis
    log_slow_queries: bool = True  # Log queries taking > slow_query_threshold
    slow_query_threshold_ms: int = 200  # What we consider "slow" for NAICS

    def __post_init__(self):
        """Set up database path after initialization."""
        if self.database_path is None:
            # Check environment variable first
            if env_path := os.getenv("NAICS_DATABASE_PATH"):
                self.database_path = Path(env_path)
                logger.info(f"Using database from environment: {self.database_path}")
            else:
                # Try to use bundled database
                try:
                    import naics_mcp_server
                    package_dir = Path(naics_mcp_server.__file__).parent
                    bundled_path = package_dir / "data" / "naics.duckdb"

                    if bundled_path.exists():
                        self.database_path = bundled_path
                        logger.info(f"Using bundled database: {self.database_path}")
                    else:
                        # Fall back to local data directory
                        local_path = Path("data/naics.duckdb")
                        if local_path.exists():
                            self.database_path = local_path
                            logger.info(f"Using local database: {self.database_path}")
                        else:
                            # Default path for new installations
                            self.database_path = package_dir / "data" / "naics.duckdb"
                            logger.info(f"Database will be created at: {self.database_path}")
                except ImportError:
                    # Package not properly installed, try local
                    self.database_path = Path("data/naics.duckdb")

    @classmethod
    def from_env(cls) -> "SearchConfig":
        """
        Create configuration from environment variables.

        This allows for easy customization without code changes.
        """
        from dotenv import load_dotenv

        load_dotenv()

        config = cls()

        # Override from environment if present
        if db_path := os.getenv("NAICS_DATABASE_PATH"):
            config.database_path = Path(db_path)

        if model := os.getenv("NAICS_EMBEDDING_MODEL"):
            config.embedding_model = model

        if semantic_weight := os.getenv("NAICS_SEMANTIC_WEIGHT"):
            config.hybrid_weight_semantic = float(semantic_weight)
            config.hybrid_weight_lexical = 1.0 - config.hybrid_weight_semantic

        if min_conf := os.getenv("NAICS_MIN_CONFIDENCE"):
            config.min_confidence = float(min_conf)

        if boost := os.getenv("NAICS_BOOST_INDEX_TERMS"):
            config.boost_index_terms = float(boost)

        if os.getenv("NAICS_ENABLE_AUDIT", "true").lower() == "false":
            config.enable_audit_log = False

        return config


@dataclass
class ServerConfig:
    """
    Configuration for the MCP server itself.
    """

    name: str = "NAICS Classification Assistant"
    description: str = (
        "An intelligent industry classification service for NAICS 2022. "
        "I help you find the right NAICS codes using natural language descriptions, "
        "with support for hierarchical navigation and cross-reference lookup."
    )
    version: str = "0.1.0"

    # MCP server settings
    stateless_http: bool = False  # Maintain session state for context
    enable_cors: bool = True  # Allow browser-based clients
    debug: bool = False  # Set via environment

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create server configuration from environment."""
        config = cls()
        config.debug = os.getenv("DEBUG", "false").lower() == "true"

        return config
