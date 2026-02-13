"""
NAICS MCP Server

An intelligent industry classification service for NAICS 2022,
providing semantic search, hierarchical navigation, and cross-reference lookup.

Usage:
    python -m naics_mcp_server              # Run the MCP server
    python -m naics_mcp_server search ...   # CLI search
    python -m naics_mcp_server stats        # Database statistics
"""

__version__ = "0.1.0"
__author__ = "NAICS MCP Team"

from .config import (
    AppConfig,
    SearchConfig,
    ServerConfig,
    get_config,
    get_search_config,
    get_server_config,
    reset_config,
)
from .core.classification_workbook import ClassificationWorkbook, FormType
from .core.database import NAICSDatabase
from .core.embeddings import EmbeddingCache, TextEmbedder
from .core.search_engine import NAICSSearchEngine
from .models.naics_models import CrossReference, IndexTerm, NAICSCode, NAICSLevel
from .models.search_models import ConfidenceScore, NAICSMatch, SearchResults, SearchStrategy

__all__ = [
    # Version
    "__version__",
    # Config
    "SearchConfig",
    "ServerConfig",
    "AppConfig",
    "get_config",
    "get_search_config",
    "get_server_config",
    "reset_config",
    # Core
    "NAICSDatabase",
    "NAICSSearchEngine",
    "TextEmbedder",
    "EmbeddingCache",
    "ClassificationWorkbook",
    "FormType",
    # Models
    "NAICSCode",
    "NAICSLevel",
    "CrossReference",
    "IndexTerm",
    "SearchStrategy",
    "NAICSMatch",
    "SearchResults",
    "ConfidenceScore",
]
