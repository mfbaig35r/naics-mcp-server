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

from .config import SearchConfig, ServerConfig
from .core.database import NAICSDatabase
from .core.search_engine import NAICSSearchEngine
from .core.embeddings import TextEmbedder, EmbeddingCache
from .core.classification_workbook import ClassificationWorkbook, FormType
from .models.naics_models import NAICSCode, NAICSLevel, CrossReference, IndexTerm
from .models.search_models import (
    SearchStrategy,
    NAICSMatch,
    SearchResults,
    ConfidenceScore
)

__all__ = [
    # Version
    "__version__",
    # Config
    "SearchConfig",
    "ServerConfig",
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
