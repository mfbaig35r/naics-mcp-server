"""
Core modules for NAICS MCP Server.
"""

from .database import NAICSDatabase, get_database
from .embeddings import TextEmbedder, EmbeddingCache
from .search_engine import NAICSSearchEngine, generate_search_guidance
from .query_expansion import QueryExpander, SmartQueryParser
from .cross_reference import CrossReferenceParser, CrossReferenceService
from .classification_workbook import ClassificationWorkbook, FormType, WorkbookEntry
from .validation import (
    ValidationConfig,
    ValidationResult,
    validate_description,
    validate_naics_code,
    validate_naics_code_exists,
    validate_search_query,
    validate_limit,
    validate_confidence,
    validate_batch_descriptions,
    validate_batch_codes,
    validate_strategy,
    normalize_text,
)
from .errors import (
    NAICSException,
    DatabaseError,
    ConnectionError,
    QueryError,
    ValidationError,
    NotFoundError,
    ConfigurationError,
    EmbeddingError,
    TimeoutError,
    SearchError,
    ErrorCategory,
    RetryConfig,
    ErrorResponse,
    retry_sync,
    retry_async,
    with_fallback,
    handle_tool_error,
    DATABASE_RETRY,
    EMBEDDING_RETRY,
)

__all__ = [
    # Database
    "NAICSDatabase",
    "get_database",
    # Embeddings
    "TextEmbedder",
    "EmbeddingCache",
    # Search
    "NAICSSearchEngine",
    "generate_search_guidance",
    # Query
    "QueryExpander",
    "SmartQueryParser",
    # Cross-reference
    "CrossReferenceParser",
    "CrossReferenceService",
    # Workbook
    "ClassificationWorkbook",
    "FormType",
    "WorkbookEntry",
    # Validation
    "ValidationConfig",
    "ValidationResult",
    "validate_description",
    "validate_naics_code",
    "validate_naics_code_exists",
    "validate_search_query",
    "validate_limit",
    "validate_confidence",
    "validate_batch_descriptions",
    "validate_batch_codes",
    "validate_strategy",
    "normalize_text",
    # Errors
    "NAICSException",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "ValidationError",
    "NotFoundError",
    "ConfigurationError",
    "EmbeddingError",
    "TimeoutError",
    "SearchError",
    "ErrorCategory",
    "RetryConfig",
    "ErrorResponse",
    "retry_sync",
    "retry_async",
    "with_fallback",
    "handle_tool_error",
    "DATABASE_RETRY",
    "EMBEDDING_RETRY",
]
