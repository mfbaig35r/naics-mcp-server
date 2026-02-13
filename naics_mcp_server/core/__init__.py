"""
Core modules for NAICS MCP Server.
"""

from .classification_workbook import ClassificationWorkbook, FormType, WorkbookEntry
from .cross_reference import CrossReferenceParser, CrossReferenceService
from .database import NAICSDatabase, get_database
from .embeddings import EmbeddingCache, TextEmbedder
from .errors import (
    DATABASE_RETRY,
    EMBEDDING_RETRY,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    EmbeddingError,
    ErrorCategory,
    ErrorResponse,
    NAICSException,
    NotFoundError,
    QueryError,
    RetryConfig,
    SearchError,
    TimeoutError,
    ValidationError,
    handle_tool_error,
    retry_async,
    retry_sync,
    with_fallback,
)
from .health import (
    ComponentHealth,
    ComponentStatus,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    liveness_check,
    readiness_check,
)
from .query_expansion import QueryExpander, SmartQueryParser
from .shutdown import (
    RequestTracker,
    ShutdownConfig,
    ShutdownManager,
    ShutdownResult,
    ShutdownState,
    create_shutdown_manager,
    get_shutdown_manager,
    reset_shutdown_manager,
)
from .search_engine import NAICSSearchEngine, generate_search_guidance
from .validation import (
    ValidationConfig,
    ValidationResult,
    normalize_text,
    validate_batch_codes,
    validate_batch_descriptions,
    validate_confidence,
    validate_description,
    validate_limit,
    validate_naics_code,
    validate_naics_code_exists,
    validate_search_query,
    validate_strategy,
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
    # Health
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "ComponentStatus",
    "ComponentHealth",
    "liveness_check",
    "readiness_check",
    # Shutdown
    "ShutdownManager",
    "ShutdownConfig",
    "ShutdownState",
    "ShutdownResult",
    "RequestTracker",
    "get_shutdown_manager",
    "create_shutdown_manager",
    "reset_shutdown_manager",
]
