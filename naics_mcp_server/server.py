#!/usr/bin/env python3
"""
NAICS MCP Server

An intelligent industry classification service for NAICS 2022,
built with clarity and purpose.
"""

import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from .config import SearchConfig, ServerConfig
from .core.database import NAICSDatabase
from .core.embeddings import TextEmbedder
from .core.search_engine import NAICSSearchEngine, generate_search_guidance
from .core.classification_workbook import ClassificationWorkbook, FormType
from .core.cross_reference import CrossReferenceService
from .models.search_models import SearchStrategy
from .observability.audit import SearchAuditLog, SearchEvent
from .observability.logging import (
    setup_logging,
    get_logger,
    set_request_context,
    generate_request_id,
    sanitize_text,
    log_server_start,
    log_server_ready,
    log_server_shutdown,
)
from .core.errors import (
    NAICSException, DatabaseError, NotFoundError, ValidationError,
    SearchError, handle_tool_error
)
from .core.health import HealthChecker, HealthStatus
from .core.validation import (
    validate_description,
    validate_naics_code,
    validate_search_query,
    validate_limit,
    validate_confidence,
    validate_batch_descriptions,
    validate_batch_codes,
    validate_strategy,
    ValidationConfig,
)
from .tools.workbook_tools import (
    WorkbookWriteRequest, WorkbookSearchRequest, WorkbookTemplateRequest
)

# Configure structured logging
setup_logging(
    level=os.getenv("NAICS_LOG_LEVEL", "INFO"),
    format=os.getenv("NAICS_LOG_FORMAT", "text"),  # Use "json" for production
    log_file=os.getenv("NAICS_LOG_FILE")
)
logger = get_logger(__name__)


# Application context for dependency injection
class AppContext:
    """Application context with all initialized services."""

    def __init__(
        self,
        database: NAICSDatabase,
        embedder: TextEmbedder,
        search_engine: NAICSSearchEngine,
        audit_log: SearchAuditLog,
        cross_ref_service: CrossReferenceService,
        classification_workbook: ClassificationWorkbook,
        health_checker: HealthChecker
    ):
        self.database = database
        self.embedder = embedder
        self.search_engine = search_engine
        self.audit_log = audit_log
        self.cross_ref_service = cross_ref_service
        self.classification_workbook = classification_workbook
        self.health_checker = health_checker


# Request/Response models
class SearchRequest(BaseModel):
    """Parameters for NAICS code search."""

    query: str = Field(description="Natural language description of the business or activity")
    strategy: str = Field(
        default="hybrid",
        description="Search strategy: 'hybrid' (best match), 'semantic' (meaning), or 'lexical' (exact)"
    )
    limit: int = Field(default=10, description="Maximum results to return", ge=1, le=50)
    min_confidence: float = Field(
        default=0.3,
        description="Minimum confidence threshold (0-1)",
        ge=0.0,
        le=1.0
    )
    include_cross_refs: bool = Field(
        default=True,
        description="Include cross-reference checks for exclusions"
    )


class NAICSResult(BaseModel):
    """A single NAICS search result."""

    code: str
    title: str
    description: Optional[str] = None
    level: str
    confidence: float
    explanation: Optional[str] = None
    hierarchy: List[str]
    matched_index_terms: List[str] = []
    exclusion_warnings: List[str] = []


class SearchResponse(BaseModel):
    """Response from NAICS search."""

    query: str
    results: List[NAICSResult]
    expanded: bool
    strategy_used: str
    total_found: int
    search_time_ms: int
    guidance: List[str] = Field(default_factory=list)


class SimilarityRequest(BaseModel):
    """Parameters for finding similar NAICS codes."""

    naics_code: str = Field(description="NAICS code to find similar codes for")
    limit: int = Field(default=5, description="Maximum results", ge=1, le=20)
    min_similarity: float = Field(default=0.7, description="Minimum similarity", ge=0.0, le=1.0)


class ClassifyRequest(BaseModel):
    """Parameters for classifying a business."""

    description: str = Field(description="Business or activity description")
    include_reasoning: bool = Field(default=True, description="Include detailed reasoning")
    check_cross_refs: bool = Field(default=True, description="Check cross-references for exclusions")


class BatchClassifyRequest(BaseModel):
    """Parameters for batch classification."""

    descriptions: List[str] = Field(description="List of business descriptions to classify")
    include_confidence: bool = Field(default=True, description="Include confidence scores")


# Initialize the MCP server
@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage server lifecycle - initialization and cleanup."""
    # Load configuration
    search_config = SearchConfig.from_env()
    server_config = ServerConfig.from_env()

    # Log startup with config
    log_server_start({
        "database_path": str(search_config.database_path),
        "embedding_model": search_config.embedding_model,
        "debug": server_config.debug
    })

    # Initialize database
    database = NAICSDatabase(search_config.database_path)
    database.connect()
    logger.info("Database connected", data={"path": str(search_config.database_path)})

    # Initialize embedder
    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(
        model_name=search_config.embedding_model,
        cache_dir=cache_dir
    )
    embedder.load_model()
    logger.info("Embedding model loaded", data={"model": search_config.embedding_model})

    # Initialize search engine
    search_engine = NAICSSearchEngine(database, embedder, search_config)

    # Initialize embeddings if needed
    init_result = await search_engine.initialize_embeddings()
    logger.info("Embeddings initialized", data={
        "action": init_result.get("action"),
        "count": init_result.get("embeddings_count") or init_result.get("embeddings_generated"),
        "time_seconds": init_result.get("time_seconds")
    })

    # Initialize cross-reference service
    cross_ref_service = CrossReferenceService(database)

    # Initialize classification workbook
    classification_workbook = ClassificationWorkbook(database)
    logger.info("Classification workbook initialized")

    # Initialize audit log
    log_dir = Path.home() / ".cache" / "naics-mcp-server" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    audit_log = SearchAuditLog(
        log_dir=log_dir,
        retention_days=search_config.audit_retention_days,
        enable_file_logging=search_config.enable_audit_log
    )

    # Create health checker
    health_checker = HealthChecker(
        database=database,
        embedder=embedder,
        search_engine=search_engine,
        version=server_config.version
    )

    # Create application context
    app_context = AppContext(
        database, embedder, search_engine, audit_log,
        cross_ref_service, classification_workbook, health_checker
    )

    # Log server ready with stats
    stats = await database.get_statistics()
    stats["embeddings_count"] = init_result.get("embeddings_count") or init_result.get("embeddings_generated", 0)
    log_server_ready(stats)

    try:
        yield app_context
    finally:
        log_server_shutdown()
        database.disconnect()
        logger.info("Shutdown complete")


# Server instructions for LLM orchestration
SERVER_INSTRUCTIONS = """
# NAICS Classification Assistant - Workflow Guide

## Quick Classification (Simple Business)
For straightforward single-activity businesses:
1. `classify_business` with a clear description
2. If confidence > 60%, you're likely done
3. Optionally verify with `search_index_terms` for exact terminology

## Full Classification (Complex/Multi-Segment Business)
For companies with multiple business lines or ambiguous activities:
1. Research the company's segments and primary revenue sources
2. `classify_business` for initial assessment
3. `search_naics_codes` for each major segment/activity
4. `search_index_terms` for specific products/services
5. `compare_codes` for top candidates side-by-side
6. `get_cross_references` to check for exclusions (CRITICAL)
7. `get_siblings` or `find_similar_industries` for alternatives
8. `write_to_workbook` with form_type="classification_analysis" to document

## Boundary Cases (Activity Could Fit Multiple Codes)
When a business sits between two codes:
1. Search from multiple angles (product vs service vs customer)
2. `get_cross_references` on both candidates - exclusions are authoritative
3. `compare_codes` to see descriptions and index terms side-by-side
4. `get_code_hierarchy` to understand where each code sits conceptually
5. Document with `write_to_workbook` form_type="decision_tree"

## Batch Classification
For processing many businesses:
1. `classify_batch` for efficiency
2. Triage by confidence: >50% accept, 35-50% spot-check, <35% manual review
3. Use full classification workflow for low-confidence items

## Exploring the NAICS Hierarchy
1. `get_sector_overview` to see all 20 sectors
2. `get_children` to drill down into subsectors
3. `get_siblings` to see what else is at the same level
4. `find_similar_industries` for semantically related codes

## Key Principles
- PRIMARY ACTIVITY determines classification (largest revenue source)
- 6-digit codes are most specific and preferred
- Cross-references tell you what activities are EXCLUDED - always check
- Index term matches are strong evidence of correct classification
- When uncertain, document reasoning with the workbook tools
"""

# Create the MCP server
mcp = FastMCP(
    name="NAICS Classification Assistant",
    instructions=SERVER_INSTRUCTIONS,
    lifespan=lifespan
)

mcp.description = (
    "An intelligent industry classification service for NAICS 2022. "
    "I help you find the right NAICS codes using natural language descriptions, "
    "with support for hierarchical navigation and cross-reference lookup."
)


# === Search Tools (4) ===

@mcp.tool()
async def search_naics_codes(
    request: SearchRequest,
    ctx: Context
) -> SearchResponse:
    """
    Search for NAICS codes using natural language.

    NAICS codes classify businesses by industry using a 6-digit hierarchical system:
    Sector (2) → Subsector (3) → Industry Group (4) → NAICS Industry (5) → National Industry (6)

    SEARCH STRATEGIES:
    • hybrid (default): Balances semantic meaning with exact term matching
    • semantic: Focuses on conceptual similarity and related meanings
    • lexical: Prioritizes exact term matching

    OPTIMIZING YOUR SEARCH:
    Include relevant details about:
    - Primary business activity
    - Products or services offered
    - Customer type (business vs consumer)
    - Production process or method

    UNDERSTANDING RESULTS:
    - Confidence scores indicate relative match strength
    - Results include full hierarchy from sector to specific code
    - Cross-reference warnings indicate the activity may belong elsewhere
    - Index term matches show alignment with official NAICS terminology
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Validate inputs
    try:
        query_result = validate_search_query(request.query)
        validated_query = query_result.value

        strategy_result = validate_strategy(request.strategy)
        validated_strategy = strategy_result.value

        limit_result = validate_limit(request.limit)
        validated_limit = limit_result.value

        confidence_result = validate_confidence(request.min_confidence)
        validated_confidence = confidence_result.value
    except ValidationError as e:
        logger.warning("Search validation failed", data={
            "error": e.message,
            "field": e.details.get("field")
        })
        return SearchResponse(
            query=request.query,
            results=[],
            expanded=False,
            strategy_used=request.strategy,
            total_found=0,
            search_time_ms=0,
            guidance=[f"Validation error: {e.message}"]
        )

    # Start audit event
    search_event = SearchEvent.start(validated_query, validated_strategy)

    try:
        # Map strategy
        strategy_map = {
            "hybrid": SearchStrategy.HYBRID,
            "best_match": SearchStrategy.HYBRID,
            "semantic": SearchStrategy.SEMANTIC,
            "meaning": SearchStrategy.SEMANTIC,
            "lexical": SearchStrategy.LEXICAL,
            "exact": SearchStrategy.LEXICAL
        }
        strategy = strategy_map.get(validated_strategy, SearchStrategy.HYBRID)

        # Perform search
        results = await app_ctx.search_engine.search(
            query=validated_query,
            strategy=strategy,
            limit=validated_limit,
            min_confidence=validated_confidence,
            include_cross_refs=request.include_cross_refs
        )

        # Log search completion
        search_event.complete(results)
        await app_ctx.audit_log.log_search(search_event)

        # Generate guidance
        guidance = generate_search_guidance(results)

        # Build response
        return SearchResponse(
            query=request.query,
            results=[
                NAICSResult(
                    code=match.code.node_code,
                    title=match.code.title,
                    description=match.code.description,
                    level=match.code.level.value,
                    confidence=match.confidence.overall,
                    explanation=match.confidence.to_explanation(),
                    hierarchy=match.hierarchy_path,
                    matched_index_terms=match.matched_index_terms,
                    exclusion_warnings=match.exclusion_warnings
                )
                for match in results.matches
            ],
            expanded=results.query_metadata.was_expanded,
            strategy_used=strategy.value,
            total_found=len(results.matches),
            search_time_ms=results.query_metadata.processing_time_ms,
            guidance=guidance
        )

    except Exception as e:
        search_event.fail(str(e))
        await app_ctx.audit_log.log_search(search_event)

        logger.error("Search failed", data={
            "error_type": type(e).__name__,
            "error_message": str(e)[:200],
            "query_preview": sanitize_text(request.query, 50)
        })
        return SearchResponse(
            query=request.query,
            results=[],
            expanded=False,
            strategy_used=request.strategy,
            total_found=0,
            search_time_ms=search_event.duration_ms
        )


@mcp.tool()
async def search_index_terms(
    search_text: str,
    limit: int = 20,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search the official NAICS index terms.

    NAICS has 20,398 official index terms that map specific activities
    to NAICS codes. This is useful for finding the exact terminology
    used by the Census Bureau.

    Examples:
    - "dog grooming" → 812910
    - "software publishing" → 511210
    - "soybean farming" → 111110
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        terms = await app_ctx.database.search_index_terms(search_text, limit=limit)

        return {
            "search_text": search_text,
            "matches": [
                {
                    "index_term": t.index_term,
                    "naics_code": t.naics_code
                }
                for t in terms
            ],
            "total_found": len(terms)
        }

    except NAICSException as e:
        logger.error(f"Index term search failed: {e}")
        result = handle_tool_error(e, "search_index_terms")
        result["matches"] = []
        return result
    except Exception as e:
        logger.error(f"Index term search failed: {e}")
        result = handle_tool_error(e, "search_index_terms")
        result["matches"] = []
        return result


@mcp.tool()
async def find_similar_industries(
    request: SimilarityRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Find NAICS codes similar to a given code.

    This helps explore related classifications and alternatives.
    Uses semantic similarity based on code descriptions.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        code = await app_ctx.database.get_by_code(request.naics_code)

        if not code:
            return {
                "error": f"NAICS code {request.naics_code} not found",
                "similar_codes": []
            }

        # Use the code's description for similarity search
        search_text = code.raw_embedding_text or f"{code.title} {code.description or ''}"

        results = await app_ctx.search_engine.search(
            query=search_text,
            strategy=SearchStrategy.SEMANTIC,
            limit=request.limit + 1,
            min_confidence=request.min_similarity
        )

        # Filter out the original code
        similar = [
            {
                "code": match.code.node_code,
                "title": match.code.title,
                "similarity": match.confidence.semantic,
                "level": match.code.level.value
            }
            for match in results.matches
            if match.code.node_code != request.naics_code
        ][:request.limit]

        return {
            "original_code": request.naics_code,
            "original_title": code.title,
            "similar_codes": similar
        }

    except NAICSException as e:
        logger.error(f"Similarity search failed: {e}")
        result = handle_tool_error(e, "find_similar_industries")
        result["similar_codes"] = []
        return result
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        result = handle_tool_error(e, "find_similar_industries")
        result["similar_codes"] = []
        return result


@mcp.tool()
async def classify_batch(
    request: BatchClassifyRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Classify multiple business descriptions in batch.

    Useful for processing lists of businesses efficiently.
    Returns the best matching NAICS code for each description.

    Constraints:
    - Maximum 100 descriptions per batch
    - Each description must be 10-5000 characters
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Validate batch
    try:
        batch_result = validate_batch_descriptions(request.descriptions)
        validated_descriptions = batch_result.value
    except ValidationError as e:
        logger.warning("Batch validation failed", data={
            "error": e.message,
            "field": e.details.get("field")
        })
        return {
            "error": e.message,
            "error_category": "validation",
            "classifications": [],
            "total_processed": 0,
            "successfully_classified": 0
        }

    classifications = []

    for description in validated_descriptions:
        try:
            results = await app_ctx.search_engine.search(
                query=description,
                strategy=SearchStrategy.HYBRID,
                limit=1,
                min_confidence=0.3
            )

            if results.matches:
                best_match = results.matches[0]
                classification = {
                    "description": description,
                    "naics_code": best_match.code.node_code,
                    "naics_title": best_match.code.title,
                    "level": best_match.code.level.value
                }

                if request.include_confidence:
                    classification["confidence"] = best_match.confidence.overall
                    classification["explanation"] = best_match.confidence.to_explanation()

                classifications.append(classification)
            else:
                classifications.append({
                    "description": description,
                    "naics_code": None,
                    "naics_title": "No suitable classification found",
                    "confidence": 0.0 if request.include_confidence else None
                })

        except Exception as e:
            logger.error(f"Failed to classify '{description}': {e}")
            classifications.append({
                "description": description,
                "error": str(e)
            })

    return {
        "classifications": classifications,
        "total_processed": len(request.descriptions),
        "successfully_classified": sum(
            1 for c in classifications
            if c.get("naics_code") and "error" not in c
        )
    }


# === Hierarchy Tools (3) ===

@mcp.tool()
async def get_code_hierarchy(
    naics_code: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Get the complete hierarchical path for a NAICS code.

    Shows how a code fits into the classification system from
    Sector (2-digit) down to National Industry (6-digit).
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        hierarchy = await app_ctx.database.get_hierarchy(naics_code)

        if not hierarchy:
            return {
                "error": f"NAICS code {naics_code} not found",
                "hierarchy": []
            }

        hierarchy_list = [
            {
                "level": code.level.value,
                "code": code.node_code,
                "title": code.title,
                "description": code.description[:200] + "..." if code.description and len(code.description) > 200 else code.description
            }
            for code in hierarchy
        ]

        return {
            "naics_code": naics_code,
            "hierarchy": hierarchy_list
        }

    except NAICSException as e:
        logger.error(f"Failed to get hierarchy: {e}")
        result = handle_tool_error(e, "get_code_hierarchy")
        result["hierarchy"] = []
        return result
    except Exception as e:
        logger.error(f"Failed to get hierarchy: {e}")
        result = handle_tool_error(e, "get_code_hierarchy")
        result["hierarchy"] = []
        return result


@mcp.tool()
async def get_children(
    naics_code: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Get immediate children of a NAICS code.

    Shows the next level of detail in the classification hierarchy.
    For example, children of sector "31" would be its subsectors.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        children = await app_ctx.database.get_children(naics_code)

        return {
            "parent_code": naics_code,
            "children": [
                {
                    "code": child.node_code,
                    "title": child.title,
                    "level": child.level.value
                }
                for child in children
            ],
            "count": len(children)
        }

    except NAICSException as e:
        logger.error(f"Failed to get children: {e}")
        result = handle_tool_error(e, "get_children")
        result["children"] = []
        return result
    except Exception as e:
        logger.error(f"Failed to get children: {e}")
        result = handle_tool_error(e, "get_children")
        result["children"] = []
        return result


@mcp.tool()
async def get_siblings(
    naics_code: str,
    limit: int = 10,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get sibling codes at the same hierarchical level.

    Shows alternative codes that share the same parent.
    Useful for exploring related industries.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        code = await app_ctx.database.get_by_code(naics_code)
        if not code:
            return {
                "error": f"NAICS code {naics_code} not found",
                "siblings": []
            }

        siblings = await app_ctx.database.get_siblings(naics_code, limit=limit)

        return {
            "code": naics_code,
            "title": code.title,
            "level": code.level.value,
            "siblings": [
                {
                    "code": sib.node_code,
                    "title": sib.title
                }
                for sib in siblings
            ]
        }

    except NAICSException as e:
        logger.error(f"Failed to get siblings: {e}")
        result = handle_tool_error(e, "get_siblings")
        result["siblings"] = []
        return result
    except Exception as e:
        logger.error(f"Failed to get siblings: {e}")
        result = handle_tool_error(e, "get_siblings")
        result["siblings"] = []
        return result


# === Classification Tools (3) ===

@mcp.tool()
async def classify_business(
    request: ClassifyRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Classify a business description to NAICS with detailed reasoning.

    This performs a thorough classification analysis including:
    - Search across all strategies
    - Index term matching
    - Cross-reference checking
    - Confidence breakdown

    Returns the recommended classification with alternatives and reasoning.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Set request context for logging
    request_id = generate_request_id()
    set_request_context(request_id=request_id, tool_name="classify_business")

    # Validate input
    try:
        desc_result = validate_description(request.description)
        validated_description = desc_result.value
    except ValidationError as e:
        logger.warning("Classification validation failed", data={
            "error": e.message,
            "field": e.details.get("field")
        })
        return {
            "input": request.description[:100] if request.description else None,
            "error": e.message,
            "error_category": "validation"
        }

    logger.info("Classification requested", data={
        "description_length": len(validated_description),
        "description_preview": sanitize_text(validated_description, 50),
        "check_cross_refs": request.check_cross_refs
    })

    try:
        # Perform comprehensive search
        results = await app_ctx.search_engine.search(
            query=validated_description,
            strategy=SearchStrategy.HYBRID,
            limit=5,
            min_confidence=0.2,
            include_cross_refs=request.check_cross_refs
        )

        if not results.matches:
            return {
                "input": request.description,
                "classification": None,
                "reasoning": "No suitable NAICS codes found for this description. Consider providing more specific details about the business activity.",
                "alternatives": []
            }

        primary = results.matches[0]
        alternatives = results.matches[1:4]

        response = {
            "input": request.description,
            "classification": {
                "code": primary.code.node_code,
                "title": primary.code.title,
                "level": primary.code.level.value,
                "confidence": primary.confidence.overall,
                "confidence_breakdown": {
                    "semantic": primary.confidence.semantic,
                    "lexical": primary.confidence.lexical,
                    "index_term": primary.confidence.index_term,
                    "specificity": primary.confidence.specificity,
                    "cross_ref": primary.confidence.cross_ref
                },
                "hierarchy": primary.hierarchy_path,
                "matched_index_terms": primary.matched_index_terms
            },
            "alternatives": [
                {
                    "code": alt.code.node_code,
                    "title": alt.code.title,
                    "confidence": alt.confidence.overall,
                    "matched_index_terms": alt.matched_index_terms
                }
                for alt in alternatives
            ]
        }

        if request.include_reasoning:
            reasoning_parts = []

            # 1. Primary classification summary
            reasoning_parts.append(f"**Primary Classification:** {primary.code.node_code} - {primary.code.title}")
            reasoning_parts.append(f"**Overall Confidence:** {primary.confidence.overall:.1%}")
            reasoning_parts.append("")

            # 2. Key decision factor - identify what drove the match
            conf = primary.confidence
            decision_factors = []
            if conf.semantic > 0.7:
                decision_factors.append(f"semantic similarity ({conf.semantic:.0%})")
            if conf.lexical > 0.5:
                decision_factors.append(f"exact term matches ({conf.lexical:.0%})")
            if conf.index_term > 0.5:
                decision_factors.append(f"official index term match ({conf.index_term:.0%})")
            if conf.specificity > 0.7:
                decision_factors.append(f"most specific level ({conf.specificity:.0%})")

            if decision_factors:
                reasoning_parts.append(f"**Key Decision Factors:** {', '.join(decision_factors)}")
            else:
                reasoning_parts.append("**Key Decision Factors:** General context match")
            reasoning_parts.append("")

            # 3. Index term matches
            if primary.matched_index_terms:
                reasoning_parts.append(f"**Official Index Terms Matched:** {', '.join(primary.matched_index_terms[:5])}")
            else:
                reasoning_parts.append("**Official Index Terms Matched:** None (matched via description)")
            reasoning_parts.append("")

            # 4. Why chosen over alternatives
            if alternatives:
                reasoning_parts.append("**Why This Over Alternatives:**")
                for alt in alternatives[:3]:
                    delta = primary.confidence.overall - alt.confidence.overall
                    reasons = []
                    if primary.confidence.semantic > alt.confidence.semantic + 0.1:
                        reasons.append("better semantic fit")
                    if primary.confidence.index_term > alt.confidence.index_term:
                        reasons.append("stronger index term match")
                    if primary.confidence.specificity > alt.confidence.specificity:
                        reasons.append("more specific code")
                    if not reasons:
                        reasons.append("higher overall score")
                    reasoning_parts.append(
                        f"  - vs {alt.code.node_code} ({alt.code.title}): "
                        f"+{delta:.0%} confidence ({', '.join(reasons)})"
                    )
                reasoning_parts.append("")

            # 5. Cross-reference status
            if request.check_cross_refs:
                if primary.exclusion_warnings:
                    reasoning_parts.append("**Cross-References Checked:** Yes - WARNINGS FOUND")
                elif primary.relevant_cross_refs:
                    reasoning_parts.append(f"**Cross-References Checked:** Yes - {len(primary.relevant_cross_refs)} references reviewed, no conflicts")
                else:
                    reasoning_parts.append("**Cross-References Checked:** Yes - no applicable exclusions")
            else:
                reasoning_parts.append("**Cross-References Checked:** No (use check_cross_refs=true for full validation)")
            reasoning_parts.append("")

            # 6. Exclusion warnings (if any)
            if primary.exclusion_warnings:
                reasoning_parts.append("⚠️ **EXCLUSION WARNINGS:**")
                for warning in primary.exclusion_warnings:
                    reasoning_parts.append(f"  - {warning}")

            response["reasoning"] = "\n".join(reasoning_parts)

        if primary.exclusion_warnings:
            response["exclusion_warnings"] = primary.exclusion_warnings

        # Log successful classification
        logger.info("Classification completed", data={
            "primary_code": primary.code.node_code,
            "confidence": primary.confidence.overall,
            "alternatives_count": len(alternatives),
            "exclusion_warnings": len(primary.exclusion_warnings) if primary.exclusion_warnings else 0
        })

        return response

    except Exception as e:
        logger.error("Classification failed", data={
            "error_type": type(e).__name__,
            "error_message": str(e)[:200]
        })
        return {
            "input": request.description,
            "error": str(e)
        }


@mcp.tool()
async def get_cross_references(
    naics_code: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Get cross-references (exclusions/inclusions) for a NAICS code.

    Cross-references are CRITICAL for accurate classification.
    They tell you what activities are explicitly excluded from this code
    and where they should be classified instead.

    Example: Code 311111 (Dog Food) explicitly excludes "prepared feeds for
    cattle, hogs, poultry" which should be classified under 311119.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        code = await app_ctx.database.get_by_code(naics_code)
        if not code:
            return {
                "error": f"NAICS code {naics_code} not found",
                "cross_references": []
            }

        cross_refs = await app_ctx.database.get_cross_references(naics_code)

        return {
            "naics_code": naics_code,
            "title": code.title,
            "cross_references": [
                {
                    "type": cr.reference_type,
                    "excluded_activity": cr.excluded_activity,
                    "target_code": cr.target_code,
                    "reference_text": cr.reference_text
                }
                for cr in cross_refs
            ],
            "total": len(cross_refs)
        }

    except Exception as e:
        logger.error("Failed to get cross-references", data={
            "naics_code": naics_code,
            "error_type": type(e).__name__,
            "error_message": str(e)[:200]
        })
        return {
            "error": str(e),
            "cross_references": []
        }


@mcp.tool()
async def validate_classification(
    naics_code: str,
    business_description: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Validate if a NAICS code is correct for a business description.

    Use this to verify an existing classification or check if a code
    chosen by the user is appropriate. Returns:
    - Validation status (valid, questionable, invalid)
    - Confidence that this code matches the description
    - Cross-reference warnings (exclusions that may apply)
    - Alternative codes if the classification seems wrong

    Example use cases:
    - User says "I think I'm 541511" - validate if that's correct
    - Checking a classification before finalizing
    - Auditing existing NAICS assignments
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        # 1. Verify the code exists
        code_info = await app_ctx.database.get_by_code(naics_code)
        if not code_info:
            return {
                "naics_code": naics_code,
                "status": "invalid",
                "reason": f"NAICS code {naics_code} does not exist",
                "valid": False
            }

        # 2. Search for best matches for this description
        results = await app_ctx.search_engine.search(
            query=business_description,
            strategy=SearchStrategy.HYBRID,
            limit=5,
            min_confidence=0.2,
            include_cross_refs=True
        )

        # 3. Check if the provided code is in the top results
        provided_match = None
        provided_rank = None
        for i, match in enumerate(results.matches):
            if match.code.node_code == naics_code:
                provided_match = match
                provided_rank = i + 1
                break

        # 4. Get cross-references for the provided code
        cross_refs = await app_ctx.database.get_cross_references(naics_code)
        exclusion_warnings = []

        # Check if the description might match an exclusion
        desc_lower = business_description.lower()
        for cr in cross_refs:
            if cr.reference_type == "excludes" and cr.excluded_activity:
                activity_lower = cr.excluded_activity.lower()
                # Simple keyword overlap check
                activity_words = set(activity_lower.split())
                desc_words = set(desc_lower.split())
                overlap = activity_words & desc_words
                if len(overlap) >= 2 or any(word in desc_lower for word in activity_words if len(word) > 5):
                    exclusion_warnings.append({
                        "excluded_activity": cr.excluded_activity,
                        "should_be": cr.target_code,
                        "reference": cr.reference_text[:200]
                    })

        # 5. Determine validation status
        top_match = results.matches[0] if results.matches else None
        alternatives = []

        if provided_match:
            confidence = provided_match.confidence.overall

            if provided_rank == 1:
                if exclusion_warnings:
                    status = "questionable"
                    reason = f"Code matches well (rank #1, {confidence:.0%} confidence) but exclusion warnings apply"
                elif confidence >= 0.7:
                    status = "valid"
                    reason = f"Strong match - rank #1 with {confidence:.0%} confidence"
                else:
                    status = "valid"
                    reason = f"Best available match (rank #1) but moderate confidence ({confidence:.0%})"
            elif provided_rank <= 3:
                status = "questionable"
                reason = f"Acceptable match (rank #{provided_rank}, {confidence:.0%} confidence) but better alternatives exist"
                alternatives = [
                    {
                        "code": m.code.node_code,
                        "title": m.code.title,
                        "confidence": m.confidence.overall,
                        "rank": i + 1
                    }
                    for i, m in enumerate(results.matches[:3])
                    if m.code.node_code != naics_code
                ]
            else:
                status = "questionable"
                reason = f"Weak match (rank #{provided_rank}, {confidence:.0%} confidence) - better alternatives likely"
                alternatives = [
                    {
                        "code": m.code.node_code,
                        "title": m.code.title,
                        "confidence": m.confidence.overall,
                        "rank": i + 1
                    }
                    for i, m in enumerate(results.matches[:3])
                ]
        else:
            # Code not in top results at all
            status = "invalid"
            reason = f"Code {naics_code} ({code_info.title}) does not appear in top matches for this description"
            alternatives = [
                {
                    "code": m.code.node_code,
                    "title": m.code.title,
                    "confidence": m.confidence.overall,
                    "rank": i + 1
                }
                for i, m in enumerate(results.matches[:3])
            ]

        # 6. Build response
        response = {
            "naics_code": naics_code,
            "title": code_info.title,
            "description_checked": business_description,
            "status": status,
            "valid": status == "valid",
            "reason": reason
        }

        if provided_match:
            response["confidence"] = provided_match.confidence.overall
            response["rank_in_results"] = provided_rank
            response["confidence_breakdown"] = {
                "semantic": provided_match.confidence.semantic,
                "lexical": provided_match.confidence.lexical,
                "index_term": provided_match.confidence.index_term,
                "specificity": provided_match.confidence.specificity
            }

        if exclusion_warnings:
            response["exclusion_warnings"] = exclusion_warnings
            response["warning"] = f"Description may match {len(exclusion_warnings)} exclusion(s) for this code"

        if alternatives:
            response["suggested_alternatives"] = alternatives

        if top_match and top_match.code.node_code != naics_code:
            response["best_match"] = {
                "code": top_match.code.node_code,
                "title": top_match.code.title,
                "confidence": top_match.confidence.overall
            }

        # Log validation result
        logger.info("Validation completed", data={
            "naics_code": naics_code,
            "status": status,
            "rank": provided_rank,
            "exclusion_warnings": len(exclusion_warnings) if exclusion_warnings else 0
        })

        return response

    except Exception as e:
        logger.error("Validation failed", data={
            "naics_code": naics_code,
            "error_type": type(e).__name__,
            "error_message": str(e)[:200]
        })
        return {
            "naics_code": naics_code,
            "status": "error",
            "error": str(e)
        }


# === Analytics Tools (2) ===

@mcp.tool()
async def get_sector_overview(
    sector_code: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get an overview of NAICS sectors or a specific sector.

    Without a sector_code, returns all 20 sectors.
    With a sector_code (2-digit), returns its subsectors.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        if sector_code:
            # Get specific sector and its children
            sector = await app_ctx.database.get_by_code(sector_code)
            if not sector:
                return {"error": f"Sector {sector_code} not found"}

            children = await app_ctx.database.get_children(sector_code)

            return {
                "sector": {
                    "code": sector.node_code,
                    "title": sector.title,
                    "description": sector.description
                },
                "subsectors": [
                    {
                        "code": child.node_code,
                        "title": child.title
                    }
                    for child in children
                ]
            }
        else:
            # Get all sectors
            results = app_ctx.database.connection.execute("""
                SELECT node_code, title, description
                FROM naics_nodes
                WHERE level = 'sector'
                ORDER BY node_code
            """).fetchall()

            return {
                "sectors": [
                    {
                        "code": row[0],
                        "title": row[1],
                        "description": row[2][:150] + "..." if row[2] and len(row[2]) > 150 else row[2]
                    }
                    for row in results
                ],
                "total": len(results)
            }

    except Exception as e:
        logger.error(f"Failed to get sector overview: {e}")
        return {"error": str(e)}


@mcp.tool()
async def compare_codes(
    codes: List[str],
    ctx: Context
) -> Dict[str, Any]:
    """
    Compare multiple NAICS codes side-by-side.

    Useful for understanding the differences between similar codes
    when making a classification decision.

    Constraints:
    - Maximum 20 codes can be compared at once
    - Each code must be 2-6 digits
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Validate codes
    try:
        codes_result = validate_batch_codes(codes)
        validated_codes = codes_result.value
    except ValidationError as e:
        logger.warning("Code comparison validation failed", data={
            "error": e.message,
            "field": e.details.get("field")
        })
        return {
            "error": e.message,
            "error_category": "validation",
            "codes_compared": [],
            "comparisons": []
        }

    try:
        comparisons = []

        for code_str in validated_codes:
            code = await app_ctx.database.get_by_code(code_str)
            if code:
                cross_refs = await app_ctx.database.get_cross_references(code_str)
                index_terms = await app_ctx.database.get_index_terms_for_code(code_str)

                comparisons.append({
                    "code": code.node_code,
                    "title": code.title,
                    "level": code.level.value,
                    "description": code.description,
                    "hierarchy": code.get_hierarchy_path(),
                    "index_terms": [t.index_term for t in index_terms[:5]],
                    "exclusions": [
                        {
                            "activity": cr.excluded_activity,
                            "classified_under": cr.target_code
                        }
                        for cr in cross_refs if cr.reference_type == "excludes"
                    ][:3]
                })
            else:
                comparisons.append({
                    "code": code_str,
                    "error": "Code not found"
                })

        return {
            "codes_compared": codes,
            "comparisons": comparisons
        }

    except Exception as e:
        logger.error(f"Failed to compare codes: {e}")
        return {"error": str(e)}


# === Workbook Tools (4) ===

@mcp.tool()
async def write_to_workbook(
    request: WorkbookWriteRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Write structured classification decisions to the workbook.

    Like filing paperwork in a filing cabinet with different forms:
    • classification_analysis: NAICS classification decisions
    • industry_comparison: Comparing multiple codes
    • cross_reference_notes: Document exclusions
    • business_profile: Full business classification
    • decision_tree: Classification path
    • sic_conversion: SIC to NAICS analysis
    • research_notes: Research findings
    • custom: Freeform structured content

    Each entry is timestamped, tagged, and searchable.

    ## Linking Entries for Multi-Step Analysis

    Use `parent_entry_id` to create linked chains of analysis:

    1. **Initial Analysis** → Returns entry_id "abc123"
       ```
       write_to_workbook(
         form_type="business_profile",
         label="Acme Corp Initial Profile",
         content={...}
       )
       ```

    2. **Follow-up Research** → Links to parent
       ```
       write_to_workbook(
         form_type="research_notes",
         label="Acme Corp - Industry Research",
         content={...},
         parent_entry_id="abc123"  # Links to initial analysis
       )
       ```

    3. **Final Decision** → Links to research
       ```
       write_to_workbook(
         form_type="classification_analysis",
         label="Acme Corp Final Classification",
         content={...},
         parent_entry_id="def456"  # Links to research step
       )
       ```

    Then use `search_workbook(parent_entry_id="abc123")` to find all
    follow-up entries, or read each entry to trace the full chain.

    ## Common Linking Patterns

    • **Deep Dive**: business_profile → research_notes → classification_analysis
    • **Comparison**: business_profile → industry_comparison → decision
    • **Boundary Case**: classification_analysis → cross_reference_notes → revised classification
    • **SIC Migration**: sic_conversion → research_notes → classification_analysis
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        try:
            form_type = FormType(request.form_type.lower())
        except ValueError:
            return {
                "error": f"Invalid form type: {request.form_type}",
                "valid_types": [ft.value for ft in FormType]
            }

        entry = await app_ctx.classification_workbook.create_entry(
            form_type=form_type,
            label=request.label,
            content=request.content,
            metadata=request.metadata,
            tags=request.tags,
            parent_entry_id=request.parent_entry_id,
            confidence_score=request.confidence_score
        )

        return {
            "success": True,
            "entry_id": entry.entry_id,
            "label": entry.label,
            "form_type": entry.form_type.value,
            "created_at": entry.created_at.isoformat(),
            "session_id": entry.session_id,
            "message": f"Successfully filed '{entry.label}' in workbook"
        }

    except Exception as e:
        logger.error(f"Failed to write to workbook: {e}")
        return {
            "error": f"Failed to write to workbook: {str(e)}",
            "success": False
        }


@mcp.tool()
async def search_workbook(
    request: WorkbookSearchRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Search the classification workbook for past entries.

    Find previous decisions and analyses by form type, session,
    tags, text content, or parent entry.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        form_type = None
        if request.form_type:
            try:
                form_type = FormType(request.form_type.lower())
            except ValueError:
                return {
                    "error": f"Invalid form type: {request.form_type}",
                    "results": []
                }

        entries = await app_ctx.classification_workbook.search_entries(
            form_type=form_type,
            session_id=request.session_id,
            tags=request.tags,
            search_text=request.search_text,
            parent_entry_id=request.parent_entry_id,
            limit=request.limit
        )

        results = [
            {
                "entry_id": entry.entry_id,
                "label": entry.label,
                "form_type": entry.form_type.value,
                "created_at": entry.created_at.isoformat(),
                "tags": entry.tags,
                "confidence_score": entry.confidence_score,
                "preview": str(entry.content)[:200] + "..." if len(str(entry.content)) > 200 else str(entry.content)
            }
            for entry in entries
        ]

        return {
            "success": True,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Failed to search workbook: {e}")
        return {
            "error": f"Failed to search workbook: {str(e)}",
            "results": []
        }


@mcp.tool()
async def get_workbook_entry(
    entry_id: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Retrieve a specific workbook entry by ID.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        entry = await app_ctx.classification_workbook.get_entry(entry_id)

        if not entry:
            return {
                "error": f"Entry {entry_id} not found",
                "success": False
            }

        return {
            "success": True,
            "entry": {
                "entry_id": entry.entry_id,
                "label": entry.label,
                "form_type": entry.form_type.value,
                "content": entry.content,
                "metadata": entry.metadata,
                "created_at": entry.created_at.isoformat(),
                "session_id": entry.session_id,
                "parent_entry_id": entry.parent_entry_id,
                "tags": entry.tags,
                "confidence_score": entry.confidence_score
            }
        }

    except Exception as e:
        logger.error(f"Failed to retrieve entry: {e}")
        return {
            "error": f"Failed to retrieve entry: {str(e)}",
            "success": False
        }


@mcp.tool()
async def get_workbook_template(
    request: WorkbookTemplateRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Get a template for a specific workbook form type.

    Templates show the expected structure for different reasoning patterns.
    Fill out the template and submit with write_to_workbook.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        try:
            form_type = FormType(request.form_type.lower())
        except ValueError:
            return {
                "error": f"Invalid form type: {request.form_type}",
                "valid_types": [ft.value for ft in FormType]
            }

        template = await app_ctx.classification_workbook.get_template(form_type)

        return {
            "success": True,
            "form_type": form_type.value,
            "template": template,
            "description": f"Template for {form_type.value.replace('_', ' ').title()}"
        }

    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        return {
            "error": f"Failed to get template: {str(e)}",
            "success": False
        }


# === Server Info & Health ===

@mcp.tool()
async def get_workflow_guide() -> Dict[str, Any]:
    """
    Get the recommended workflows for NAICS classification.

    Call this to understand how to use the NAICS tools effectively.
    Returns workflow guidance for common use cases:
    - Quick classification (simple businesses)
    - Full classification (complex/multi-segment companies)
    - Boundary cases (activity fits multiple codes)
    - Batch classification
    - Hierarchy exploration

    This is especially useful on first use or when unsure
    which tools to use in what order.
    """
    return {
        "workflows": {
            "quick_classification": {
                "description": "For straightforward single-activity businesses",
                "steps": [
                    "1. classify_business with a clear description",
                    "2. If confidence > 60%, you're likely done",
                    "3. Optionally verify with search_index_terms for exact terminology"
                ]
            },
            "full_classification": {
                "description": "For companies with multiple business lines or ambiguous activities",
                "steps": [
                    "1. Research the company's segments and primary revenue sources",
                    "2. classify_business for initial assessment",
                    "3. search_naics_codes for each major segment/activity",
                    "4. search_index_terms for specific products/services",
                    "5. compare_codes for top candidates side-by-side",
                    "6. get_cross_references to check for exclusions (CRITICAL)",
                    "7. get_siblings or find_similar_industries for alternatives",
                    "8. write_to_workbook with form_type='classification_analysis' to document"
                ]
            },
            "boundary_cases": {
                "description": "When a business sits between two codes",
                "steps": [
                    "1. Search from multiple angles (product vs service vs customer)",
                    "2. get_cross_references on both candidates - exclusions are authoritative",
                    "3. compare_codes to see descriptions and index terms side-by-side",
                    "4. get_code_hierarchy to understand where each code sits conceptually",
                    "5. Document with write_to_workbook form_type='decision_tree'"
                ]
            },
            "batch_classification": {
                "description": "For processing many businesses efficiently",
                "steps": [
                    "1. classify_batch for efficiency",
                    "2. Triage by confidence: >50% accept, 35-50% spot-check, <35% manual review",
                    "3. Use full_classification workflow for low-confidence items"
                ]
            },
            "hierarchy_exploration": {
                "description": "For exploring the NAICS structure itself",
                "steps": [
                    "1. get_sector_overview to see all 20 sectors",
                    "2. get_children to drill down into subsectors",
                    "3. get_siblings to see what else is at the same level",
                    "4. find_similar_industries for semantically related codes"
                ]
            },
            "documented_analysis": {
                "description": "For complex cases requiring traceable reasoning with linked entries",
                "steps": [
                    "1. write_to_workbook with form_type='business_profile' - capture initial understanding",
                    "2. Perform research using search tools, cross-references, comparisons",
                    "3. write_to_workbook with form_type='research_notes', parent_entry_id=step1_id",
                    "4. If comparing codes: write_to_workbook form_type='industry_comparison', parent_entry_id=step3_id",
                    "5. Final decision: write_to_workbook form_type='classification_analysis', parent_entry_id=previous_id",
                    "6. Use search_workbook(parent_entry_id=root_id) to retrieve full analysis chain"
                ],
                "linking_patterns": {
                    "deep_dive": "business_profile → research_notes → classification_analysis",
                    "comparison": "business_profile → industry_comparison → classification_analysis",
                    "boundary_case": "classification_analysis → cross_reference_notes → revised_classification",
                    "sic_migration": "sic_conversion → research_notes → classification_analysis"
                }
            },
            "validation": {
                "description": "For verifying existing or user-provided classifications",
                "steps": [
                    "1. validate_classification with code and business description",
                    "2. If status='valid', classification is confirmed",
                    "3. If status='questionable', review exclusion_warnings and suggested_alternatives",
                    "4. If status='invalid', use suggested_alternatives or run classify_business",
                    "5. Document decision with write_to_workbook if needed"
                ]
            }
        },
        "key_principles": [
            "PRIMARY ACTIVITY determines classification (largest revenue source)",
            "6-digit codes are most specific and preferred",
            "Cross-references tell you what activities are EXCLUDED - always check",
            "Index term matches are strong evidence of correct classification",
            "When uncertain, document reasoning with the workbook tools"
        ],
        "tool_categories": {
            "search": ["search_naics_codes", "search_index_terms", "classify_business", "classify_batch"],
            "navigation": ["get_code_hierarchy", "get_children", "get_siblings", "get_sector_overview", "find_similar_industries"],
            "validation": ["get_cross_references", "compare_codes", "validate_classification"],
            "documentation": ["get_workbook_template", "write_to_workbook", "search_workbook", "get_workbook_entry"],
            "diagnostics": ["ping", "check_readiness", "get_server_health", "get_workflow_guide"]
        },
        "workbook_linking": {
            "description": "Use parent_entry_id to create traceable analysis chains",
            "how_it_works": [
                "1. First entry returns an entry_id (e.g., 'abc123')",
                "2. Pass that as parent_entry_id when creating follow-up entries",
                "3. Chain continues: each entry can be parent to the next",
                "4. Use search_workbook(parent_entry_id='abc123') to find all children",
                "5. Use get_workbook_entry to read any entry and see its parent_entry_id"
            ],
            "benefits": [
                "Full audit trail of classification reasoning",
                "Easy to review multi-step analysis",
                "Supports complex boundary case documentation",
                "Enables 'show your work' for compliance"
            ]
        }
    }


@mcp.tool()
async def ping() -> Dict[str, Any]:
    """
    Simple liveness check.

    Returns immediately to confirm the server process is alive.
    Use this for quick health checks or Kubernetes liveness probes.

    For detailed health information, use get_server_health instead.
    """
    from datetime import datetime, timezone
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
async def check_readiness(ctx: Context) -> Dict[str, Any]:
    """
    Check if the server is ready to handle requests.

    Returns ready/not_ready status based on critical components:
    - Database connection
    - Embedding model loaded

    Use this for Kubernetes readiness probes or before sending requests.
    For detailed diagnostics, use get_server_health instead.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    is_ready = await app_ctx.health_checker.check_readiness()
    return {
        "status": "ready" if is_ready else "not_ready",
        "uptime_seconds": round(app_ctx.health_checker.uptime_seconds, 1),
    }


@mcp.tool()
async def get_server_health(ctx: Context) -> Dict[str, Any]:
    """
    Comprehensive health check with detailed diagnostics.

    Call this first if you're getting empty results or errors.
    Returns detailed status of all critical components:
    - Database: connection, code counts, index terms
    - Embedder: model loaded, dimension
    - Search engine: cache stats, embeddings ready
    - Embeddings: coverage percentage
    - Cross-references: data availability

    Status levels:
    - healthy: All components ready
    - degraded: Some components not fully ready, but functional
    - unhealthy: Critical components unavailable

    For simple liveness checks, use ping instead.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Get comprehensive health check
    result = await app_ctx.health_checker.check_health()

    # Convert to dict and add workbook check (not in core health checker)
    health = result.to_dict()

    # Add workbook status
    try:
        count = app_ctx.database.connection.execute(
            "SELECT COUNT(*) FROM classification_workbook"
        ).fetchone()[0]
        health["components"]["workbook"] = {
            "status": "ready",
            "entries": count
        }
    except Exception as e:
        health["components"]["workbook"] = {
            "status": "error",
            "message": str(e)[:100]
        }
        if "issues" not in health:
            health["issues"] = []
        health["issues"].append(f"Workbook not available: {e}")

    return health


# === Resources ===

@mcp.resource("naics://statistics")
async def get_statistics(ctx: Context) -> Dict[str, Any]:
    """
    Get statistics about the NAICS database.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    stats = await app_ctx.database.get_statistics()

    # Add search engine stats
    stats["embedding_cache"] = app_ctx.search_engine.embedding_cache.get_stats()
    stats["search_cache"] = app_ctx.search_engine.search_cache.get_stats()

    # Add recent search patterns
    if app_ctx.audit_log:
        patterns = await app_ctx.audit_log.analyze_patterns(timeframe_hours=24)
        stats["recent_searches"] = patterns

    return stats


@mcp.resource("naics://recent_searches")
async def get_recent_searches(ctx: Context) -> List[Dict[str, Any]]:
    """
    Get recent search queries for monitoring.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    return app_ctx.audit_log.get_recent_searches(limit=20)


# === Prompts ===

@mcp.prompt()
def classification_assistant_prompt() -> str:
    """Template for NAICS classification assistance."""
    return """You are a NAICS classification expert. Help the user find the right NAICS codes by:

1. Understanding their business description
2. Asking clarifying questions if needed (primary activity, customers, products)
3. Searching for appropriate codes
4. Checking cross-references for exclusions
5. Explaining why each result matches

Always explain the hierarchy (Sector → Subsector → Industry Group → NAICS Industry → National Industry)
and highlight any important cross-reference exclusions.

Key classification principles:
- Primary activity determines classification (what generates the most revenue)
- 6-digit codes are most specific and preferred
- Cross-references tell you what activities belong elsewhere
- Index terms are the official vocabulary"""


@mcp.prompt()
def classification_methodology() -> str:
    """NAICS classification methodology guide."""
    return """NAICS CLASSIFICATION METHODOLOGY

PRIMARY RULE:
Establishments are classified based on their PRIMARY activity - the activity
that generates the largest share of their revenue.

HIERARCHY (5 levels):
- Sector (2-digit): Broadest category (e.g., 31 = Manufacturing)
- Subsector (3-digit): Industry groupings
- Industry Group (4-digit): More specific groupings
- NAICS Industry (5-digit): Industry level
- National Industry (6-digit): Most specific (US-specific detail)

BUILDING EFFECTIVE SEARCHES:
1. Start with the primary activity or product
2. Include customer type (business vs consumer)
3. Describe the production process if relevant
4. Use specific terminology from your industry

INTERPRETING RESULTS:
- Higher confidence = stronger match
- Check cross-references for exclusions
- Index term matches indicate official terminology alignment
- The most specific applicable code is preferred

CROSS-REFERENCES ARE CRITICAL:
- They tell you what activities are EXCLUDED from a code
- They point to where excluded activities SHOULD be classified
- Always check cross-references when confidence is moderate

COMMON PITFALLS:
- Classifying by product sold rather than primary activity
- Ignoring cross-reference exclusions
- Choosing a broad code when a specific one applies
- Not considering vertically integrated operations"""


def main():
    """Run the MCP server."""
    import sys

    if "--debug" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)

    mcp.run()


if __name__ == "__main__":
    main()
