#!/usr/bin/env python3
"""
NAICS MCP Server

An intelligent industry classification service for NAICS 2022,
built with clarity and purpose.
"""

import asyncio
import logging
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
from .tools.workbook_tools import (
    WorkbookWriteRequest, WorkbookSearchRequest, WorkbookTemplateRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Debug mode
import os
if os.getenv('DEBUG', '').lower() in ['true', '1', 'yes']:
    log_file = Path.home() / ".cache" / "naics-mcp-server" / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"Debug mode enabled, logging to {log_file}")


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
        classification_workbook: ClassificationWorkbook
    ):
        self.database = database
        self.embedder = embedder
        self.search_engine = search_engine
        self.audit_log = audit_log
        self.cross_ref_service = cross_ref_service
        self.classification_workbook = classification_workbook


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
    logger.info("Starting NAICS MCP Server...")

    # Load configuration
    search_config = SearchConfig.from_env()
    server_config = ServerConfig.from_env()

    # Initialize database
    database = NAICSDatabase(search_config.database_path)
    database.connect()

    # Initialize embedder
    cache_dir = Path.home() / ".cache" / "naics-mcp-server" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(
        model_name=search_config.embedding_model,
        cache_dir=cache_dir
    )
    embedder.load_model()

    # Initialize search engine
    search_engine = NAICSSearchEngine(database, embedder, search_config)

    # Initialize embeddings if needed
    init_result = await search_engine.initialize_embeddings()
    if init_result.get('action') in ['created', 'populated']:
        logger.info(f"First run: Generated {init_result.get('embeddings_generated', 0)} embeddings")
    else:
        logger.info(f"Embeddings ready: {init_result}")

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

    # Create application context
    app_context = AppContext(
        database, embedder, search_engine, audit_log,
        cross_ref_service, classification_workbook
    )

    logger.info("NAICS MCP Server ready!")

    try:
        yield app_context
    finally:
        logger.info("Shutting down NAICS MCP Server...")
        database.disconnect()
        logger.info("Shutdown complete")


# Create the MCP server
mcp = FastMCP(
    name="NAICS Classification Assistant",
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

    # Start audit event
    search_event = SearchEvent.start(request.query, request.strategy)

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
        strategy = strategy_map.get(request.strategy, SearchStrategy.HYBRID)

        # Perform search
        results = await app_ctx.search_engine.search(
            query=request.query,
            strategy=strategy,
            limit=request.limit,
            min_confidence=request.min_confidence,
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

        logger.error(f"Search failed: {e}")
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

    except Exception as e:
        logger.error(f"Index term search failed: {e}")
        return {
            "error": str(e),
            "matches": []
        }


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

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return {
            "error": str(e),
            "similar_codes": []
        }


@mcp.tool()
async def classify_batch(
    request: BatchClassifyRequest,
    ctx: Context
) -> Dict[str, Any]:
    """
    Classify multiple business descriptions in batch.

    Useful for processing lists of businesses efficiently.
    Returns the best matching NAICS code for each description.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    classifications = []

    for description in request.descriptions:
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

    except Exception as e:
        logger.error(f"Failed to get hierarchy: {e}")
        return {
            "error": str(e),
            "hierarchy": []
        }


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

    except Exception as e:
        logger.error(f"Failed to get children: {e}")
        return {
            "error": str(e),
            "children": []
        }


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

    except Exception as e:
        logger.error(f"Failed to get siblings: {e}")
        return {
            "error": str(e),
            "siblings": []
        }


# === Classification Tools (2) ===

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

    try:
        # Perform comprehensive search
        results = await app_ctx.search_engine.search(
            query=request.description,
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
                "hierarchy": primary.hierarchy_path
            },
            "alternatives": [
                {
                    "code": alt.code.node_code,
                    "title": alt.code.title,
                    "confidence": alt.confidence.overall
                }
                for alt in alternatives
            ]
        }

        if request.include_reasoning:
            reasoning_parts = [
                f"Primary classification: {primary.code.node_code} - {primary.code.title}",
                f"Confidence: {primary.confidence.overall:.1%}",
                primary.confidence.to_explanation()
            ]

            if primary.matched_index_terms:
                reasoning_parts.append(f"Matches official index terms: {', '.join(primary.matched_index_terms[:3])}")

            if primary.exclusion_warnings:
                reasoning_parts.append("WARNINGS:")
                reasoning_parts.extend(primary.exclusion_warnings)

            response["reasoning"] = "\n".join(reasoning_parts)

        if primary.exclusion_warnings:
            response["exclusion_warnings"] = primary.exclusion_warnings

        return response

    except Exception as e:
        logger.error(f"Classification failed: {e}")
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
        logger.error(f"Failed to get cross-references: {e}")
        return {
            "error": str(e),
            "cross_references": []
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
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    try:
        comparisons = []

        for code_str in codes:
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
