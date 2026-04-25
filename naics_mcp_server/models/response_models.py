"""
Typed response models for NAICS MCP Server tools.

Each tool returns a dataclass with a to_dict() method, providing
typed contracts for AI agent consumers.
"""

from dataclasses import dataclass, field
from typing import Any

# === Search Tools ===


@dataclass
class IndexTermSearchResponse:
    """Response from search_index_terms."""

    search_text: str
    matches: list[dict[str, str]]
    total_found: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_text": self.search_text,
            "matches": self.matches,
            "total_found": self.total_found,
        }


@dataclass
class SimilarIndustriesResponse:
    """Response from find_similar_industries."""

    original_code: str
    original_title: str
    similar_codes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_code": self.original_code,
            "original_title": self.original_title,
            "similar_codes": self.similar_codes,
        }


# === Hierarchy Tools ===


@dataclass
class HierarchyResponse:
    """Response from get_code_hierarchy."""

    naics_code: str
    hierarchy: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "naics_code": self.naics_code,
            "hierarchy": self.hierarchy,
        }


@dataclass
class ChildrenResponse:
    """Response from get_children."""

    parent_code: str
    children: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_code": self.parent_code,
            "children": self.children,
            "count": self.count,
        }


@dataclass
class SiblingsResponse:
    """Response from get_siblings."""

    code: str
    title: str
    level: str
    siblings: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "level": self.level,
            "siblings": self.siblings,
        }


# === Classification Tools ===


@dataclass
class CrossReferencesResponse:
    """Response from get_cross_references."""

    naics_code: str
    title: str
    cross_references: list[dict[str, Any]]
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "naics_code": self.naics_code,
            "title": self.title,
            "cross_references": self.cross_references,
            "total": self.total,
        }


@dataclass
class ValidationResponse:
    """Response from validate_classification."""

    naics_code: str
    title: str | None
    status: str  # "valid", "questionable", "invalid", "error"
    valid: bool
    reason: str
    description_checked: str | None = None
    confidence: float | None = None
    rank_in_results: int | None = None
    confidence_breakdown: dict[str, float] | None = None
    exclusion_warnings: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    suggested_alternatives: list[dict[str, Any]] = field(default_factory=list)
    best_match: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "naics_code": self.naics_code,
            "title": self.title,
            "status": self.status,
            "valid": self.valid,
            "reason": self.reason,
        }
        if self.description_checked:
            result["description_checked"] = self.description_checked
        if self.confidence is not None:
            result["confidence"] = self.confidence
            result["rank_in_results"] = self.rank_in_results
        if self.confidence_breakdown:
            result["confidence_breakdown"] = self.confidence_breakdown
        if self.exclusion_warnings:
            result["exclusion_warnings"] = self.exclusion_warnings
            result["warning"] = self.warning
        if self.suggested_alternatives:
            result["suggested_alternatives"] = self.suggested_alternatives
        if self.best_match:
            result["best_match"] = self.best_match
        return result


@dataclass
class BatchClassifyResponse:
    """Response from classify_batch."""

    classifications: list[dict[str, Any]]
    total_processed: int
    successfully_classified: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifications": self.classifications,
            "total_processed": self.total_processed,
            "successfully_classified": self.successfully_classified,
        }


# === Analytics Tools ===


@dataclass
class SectorOverviewResponse:
    """Response from get_sector_overview."""

    sectors: list[dict[str, Any]] | None = None
    sector: dict[str, Any] | None = None
    subsectors: list[dict[str, Any]] | None = None
    total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.sector is not None:
            return {"sector": self.sector, "subsectors": self.subsectors or []}
        return {"sectors": self.sectors or [], "total": self.total or 0}


@dataclass
class CompareCodesResponse:
    """Response from compare_codes."""

    codes_compared: list[str]
    comparisons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "codes_compared": self.codes_compared,
            "comparisons": self.comparisons,
        }


# === Workbook Tools ===


@dataclass
class WorkbookWriteResponse:
    """Response from write_to_workbook."""

    success: bool
    entry_id: str | None = None
    label: str | None = None
    form_type: str | None = None
    created_at: str | None = None
    session_id: str | None = None
    message: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if not self.success:
            return {"error": self.error, "success": False}
        return {
            "success": True,
            "entry_id": self.entry_id,
            "label": self.label,
            "form_type": self.form_type,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "message": self.message,
        }


@dataclass
class WorkbookSearchResponse:
    """Response from search_workbook."""

    results: list[dict[str, Any]]
    total_found: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "total_found": self.total_found,
        }


# === Diagnostics Tools ===


@dataclass
class PingResponse:
    """Response from ping."""

    status: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "timestamp": self.timestamp}


@dataclass
class ReadinessResponse:
    """Response from check_readiness."""

    status: str  # "ready" or "not_ready"
    uptime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
        }
