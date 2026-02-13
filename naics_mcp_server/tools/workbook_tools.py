"""
Workbook tool request models.

Pydantic models for workbook-related MCP tool parameters.
"""

from typing import Any

from pydantic import BaseModel, Field


class WorkbookWriteRequest(BaseModel):
    """Parameters for writing to the workbook."""

    form_type: str = Field(
        description=(
            "Type of form to use: classification_analysis, industry_comparison, "
            "cross_reference_notes, business_profile, decision_tree, sic_conversion, "
            "research_notes, or custom"
        )
    )
    label: str = Field(
        description="Human-readable label for this entry (e.g., 'Software Company Classification')"
    )
    content: dict[str, Any] = Field(description="Structured content following the form template")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (source, context, etc.)"
    )
    tags: list[str] | None = Field(default=None, description="Tags for categorization and search")
    parent_entry_id: str | None = Field(
        default=None, description="Link to a parent entry if this is a follow-up"
    )
    confidence_score: float | None = Field(
        default=None, description="Confidence score for the decision (0-1)", ge=0.0, le=1.0
    )


class WorkbookSearchRequest(BaseModel):
    """Parameters for searching the workbook."""

    form_type: str | None = Field(default=None, description="Filter by form type")
    session_id: str | None = Field(default=None, description="Filter by session ID")
    tags: list[str] | None = Field(default=None, description="Filter by tags (any match)")
    search_text: str | None = Field(default=None, description="Full-text search in content")
    parent_entry_id: str | None = Field(default=None, description="Filter by parent entry")
    limit: int = Field(default=20, description="Maximum results to return", ge=1, le=100)


class WorkbookTemplateRequest(BaseModel):
    """Parameters for getting a workbook template."""

    form_type: str = Field(
        description=(
            "Type of form to get template for: classification_analysis, "
            "industry_comparison, cross_reference_notes, business_profile, "
            "decision_tree, sic_conversion, research_notes, or custom"
        )
    )
