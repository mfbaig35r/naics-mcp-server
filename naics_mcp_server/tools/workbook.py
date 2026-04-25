"""
Workbook tools for NAICS MCP Server.

Provides: write_to_workbook, search_workbook, get_workbook_entry, get_workbook_template
"""

from typing import Any

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from ..app_context import get_app_context
from ..core.classification_workbook import FormType
from ..observability.logging import get_logger

logger = get_logger(__name__)


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


def register_tools(mcp):
    """Register workbook tools on the MCP server."""

    @mcp.tool()
    async def write_to_workbook(request: WorkbookWriteRequest, ctx: Context) -> dict[str, Any]:
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
        app_ctx = get_app_context(ctx)

        try:
            try:
                form_type = FormType(request.form_type.lower())
            except ValueError:
                return {
                    "error": f"Invalid form type: {request.form_type}",
                    "valid_types": [ft.value for ft in FormType],
                }

            entry = await app_ctx.classification_workbook.create_entry(
                form_type=form_type,
                label=request.label,
                content=request.content,
                metadata=request.metadata,
                tags=request.tags,
                parent_entry_id=request.parent_entry_id,
                confidence_score=request.confidence_score,
            )

            return {
                "success": True,
                "entry_id": entry.entry_id,
                "label": entry.label,
                "form_type": entry.form_type.value,
                "created_at": entry.created_at.isoformat(),
                "session_id": entry.session_id,
                "message": f"Successfully filed '{entry.label}' in workbook",
            }

        except Exception as e:
            logger.error(f"Failed to write to workbook: {e}")
            return {"error": f"Failed to write to workbook: {str(e)}", "success": False}

    @mcp.tool()
    async def search_workbook(request: WorkbookSearchRequest, ctx: Context) -> dict[str, Any]:
        """
        Search the classification workbook for past entries.

        Find previous decisions and analyses by form type, session,
        tags, text content, or parent entry.
        """
        app_ctx = get_app_context(ctx)

        try:
            form_type = None
            if request.form_type:
                try:
                    form_type = FormType(request.form_type.lower())
                except ValueError:
                    return {"error": f"Invalid form type: {request.form_type}", "results": []}

            entries = await app_ctx.classification_workbook.search_entries(
                form_type=form_type,
                session_id=request.session_id,
                tags=request.tags,
                search_text=request.search_text,
                parent_entry_id=request.parent_entry_id,
                limit=request.limit,
            )

            results = [
                {
                    "entry_id": entry.entry_id,
                    "label": entry.label,
                    "form_type": entry.form_type.value,
                    "created_at": entry.created_at.isoformat(),
                    "tags": entry.tags,
                    "confidence_score": entry.confidence_score,
                    "preview": str(entry.content)[:200] + "..."
                    if len(str(entry.content)) > 200
                    else str(entry.content),
                }
                for entry in entries
            ]

            return {"success": True, "count": len(results), "results": results}

        except Exception as e:
            logger.error(f"Failed to search workbook: {e}")
            return {"error": f"Failed to search workbook: {str(e)}", "results": []}

    @mcp.tool()
    async def get_workbook_entry(entry_id: str, ctx: Context) -> dict[str, Any]:
        """
        Retrieve a specific workbook entry by ID.
        """
        app_ctx = get_app_context(ctx)

        try:
            entry = await app_ctx.classification_workbook.get_entry(entry_id)

            if not entry:
                return {"error": f"Entry {entry_id} not found", "success": False}

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
                    "confidence_score": entry.confidence_score,
                },
            }

        except Exception as e:
            logger.error(f"Failed to retrieve entry: {e}")
            return {"error": f"Failed to retrieve entry: {str(e)}", "success": False}

    @mcp.tool()
    async def get_workbook_template(
        request: WorkbookTemplateRequest, ctx: Context
    ) -> dict[str, Any]:
        """
        Get a template for a specific workbook form type.

        Templates show the expected structure for different reasoning patterns.
        Fill out the template and submit with write_to_workbook.
        """
        app_ctx = get_app_context(ctx)

        try:
            try:
                form_type = FormType(request.form_type.lower())
            except ValueError:
                return {
                    "error": f"Invalid form type: {request.form_type}",
                    "valid_types": [ft.value for ft in FormType],
                }

            template = await app_ctx.classification_workbook.get_template(form_type)

            return {
                "success": True,
                "form_type": form_type.value,
                "template": template,
                "description": f"Template for {form_type.value.replace('_', ' ').title()}",
            }

        except Exception as e:
            logger.error(f"Failed to get template: {e}")
            return {"error": f"Failed to get template: {str(e)}", "success": False}
