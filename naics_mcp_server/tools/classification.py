"""
Classification tools for NAICS MCP Server.

Provides: classify_business, classify_batch, get_cross_references, validate_classification
"""

from typing import Any

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from ..app_context import check_rate_limit, get_app_context
from ..core.errors import ValidationError
from ..core.validation import validate_batch_descriptions, validate_description
from ..models.response_models import (
    BatchClassifyResponse,
    CrossReferencesResponse,
    ValidationResponse,
)
from ..models.search_models import ClassificationResult, SearchStrategy
from ..observability.logging import (
    generate_request_id,
    get_logger,
    sanitize_text,
    set_request_context,
)

logger = get_logger(__name__)


class ClassifyRequest(BaseModel):
    """Parameters for classifying a business."""

    description: str = Field(description="Business or activity description")
    include_reasoning: bool = Field(default=True, description="Include detailed reasoning")
    check_cross_refs: bool = Field(
        default=True, description="Check cross-references for exclusions"
    )


class BatchClassifyRequest(BaseModel):
    """Parameters for batch classification."""

    descriptions: list[str] = Field(description="List of business descriptions to classify")
    include_confidence: bool = Field(default=True, description="Include confidence scores")


def register_tools(mcp):
    """Register classification tools on the MCP server."""

    @mcp.tool()
    async def classify_business(request: ClassifyRequest, ctx: Context) -> dict[str, Any]:
        """
        Classify a business description to NAICS with detailed reasoning.

        This performs a thorough classification analysis including:
        - Search across all strategies
        - Index term matching
        - Cross-reference checking
        - Confidence breakdown

        Returns the recommended classification with alternatives and reasoning.
        """
        app_ctx = get_app_context(ctx)

        # Check rate limit
        await check_rate_limit(ctx, "classify_business")

        # Set request context for logging
        request_id = generate_request_id()
        set_request_context(request_id=request_id, tool_name="classify_business")

        # Validate input
        try:
            desc_result = validate_description(request.description)
            validated_description = desc_result.value
        except ValidationError as e:
            logger.warning(
                "Classification validation failed",
                data={"error": e.message, "field": e.details.get("field")},
            )
            return {
                "input": request.description[:100] if request.description else None,
                "error": e.message,
                "error_category": "validation",
            }

        logger.info(
            "Classification requested",
            data={
                "description_length": len(validated_description),
                "description_preview": sanitize_text(validated_description, 50),
                "check_cross_refs": request.check_cross_refs,
            },
        )

        try:
            # Perform comprehensive search
            results = await app_ctx.search_engine.search(
                query=validated_description,
                strategy=SearchStrategy.HYBRID,
                limit=5,
                min_confidence=0.2,
                include_cross_refs=request.check_cross_refs,
            )

            if not results.matches:
                return {
                    "input": request.description,
                    "classification": None,
                    "reasoning": "No suitable NAICS codes found for this description. Consider providing more specific details about the business activity.",
                    "alternatives": [],
                }

            primary = results.matches[0]
            alternatives = results.matches[1:4]

            reasoning = (
                ClassificationResult.build_reasoning(primary, alternatives, request.check_cross_refs)
                if request.include_reasoning
                else None
            )

            classification = ClassificationResult(
                input_description=request.description,
                primary_classification=primary,
                alternative_classifications=alternatives,
                reasoning=reasoning or "",
                cross_ref_notes=primary.exclusion_warnings,
            )
            response = classification.to_dict()

            # Log successful classification
            logger.info(
                "Classification completed",
                data={
                    "primary_code": primary.code.node_code,
                    "confidence": primary.confidence.overall,
                    "alternatives_count": len(alternatives),
                    "exclusion_warnings": len(primary.exclusion_warnings)
                    if primary.exclusion_warnings
                    else 0,
                },
            )

            return response

        except Exception as e:
            logger.error(
                "Classification failed",
                data={"error_type": type(e).__name__, "error_message": str(e)[:200]},
            )
            return {"input": request.description, "error": str(e)}

    @mcp.tool()
    async def classify_batch(request: BatchClassifyRequest, ctx: Context) -> dict[str, Any]:
        """
        Classify multiple business descriptions in batch.

        Useful for processing lists of businesses efficiently.
        Returns the best matching NAICS code for each description.

        Constraints:
        - Maximum 100 descriptions per batch
        - Each description must be 10-5000 characters
        """
        app_ctx = get_app_context(ctx)

        # Check rate limit (batch operations have stricter limits)
        await check_rate_limit(ctx, "classify_batch")

        # Validate batch
        try:
            batch_result = validate_batch_descriptions(request.descriptions)
            validated_descriptions = batch_result.value
        except ValidationError as e:
            logger.warning(
                "Batch validation failed", data={"error": e.message, "field": e.details.get("field")}
            )
            return {
                "error": e.message,
                "error_category": "validation",
                "classifications": [],
                "total_processed": 0,
                "successfully_classified": 0,
            }

        classifications = []

        for description in validated_descriptions:
            try:
                results = await app_ctx.search_engine.search(
                    query=description, strategy=SearchStrategy.HYBRID, limit=1, min_confidence=0.3
                )

                if results.matches:
                    best_match = results.matches[0]
                    classification = {
                        "description": description,
                        "naics_code": best_match.code.node_code,
                        "naics_title": best_match.code.title,
                        "level": best_match.code.level.value,
                    }

                    if request.include_confidence:
                        classification["confidence"] = best_match.confidence.overall
                        classification["explanation"] = best_match.confidence.to_explanation()

                    classifications.append(classification)
                else:
                    classifications.append(
                        {
                            "description": description,
                            "naics_code": None,
                            "naics_title": "No suitable classification found",
                            "confidence": 0.0 if request.include_confidence else None,
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to classify '{description}': {e}")
                classifications.append({"description": description, "error": str(e)})

        return BatchClassifyResponse(
            classifications=classifications,
            total_processed=len(request.descriptions),
            successfully_classified=sum(
                1 for c in classifications if c.get("naics_code") and "error" not in c
            ),
        ).to_dict()

    @mcp.tool()
    async def get_cross_references(naics_code: str, ctx: Context) -> dict[str, Any]:
        """
        Get cross-references (exclusions/inclusions) for a NAICS code.

        Cross-references are CRITICAL for accurate classification.
        They tell you what activities are explicitly excluded from this code
        and where they should be classified instead.

        Example: Code 311111 (Dog Food) explicitly excludes "prepared feeds for
        cattle, hogs, poultry" which should be classified under 311119.
        """
        app_ctx = get_app_context(ctx)

        try:
            code = await app_ctx.database.get_by_code(naics_code)
            if not code:
                return {"error": f"NAICS code {naics_code} not found", "cross_references": []}

            cross_refs = await app_ctx.database.get_cross_references(naics_code)

            return CrossReferencesResponse(
                naics_code=naics_code,
                title=code.title,
                cross_references=[
                    {
                        "type": cr.reference_type,
                        "excluded_activity": cr.excluded_activity,
                        "target_code": cr.target_code,
                        "reference_text": cr.reference_text,
                    }
                    for cr in cross_refs
                ],
                total=len(cross_refs),
            ).to_dict()

        except Exception as e:
            logger.error(
                "Failed to get cross-references",
                data={
                    "naics_code": naics_code,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                },
            )
            return {"error": str(e), "cross_references": []}

    @mcp.tool()
    async def validate_classification(
        naics_code: str, business_description: str, ctx: Context
    ) -> dict[str, Any]:
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
        app_ctx = get_app_context(ctx)

        try:
            # 1. Verify the code exists
            code_info = await app_ctx.database.get_by_code(naics_code)
            if not code_info:
                return {
                    "naics_code": naics_code,
                    "status": "invalid",
                    "reason": f"NAICS code {naics_code} does not exist",
                    "valid": False,
                }

            # 2. Search for best matches for this description
            results = await app_ctx.search_engine.search(
                query=business_description,
                strategy=SearchStrategy.HYBRID,
                limit=5,
                min_confidence=0.2,
                include_cross_refs=True,
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
                    if len(overlap) >= 2 or any(
                        word in desc_lower for word in activity_words if len(word) > 5
                    ):
                        exclusion_warnings.append(
                            {
                                "excluded_activity": cr.excluded_activity,
                                "should_be": cr.target_code,
                                "reference": cr.reference_text[:200],
                            }
                        )

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
                        reason = (
                            f"Best available match (rank #1) but moderate confidence ({confidence:.0%})"
                        )
                elif provided_rank <= 3:
                    status = "questionable"
                    reason = f"Acceptable match (rank #{provided_rank}, {confidence:.0%} confidence) but better alternatives exist"
                    alternatives = [
                        {
                            "code": m.code.node_code,
                            "title": m.code.title,
                            "confidence": m.confidence.overall,
                            "rank": i + 1,
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
                            "rank": i + 1,
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
                        "rank": i + 1,
                    }
                    for i, m in enumerate(results.matches[:3])
                ]

            # 6. Build response
            best_match = None
            if top_match and top_match.code.node_code != naics_code:
                best_match = {
                    "code": top_match.code.node_code,
                    "title": top_match.code.title,
                    "confidence": top_match.confidence.overall,
                }

            response = ValidationResponse(
                naics_code=naics_code,
                title=code_info.title,
                status=status,
                valid=status == "valid",
                reason=reason,
                description_checked=business_description,
                confidence=provided_match.confidence.overall if provided_match else None,
                rank_in_results=provided_rank,
                confidence_breakdown={
                    "semantic": provided_match.confidence.semantic,
                    "lexical": provided_match.confidence.lexical,
                    "index_term": provided_match.confidence.index_term,
                    "specificity": provided_match.confidence.specificity,
                } if provided_match else None,
                exclusion_warnings=exclusion_warnings,
                warning=f"Description may match {len(exclusion_warnings)} exclusion(s) for this code" if exclusion_warnings else None,
                suggested_alternatives=alternatives,
                best_match=best_match,
            )

            # Log validation result
            logger.info(
                "Validation completed",
                data={
                    "naics_code": naics_code,
                    "status": status,
                    "rank": provided_rank,
                    "exclusion_warnings": len(exclusion_warnings) if exclusion_warnings else 0,
                },
            )

            return response.to_dict()

        except Exception as e:
            logger.error(
                "Validation failed",
                data={
                    "naics_code": naics_code,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:200],
                },
            )
            return {"naics_code": naics_code, "status": "error", "error": str(e)}
