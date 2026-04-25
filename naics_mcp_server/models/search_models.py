"""
Search-related models for NAICS classification.

Clear, purposeful data structures for search operations and results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .naics_models import CrossReference, NAICSCode


class SearchStrategy(str, Enum):
    """
    Search strategies with clear purposes.
    """

    HYBRID = "hybrid"  # Best of both worlds (default)
    SEMANTIC = "semantic"  # Meaning-based search
    LEXICAL = "lexical"  # Exact term matching

    def to_user_friendly(self) -> str:
        """Convert to user-friendly description."""
        descriptions = {
            self.HYBRID: "best match (semantic + exact)",
            self.SEMANTIC: "meaning-based search",
            self.LEXICAL: "exact term matching",
        }
        return descriptions.get(self, self.value)


@dataclass
class ConfidenceScore:
    """
    Transparent confidence scoring with explainable components.

    Each factor is explicit and can be explained to users.
    Based on the NAICS-specific formula from requirements:

    overall = (
        0.40 * semantic_score +
        0.20 * lexical_score +
        0.15 * index_term_match +
        0.15 * specificity_preference +
        0.10 * cross_ref_relevance
    )
    """

    semantic: float  # Semantic similarity (0-1)
    lexical: float  # Exact term matching (0-1)
    index_term: float  # Official index term match (0-1)
    specificity: float  # 6-digit > 2-digit preference (0-1)
    cross_ref: float  # Cross-reference relevance (0-1)
    overall: float  # Weighted combination

    def to_explanation(self) -> str:
        """
        Generate a human-readable explanation of the confidence score.
        """
        explanations = []

        if self.semantic > 0.8:
            explanations.append("Strong semantic match")
        elif self.semantic > 0.6:
            explanations.append("Good semantic similarity")
        elif self.semantic > 0.4:
            explanations.append("Moderate semantic similarity")

        if self.lexical > 0.7:
            explanations.append("contains exact terms")

        if self.index_term > 0.5:
            explanations.append("matches official index term")

        if self.specificity > 0.8:
            explanations.append("most specific level")
        elif self.specificity < 0.3:
            explanations.append("broad category")

        if self.cross_ref < 0:
            explanations.append("WARNING: may match exclusion criteria")

        if not explanations:
            explanations.append("Possible match based on context")

        return f"{', '.join(explanations)} (confidence: {self.overall:.0%})"


@dataclass
class NAICSMatch:
    """
    A single search result with all context needed for understanding.
    """

    code: NAICSCode
    confidence: ConfidenceScore

    # Why this result was returned
    match_reasons: list[str] = field(default_factory=list)

    # Distance metrics for transparency
    embedding_similarity: float = 0.0
    text_similarity: float = 0.0

    # Index term matches
    matched_index_terms: list[str] = field(default_factory=list)

    # Cross-reference context (critical for NAICS)
    relevant_cross_refs: list[CrossReference] = field(default_factory=list)
    exclusion_warnings: list[str] = field(default_factory=list)

    # Additional context
    hierarchy_path: list[str] = field(default_factory=list)
    rank: int = 0  # Position in results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code.node_code,
            "title": self.code.title,
            "description": self.code.description,
            "level": self.code.level.value,
            "confidence": self.confidence.overall,
            "explanation": self.confidence.to_explanation(),
            "hierarchy": self.hierarchy_path,
            "matched_index_terms": self.matched_index_terms,
            "exclusion_warnings": self.exclusion_warnings,
            "rank": self.rank,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Compact dict for use in alternative listings."""
        return {
            "code": self.code.node_code,
            "title": self.code.title,
            "confidence": self.confidence.overall,
            "matched_index_terms": self.matched_index_terms,
        }


@dataclass
class QueryTerms:
    """
    Expanded query terms for comprehensive search.

    Makes the query expansion process transparent.
    """

    original: str  # User's exact input
    synonyms: list[str] = field(default_factory=list)  # Alternative words
    expanded: list[str] = field(default_factory=list)  # Spelled-out abbreviations
    related: list[str] = field(default_factory=list)  # Conceptually connected terms

    def all_terms(self) -> list[str]:
        """Get all terms for search."""
        terms = [self.original]
        terms.extend(self.synonyms)
        terms.extend(self.expanded)
        terms.extend(self.related)
        return list(set(terms))  # Remove duplicates

    def was_expanded(self) -> bool:
        """Check if any expansion occurred."""
        return bool(self.synonyms or self.expanded or self.related)


@dataclass
class QueryMetadata:
    """
    Metadata about how a query was processed.

    Helps users understand what happened behind the scenes.
    """

    original_query: str
    expanded_terms: QueryTerms
    strategy_used: str
    was_expanded: bool
    processing_time_ms: int
    total_candidates_considered: int
    index_terms_searched: int = 0  # How many index terms were checked
    cross_refs_checked: int = 0  # How many cross-refs were checked
    fallback_used: str | None = None  # If primary strategy failed, what fallback was used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.original_query,
            "expanded": self.was_expanded,
            "strategy": self.strategy_used,
            "processing_time_ms": self.processing_time_ms,
            "candidates_considered": self.total_candidates_considered,
            "index_terms_searched": self.index_terms_searched,
            "cross_refs_checked": self.cross_refs_checked,
        }
        if self.fallback_used:
            result["fallback_used"] = self.fallback_used
        return result


@dataclass
class SearchResults:
    """
    Complete search results with metadata for transparency.
    """

    matches: list[NAICSMatch]
    query_metadata: QueryMetadata

    def get_top_results(self, n: int = 10) -> list[NAICSMatch]:
        """Get the top N results."""
        return self.matches[:n]

    def filter_by_confidence(self, min_confidence: float) -> list[NAICSMatch]:
        """Filter results by minimum confidence score."""
        return [m for m in self.matches if m.confidence.overall >= min_confidence]

    def get_exclusion_warnings(self) -> list[dict[str, Any]]:
        """Get all exclusion warnings from results."""
        warnings = []
        for match in self.matches:
            if match.exclusion_warnings:
                warnings.append(
                    {"code": match.code.node_code, "warnings": match.exclusion_warnings}
                )
        return warnings

    def to_dict(self, guidance: list[str] | None = None) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query_metadata.original_query,
            "results": [m.to_dict() for m in self.matches],
            "expanded": self.query_metadata.was_expanded,
            "strategy_used": self.query_metadata.strategy_used,
            "total_found": len(self.matches),
            "search_time_ms": self.query_metadata.processing_time_ms,
            "guidance": guidance or [],
        }


@dataclass
class ClassificationResult:
    """
    Result of a business classification operation.

    Used by the classify_business tool.
    """

    input_description: str
    primary_classification: NAICSMatch
    alternative_classifications: list[NAICSMatch]
    reasoning: str
    cross_ref_notes: list[str] = field(default_factory=list)
    confidence_level: str = "medium"  # "high", "medium", "low"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "input": self.input_description,
            "classification": self.primary_classification.to_dict()
            if self.primary_classification
            else None,
            "alternatives": [m.to_summary_dict() for m in self.alternative_classifications],
        }
        if self.primary_classification:
            # Add confidence breakdown to classification
            conf = self.primary_classification.confidence
            result["classification"]["confidence_breakdown"] = {
                "semantic": conf.semantic,
                "lexical": conf.lexical,
                "index_term": conf.index_term,
                "specificity": conf.specificity,
                "cross_ref": conf.cross_ref,
            }
        if self.reasoning:
            result["reasoning"] = self.reasoning
        if self.cross_ref_notes:
            result["exclusion_warnings"] = self.cross_ref_notes
        return result

    @staticmethod
    def build_reasoning(
        primary: "NAICSMatch",
        alternatives: list["NAICSMatch"],
        check_cross_refs: bool = True,
    ) -> str:
        """Build detailed reasoning text for a classification."""
        parts = []

        parts.append(f"**Primary Classification:** {primary.code.node_code} - {primary.code.title}")
        parts.append(f"**Overall Confidence:** {primary.confidence.overall:.1%}")
        parts.append("")

        # Key decision factors
        conf = primary.confidence
        factors = []
        if conf.semantic > 0.7:
            factors.append(f"semantic similarity ({conf.semantic:.0%})")
        if conf.lexical > 0.5:
            factors.append(f"exact term matches ({conf.lexical:.0%})")
        if conf.index_term > 0.5:
            factors.append(f"official index term match ({conf.index_term:.0%})")
        if conf.specificity > 0.7:
            factors.append(f"most specific level ({conf.specificity:.0%})")
        parts.append(
            f"**Key Decision Factors:** {', '.join(factors) if factors else 'General context match'}"
        )
        parts.append("")

        # Index term matches
        if primary.matched_index_terms:
            parts.append(
                f"**Official Index Terms Matched:** {', '.join(primary.matched_index_terms[:5])}"
            )
        else:
            parts.append("**Official Index Terms Matched:** None (matched via description)")
        parts.append("")

        # Why chosen over alternatives
        if alternatives:
            parts.append("**Why This Over Alternatives:**")
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
                parts.append(
                    f"  - vs {alt.code.node_code} ({alt.code.title}): "
                    f"+{delta:.0%} confidence ({', '.join(reasons)})"
                )
            parts.append("")

        # Cross-reference status
        if check_cross_refs:
            if primary.exclusion_warnings:
                parts.append("**Cross-References Checked:** Yes - WARNINGS FOUND")
            elif primary.relevant_cross_refs:
                parts.append(
                    f"**Cross-References Checked:** Yes - {len(primary.relevant_cross_refs)} references reviewed, no conflicts"
                )
            else:
                parts.append("**Cross-References Checked:** Yes - no applicable exclusions")
        else:
            parts.append(
                "**Cross-References Checked:** No (use check_cross_refs=true for full validation)"
            )
        parts.append("")

        if primary.exclusion_warnings:
            parts.append("**EXCLUSION WARNINGS:**")
            for warning in primary.exclusion_warnings:
                parts.append(f"  - {warning}")

        return "\n".join(parts)
