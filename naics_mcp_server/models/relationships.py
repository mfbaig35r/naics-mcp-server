"""
Domain models for NAICS code relationships.

These models represent pre-computed semantic similarity relationships
between NAICS codes, enabling:
- Cross-sector alternative discovery
- Same-sector alternative discovery
- Relationship statistics and analysis
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SimilarCode:
    """
    A semantically similar NAICS code.

    Represents a single alternative code with similarity information.
    """

    target_code: str  # The similar NAICS code
    target_title: str  # Title of the similar code
    target_level: str  # Level (sector, subsector, etc.)
    similarity_score: float  # Cosine similarity [0, 1]
    target_sector: str | None = None  # Sector of the target code
    target_hierarchy: list[str] = field(default_factory=list)  # Full hierarchy path
    relationship_note: str | None = None  # E.g., "Cross-sector: 31 -> 72"

    def __str__(self) -> str:
        return f"[{self.target_code}] {self.target_title} ({self.similarity_score:.2f})"


@dataclass
class RelationshipStats:
    """
    Statistics about a code's relationships.

    Provides aggregate metrics for relationship analysis.
    """

    same_sector_count: int  # Number of same-sector alternatives
    cross_sector_count: int  # Number of cross-sector alternatives
    total_count: int  # Total alternatives
    max_similarity: float  # Highest similarity score
    avg_similarity: float  # Average similarity score

    @property
    def has_cross_sector(self) -> bool:
        """Whether this code has any cross-sector relationships."""
        return self.cross_sector_count > 0


@dataclass
class NAICSRelationship:
    """
    Complete relationship document for a NAICS code.

    Contains all pre-computed similarity relationships and metadata.
    This is the primary data structure returned by relationship queries.
    """

    node_code: str  # The NAICS code
    level: str  # Hierarchical level
    title: str  # Code title
    sector_code: str | None  # 2-digit sector
    hierarchy_path: list[str]  # Full hierarchy [sector, subsector, ...]
    same_sector_alternatives: list[SimilarCode]  # Alternatives in same sector
    cross_sector_alternatives: list[SimilarCode]  # Alternatives in other sectors
    relationship_stats: RelationshipStats  # Aggregate statistics
    created_at: datetime | None = None  # When relationships were computed

    @classmethod
    def from_dict(cls, data: dict) -> "NAICSRelationship":
        """Create from JSON/dict representation (from database)."""
        same_sector = [
            SimilarCode(
                target_code=alt["target_code"],
                target_title=alt["target_title"],
                target_level=alt["target_level"],
                similarity_score=alt["similarity_score"],
                target_sector=alt.get("target_sector"),
                target_hierarchy=alt.get("target_hierarchy", []),
                relationship_note=alt.get("relationship_note"),
            )
            for alt in data.get("same_sector_alternatives", [])
        ]

        cross_sector = [
            SimilarCode(
                target_code=alt["target_code"],
                target_title=alt["target_title"],
                target_level=alt["target_level"],
                similarity_score=alt["similarity_score"],
                target_sector=alt.get("target_sector"),
                target_hierarchy=alt.get("target_hierarchy", []),
                relationship_note=alt.get("relationship_note"),
            )
            for alt in data.get("cross_sector_alternatives", [])
        ]

        stats_data = data.get("relationship_stats", {})
        stats = RelationshipStats(
            same_sector_count=stats_data.get("same_sector_count", 0),
            cross_sector_count=stats_data.get("cross_sector_count", 0),
            total_count=stats_data.get("total_count", 0),
            max_similarity=stats_data.get("max_similarity", 0.0),
            avg_similarity=stats_data.get("avg_similarity", 0.0),
        )

        return cls(
            node_code=data["node_code"],
            level=data["level"],
            title=data["title"],
            sector_code=data.get("sector_code"),
            hierarchy_path=data.get("hierarchy_path", []),
            same_sector_alternatives=same_sector,
            cross_sector_alternatives=cross_sector,
            relationship_stats=stats,
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "node_code": self.node_code,
            "level": self.level,
            "title": self.title,
            "sector_code": self.sector_code,
            "hierarchy_path": self.hierarchy_path,
            "same_sector_alternatives": [
                {
                    "target_code": alt.target_code,
                    "target_title": alt.target_title,
                    "target_level": alt.target_level,
                    "similarity_score": alt.similarity_score,
                    "target_sector": alt.target_sector,
                    "target_hierarchy": alt.target_hierarchy,
                    "relationship_note": alt.relationship_note,
                }
                for alt in self.same_sector_alternatives
            ],
            "cross_sector_alternatives": [
                {
                    "target_code": alt.target_code,
                    "target_title": alt.target_title,
                    "target_level": alt.target_level,
                    "similarity_score": alt.similarity_score,
                    "target_sector": alt.target_sector,
                    "target_hierarchy": alt.target_hierarchy,
                    "relationship_note": alt.relationship_note,
                }
                for alt in self.cross_sector_alternatives
            ],
            "relationship_stats": {
                "same_sector_count": self.relationship_stats.same_sector_count,
                "cross_sector_count": self.relationship_stats.cross_sector_count,
                "total_count": self.relationship_stats.total_count,
                "max_similarity": self.relationship_stats.max_similarity,
                "avg_similarity": self.relationship_stats.avg_similarity,
            },
        }

    def get_filtered_alternatives(
        self,
        min_similarity: float = 0.75,
        include_cross_sector: bool = True,
        limit: int | None = None,
    ) -> tuple[list[SimilarCode], list[SimilarCode]]:
        """
        Get alternatives filtered by similarity threshold.

        Args:
            min_similarity: Minimum similarity score (0-1)
            include_cross_sector: Whether to include cross-sector alternatives
            limit: Maximum alternatives per category (None = no limit)

        Returns:
            Tuple of (same_sector_alternatives, cross_sector_alternatives)
        """
        same_sector = [
            alt for alt in self.same_sector_alternatives if alt.similarity_score >= min_similarity
        ]
        if limit:
            same_sector = same_sector[:limit]

        cross_sector = []
        if include_cross_sector:
            cross_sector = [
                alt
                for alt in self.cross_sector_alternatives
                if alt.similarity_score >= min_similarity
            ]
            if limit:
                cross_sector = cross_sector[:limit]

        return same_sector, cross_sector

    def __str__(self) -> str:
        return (
            f"NAICSRelationship({self.node_code}: "
            f"{self.relationship_stats.same_sector_count} same-sector, "
            f"{self.relationship_stats.cross_sector_count} cross-sector)"
        )
