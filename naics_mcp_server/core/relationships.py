"""
Relationship service for NAICS codes.

Provides access to pre-computed semantic similarity relationships
between NAICS codes, enabling discovery of:
- Same-sector alternatives
- Cross-sector alternatives (codes in different sectors that are semantically similar)
- Relationship statistics
"""

import json
import logging
from typing import TYPE_CHECKING

from ..models.relationships import NAICSRelationship

if TYPE_CHECKING:
    from .database import NAICSDatabase

logger = logging.getLogger(__name__)


class RelationshipService:
    """
    Service for querying pre-computed NAICS code relationships.

    Relationships are pre-computed using FAISS similarity search and stored
    in the naics_relationships table as JSON documents. This service provides
    a clean interface for querying and filtering those relationships.
    """

    def __init__(self, database: "NAICSDatabase"):
        """
        Initialize the relationship service.

        Args:
            database: The NAICS database connection
        """
        self.database = database

    async def get_relationships(self, code: str) -> NAICSRelationship | None:
        """
        Get all relationships for a NAICS code.

        Args:
            code: The NAICS code to get relationships for

        Returns:
            NAICSRelationship object or None if not found
        """
        self.database._ensure_connected()

        try:
            result = self.database.connection.execute(
                """
                SELECT json_data
                FROM naics_relationships
                WHERE node_code = ?
                """,
                [code],
            ).fetchone()

            if not result:
                logger.debug(f"No relationships found for code {code}")
                return None

            data = json.loads(result[0])
            return NAICSRelationship.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to get relationships for {code}: {e}")
            return None

    async def get_similar_codes(
        self,
        code: str,
        min_similarity: float = 0.75,
        include_cross_sector: bool = True,
        limit: int = 10,
    ) -> dict:
        """
        Get pre-computed similar NAICS codes.

        Args:
            code: The NAICS code to find similar codes for
            min_similarity: Minimum similarity threshold (0-1)
            include_cross_sector: Whether to include cross-sector alternatives
            limit: Maximum results per category

        Returns:
            Dict with similar codes information
        """
        relationships = await self.get_relationships(code)

        if not relationships:
            return {
                "error": f"NAICS code {code} not found in relationships",
                "same_sector_alternatives": [],
                "cross_sector_alternatives": [],
            }

        same_sector, cross_sector = relationships.get_filtered_alternatives(
            min_similarity=min_similarity,
            include_cross_sector=include_cross_sector,
            limit=limit,
        )

        return {
            "node_code": code,
            "title": relationships.title,
            "level": relationships.level,
            "sector_code": relationships.sector_code,
            "same_sector_alternatives": [
                {
                    "code": alt.target_code,
                    "title": alt.target_title,
                    "level": alt.target_level,
                    "similarity": alt.similarity_score,
                }
                for alt in same_sector
            ],
            "cross_sector_alternatives": [
                {
                    "code": alt.target_code,
                    "title": alt.target_title,
                    "level": alt.target_level,
                    "similarity": alt.similarity_score,
                    "target_sector": alt.target_sector,
                    "note": alt.relationship_note,
                }
                for alt in cross_sector
            ],
            "stats": {
                "same_sector_count": len(same_sector),
                "cross_sector_count": len(cross_sector),
            },
        }

    async def get_cross_sector_alternatives(
        self,
        code: str,
        min_similarity: float = 0.70,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Get codes in different sectors that are semantically similar.

        This is useful for discovering classification alternatives
        that span sector boundaries.

        Args:
            code: The NAICS code to find alternatives for
            min_similarity: Minimum similarity threshold
            top_n: Maximum results to return

        Returns:
            List of cross-sector alternative dicts
        """
        relationships = await self.get_relationships(code)

        if not relationships:
            return []

        # Filter and limit cross-sector alternatives
        alternatives = [
            alt
            for alt in relationships.cross_sector_alternatives
            if alt.similarity_score >= min_similarity
        ][:top_n]

        return [
            {
                "code": alt.target_code,
                "title": alt.target_title,
                "level": alt.target_level,
                "similarity": alt.similarity_score,
                "target_sector": alt.target_sector,
                "relationship_note": alt.relationship_note,
                "hierarchy": alt.target_hierarchy,
            }
            for alt in alternatives
        ]

    async def get_same_sector_alternatives(
        self,
        code: str,
        min_similarity: float = 0.75,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Get similar codes within the same sector.

        Args:
            code: The NAICS code to find alternatives for
            min_similarity: Minimum similarity threshold
            top_n: Maximum results to return

        Returns:
            List of same-sector alternative dicts
        """
        relationships = await self.get_relationships(code)

        if not relationships:
            return []

        # Filter and limit same-sector alternatives
        alternatives = [
            alt
            for alt in relationships.same_sector_alternatives
            if alt.similarity_score >= min_similarity
        ][:top_n]

        return [
            {
                "code": alt.target_code,
                "title": alt.target_title,
                "level": alt.target_level,
                "similarity": alt.similarity_score,
                "hierarchy": alt.target_hierarchy,
            }
            for alt in alternatives
        ]

    async def get_relationship_stats(self, code: str) -> dict:
        """
        Get statistics about a code's relationships.

        Args:
            code: The NAICS code to get stats for

        Returns:
            Dict with relationship statistics
        """
        relationships = await self.get_relationships(code)

        if not relationships:
            return {"error": f"NAICS code {code} not found in relationships"}

        stats = relationships.relationship_stats

        return {
            "node_code": code,
            "title": relationships.title,
            "level": relationships.level,
            "sector_code": relationships.sector_code,
            "stats": {
                "same_sector_count": stats.same_sector_count,
                "cross_sector_count": stats.cross_sector_count,
                "total_count": stats.total_count,
                "max_similarity": stats.max_similarity,
                "avg_similarity": stats.avg_similarity,
                "has_cross_sector": stats.has_cross_sector,
            },
        }

    async def check_relationships_available(self) -> bool:
        """
        Check if relationship data is available in the database.

        Returns:
            True if relationships table exists and has data
        """
        self.database._ensure_connected()

        try:
            result = self.database.connection.execute(
                "SELECT COUNT(*) FROM naics_relationships"
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    async def get_relationship_statistics(self) -> dict:
        """
        Get aggregate statistics about all relationships in the database.

        Returns:
            Dict with overall relationship statistics
        """
        self.database._ensure_connected()

        try:
            result = self.database.connection.execute("""
                SELECT
                    COUNT(*) as total_codes,
                    AVG(CAST(json_extract(json_data, '$.relationship_stats.same_sector_count') AS DOUBLE)) as avg_same_sector,
                    AVG(CAST(json_extract(json_data, '$.relationship_stats.cross_sector_count') AS DOUBLE)) as avg_cross_sector,
                    SUM(CAST(json_extract(json_data, '$.relationship_stats.cross_sector_count') AS INT) > 0) as codes_with_cross_sector
                FROM naics_relationships
            """).fetchone()

            return {
                "total_codes": result[0],
                "avg_same_sector_per_code": round(result[1], 2) if result[1] else 0,
                "avg_cross_sector_per_code": round(result[2], 2) if result[2] else 0,
                "codes_with_cross_sector": result[3] if result[3] else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get relationship statistics: {e}")
            return {"error": str(e)}
