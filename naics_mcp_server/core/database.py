"""
Database access layer for NAICS codes.

Clear, purposeful database operations with explicit error handling.
Adapted for NAICS with 5-level hierarchy, cross-references, and index terms.
"""

import duckdb
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import logging

from ..models.naics_models import NAICSCode, NAICSLevel, CrossReference, IndexTerm
from ..models.search_models import SearchStrategy

logger = logging.getLogger(__name__)


# SQL for creating the NAICS schema
CREATE_SCHEMA_SQL = """
-- Primary codes table
CREATE TABLE IF NOT EXISTS naics_nodes (
    node_code VARCHAR PRIMARY KEY,        -- '31', '311', etc.
    level VARCHAR NOT NULL,               -- 'sector' through 'national_industry'
    title VARCHAR NOT NULL,
    description TEXT,

    -- Hierarchy (explicit for fast traversal)
    sector_code VARCHAR,
    subsector_code VARCHAR,
    industry_group_code VARCHAR,
    naics_industry_code VARCHAR,

    -- Embedding source
    raw_embedding_text TEXT,

    -- Metadata
    change_indicator VARCHAR,             -- Change markers from 2017
    is_trilateral BOOLEAN DEFAULT true
);

-- Vector embeddings (384-dim)
CREATE TABLE IF NOT EXISTS naics_embeddings (
    node_code VARCHAR PRIMARY KEY,
    embedding FLOAT[384],
    embedding_text VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Official index terms (20,398 entries)
CREATE TABLE IF NOT EXISTS naics_index_terms (
    term_id INTEGER PRIMARY KEY,
    naics_code VARCHAR NOT NULL,
    index_term VARCHAR NOT NULL,
    term_normalized VARCHAR               -- Lowercase for search
);

-- Cross-references (critical for classification)
CREATE TABLE IF NOT EXISTS naics_cross_references (
    ref_id INTEGER PRIMARY KEY,
    source_code VARCHAR NOT NULL,
    reference_type VARCHAR,               -- 'excludes', 'see_also', 'includes'
    reference_text TEXT NOT NULL,
    target_code VARCHAR,                  -- Parsed target if available
    excluded_activity VARCHAR             -- What activity is excluded
);

-- SIC crosswalk (legacy integration)
CREATE TABLE IF NOT EXISTS sic_naics_crosswalk (
    sic_code VARCHAR NOT NULL,
    naics_code VARCHAR NOT NULL,
    relationship_type VARCHAR,
    PRIMARY KEY (sic_code, naics_code)
);

-- Classification workbook
CREATE TABLE IF NOT EXISTS classification_workbook (
    entry_id VARCHAR PRIMARY KEY,
    form_type VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    content JSON NOT NULL,
    metadata JSON,
    created_at TIMESTAMP NOT NULL,
    session_id VARCHAR,
    parent_entry_id VARCHAR,
    tags JSON,
    search_text TEXT,
    confidence_score FLOAT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_naics_level ON naics_nodes (level);
CREATE INDEX IF NOT EXISTS idx_naics_sector ON naics_nodes (sector_code);
CREATE INDEX IF NOT EXISTS idx_naics_subsector ON naics_nodes (subsector_code);
CREATE INDEX IF NOT EXISTS idx_index_terms_code ON naics_index_terms (naics_code);
CREATE INDEX IF NOT EXISTS idx_index_terms_normalized ON naics_index_terms (term_normalized);
CREATE INDEX IF NOT EXISTS idx_crossref_source ON naics_cross_references (source_code);
CREATE INDEX IF NOT EXISTS idx_crossref_target ON naics_cross_references (target_code);
CREATE INDEX IF NOT EXISTS idx_workbook_session ON classification_workbook (session_id);
CREATE INDEX IF NOT EXISTS idx_workbook_form_type ON classification_workbook (form_type);
"""


class NAICSDatabase:
    """
    Database access for NAICS codes.

    This class manages all database operations with clear methods
    and explicit error handling. Includes NAICS-specific features:
    - 5-level hierarchy
    - Cross-reference lookup
    - Index term search
    - SIC crosswalk
    """

    def __init__(self, database_path: Path, pool_size: int = 5):
        """
        Initialize database connection.

        Args:
            database_path: Path to the DuckDB database file
            pool_size: Number of connections to maintain
        """
        self.database_path = Path(database_path)
        self.pool_size = pool_size
        self.connection = None

    def connect(self) -> None:
        """Establish database connection."""
        try:
            # Ensure parent directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect with read-write access
            self.connection = duckdb.connect(str(self.database_path))
            logger.info(f"Connected to database at {self.database_path}")

            # Initialize schema if needed
            self._initialize_schema()

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _initialize_schema(self) -> None:
        """Initialize database schema if tables don't exist."""
        try:
            tables = self.connection.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            if "naics_nodes" not in table_names:
                logger.info("Initializing NAICS database schema...")
                self.connection.execute(CREATE_SCHEMA_SQL)
                logger.info("Schema initialized successfully")
            else:
                logger.info("Database schema already exists")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def disconnect(self) -> None:
        """Close database connection cleanly."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    async def get_by_code(self, node_code: str) -> Optional[NAICSCode]:
        """
        Retrieve a specific NAICS code.

        Args:
            node_code: The NAICS code to retrieve

        Returns:
            NAICSCode object or None if not found
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            query = """
                SELECT * FROM naics_nodes
                WHERE node_code = ?
                LIMIT 1
            """

            result = self.connection.execute(query, [node_code]).fetchone()

            if result:
                return self._row_to_naics_code(result)
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve code {node_code}: {e}")
            return None

    async def search_by_text(
        self,
        search_terms: List[str],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for NAICS codes by text matching.

        Args:
            search_terms: Terms to search for
            limit: Maximum results to return

        Returns:
            List of matches with relevance scores
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            # Build search condition for multiple terms
            conditions = []
            params = []

            for term in search_terms:
                conditions.append(
                    "(LOWER(title) LIKE ? OR LOWER(description) LIKE ? OR LOWER(raw_embedding_text) LIKE ?)"
                )
                pattern = f"%{term.lower()}%"
                params.extend([pattern, pattern, pattern])

            where_clause = " OR ".join(conditions)

            query = f"""
                SELECT *
                FROM naics_nodes
                WHERE {where_clause}
                LIMIT ?
            """

            params.append(limit)
            results = self.connection.execute(query, params).fetchall()

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def search_index_terms(
        self,
        search_text: str,
        limit: int = 50
    ) -> List[IndexTerm]:
        """
        Search the official NAICS index terms.

        Args:
            search_text: Text to search for in index terms
            limit: Maximum results

        Returns:
            List of matching IndexTerm objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            query = """
                SELECT term_id, naics_code, index_term, term_normalized
                FROM naics_index_terms
                WHERE term_normalized LIKE ?
                ORDER BY LENGTH(index_term)
                LIMIT ?
            """

            pattern = f"%{search_text.lower()}%"
            results = self.connection.execute(query, [pattern, limit]).fetchall()

            return [
                IndexTerm(
                    term_id=row[0],
                    naics_code=row[1],
                    index_term=row[2],
                    term_normalized=row[3]
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Index term search failed: {e}")
            return []

    async def get_index_terms_for_code(self, naics_code: str) -> List[IndexTerm]:
        """
        Get all index terms for a specific NAICS code.

        Args:
            naics_code: The NAICS code

        Returns:
            List of IndexTerm objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            query = """
                SELECT term_id, naics_code, index_term, term_normalized
                FROM naics_index_terms
                WHERE naics_code = ?
                ORDER BY index_term
            """

            results = self.connection.execute(query, [naics_code]).fetchall()

            return [
                IndexTerm(
                    term_id=row[0],
                    naics_code=row[1],
                    index_term=row[2],
                    term_normalized=row[3]
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to get index terms for {naics_code}: {e}")
            return []

    async def get_cross_references(self, source_code: str) -> List[CrossReference]:
        """
        Get cross-references for a NAICS code.

        Cross-references are CRITICAL for accurate classification.
        They tell you what activities are excluded from this code.

        Args:
            source_code: The NAICS code to get cross-refs for

        Returns:
            List of CrossReference objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            query = """
                SELECT ref_id, source_code, reference_type, reference_text,
                       target_code, excluded_activity
                FROM naics_cross_references
                WHERE source_code = ?
                ORDER BY reference_type, ref_id
            """

            results = self.connection.execute(query, [source_code]).fetchall()

            return [
                CrossReference(
                    source_code=row[1],
                    reference_type=row[2],
                    reference_text=row[3],
                    target_code=row[4],
                    excluded_activity=row[5]
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to get cross-refs for {source_code}: {e}")
            return []

    async def search_cross_references(
        self,
        search_text: str,
        limit: int = 20
    ) -> List[CrossReference]:
        """
        Search cross-references by activity text.

        This helps find where a specific activity should be classified.

        Args:
            search_text: Activity description to search for
            limit: Maximum results

        Returns:
            List of matching CrossReference objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            query = """
                SELECT ref_id, source_code, reference_type, reference_text,
                       target_code, excluded_activity
                FROM naics_cross_references
                WHERE LOWER(reference_text) LIKE ?
                   OR LOWER(excluded_activity) LIKE ?
                LIMIT ?
            """

            pattern = f"%{search_text.lower()}%"
            results = self.connection.execute(query, [pattern, pattern, limit]).fetchall()

            return [
                CrossReference(
                    source_code=row[1],
                    reference_type=row[2],
                    reference_text=row[3],
                    target_code=row[4],
                    excluded_activity=row[5]
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Cross-reference search failed: {e}")
            return []

    async def get_hierarchy(self, node_code: str) -> List[NAICSCode]:
        """
        Get the complete hierarchy for a given code.

        Args:
            node_code: The NAICS code to get hierarchy for

        Returns:
            List of NAICSCode objects from sector to specific code
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            # First get the code to understand its hierarchy
            code = await self.get_by_code(node_code)
            if not code:
                return []

            # Get all codes in the hierarchy
            hierarchy_codes = code.get_hierarchy_path()

            if not hierarchy_codes:
                return [code]

            placeholders = ",".join(["?"] * len(hierarchy_codes))
            query = f"""
                SELECT * FROM naics_nodes
                WHERE node_code IN ({placeholders})
                ORDER BY
                    CASE level
                        WHEN 'sector' THEN 1
                        WHEN 'subsector' THEN 2
                        WHEN 'industry_group' THEN 3
                        WHEN 'naics_industry' THEN 4
                        WHEN 'national_industry' THEN 5
                    END
            """

            results = self.connection.execute(query, hierarchy_codes).fetchall()

            return [self._row_to_naics_code(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get hierarchy for {node_code}: {e}")
            return []

    async def get_children(self, parent_code: str) -> List[NAICSCode]:
        """
        Get immediate children of a NAICS code.

        Args:
            parent_code: The parent code

        Returns:
            List of child NAICSCode objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            parent = await self.get_by_code(parent_code)
            if not parent:
                return []

            # Determine which field to query based on parent level
            parent_field_map = {
                NAICSLevel.SECTOR: "sector_code",
                NAICSLevel.SUBSECTOR: "subsector_code",
                NAICSLevel.INDUSTRY_GROUP: "industry_group_code",
                NAICSLevel.NAICS_INDUSTRY: "naics_industry_code",
            }

            child_level_map = {
                NAICSLevel.SECTOR: "subsector",
                NAICSLevel.SUBSECTOR: "industry_group",
                NAICSLevel.INDUSTRY_GROUP: "naics_industry",
                NAICSLevel.NAICS_INDUSTRY: "national_industry",
            }

            parent_field = parent_field_map.get(parent.level)
            child_level = child_level_map.get(parent.level)

            if not parent_field or not child_level:
                return []

            query = f"""
                SELECT * FROM naics_nodes
                WHERE {parent_field} = ?
                AND level = ?
                ORDER BY node_code
            """

            results = self.connection.execute(query, [parent_code, child_level]).fetchall()

            return [self._row_to_naics_code(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get children for {parent_code}: {e}")
            return []

    async def get_siblings(self, node_code: str, limit: int = 10) -> List[NAICSCode]:
        """
        Get sibling codes at the same hierarchical level.

        Args:
            node_code: The NAICS code to find siblings for
            limit: Maximum number of siblings to return

        Returns:
            List of sibling NAICSCode objects
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            code = await self.get_by_code(node_code)
            if not code:
                return []

            # Determine parent based on level
            parent_field = None
            parent_value = None

            if code.level == NAICSLevel.NATIONAL_INDUSTRY and code.naics_industry_code:
                parent_field = "naics_industry_code"
                parent_value = code.naics_industry_code
            elif code.level == NAICSLevel.NAICS_INDUSTRY and code.industry_group_code:
                parent_field = "industry_group_code"
                parent_value = code.industry_group_code
            elif code.level == NAICSLevel.INDUSTRY_GROUP and code.subsector_code:
                parent_field = "subsector_code"
                parent_value = code.subsector_code
            elif code.level == NAICSLevel.SUBSECTOR and code.sector_code:
                parent_field = "sector_code"
                parent_value = code.sector_code

            if not parent_field:
                return []

            query = f"""
                SELECT * FROM naics_nodes
                WHERE level = ?
                AND {parent_field} = ?
                AND node_code != ?
                LIMIT ?
            """

            results = self.connection.execute(
                query,
                [code.level.value, parent_value, node_code, limit]
            ).fetchall()

            return [self._row_to_naics_code(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get siblings for {node_code}: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring and debugging.

        Returns:
            Dictionary of statistics about the database
        """
        if not self.connection:
            raise RuntimeError("Database not connected")

        try:
            stats = {}

            # Total counts by level
            counts = self.connection.execute("""
                SELECT level, COUNT(*) as count
                FROM naics_nodes
                GROUP BY level
                ORDER BY
                    CASE level
                        WHEN 'sector' THEN 1
                        WHEN 'subsector' THEN 2
                        WHEN 'industry_group' THEN 3
                        WHEN 'naics_industry' THEN 4
                        WHEN 'national_industry' THEN 5
                    END
            """).fetchall()

            stats["counts_by_level"] = {level: count for level, count in counts}

            # Total records
            total = self.connection.execute(
                "SELECT COUNT(*) FROM naics_nodes"
            ).fetchone()[0]
            stats["total_codes"] = total

            # Index terms count
            try:
                index_count = self.connection.execute(
                    "SELECT COUNT(*) FROM naics_index_terms"
                ).fetchone()[0]
                stats["total_index_terms"] = index_count
            except:
                stats["total_index_terms"] = 0

            # Cross-references count
            try:
                crossref_count = self.connection.execute(
                    "SELECT COUNT(*) FROM naics_cross_references"
                ).fetchone()[0]
                stats["total_cross_references"] = crossref_count
            except:
                stats["total_cross_references"] = 0

            # Check embedding coverage
            try:
                with_embeddings = self.connection.execute("""
                    SELECT COUNT(*) FROM naics_embeddings
                """).fetchone()[0]

                stats["embedding_coverage"] = {
                    "with_embeddings": with_embeddings,
                    "without_embeddings": total - with_embeddings,
                    "coverage_percent": (with_embeddings / total * 100) if total > 0 else 0
                }
            except:
                stats["embedding_coverage"] = {
                    "with_embeddings": 0,
                    "without_embeddings": total,
                    "coverage_percent": 0
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def _row_to_naics_code(self, row: tuple) -> NAICSCode:
        """
        Convert a database row to a NAICSCode object.

        Expected column order:
        node_code, level, title, description, sector_code, subsector_code,
        industry_group_code, naics_industry_code, raw_embedding_text,
        change_indicator, is_trilateral
        """
        return NAICSCode(
            node_code=row[0],
            level=NAICSLevel(row[1]),
            title=row[2],
            description=row[3],
            sector_code=row[4],
            subsector_code=row[5],
            industry_group_code=row[6],
            naics_industry_code=row[7],
            raw_embedding_text=row[8],
            change_indicator=row[9] if len(row) > 9 else None,
            is_trilateral=row[10] if len(row) > 10 else True
        )

    def _format_results(self, results: List[tuple]) -> List[Dict[str, Any]]:
        """
        Format database results into dictionaries.
        """
        formatted = []

        for row in results:
            formatted.append({
                "node_code": row[0],
                "level": row[1],
                "title": row[2],
                "description": row[3],
                "sector_code": row[4],
                "subsector_code": row[5],
                "industry_group_code": row[6],
                "naics_industry_code": row[7],
                "raw_embedding_text": row[8],
            })

        return formatted


@asynccontextmanager
async def get_database(database_path: Path):
    """
    Context manager for database connections.

    Ensures clean connection management.
    """
    db = NAICSDatabase(database_path)
    try:
        db.connect()
        yield db
    finally:
        db.disconnect()
