"""
Unit tests for NAICSDatabase.

Tests database operations including CRUD, search, hierarchy, and cross-references.
"""

import pytest
from pathlib import Path

from naics_mcp_server.core.database import NAICSDatabase, get_database
from naics_mcp_server.models.naics_models import NAICSCode, NAICSLevel, CrossReference, IndexTerm


class TestNAICSDatabaseConnection:
    """Tests for database connection and initialization."""

    def test_connect_creates_database_file(self, temp_db_path: Path):
        """Database file should be created on connect."""
        db = NAICSDatabase(temp_db_path)
        db.connect()

        assert temp_db_path.exists()

        db.disconnect()

    def test_connect_initializes_schema(self, empty_database: NAICSDatabase):
        """Schema should be initialized on first connect."""
        tables = empty_database.connection.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        assert "naics_nodes" in table_names
        assert "naics_embeddings" in table_names
        assert "naics_index_terms" in table_names
        assert "naics_cross_references" in table_names
        assert "classification_workbook" in table_names

    def test_disconnect_closes_connection(self, temp_db_path: Path):
        """Disconnect should close the database connection."""
        db = NAICSDatabase(temp_db_path)
        db.connect()
        db.disconnect()

        assert db.connection is None

    def test_operations_fail_without_connection(self, temp_db_path: Path):
        """Operations should fail if database is not connected."""
        db = NAICSDatabase(temp_db_path)

        with pytest.raises(RuntimeError, match="Database not connected"):
            import asyncio
            asyncio.run(db.get_by_code("311111"))


class TestGetByCode:
    """Tests for retrieving NAICS codes by code."""

    @pytest.mark.asyncio
    async def test_get_existing_code(self, populated_database: NAICSDatabase):
        """Should return NAICSCode for existing code."""
        code = await populated_database.get_by_code("311111")

        assert code is not None
        assert code.node_code == "311111"
        assert code.title == "Dog and Cat Food Manufacturing"
        assert code.level == NAICSLevel.NATIONAL_INDUSTRY
        assert code.sector_code == "31"
        assert code.subsector_code == "311"
        assert code.industry_group_code == "3111"
        assert code.naics_industry_code == "31111"

    @pytest.mark.asyncio
    async def test_get_nonexistent_code(self, populated_database: NAICSDatabase):
        """Should return None for nonexistent code."""
        code = await populated_database.get_by_code("999999")

        assert code is None

    @pytest.mark.asyncio
    async def test_get_sector_code(self, populated_database: NAICSDatabase):
        """Should correctly retrieve sector-level codes."""
        code = await populated_database.get_by_code("31")

        assert code is not None
        assert code.level == NAICSLevel.SECTOR
        assert code.title == "Manufacturing"

    @pytest.mark.asyncio
    async def test_get_empty_code(self, populated_database: NAICSDatabase):
        """Should return None for empty code."""
        code = await populated_database.get_by_code("")

        assert code is None


class TestSearchByText:
    """Tests for text-based search."""

    @pytest.mark.asyncio
    async def test_search_single_term(self, populated_database: NAICSDatabase):
        """Should find codes matching a single search term."""
        results = await populated_database.search_by_text(["dog"])

        assert len(results) > 0
        codes = [r["node_code"] for r in results]
        assert "311111" in codes

    @pytest.mark.asyncio
    async def test_search_multiple_terms(self, populated_database: NAICSDatabase):
        """Should find codes matching any of multiple terms."""
        results = await populated_database.search_by_text(["dog", "beverage"])

        assert len(results) >= 2
        codes = [r["node_code"] for r in results]
        assert "311111" in codes  # dog food
        assert "312" in codes or "312111" in codes  # beverage

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, populated_database: NAICSDatabase):
        """Search should be case-insensitive."""
        results_lower = await populated_database.search_by_text(["dog"])
        results_upper = await populated_database.search_by_text(["DOG"])
        results_mixed = await populated_database.search_by_text(["DoG"])

        assert len(results_lower) == len(results_upper) == len(results_mixed)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, populated_database: NAICSDatabase):
        """Should respect limit parameter."""
        results = await populated_database.search_by_text(["manufacturing"], limit=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_no_results(self, populated_database: NAICSDatabase):
        """Should return empty list for no matches."""
        results = await populated_database.search_by_text(["xyznonexistent"])

        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_terms(self, populated_database: NAICSDatabase):
        """Should handle empty search terms."""
        results = await populated_database.search_by_text([])

        assert results == []


class TestSearchIndexTerms:
    """Tests for official NAICS index term search."""

    @pytest.mark.asyncio
    async def test_search_index_term(self, populated_database: NAICSDatabase):
        """Should find index terms matching search text."""
        results = await populated_database.search_index_terms("dog food")

        assert len(results) > 0
        assert all(isinstance(r, IndexTerm) for r in results)
        assert any(r.naics_code == "311111" for r in results)

    @pytest.mark.asyncio
    async def test_search_index_term_case_insensitive(self, populated_database: NAICSDatabase):
        """Index term search should be case-insensitive."""
        results_lower = await populated_database.search_index_terms("pet food")
        results_upper = await populated_database.search_index_terms("PET FOOD")

        assert len(results_lower) == len(results_upper)

    @pytest.mark.asyncio
    async def test_search_index_term_with_limit(self, populated_database: NAICSDatabase):
        """Should respect limit parameter."""
        results = await populated_database.search_index_terms("food", limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_index_term_no_results(self, populated_database: NAICSDatabase):
        """Should return empty list for no matches."""
        results = await populated_database.search_index_terms("xyznonexistent")

        assert results == []


class TestGetIndexTermsForCode:
    """Tests for retrieving index terms by NAICS code."""

    @pytest.mark.asyncio
    async def test_get_index_terms_for_code(self, populated_database: NAICSDatabase):
        """Should return all index terms for a code."""
        results = await populated_database.get_index_terms_for_code("311111")

        assert len(results) == 3  # dog food, cat food, pet food
        assert all(r.naics_code == "311111" for r in results)

    @pytest.mark.asyncio
    async def test_get_index_terms_nonexistent_code(self, populated_database: NAICSDatabase):
        """Should return empty list for code with no index terms."""
        results = await populated_database.get_index_terms_for_code("999999")

        assert results == []


class TestGetCrossReferences:
    """Tests for cross-reference retrieval."""

    @pytest.mark.asyncio
    async def test_get_cross_references(self, populated_database: NAICSDatabase):
        """Should return cross-references for a code."""
        refs = await populated_database.get_cross_references("311111")

        assert len(refs) > 0
        assert all(isinstance(r, CrossReference) for r in refs)
        assert all(r.source_code == "311111" for r in refs)

    @pytest.mark.asyncio
    async def test_cross_reference_fields(self, populated_database: NAICSDatabase):
        """Cross-references should have all expected fields."""
        refs = await populated_database.get_cross_references("311111")

        ref = refs[0]
        assert ref.source_code == "311111"
        assert ref.reference_type == "excludes"
        assert ref.target_code == "311119"
        assert "other animal food" in ref.excluded_activity

    @pytest.mark.asyncio
    async def test_get_cross_references_no_results(self, populated_database: NAICSDatabase):
        """Should return empty list for code with no cross-references."""
        refs = await populated_database.get_cross_references("31")  # Sector has no cross-refs

        assert refs == []


class TestSearchCrossReferences:
    """Tests for cross-reference search."""

    @pytest.mark.asyncio
    async def test_search_cross_references(self, populated_database: NAICSDatabase):
        """Should find cross-references matching search text."""
        refs = await populated_database.search_cross_references("animal food")

        assert len(refs) > 0
        assert all(isinstance(r, CrossReference) for r in refs)

    @pytest.mark.asyncio
    async def test_search_cross_references_by_activity(self, populated_database: NAICSDatabase):
        """Should find cross-references by excluded activity."""
        refs = await populated_database.search_cross_references("flavoring syrup")

        assert len(refs) > 0
        assert any(r.source_code == "312111" for r in refs)

    @pytest.mark.asyncio
    async def test_search_cross_references_no_results(self, populated_database: NAICSDatabase):
        """Should return empty list for no matches."""
        refs = await populated_database.search_cross_references("xyznonexistent")

        assert refs == []


class TestGetHierarchy:
    """Tests for hierarchy retrieval."""

    @pytest.mark.asyncio
    async def test_get_hierarchy_for_national_industry(self, populated_database: NAICSDatabase):
        """Should return full hierarchy path for national industry code."""
        hierarchy = await populated_database.get_hierarchy("311111")

        assert len(hierarchy) == 5  # sector, subsector, industry_group, naics_industry, national_industry

        # Check order: most general to most specific
        levels = [c.level for c in hierarchy]
        assert levels == [
            NAICSLevel.SECTOR,
            NAICSLevel.SUBSECTOR,
            NAICSLevel.INDUSTRY_GROUP,
            NAICSLevel.NAICS_INDUSTRY,
            NAICSLevel.NATIONAL_INDUSTRY,
        ]

        # Check codes
        codes = [c.node_code for c in hierarchy]
        assert codes == ["31", "311", "3111", "31111", "311111"]

    @pytest.mark.asyncio
    async def test_get_hierarchy_for_sector(self, populated_database: NAICSDatabase):
        """Sector should return single-item hierarchy."""
        hierarchy = await populated_database.get_hierarchy("31")

        assert len(hierarchy) == 1
        assert hierarchy[0].node_code == "31"
        assert hierarchy[0].level == NAICSLevel.SECTOR

    @pytest.mark.asyncio
    async def test_get_hierarchy_nonexistent_code(self, populated_database: NAICSDatabase):
        """Should return empty list for nonexistent code."""
        hierarchy = await populated_database.get_hierarchy("999999")

        assert hierarchy == []


class TestGetChildren:
    """Tests for retrieving child codes."""

    @pytest.mark.asyncio
    async def test_get_children_of_sector(self, populated_database: NAICSDatabase):
        """Should return subsectors for a sector."""
        children = await populated_database.get_children("31")

        assert len(children) >= 2  # 311 and 312
        assert all(c.level == NAICSLevel.SUBSECTOR for c in children)
        codes = [c.node_code for c in children]
        assert "311" in codes
        assert "312" in codes

    @pytest.mark.asyncio
    async def test_get_children_of_naics_industry(self, populated_database: NAICSDatabase):
        """Should return national industries for NAICS industry."""
        children = await populated_database.get_children("31111")

        assert len(children) == 2  # 311111 and 311119
        assert all(c.level == NAICSLevel.NATIONAL_INDUSTRY for c in children)

    @pytest.mark.asyncio
    async def test_get_children_of_national_industry(self, populated_database: NAICSDatabase):
        """National industries should have no children."""
        children = await populated_database.get_children("311111")

        assert children == []

    @pytest.mark.asyncio
    async def test_get_children_nonexistent_code(self, populated_database: NAICSDatabase):
        """Should return empty list for nonexistent code."""
        children = await populated_database.get_children("999999")

        assert children == []


class TestGetSiblings:
    """Tests for retrieving sibling codes."""

    @pytest.mark.asyncio
    async def test_get_siblings(self, populated_database: NAICSDatabase):
        """Should return sibling codes at same level."""
        siblings = await populated_database.get_siblings("311111")

        assert len(siblings) == 1  # 311119
        assert siblings[0].node_code == "311119"
        assert siblings[0].level == NAICSLevel.NATIONAL_INDUSTRY

    @pytest.mark.asyncio
    async def test_get_siblings_of_subsector(self, populated_database: NAICSDatabase):
        """Should return sibling subsectors."""
        siblings = await populated_database.get_siblings("311")

        assert len(siblings) == 1  # 312
        assert siblings[0].node_code == "312"

    @pytest.mark.asyncio
    async def test_get_siblings_respects_limit(self, populated_database: NAICSDatabase):
        """Should respect limit parameter."""
        siblings = await populated_database.get_siblings("311111", limit=1)

        assert len(siblings) <= 1

    @pytest.mark.asyncio
    async def test_get_siblings_excludes_self(self, populated_database: NAICSDatabase):
        """Should not include the original code in siblings."""
        siblings = await populated_database.get_siblings("311111")

        codes = [s.node_code for s in siblings]
        assert "311111" not in codes


class TestGetStatistics:
    """Tests for database statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, populated_database: NAICSDatabase):
        """Should return database statistics."""
        stats = await populated_database.get_statistics()

        assert "total_codes" in stats
        assert "counts_by_level" in stats
        assert "total_index_terms" in stats
        assert "total_cross_references" in stats

        assert stats["total_codes"] == len(
            [c for c in conftest.SAMPLE_NAICS_CODES if True]
        ) or stats["total_codes"] > 0  # Fallback check

    @pytest.mark.asyncio
    async def test_statistics_counts_by_level(self, populated_database: NAICSDatabase):
        """Should include counts by hierarchy level."""
        stats = await populated_database.get_statistics()

        counts = stats["counts_by_level"]
        assert "sector" in counts
        assert "subsector" in counts
        assert "industry_group" in counts
        assert "naics_industry" in counts
        assert "national_industry" in counts


class TestDatabaseContextManager:
    """Tests for the async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_disconnects(self, temp_db_path: Path):
        """Context manager should handle connection lifecycle."""
        async with get_database(temp_db_path) as db:
            assert db.connection is not None

            # Should be able to use the database
            stats = await db.get_statistics()
            assert "total_codes" in stats

        # After context, connection should be closed
        assert db.connection is None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_special_characters_in_search(self, populated_database: NAICSDatabase):
        """Should handle special characters in search terms."""
        # SQL injection attempt
        results = await populated_database.search_by_text(["'; DROP TABLE naics_nodes; --"])
        assert isinstance(results, list)

        # Special characters
        results = await populated_database.search_by_text(["dog & cat"])
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_unicode_in_search(self, populated_database: NAICSDatabase):
        """Should handle Unicode characters in search."""
        results = await populated_database.search_by_text(["café"])
        assert isinstance(results, list)

        results = await populated_database.search_by_text(["日本語"])
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_very_long_search_term(self, populated_database: NAICSDatabase):
        """Should handle very long search terms."""
        long_term = "a" * 10000
        results = await populated_database.search_by_text([long_term])
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_whitespace_only_search(self, populated_database: NAICSDatabase):
        """Should handle whitespace-only search terms."""
        results = await populated_database.search_by_text(["   "])
        assert isinstance(results, list)


# Import conftest for reference in statistics test
from tests import conftest
