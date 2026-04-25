"""
Tests for the NAICS relationship service.

Tests pre-computed semantic similarity relationships between NAICS codes.
"""

import json

import pytest

from naics_mcp_server.core.database import NAICSDatabase
from naics_mcp_server.core.relationships import RelationshipService
from naics_mcp_server.models.relationships import (
    NAICSRelationship,
    RelationshipStats,
    SimilarCode,
)

# --- Model Tests ---


class TestSimilarCode:
    """Tests for SimilarCode dataclass."""

    def test_similar_code_creation(self):
        """Test creating a SimilarCode instance."""
        code = SimilarCode(
            target_code="722511",
            target_title="Full-Service Restaurants",
            target_level="national_industry",
            similarity_score=0.85,
            target_sector="72",
        )
        assert code.target_code == "722511"
        assert code.similarity_score == 0.85

    def test_similar_code_with_hierarchy(self):
        """Test SimilarCode with full hierarchy path."""
        code = SimilarCode(
            target_code="722511",
            target_title="Full-Service Restaurants",
            target_level="national_industry",
            similarity_score=0.85,
            target_hierarchy=["72", "722", "7225", "72251", "722511"],
        )
        assert len(code.target_hierarchy) == 5

    def test_similar_code_str(self):
        """Test string representation."""
        code = SimilarCode(
            target_code="722511",
            target_title="Full-Service Restaurants",
            target_level="national_industry",
            similarity_score=0.85,
        )
        assert "722511" in str(code)
        assert "0.85" in str(code)


class TestRelationshipStats:
    """Tests for RelationshipStats dataclass."""

    def test_stats_creation(self):
        """Test creating RelationshipStats."""
        stats = RelationshipStats(
            same_sector_count=10,
            cross_sector_count=5,
            total_count=15,
            max_similarity=0.92,
            avg_similarity=0.78,
        )
        assert stats.same_sector_count == 10
        assert stats.cross_sector_count == 5

    def test_has_cross_sector_true(self):
        """Test has_cross_sector when cross-sector exists."""
        stats = RelationshipStats(
            same_sector_count=10,
            cross_sector_count=5,
            total_count=15,
            max_similarity=0.92,
            avg_similarity=0.78,
        )
        assert stats.has_cross_sector is True

    def test_has_cross_sector_false(self):
        """Test has_cross_sector when no cross-sector exists."""
        stats = RelationshipStats(
            same_sector_count=10,
            cross_sector_count=0,
            total_count=10,
            max_similarity=0.92,
            avg_similarity=0.78,
        )
        assert stats.has_cross_sector is False


class TestNAICSRelationship:
    """Tests for NAICSRelationship dataclass."""

    @pytest.fixture
    def sample_relationship_data(self):
        """Sample relationship data as dict."""
        return {
            "node_code": "722511",
            "level": "national_industry",
            "title": "Full-Service Restaurants",
            "sector_code": "72",
            "hierarchy_path": ["72", "722", "7225", "72251", "722511"],
            "same_sector_alternatives": [
                {
                    "target_code": "722513",
                    "target_title": "Limited-Service Restaurants",
                    "target_level": "national_industry",
                    "similarity_score": 0.92,
                    "target_sector": "72",
                    "target_hierarchy": ["72", "722", "7225", "72251", "722513"],
                },
                {
                    "target_code": "722514",
                    "target_title": "Cafeterias",
                    "target_level": "national_industry",
                    "similarity_score": 0.88,
                    "target_sector": "72",
                    "target_hierarchy": ["72", "722", "7225", "72251", "722514"],
                },
            ],
            "cross_sector_alternatives": [
                {
                    "target_code": "311811",
                    "target_title": "Retail Bakeries",
                    "target_level": "national_industry",
                    "similarity_score": 0.78,
                    "target_sector": "31",
                    "target_hierarchy": ["31", "311", "3118", "31181", "311811"],
                    "relationship_note": "Cross-sector: 72 -> 31",
                }
            ],
            "relationship_stats": {
                "same_sector_count": 2,
                "cross_sector_count": 1,
                "total_count": 3,
                "max_similarity": 0.92,
                "avg_similarity": 0.86,
            },
        }

    def test_from_dict(self, sample_relationship_data):
        """Test creating NAICSRelationship from dict."""
        rel = NAICSRelationship.from_dict(sample_relationship_data)

        assert rel.node_code == "722511"
        assert rel.title == "Full-Service Restaurants"
        assert rel.sector_code == "72"
        assert len(rel.same_sector_alternatives) == 2
        assert len(rel.cross_sector_alternatives) == 1
        assert rel.relationship_stats.same_sector_count == 2

    def test_to_dict(self, sample_relationship_data):
        """Test converting NAICSRelationship to dict."""
        rel = NAICSRelationship.from_dict(sample_relationship_data)
        result = rel.to_dict()

        assert result["node_code"] == "722511"
        assert len(result["same_sector_alternatives"]) == 2
        assert len(result["cross_sector_alternatives"]) == 1

    def test_get_filtered_alternatives(self, sample_relationship_data):
        """Test filtering alternatives by similarity."""
        rel = NAICSRelationship.from_dict(sample_relationship_data)

        # Filter at 0.90 - should get only the 0.92 match
        same, cross = rel.get_filtered_alternatives(min_similarity=0.90)
        assert len(same) == 1
        assert same[0].similarity_score == 0.92

        # Cross-sector is below 0.90, so should be empty
        assert len(cross) == 0

    def test_get_filtered_alternatives_with_limit(self, sample_relationship_data):
        """Test filtering alternatives with limit."""
        rel = NAICSRelationship.from_dict(sample_relationship_data)

        same, cross = rel.get_filtered_alternatives(min_similarity=0.70, limit=1)
        assert len(same) == 1
        assert len(cross) == 1

    def test_get_filtered_alternatives_exclude_cross_sector(self, sample_relationship_data):
        """Test excluding cross-sector alternatives."""
        rel = NAICSRelationship.from_dict(sample_relationship_data)

        same, cross = rel.get_filtered_alternatives(min_similarity=0.70, include_cross_sector=False)
        assert len(same) == 2
        assert len(cross) == 0


# --- Service Tests ---


class TestRelationshipService:
    """Tests for RelationshipService."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_naics.duckdb"
        db = NAICSDatabase(db_path)
        db.connect()

        # Create the relationships table
        db.connection.execute("""
            CREATE TABLE IF NOT EXISTS naics_relationships (
                node_code VARCHAR PRIMARY KEY,
                json_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        yield db
        db.disconnect()

    @pytest.fixture
    def sample_json_data(self):
        """Sample relationship JSON data."""
        return {
            "node_code": "722511",
            "level": "national_industry",
            "title": "Full-Service Restaurants",
            "sector_code": "72",
            "hierarchy_path": ["72", "722", "7225", "72251", "722511"],
            "same_sector_alternatives": [
                {
                    "target_code": "722513",
                    "target_title": "Limited-Service Restaurants",
                    "target_level": "national_industry",
                    "similarity_score": 0.92,
                    "target_sector": "72",
                    "target_hierarchy": ["72", "722", "7225", "72251", "722513"],
                }
            ],
            "cross_sector_alternatives": [
                {
                    "target_code": "311811",
                    "target_title": "Retail Bakeries",
                    "target_level": "national_industry",
                    "similarity_score": 0.78,
                    "target_sector": "31",
                    "target_hierarchy": ["31", "311", "3118", "31181", "311811"],
                    "relationship_note": "Cross-sector: 72 -> 31",
                }
            ],
            "relationship_stats": {
                "same_sector_count": 1,
                "cross_sector_count": 1,
                "total_count": 2,
                "max_similarity": 0.92,
                "avg_similarity": 0.85,
            },
        }

    @pytest.mark.asyncio
    async def test_check_relationships_available_empty(self, temp_db):
        """Test checking for relationships when none exist."""
        service = RelationshipService(temp_db)
        result = await service.check_relationships_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_relationships_available_with_data(self, temp_db, sample_json_data):
        """Test checking for relationships when data exists."""
        # Insert test data
        temp_db.connection.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            ["722511", json.dumps(sample_json_data)],
        )

        service = RelationshipService(temp_db)
        result = await service.check_relationships_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_relationships(self, temp_db, sample_json_data):
        """Test getting relationships for a code."""
        temp_db.connection.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            ["722511", json.dumps(sample_json_data)],
        )

        service = RelationshipService(temp_db)
        result = await service.get_relationships("722511")

        assert result is not None
        assert result.node_code == "722511"
        assert len(result.same_sector_alternatives) == 1
        assert len(result.cross_sector_alternatives) == 1

    @pytest.mark.asyncio
    async def test_get_relationships_not_found(self, temp_db):
        """Test getting relationships for non-existent code."""
        service = RelationshipService(temp_db)
        result = await service.get_relationships("999999")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_similar_codes(self, temp_db, sample_json_data):
        """Test getting similar codes."""
        temp_db.connection.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            ["722511", json.dumps(sample_json_data)],
        )

        service = RelationshipService(temp_db)
        result = await service.get_similar_codes("722511", min_similarity=0.75)

        assert "node_code" in result
        assert result["node_code"] == "722511"
        assert len(result["same_sector_alternatives"]) == 1
        assert result["same_sector_alternatives"][0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_get_cross_sector_alternatives(self, temp_db, sample_json_data):
        """Test getting cross-sector alternatives."""
        temp_db.connection.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            ["722511", json.dumps(sample_json_data)],
        )

        service = RelationshipService(temp_db)
        result = await service.get_cross_sector_alternatives("722511", min_similarity=0.70)

        assert len(result) == 1
        assert result[0]["code"] == "311811"
        assert result[0]["similarity"] == 0.78

    @pytest.mark.asyncio
    async def test_get_relationship_stats(self, temp_db, sample_json_data):
        """Test getting relationship statistics."""
        temp_db.connection.execute(
            "INSERT INTO naics_relationships (node_code, json_data) VALUES (?, ?)",
            ["722511", json.dumps(sample_json_data)],
        )

        service = RelationshipService(temp_db)
        result = await service.get_relationship_stats("722511")

        assert result["node_code"] == "722511"
        assert result["stats"]["same_sector_count"] == 1
        assert result["stats"]["cross_sector_count"] == 1
        assert result["stats"]["has_cross_sector"] is True
