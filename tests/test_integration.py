"""
Integration tests for NAICS MCP Server.

Tests complete workflows combining multiple components.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from naics_mcp_server.config import SearchConfig
from naics_mcp_server.core.classification_workbook import ClassificationWorkbook, FormType
from naics_mcp_server.core.database import NAICSDatabase
from naics_mcp_server.core.embeddings import TextEmbedder
from naics_mcp_server.core.search_engine import NAICSSearchEngine
from naics_mcp_server.models.naics_models import NAICSLevel
from naics_mcp_server.models.search_models import SearchStrategy


class TestClassificationWorkflow:
    """Integration tests for the complete classification workflow."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for tests."""
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        embedder.embed_batch.return_value = np.random.rand(10, 384).astype(np.float32)
        return embedder

    @pytest.fixture
    def search_engine(self, populated_database: NAICSDatabase, mock_embedder):
        """Create search engine with populated database."""
        config = SearchConfig()
        engine = NAICSSearchEngine(
            database=populated_database, embedder=mock_embedder, config=config
        )
        engine.embeddings_ready = False  # Use lexical search
        return engine

    @pytest.fixture
    def workbook(self, populated_database: NAICSDatabase) -> ClassificationWorkbook:
        """Create workbook with populated database."""
        return ClassificationWorkbook(populated_database)

    @pytest.mark.asyncio
    async def test_search_and_document_workflow(
        self, search_engine: NAICSSearchEngine, workbook: ClassificationWorkbook
    ):
        """Test: Search for codes, then document the classification decision."""
        # Step 1: Search for matching NAICS codes
        # Use lower min_confidence for test data with simpler lexical matching
        search_results = await search_engine.search(
            query="dog food",
            strategy=SearchStrategy.LEXICAL,
            limit=5,
            min_confidence=0.1,  # Lower threshold for test data
        )

        assert len(search_results.matches) > 0

        # Step 2: Get the top result
        top_match = search_results.matches[0]
        assert top_match.code.node_code == "311111"

        # Step 3: Document the classification decision
        entry = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Dog Food Business Classification",
            content={
                "business_description": "Company manufactures premium dog food",
                "search_query": "dog food manufacturing",
                "candidate_codes": [
                    {
                        "code": match.code.node_code,
                        "title": match.code.title,
                        "confidence": match.confidence.overall,
                    }
                    for match in search_results.matches[:3]
                ],
                "decision": {
                    "selected_code": top_match.code.node_code,
                    "reasoning": f"Best match with {top_match.confidence.overall:.0%} confidence",
                },
            },
            confidence_score=top_match.confidence.overall,
        )

        # Step 4: Verify the entry was created
        retrieved = await workbook.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.content["decision"]["selected_code"] == "311111"

    @pytest.mark.asyncio
    async def test_hierarchy_exploration_workflow(
        self, populated_database: NAICSDatabase, workbook: ClassificationWorkbook
    ):
        """Test: Explore hierarchy from sector to specific code."""
        # Step 1: Start at sector level
        sector = await populated_database.get_by_code("31")
        assert sector is not None
        assert sector.level == NAICSLevel.SECTOR

        # Step 2: Get subsectors
        subsectors = await populated_database.get_children("31")
        assert len(subsectors) >= 2  # 311 and 312

        # Step 3: Drill down to national industry
        national_industry = await populated_database.get_by_code("311111")
        assert national_industry is not None
        assert national_industry.level == NAICSLevel.NATIONAL_INDUSTRY

        # Step 4: Get complete hierarchy
        hierarchy = await populated_database.get_hierarchy("311111")
        assert (
            len(hierarchy) == 5
        )  # sector → subsector → industry_group → naics_industry → national_industry

        # Step 5: Document the exploration
        entry = await workbook.create_entry(
            form_type=FormType.DECISION_TREE,
            label="Manufacturing Hierarchy Exploration",
            content={
                "initial_question": "What type of manufacturing?",
                "decision_points": [
                    {"question": "Sector?", "answer": "31 - Manufacturing"},
                    {"question": "Subsector?", "answer": "311 - Food Manufacturing"},
                    {"question": "Industry Group?", "answer": "3111 - Animal Food"},
                    {"question": "NAICS Industry?", "answer": "31111 - Animal Food Mfg"},
                    {"question": "National Industry?", "answer": "311111 - Dog and Cat Food"},
                ],
                "final_classification": {"code": "311111", "title": national_industry.title},
            },
        )

        assert entry is not None

    @pytest.mark.asyncio
    async def test_cross_reference_check_workflow(
        self, populated_database: NAICSDatabase, workbook: ClassificationWorkbook
    ):
        """Test: Check cross-references for exclusions."""
        # Step 1: Get cross-references for a code
        cross_refs = await populated_database.get_cross_references("311111")
        assert len(cross_refs) > 0

        # Step 2: Verify exclusion details
        exclusion = cross_refs[0]
        assert exclusion.reference_type == "excludes"
        assert exclusion.target_code == "311119"

        # Step 3: Document the cross-reference check
        entry = await workbook.create_entry(
            form_type=FormType.CROSS_REFERENCE_NOTES,
            label="311111 Cross-Reference Check",
            content={
                "source_code": "311111",
                "source_title": "Dog and Cat Food Manufacturing",
                "activity_in_question": "Other animal food",
                "exclusions_found": [
                    {
                        "excluded_activity": exclusion.excluded_activity,
                        "target_code": exclusion.target_code,
                        "reference_text": exclusion.reference_text,
                    }
                ],
                "conclusion": "Other animal food excluded, classified under 311119",
            },
        )

        assert entry is not None

    @pytest.mark.asyncio
    async def test_index_term_search_workflow(
        self, populated_database: NAICSDatabase, search_engine: NAICSSearchEngine
    ):
        """Test: Search using official index terms."""
        # Step 1: Search index terms
        index_terms = await populated_database.search_index_terms("pet food")
        assert len(index_terms) > 0

        # Step 2: Verify we found the right code
        codes_found = {term.naics_code for term in index_terms}
        assert "311111" in codes_found

        # Step 3: Get index terms for a specific code
        code_terms = await populated_database.get_index_terms_for_code("311111")
        assert len(code_terms) >= 2  # dog food, cat food, pet food

        term_texts = [t.index_term.lower() for t in code_terms]
        assert any("dog" in t for t in term_texts)
        assert any("cat" in t for t in term_texts)

    @pytest.mark.asyncio
    async def test_sibling_comparison_workflow(
        self, populated_database: NAICSDatabase, workbook: ClassificationWorkbook
    ):
        """Test: Compare sibling codes for classification decision."""
        # Step 1: Get code and its siblings
        code = await populated_database.get_by_code("311111")
        siblings = await populated_database.get_siblings("311111")

        assert len(siblings) >= 1  # 311119

        # Step 2: Document the comparison
        comparison_data = {
            "comparison_title": "Animal Food Manufacturing Comparison",
            "codes_compared": [
                {"code": code.node_code, "title": code.title, "description": code.description}
            ]
            + [
                {"code": s.node_code, "title": s.title, "description": s.description}
                for s in siblings
            ],
            "analysis": {
                "key_differentiators": [
                    "311111: Specifically dog and cat food",
                    "311119: All other animal food (livestock, poultry)",
                ],
                "recommendation": "Use 311111 for pets, 311119 for farm animals",
            },
        }

        entry = await workbook.create_entry(
            form_type=FormType.INDUSTRY_COMPARISON,
            label="Animal Food Codes Comparison",
            content=comparison_data,
        )

        assert entry is not None
        assert len(entry.content["codes_compared"]) >= 2


class TestDatabaseIntegrity:
    """Tests for database integrity and relationships."""

    @pytest.mark.asyncio
    async def test_hierarchy_consistency(self, populated_database: NAICSDatabase):
        """Verify hierarchy relationships are consistent."""
        # Get a 6-digit code
        code = await populated_database.get_by_code("311111")
        assert code is not None

        # Verify parent codes exist and match
        if code.naics_industry_code:
            parent = await populated_database.get_by_code(code.naics_industry_code)
            assert parent is not None
            assert parent.level == NAICSLevel.NAICS_INDUSTRY

        if code.industry_group_code:
            parent = await populated_database.get_by_code(code.industry_group_code)
            assert parent is not None
            assert parent.level == NAICSLevel.INDUSTRY_GROUP

        if code.subsector_code:
            parent = await populated_database.get_by_code(code.subsector_code)
            assert parent is not None
            assert parent.level == NAICSLevel.SUBSECTOR

        if code.sector_code:
            parent = await populated_database.get_by_code(code.sector_code)
            assert parent is not None
            assert parent.level == NAICSLevel.SECTOR

    @pytest.mark.asyncio
    async def test_cross_reference_targets_exist(self, populated_database: NAICSDatabase):
        """Verify cross-reference targets exist in database."""
        cross_refs = await populated_database.get_cross_references("311111")

        for ref in cross_refs:
            if ref.target_code:
                target = await populated_database.get_by_code(ref.target_code)
                assert target is not None, f"Cross-ref target {ref.target_code} not found"

    @pytest.mark.asyncio
    async def test_index_terms_reference_valid_codes(self, populated_database: NAICSDatabase):
        """Verify index terms reference existing codes."""
        terms = await populated_database.search_index_terms("food")

        for term in terms:
            code = await populated_database.get_by_code(term.naics_code)
            assert code is not None, f"Index term references invalid code {term.naics_code}"


class TestSearchConsistency:
    """Tests for search consistency and accuracy."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        return embedder

    @pytest.fixture
    def search_engine(self, populated_database: NAICSDatabase, mock_embedder):
        config = SearchConfig()
        engine = NAICSSearchEngine(
            database=populated_database, embedder=mock_embedder, config=config
        )
        engine.embeddings_ready = False
        return engine

    @pytest.mark.asyncio
    async def test_search_results_are_deterministic(self, search_engine: NAICSSearchEngine):
        """Same query should return same results."""
        results1 = await search_engine.search(
            query="dog food", strategy=SearchStrategy.LEXICAL, limit=5
        )

        # Clear cache to ensure fresh search
        search_engine.search_cache.clear()

        results2 = await search_engine.search(
            query="dog food", strategy=SearchStrategy.LEXICAL, limit=5
        )

        assert len(results1.matches) == len(results2.matches)
        for i in range(len(results1.matches)):
            assert results1.matches[i].code.node_code == results2.matches[i].code.node_code

    @pytest.mark.asyncio
    async def test_search_ranking_consistency(self, search_engine: NAICSSearchEngine):
        """Higher confidence matches should always rank higher."""
        results = await search_engine.search(
            query="manufacturing food dog", strategy=SearchStrategy.LEXICAL, limit=10
        )

        if len(results.matches) >= 2:
            confidences = [m.confidence.overall for m in results.matches]
            # Verify descending order
            assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_search_cache_consistency(self, search_engine: NAICSSearchEngine):
        """Cached results should match fresh results."""
        # First search (cache miss)
        results1 = await search_engine.search(
            query="beverage manufacturing", strategy=SearchStrategy.LEXICAL, limit=5
        )

        # Second search (cache hit)
        results2 = await search_engine.search(
            query="beverage manufacturing", strategy=SearchStrategy.LEXICAL, limit=5
        )

        assert len(results1.matches) == len(results2.matches)
        for i in range(len(results1.matches)):
            assert results1.matches[i].code.node_code == results2.matches[i].code.node_code


class TestWorkbookPersistence:
    """Tests for workbook data persistence."""

    @pytest.mark.asyncio
    async def test_entries_persist_across_instances(self, populated_database: NAICSDatabase):
        """Entries created by one workbook instance should be readable by another."""
        workbook1 = ClassificationWorkbook(populated_database)

        entry = await workbook1.create_entry(
            form_type=FormType.CUSTOM,
            label="Persistent Entry",
            content={"data": "test persistence"},
        )

        # Create new workbook instance (same database)
        workbook2 = ClassificationWorkbook(populated_database)

        # Should be able to retrieve entry
        retrieved = await workbook2.get_entry(entry.entry_id)

        assert retrieved is not None
        assert retrieved.label == "Persistent Entry"
        assert retrieved.content["data"] == "test persistence"

    @pytest.mark.asyncio
    async def test_complex_content_roundtrip(self, populated_database: NAICSDatabase):
        """Complex nested content should survive storage and retrieval."""
        workbook = ClassificationWorkbook(populated_database)

        complex_content = {
            "nested": {"deeply": {"nested": {"value": 12345}}},
            "array": [1, 2, 3, {"key": "value"}],
            "unicode": "Ünïcödé tëst 日本語",
            "special": "Quotes 'single' and \"double\"",
            "boolean": True,
            "null": None,
            "float": 3.14159,
        }

        entry = await workbook.create_entry(
            form_type=FormType.CUSTOM, label="Complex Content Test", content=complex_content
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert retrieved.content["nested"]["deeply"]["nested"]["value"] == 12345
        assert retrieved.content["array"][3]["key"] == "value"
        assert retrieved.content["unicode"] == "Ünïcödé tëst 日本語"
        assert retrieved.content["boolean"] is True
        assert retrieved.content["null"] is None
        assert abs(retrieved.content["float"] - 3.14159) < 0.0001


class TestEndToEndScenarios:
    """End-to-end test scenarios."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        return embedder

    @pytest.mark.asyncio
    async def test_full_classification_session(
        self, populated_database: NAICSDatabase, mock_embedder
    ):
        """Test a complete classification session from start to finish."""
        # Initialize components
        search_engine = NAICSSearchEngine(
            database=populated_database, embedder=mock_embedder, config=SearchConfig()
        )
        search_engine.embeddings_ready = False

        workbook = ClassificationWorkbook(populated_database)

        # Business description
        business_description = "Company manufactures premium dog and cat food products"

        # Step 1: Initial search
        # Use simpler query matching sample data and lower threshold
        search_results = await search_engine.search(
            query="dog food",  # Simpler query for test data
            strategy=SearchStrategy.LEXICAL,
            limit=5,
            min_confidence=0.1,  # Lower threshold for test data
        )

        assert len(search_results.matches) > 0
        top_code = search_results.matches[0].code

        # Step 2: Create business profile
        profile = await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Pet Food Company Profile",
            content={
                "business_name": "Premium Pet Foods Inc.",
                "business_description": business_description,
                "primary_products_services": ["Dog food", "Cat food"],
                "primary_customers": "Pet retailers",
            },
        )

        # Step 3: Get hierarchy context
        hierarchy = await populated_database.get_hierarchy(top_code.node_code)
        assert len(hierarchy) > 0

        # Step 4: Check cross-references
        cross_refs = await populated_database.get_cross_references(top_code.node_code)

        # Step 5: Get siblings for comparison
        siblings = await populated_database.get_siblings(top_code.node_code)

        # Step 6: Document classification decision
        classification = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Pet Food Classification Analysis",
            content={
                "business_description": business_description,
                "candidate_codes": [
                    {
                        "code": m.code.node_code,
                        "title": m.code.title,
                        "confidence": m.confidence.overall,
                    }
                    for m in search_results.matches[:3]
                ],
                "hierarchy_context": [{"code": h.node_code, "title": h.title} for h in hierarchy],
                "cross_reference_check": {
                    "exclusions_reviewed": len(cross_refs),
                    "relevant_exclusions": [
                        {"activity": ref.excluded_activity, "target": ref.target_code}
                        for ref in cross_refs
                    ],
                },
                "alternatives_considered": [
                    {"code": s.node_code, "title": s.title} for s in siblings
                ],
                "decision": {
                    "selected_code": top_code.node_code,
                    "reasoning": "Primary activity is dog and cat food manufacturing",
                },
            },
            parent_entry_id=profile.entry_id,
            confidence_score=search_results.matches[0].confidence.overall,
        )

        # Verify complete session
        assert profile is not None
        assert classification is not None
        assert classification.parent_entry_id == profile.entry_id

        # Verify we can retrieve all related entries
        session_entries = await workbook.search_entries(session_id=workbook.current_session_id)
        assert len(session_entries) >= 2

        # Verify parent-child relationship
        children = await workbook.search_entries(parent_entry_id=profile.entry_id)
        assert len(children) == 1
        assert children[0].entry_id == classification.entry_id
