"""
Unit tests for ClassificationWorkbook.

Tests workbook operations including entry creation, retrieval, search, and templates.
"""

from datetime import datetime

import pytest

from naics_mcp_server.core.classification_workbook import (
    FORM_TEMPLATES,
    ClassificationWorkbook,
    FormType,
    WorkbookEntry,
)
from naics_mcp_server.core.database import NAICSDatabase


class TestWorkbookEntry:
    """Tests for WorkbookEntry dataclass."""

    def test_entry_creation(self):
        """Should create entry with required fields."""
        entry = WorkbookEntry(
            entry_id="test123",
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Test Classification",
            content={"business_description": "Dog food manufacturer"},
        )

        assert entry.entry_id == "test123"
        assert entry.form_type == FormType.CLASSIFICATION_ANALYSIS
        assert entry.label == "Test Classification"
        assert entry.content["business_description"] == "Dog food manufacturer"

    def test_entry_default_values(self):
        """Should have correct default values."""
        entry = WorkbookEntry(
            entry_id="test123", form_type=FormType.CUSTOM, label="Test", content={}
        )

        assert entry.metadata == {}
        assert entry.session_id is None
        assert entry.parent_entry_id is None
        assert entry.tags == []
        assert entry.search_text is None
        assert entry.confidence_score is None
        assert isinstance(entry.created_at, datetime)

    def test_entry_to_dict(self):
        """to_dict should serialize all fields."""
        entry = WorkbookEntry(
            entry_id="test123",
            form_type=FormType.BUSINESS_PROFILE,
            label="Test Business",
            content={"business_name": "Test Corp"},
            metadata={"source": "manual"},
            session_id="sess_abc",
            tags=["retail", "small-business"],
            confidence_score=0.85,
        )

        result = entry.to_dict()

        assert result["entry_id"] == "test123"
        assert result["form_type"] == "business_profile"
        assert result["label"] == "Test Business"
        assert result["content"]["business_name"] == "Test Corp"
        assert result["metadata"]["source"] == "manual"
        assert result["session_id"] == "sess_abc"
        assert result["tags"] == ["retail", "small-business"]
        assert result["confidence_score"] == 0.85
        assert "created_at" in result


class TestFormType:
    """Tests for FormType enum."""

    def test_all_form_types_have_templates(self):
        """All form types (except CUSTOM) should have templates."""
        for form_type in FormType:
            if form_type != FormType.CUSTOM:
                assert form_type in FORM_TEMPLATES, f"Missing template for {form_type}"

    def test_classification_analysis_template(self):
        """Classification analysis template should have required fields."""
        template = FORM_TEMPLATES[FormType.CLASSIFICATION_ANALYSIS]

        assert "business_description" in template
        assert "candidate_codes" in template
        assert "cross_reference_check" in template
        assert "decision" in template

    def test_business_profile_template(self):
        """Business profile template should have required fields."""
        template = FORM_TEMPLATES[FormType.BUSINESS_PROFILE]

        assert "business_name" in template
        assert "business_description" in template
        assert "primary_products_services" in template
        assert "naics_analysis" in template


class TestClassificationWorkbook:
    """Tests for ClassificationWorkbook service."""

    @pytest.fixture
    def workbook(self, populated_database: NAICSDatabase) -> ClassificationWorkbook:
        """Create workbook instance with populated database."""
        return ClassificationWorkbook(populated_database)

    def test_initialization(self, workbook: ClassificationWorkbook):
        """Should initialize with session ID."""
        assert workbook.current_session_id is not None
        assert len(workbook.current_session_id) == 12  # MD5 hash truncated

    @pytest.mark.asyncio
    async def test_create_entry(self, workbook: ClassificationWorkbook):
        """Should create and store workbook entry."""
        entry = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Dog Food Analysis",
            content={
                "business_description": "Company manufactures dog food",
                "primary_activity": "Dog food manufacturing",
            },
            tags=["pet-food", "manufacturing"],
            confidence_score=0.85,
        )

        assert entry is not None
        assert entry.entry_id is not None
        assert entry.form_type == FormType.CLASSIFICATION_ANALYSIS
        assert entry.label == "Dog Food Analysis"
        assert entry.session_id == workbook.current_session_id
        assert "pet-food" in entry.tags
        assert entry.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_get_entry(self, workbook: ClassificationWorkbook):
        """Should retrieve created entry."""
        created = await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Test Business",
            content={"business_name": "Test Corp"},
        )

        retrieved = await workbook.get_entry(created.entry_id)

        assert retrieved is not None
        assert retrieved.entry_id == created.entry_id
        assert retrieved.label == "Test Business"
        assert retrieved.content["business_name"] == "Test Corp"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entry(self, workbook: ClassificationWorkbook):
        """Should return None for nonexistent entry."""
        result = await workbook.get_entry("nonexistent_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_search_by_form_type(self, workbook: ClassificationWorkbook):
        """Should find entries by form type."""
        # Create entries of different types
        await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Classification 1",
            content={"test": "data"},
        )

        await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE, label="Business 1", content={"test": "data"}
        )

        await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Classification 2",
            content={"test": "data"},
        )

        results = await workbook.search_entries(form_type=FormType.CLASSIFICATION_ANALYSIS)

        assert len(results) == 2
        assert all(e.form_type == FormType.CLASSIFICATION_ANALYSIS for e in results)

    @pytest.mark.asyncio
    async def test_search_by_session(self, workbook: ClassificationWorkbook):
        """Should find entries by session ID."""
        await workbook.create_entry(form_type=FormType.CUSTOM, label="Session Entry", content={})

        results = await workbook.search_entries(session_id=workbook.current_session_id)

        assert len(results) >= 1
        assert all(e.session_id == workbook.current_session_id for e in results)

    @pytest.mark.asyncio
    async def test_search_by_text(self, workbook: ClassificationWorkbook):
        """Should find entries by search text."""
        await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Dog Food Classification",
            content={"business_description": "Manufacturing premium dog food products"},
        )

        await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Soft Drink Classification",
            content={"business_description": "Bottling carbonated beverages"},
        )

        results = await workbook.search_entries(search_text="dog food")

        assert len(results) >= 1
        assert any("Dog Food" in e.label for e in results)

    @pytest.mark.asyncio
    async def test_search_by_tags(self, workbook: ClassificationWorkbook):
        """Should find entries by tags."""
        await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Manufacturing Business",
            content={},
            tags=["manufacturing", "food"],
        )

        await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Retail Business",
            content={},
            tags=["retail", "services"],
        )

        results = await workbook.search_entries(tags=["manufacturing"])

        assert len(results) >= 1
        assert any("manufacturing" in e.tags for e in results)

    @pytest.mark.asyncio
    async def test_search_by_parent_entry(self, workbook: ClassificationWorkbook):
        """Should find child entries by parent ID."""
        parent = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS, label="Parent Analysis", content={}
        )

        await workbook.create_entry(
            form_type=FormType.CROSS_REFERENCE_NOTES,
            label="Follow-up Notes",
            content={},
            parent_entry_id=parent.entry_id,
        )

        results = await workbook.search_entries(parent_entry_id=parent.entry_id)

        assert len(results) == 1
        assert results[0].parent_entry_id == parent.entry_id

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, workbook: ClassificationWorkbook):
        """Should respect limit parameter."""
        # Create multiple entries
        for i in range(5):
            await workbook.create_entry(form_type=FormType.CUSTOM, label=f"Entry {i}", content={})

        results = await workbook.search_entries(limit=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_get_template(self, workbook: ClassificationWorkbook):
        """Should return correct template for form type."""
        template = await workbook.get_template(FormType.CLASSIFICATION_ANALYSIS)

        assert "business_description" in template
        assert "candidate_codes" in template

    @pytest.mark.asyncio
    async def test_get_custom_template(self, workbook: ClassificationWorkbook):
        """CUSTOM form type should return empty template."""
        template = await workbook.get_template(FormType.CUSTOM)

        assert template == {}

    @pytest.mark.asyncio
    async def test_entry_with_metadata(self, workbook: ClassificationWorkbook):
        """Should store and retrieve metadata."""
        entry = await workbook.create_entry(
            form_type=FormType.RESEARCH_NOTES,
            label="Research Entry",
            content={"research_question": "What NAICS code for dog food?"},
            metadata={"analyst": "Test User", "priority": "high", "reviewed": False},
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert retrieved.metadata["analyst"] == "Test User"
        assert retrieved.metadata["priority"] == "high"
        assert retrieved.metadata["reviewed"] is False

    @pytest.mark.asyncio
    async def test_entry_preserves_complex_content(self, workbook: ClassificationWorkbook):
        """Should preserve complex nested content."""
        complex_content = {
            "business_description": "Test business",
            "candidate_codes": [
                {
                    "code": "311111",
                    "title": "Dog Food Manufacturing",
                    "rationale": "Primary activity matches",
                    "pros": ["Exact match for activity", "High confidence"],
                    "cons": ["Consider if primary revenue source"],
                    "confidence": 0.92,
                },
                {
                    "code": "311119",
                    "title": "Other Animal Food",
                    "rationale": "Alternative if not primarily dog/cat",
                    "pros": ["Broader category"],
                    "cons": ["Less specific"],
                    "confidence": 0.75,
                },
            ],
            "decision": {
                "selected_code": "311111",
                "reasoning": "Primary activity is dog food manufacturing",
            },
        }

        entry = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Complex Analysis",
            content=complex_content,
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert len(retrieved.content["candidate_codes"]) == 2
        assert retrieved.content["candidate_codes"][0]["confidence"] == 0.92
        assert retrieved.content["decision"]["selected_code"] == "311111"


class TestExtractSearchText:
    """Tests for search text extraction."""

    @pytest.fixture
    def workbook(self, populated_database: NAICSDatabase) -> ClassificationWorkbook:
        return ClassificationWorkbook(populated_database)

    def test_extract_from_description(self, workbook: ClassificationWorkbook):
        """Should extract text from description fields."""
        content = {"business_description": "Manufacturing premium dog food"}

        search_text = workbook._extract_search_text(content)

        assert "Manufacturing premium dog food" in search_text

    def test_extract_from_nested_content(self, workbook: ClassificationWorkbook):
        """Should extract text from nested structures with recognized keys."""
        content = {
            # These are top-level keys that get extracted
            "reasoning": "This is the primary reasoning",
            "conclusion": "Final conclusion text",
            "analysis": "Detailed analysis here",
        }

        search_text = workbook._extract_search_text(content)

        assert "primary reasoning" in search_text
        assert "Final conclusion" in search_text

    def test_extract_truncates_long_text(self, workbook: ClassificationWorkbook):
        """Should truncate very long text."""
        content = {
            "business_description": "x" * 5000  # Very long description
        }

        search_text = workbook._extract_search_text(content)

        assert len(search_text) <= 2000

    def test_extract_handles_empty_content(self, workbook: ClassificationWorkbook):
        """Should handle empty content."""
        search_text = workbook._extract_search_text({})

        assert search_text == ""


class TestWorkbookEdgeCases:
    """Edge case tests for workbook."""

    @pytest.fixture
    def workbook(self, populated_database: NAICSDatabase) -> ClassificationWorkbook:
        return ClassificationWorkbook(populated_database)

    @pytest.mark.asyncio
    async def test_create_entry_with_unicode(self, workbook: ClassificationWorkbook):
        """Should handle Unicode content."""
        entry = await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Café Business",
            content={
                "business_name": "Café Résumé",
                "business_description": "Serving café au lait and croissants",
            },
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert retrieved.content["business_name"] == "Café Résumé"

    @pytest.mark.asyncio
    async def test_create_entry_with_special_characters(self, workbook: ClassificationWorkbook):
        """Should handle special characters."""
        entry = await workbook.create_entry(
            form_type=FormType.CUSTOM,
            label="Test & Analysis <2024>",
            content={"notes": "Contains 'quotes' and \"double quotes\""},
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert "quotes" in retrieved.content["notes"]

    @pytest.mark.asyncio
    async def test_create_entry_empty_content(self, workbook: ClassificationWorkbook):
        """Should handle empty content."""
        entry = await workbook.create_entry(
            form_type=FormType.CUSTOM, label="Empty Entry", content={}
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert retrieved.content == {}

    @pytest.mark.asyncio
    async def test_create_entry_null_values(self, workbook: ClassificationWorkbook):
        """Should handle None values in content."""
        entry = await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Partial Entry",
            content={"business_name": "Test", "business_description": None, "notes": None},
        )

        retrieved = await workbook.get_entry(entry.entry_id)

        assert retrieved.content["business_description"] is None

    @pytest.mark.asyncio
    async def test_search_no_results(self, workbook: ClassificationWorkbook):
        """Should return empty list when no matches."""
        results = await workbook.search_entries(search_text="xyznonexistentterm123")

        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_tag_search(self, workbook: ClassificationWorkbook):
        """Should find entries matching any of multiple tags."""
        await workbook.create_entry(
            form_type=FormType.CUSTOM, label="Entry A", content={}, tags=["tag1", "tag2"]
        )

        await workbook.create_entry(
            form_type=FormType.CUSTOM, label="Entry B", content={}, tags=["tag3", "tag4"]
        )

        # Should find Entry A (has tag1)
        results = await workbook.search_entries(tags=["tag1", "tag3"])

        assert len(results) >= 1


class TestWorkbookIntegration:
    """Integration tests for workbook with database."""

    @pytest.fixture
    def workbook(self, populated_database: NAICSDatabase) -> ClassificationWorkbook:
        return ClassificationWorkbook(populated_database)

    @pytest.mark.asyncio
    async def test_full_classification_workflow(self, workbook: ClassificationWorkbook):
        """Test complete classification workflow."""
        # Step 1: Create business profile
        profile = await workbook.create_entry(
            form_type=FormType.BUSINESS_PROFILE,
            label="Acme Pet Foods",
            content={
                "business_name": "Acme Pet Foods Inc.",
                "business_description": "Manufacturing premium dog and cat food",
                "primary_products_services": ["Dog food", "Cat food"],
                "primary_customers": "Pet retailers and distributors",
            },
            tags=["pet-food", "manufacturing", "B2B"],
        )

        # Step 2: Create classification analysis
        analysis = await workbook.create_entry(
            form_type=FormType.CLASSIFICATION_ANALYSIS,
            label="Acme Pet Foods - Classification",
            content={
                "business_description": "Manufacturing premium dog and cat food",
                "candidate_codes": [
                    {
                        "code": "311111",
                        "title": "Dog and Cat Food Manufacturing",
                        "confidence": 0.95,
                    }
                ],
                "decision": {
                    "selected_code": "311111",
                    "reasoning": "Primary activity is dog and cat food manufacturing",
                },
            },
            parent_entry_id=profile.entry_id,
            confidence_score=0.95,
        )

        # Step 3: Add cross-reference notes
        notes = await workbook.create_entry(
            form_type=FormType.CROSS_REFERENCE_NOTES,
            label="Acme - Cross-Reference Check",
            content={
                "source_code": "311111",
                "exclusions_found": [],
                "conclusion": "No exclusions apply",
            },
            parent_entry_id=analysis.entry_id,
        )

        # Verify chain
        _retrieved_profile = await workbook.get_entry(profile.entry_id)
        retrieved_analysis = await workbook.get_entry(analysis.entry_id)
        retrieved_notes = await workbook.get_entry(notes.entry_id)

        assert retrieved_analysis.parent_entry_id == profile.entry_id
        assert retrieved_notes.parent_entry_id == analysis.entry_id

        # Search for related entries
        children = await workbook.search_entries(parent_entry_id=profile.entry_id)
        assert len(children) == 1
        assert children[0].entry_id == analysis.entry_id

    @pytest.mark.asyncio
    async def test_session_isolation(self, populated_database: NAICSDatabase):
        """Different workbook instances should have different sessions."""
        workbook1 = ClassificationWorkbook(populated_database)
        workbook2 = ClassificationWorkbook(populated_database)

        assert workbook1.current_session_id != workbook2.current_session_id

        await workbook1.create_entry(form_type=FormType.CUSTOM, label="Session 1 Entry", content={})

        await workbook2.create_entry(form_type=FormType.CUSTOM, label="Session 2 Entry", content={})

        session1_entries = await workbook1.search_entries(session_id=workbook1.current_session_id)

        session2_entries = await workbook2.search_entries(session_id=workbook2.current_session_id)

        # Each session should only find its own entries
        assert all(e.session_id == workbook1.current_session_id for e in session1_entries)
        assert all(e.session_id == workbook2.current_session_id for e in session2_entries)
