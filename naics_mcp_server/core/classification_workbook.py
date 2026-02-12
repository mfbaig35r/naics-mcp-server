"""
Classification Workbook Service

A structured filing system for AI reasoning, decisions, and analysis.
Like a filing cabinet with different forms for different types of work.

Adapted for NAICS classification with industry-specific form types.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.database import NAICSDatabase

logger = logging.getLogger(__name__)


class FormType(str, Enum):
    """
    Types of forms that can be filed in the workbook.
    Each represents a different reasoning structure.

    NAICS-specific forms added for industry classification.
    """

    CLASSIFICATION_ANALYSIS = "classification_analysis"
    INDUSTRY_COMPARISON = "industry_comparison"
    CROSS_REFERENCE_NOTES = "cross_reference_notes"
    BUSINESS_PROFILE = "business_profile"
    DECISION_TREE = "decision_tree"
    SIC_CONVERSION = "sic_conversion"
    RESEARCH_NOTES = "research_notes"
    CUSTOM = "custom"


@dataclass
class WorkbookEntry:
    """
    A single entry in the classification workbook.
    """

    entry_id: str
    form_type: FormType
    label: str  # Human-readable label for quick reference
    content: Dict[str, Any]  # The structured content
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Auto-filled fields
    created_at: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    parent_entry_id: Optional[str] = None  # For linked entries
    tags: List[str] = field(default_factory=list)

    # Search and retrieval helpers
    search_text: Optional[str] = None  # Denormalized text for full-text search
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "form_type": self.form_type.value,
            "label": self.label,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "parent_entry_id": self.parent_entry_id,
            "tags": self.tags,
            "search_text": self.search_text,
            "confidence_score": self.confidence_score
        }


# NAICS-specific form templates
FORM_TEMPLATES = {
    FormType.CLASSIFICATION_ANALYSIS: {
        "business_description": "",
        "primary_activity": "",
        "secondary_activities": [],
        "key_characteristics": [],
        "candidate_codes": [
            {
                "code": "",
                "title": "",
                "rationale": "",
                "pros": [],
                "cons": [],
                "confidence": 0.0
            }
        ],
        "cross_reference_check": {
            "exclusions_reviewed": [],
            "relevant_exclusions": []
        },
        "decision": {
            "selected_code": "",
            "reasoning": "",
            "alternative_if_uncertain": "",
            "notes": []
        }
    },

    FormType.INDUSTRY_COMPARISON: {
        "comparison_title": "",
        "codes_compared": [],  # List of NAICS codes being compared
        "comparison_criteria": [
            "primary_activity",
            "production_process",
            "customer_type",
            "output_type"
        ],
        "matrix": {},  # Code -> Criteria -> Assessment
        "analysis": {
            "best_fit": "",
            "reasoning": "",
            "key_differentiators": [],
            "recommendation": ""
        }
    },

    FormType.CROSS_REFERENCE_NOTES: {
        "source_code": "",
        "source_title": "",
        "activity_in_question": "",
        "exclusions_found": [
            {
                "excluded_activity": "",
                "classified_under": "",
                "target_code": "",
                "target_title": ""
            }
        ],
        "conclusion": "",
        "correct_classification": ""
    },

    FormType.BUSINESS_PROFILE: {
        "business_name": "",
        "business_description": "",
        "primary_products_services": [],
        "primary_customers": "",
        "production_process": "",
        "location_type": "",  # office, retail, warehouse, factory, etc.
        "employee_count_range": "",
        "revenue_source": "",
        "naics_analysis": {
            "primary_code": "",
            "primary_title": "",
            "secondary_codes": [],
            "reasoning": ""
        },
        "sic_codes_if_known": [],
        "notes": ""
    },

    FormType.DECISION_TREE: {
        "initial_question": "",
        "decision_points": [
            {
                "question": "",
                "answer": "",
                "reasoning": "",
                "naics_codes_considered": [],
                "leads_to": ""
            }
        ],
        "final_classification": {
            "code": "",
            "title": "",
            "confidence": 0.0
        },
        "path_summary": []
    },

    FormType.SIC_CONVERSION: {
        "sic_code": "",
        "sic_title": "",
        "business_description": "",
        "naics_mappings": [
            {
                "naics_code": "",
                "naics_title": "",
                "relationship": "",  # direct, partial, split
                "notes": ""
            }
        ],
        "recommended_naics": "",
        "reasoning": "",
        "additional_context": ""
    },

    FormType.RESEARCH_NOTES: {
        "research_question": "",
        "sources_consulted": [
            {
                "type": "",  # NAICS manual, Census, industry guide
                "reference": "",
                "key_findings": [],
                "reliability": 0.0
            }
        ],
        "findings": {
            "primary": [],
            "secondary": [],
            "contradictions": [],
            "gaps": []
        },
        "synthesis": "",
        "conclusion": "",
        "follow_up_needed": []
    }
}


class ClassificationWorkbook:
    """
    Service for managing the classification workbook.

    This acts like a filing cabinet where structured reasoning
    about NAICS classification can be stored and retrieved.
    """

    def __init__(self, database: NAICSDatabase):
        """
        Initialize the workbook service.

        Args:
            database: NAICSDatabase instance for storage
        """
        self.database = database
        self.current_session_id = self._generate_session_id()

        # Initialize the workbook table if needed
        self._initialize_table()

    def _initialize_table(self) -> None:
        """Create the workbook table if it doesn't exist, and migrate if needed."""
        try:
            self.database.connection.execute("""
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
                )
            """)

            # Migration: Add search_text column if missing (for existing databases)
            columns = self.database.connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'classification_workbook'"
            ).fetchall()
            column_names = [c[0] for c in columns]

            if 'search_text' not in column_names:
                logger.info("Migrating classification_workbook: adding search_text column")
                self.database.connection.execute(
                    "ALTER TABLE classification_workbook ADD COLUMN search_text TEXT"
                )

            logger.info("Classification workbook table initialized")
        except Exception as e:
            logger.debug(f"Table initialization note: {e}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _generate_entry_id(self, label: str) -> str:
        """Generate a unique entry ID."""
        timestamp = datetime.now().isoformat()
        content = f"{label}:{timestamp}:{self.current_session_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def create_entry(
        self,
        form_type: FormType,
        label: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_entry_id: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> WorkbookEntry:
        """
        Create a new workbook entry.

        Args:
            form_type: Type of form to use
            label: Human-readable label
            content: Structured content (should match form template)
            metadata: Additional metadata
            tags: Tags for categorization
            parent_entry_id: Link to parent entry if this is a follow-up
            confidence_score: Optional confidence in the decision

        Returns:
            Created WorkbookEntry
        """
        entry = WorkbookEntry(
            entry_id=self._generate_entry_id(label),
            form_type=form_type,
            label=label,
            content=content,
            metadata=metadata or {},
            session_id=self.current_session_id,
            parent_entry_id=parent_entry_id,
            tags=tags or [],
            confidence_score=confidence_score,
            search_text=self._extract_search_text(content)
        )

        # Store in database
        try:
            self.database.connection.execute("""
                INSERT INTO classification_workbook (
                    entry_id, form_type, label, content, metadata,
                    created_at, session_id, parent_entry_id, tags,
                    search_text, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                entry.entry_id,
                entry.form_type.value,
                entry.label,
                json.dumps(entry.content),
                json.dumps(entry.metadata),
                entry.created_at,
                entry.session_id,
                entry.parent_entry_id,
                json.dumps(entry.tags),
                entry.search_text,
                entry.confidence_score
            ])

            logger.info(f"Created workbook entry: {entry.label} ({entry.entry_id})")
            return entry

        except Exception as e:
            logger.error(f"Failed to create workbook entry: {e}")
            raise

    async def get_entry(self, entry_id: str) -> Optional[WorkbookEntry]:
        """
        Retrieve a specific workbook entry.

        Args:
            entry_id: The entry ID to retrieve

        Returns:
            WorkbookEntry or None if not found
        """
        try:
            result = self.database.connection.execute("""
                SELECT * FROM classification_workbook
                WHERE entry_id = ?
            """, [entry_id]).fetchone()

            if result:
                return self._row_to_entry(result)
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve entry {entry_id}: {e}")
            return None

    async def search_entries(
        self,
        form_type: Optional[FormType] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search_text: Optional[str] = None,
        parent_entry_id: Optional[str] = None,
        limit: int = 50
    ) -> List[WorkbookEntry]:
        """
        Search for workbook entries.

        Args:
            form_type: Filter by form type
            session_id: Filter by session
            tags: Filter by tags (any match)
            search_text: Full-text search
            parent_entry_id: Filter by parent entry
            limit: Maximum results

        Returns:
            List of matching entries
        """
        conditions = []
        params = []

        if form_type:
            conditions.append("form_type = ?")
            params.append(form_type.value)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if parent_entry_id:
            conditions.append("parent_entry_id = ?")
            params.append(parent_entry_id)

        if search_text:
            conditions.append("search_text LIKE ?")
            params.append(f"%{search_text}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        try:
            results = self.database.connection.execute(f"""
                SELECT * FROM classification_workbook
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """, params).fetchall()

            entries = [self._row_to_entry(row) for row in results]

            # Filter by tags in Python (JSON array handling)
            if tags:
                entries = [
                    e for e in entries
                    if any(tag in e.tags for tag in tags)
                ]

            return entries

        except Exception as e:
            logger.error(f"Failed to search entries: {e}")
            return []

    async def get_template(self, form_type: FormType) -> Dict[str, Any]:
        """
        Get a template for a specific form type.

        Args:
            form_type: The form type to get template for

        Returns:
            Template dictionary
        """
        if form_type == FormType.CUSTOM:
            return {}

        return FORM_TEMPLATES.get(form_type, {}).copy()

    def _extract_search_text(self, content: Dict[str, Any]) -> str:
        """
        Extract searchable text from structured content.

        Args:
            content: Structured content dictionary

        Returns:
            Concatenated searchable text
        """
        text_parts = []

        def extract_text(obj, prefix=""):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ["description", "reasoning", "synthesis", "conclusion",
                              "recommendation", "notes", "findings", "analysis",
                              "business_description", "primary_activity", "rationale"]:
                        extract_text(value)

        extract_text(content)
        return " ".join(text_parts)[:2000]

    def _row_to_entry(self, row: tuple) -> WorkbookEntry:
        """
        Convert a database row to a WorkbookEntry.
        """
        return WorkbookEntry(
            entry_id=row[0],
            form_type=FormType(row[1]),
            label=row[2],
            content=json.loads(row[3]) if row[3] else {},
            metadata=json.loads(row[4]) if row[4] else {},
            created_at=datetime.fromisoformat(row[5]) if isinstance(row[5], str) else row[5],
            session_id=row[6],
            parent_entry_id=row[7],
            tags=json.loads(row[8]) if row[8] else [],
            search_text=row[9],
            confidence_score=row[10]
        )
