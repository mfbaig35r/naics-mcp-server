"""
Pytest fixtures for NAICS MCP Server tests.

Provides reusable test fixtures for database, search engine, and sample data.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import duckdb

from naics_mcp_server.core.database import NAICSDatabase
from naics_mcp_server.models.naics_models import NAICSLevel


# --- Sample Data ---

SAMPLE_NAICS_CODES = [
    # Sector
    ("31", "sector", "Manufacturing", "Manufacturing sector description",
     None, None, None, None, "Manufacturing sector", None, True),
    # Subsector
    ("311", "subsector", "Food Manufacturing", "Food manufacturing subsector",
     "31", None, None, None, "Food manufacturing", None, True),
    # Industry Group
    ("3111", "industry_group", "Animal Food Manufacturing", "Animal food industry group",
     "31", "311", None, None, "Animal food manufacturing", None, True),
    # NAICS Industry
    ("31111", "naics_industry", "Animal Food Manufacturing", "Animal food NAICS industry",
     "31", "311", "3111", None, "Animal food manufacturing industry", None, True),
    # National Industries
    ("311111", "national_industry", "Dog and Cat Food Manufacturing",
     "Manufacturing dog and cat food from purchased ingredients",
     "31", "311", "3111", "31111", "Dog food cat food pet food manufacturing", None, True),
    ("311119", "national_industry", "Other Animal Food Manufacturing",
     "Manufacturing animal food (except dog and cat) from purchased ingredients",
     "31", "311", "3111", "31111", "Animal feed livestock feed poultry feed", None, True),
    # Another branch - Beverage
    ("312", "subsector", "Beverage and Tobacco Product Manufacturing",
     "Beverage and tobacco manufacturing subsector",
     "31", None, None, None, "Beverage tobacco manufacturing", None, True),
    ("3121", "industry_group", "Beverage Manufacturing", "Beverage manufacturing",
     "31", "312", None, None, "Beverage manufacturing", None, True),
    ("31211", "naics_industry", "Soft Drink and Ice Manufacturing",
     "Soft drink and ice manufacturing",
     "31", "312", "3121", None, "Soft drinks ice manufacturing", None, True),
    ("312111", "national_industry", "Soft Drink Manufacturing",
     "Manufacturing soft drinks, bottled water, and ice",
     "31", "312", "3121", "31211", "Soft drinks bottled water soda pop", None, True),
    # Different sector - Retail
    ("44", "sector", "Retail Trade", "Retail trade sector",
     None, None, None, None, "Retail trade stores", None, True),
    ("441", "subsector", "Motor Vehicle and Parts Dealers",
     "Motor vehicle dealers",
     "44", None, None, None, "Car dealers auto dealers", None, True),
]

SAMPLE_INDEX_TERMS = [
    (1, "311111", "Dog food manufacturing", "dog food manufacturing"),
    (2, "311111", "Cat food manufacturing", "cat food manufacturing"),
    (3, "311111", "Pet food manufacturing", "pet food manufacturing"),
    (4, "311119", "Animal feed manufacturing", "animal feed manufacturing"),
    (5, "311119", "Livestock feed manufacturing", "livestock feed manufacturing"),
    (6, "312111", "Soft drink bottling", "soft drink bottling"),
    (7, "312111", "Bottled water manufacturing", "bottled water manufacturing"),
]

SAMPLE_CROSS_REFERENCES = [
    (1, "311111", "excludes", "See 311119 for other animal food", "311119", "other animal food"),
    (2, "311119", "excludes", "See 311111 for dog and cat food", "311111", "dog and cat food"),
    (3, "312111", "excludes", "See 311930 for flavoring syrup", "311930", "flavoring syrup"),
]


# --- Fixtures ---

@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary database path that's cleaned up after tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_naics.db"
        yield db_path
        # Cleanup happens automatically when temp_dir is deleted


@pytest.fixture
def empty_database(temp_db_path: Path) -> Generator[NAICSDatabase, None, None]:
    """Provide an empty database with schema initialized."""
    db = NAICSDatabase(temp_db_path)
    db.connect()

    yield db

    db.disconnect()


@pytest.fixture
def populated_database(temp_db_path: Path) -> Generator[NAICSDatabase, None, None]:
    """Provide a database populated with sample NAICS data."""
    db = NAICSDatabase(temp_db_path)
    db.connect()

    # Insert sample NAICS codes
    for row in SAMPLE_NAICS_CODES:
        db.connection.execute("""
            INSERT INTO naics_nodes
            (node_code, level, title, description, sector_code, subsector_code,
             industry_group_code, naics_industry_code, raw_embedding_text,
             change_indicator, is_trilateral)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)

    # Insert sample index terms
    for row in SAMPLE_INDEX_TERMS:
        db.connection.execute("""
            INSERT INTO naics_index_terms (term_id, naics_code, index_term, term_normalized)
            VALUES (?, ?, ?, ?)
        """, row)

    # Insert sample cross-references
    for row in SAMPLE_CROSS_REFERENCES:
        db.connection.execute("""
            INSERT INTO naics_cross_references
            (ref_id, source_code, reference_type, reference_text, target_code, excluded_activity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row)

    yield db

    db.disconnect()


@pytest.fixture
def sample_naics_code():
    """Provide a sample NAICSCode for testing."""
    from naics_mcp_server.models.naics_models import NAICSCode, NAICSLevel

    return NAICSCode(
        node_code="311111",
        title="Dog and Cat Food Manufacturing",
        level=NAICSLevel.NATIONAL_INDUSTRY,
        description="Manufacturing dog and cat food from purchased ingredients",
        sector_code="31",
        subsector_code="311",
        industry_group_code="3111",
        naics_industry_code="31111",
        raw_embedding_text="Dog food cat food pet food manufacturing",
    )


@pytest.fixture
def sample_cross_reference():
    """Provide a sample CrossReference for testing."""
    from naics_mcp_server.models.naics_models import CrossReference

    return CrossReference(
        source_code="311111",
        reference_type="excludes",
        reference_text="See 311119 for other animal food",
        target_code="311119",
        excluded_activity="other animal food",
    )


@pytest.fixture
def sample_index_term():
    """Provide a sample IndexTerm for testing."""
    from naics_mcp_server.models.naics_models import IndexTerm

    return IndexTerm(
        term_id=1,
        naics_code="311111",
        index_term="Dog food manufacturing",
        term_normalized="dog food manufacturing",
    )


# --- Environment Fixtures ---

@pytest.fixture
def clean_env():
    """Ensure clean environment variables for testing."""
    # Store original values
    original_env = {}
    env_vars = [
        "NAICS_LOG_LEVEL",
        "NAICS_LOG_FORMAT",
        "NAICS_LOG_FILE",
        "NAICS_DATABASE_PATH",
    ]

    for var in env_vars:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]
