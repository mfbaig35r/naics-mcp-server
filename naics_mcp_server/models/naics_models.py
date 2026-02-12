"""
Domain models for NAICS codes.

Clear, purposeful data structures that represent the NAICS classification system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class NAICSLevel(str, Enum):
    """
    The 5 hierarchical levels in the NAICS classification system.

    Unlike HTS (4 levels), NAICS has 5 levels from broad sectors
    to specific national industries.
    """

    SECTOR = "sector"                    # 2-digit (e.g., "31" - Manufacturing)
    SUBSECTOR = "subsector"              # 3-digit (e.g., "311" - Food Manufacturing)
    INDUSTRY_GROUP = "industry_group"    # 4-digit (e.g., "3111" - Animal Food)
    NAICS_INDUSTRY = "naics_industry"    # 5-digit (e.g., "31111" - Animal Food Mfg)
    NATIONAL_INDUSTRY = "national_industry"  # 6-digit (e.g., "311111" - Dog Food)

    @classmethod
    def from_code_length(cls, code: str) -> "NAICSLevel":
        """Determine level from code length."""
        length = len(code.strip())
        level_map = {
            2: cls.SECTOR,
            3: cls.SUBSECTOR,
            4: cls.INDUSTRY_GROUP,
            5: cls.NAICS_INDUSTRY,
            6: cls.NATIONAL_INDUSTRY
        }
        return level_map.get(length, cls.NATIONAL_INDUSTRY)


@dataclass
class NAICSCode:
    """
    Represents a single NAICS classification code.

    This is the core domain object - every code in the system.
    """

    node_code: str
    title: str
    level: NAICSLevel
    description: Optional[str] = None

    # Hierarchical relationships - explicit and clear
    sector_code: Optional[str] = None
    subsector_code: Optional[str] = None
    industry_group_code: Optional[str] = None
    naics_industry_code: Optional[str] = None

    # Text used for embedding generation
    raw_embedding_text: Optional[str] = None

    # Metadata
    change_indicator: Optional[str] = None  # Change from 2017 version
    is_trilateral: bool = True  # US-Canada-Mexico common code

    def get_hierarchy_path(self) -> List[str]:
        """
        Returns the complete hierarchical path for this code.

        Example: ["31", "311", "3111", "31111", "311111"]
        """
        path = []

        if self.sector_code:
            path.append(self.sector_code)
        if self.subsector_code:
            path.append(self.subsector_code)
        if self.industry_group_code:
            path.append(self.industry_group_code)
        if self.naics_industry_code:
            path.append(self.naics_industry_code)

        # Add self if not already in path
        if self.node_code and self.node_code not in path:
            path.append(self.node_code)

        return path

    def get_parent_code(self) -> Optional[str]:
        """Get the immediate parent code based on level."""
        if self.level == NAICSLevel.NATIONAL_INDUSTRY:
            return self.naics_industry_code
        elif self.level == NAICSLevel.NAICS_INDUSTRY:
            return self.industry_group_code
        elif self.level == NAICSLevel.INDUSTRY_GROUP:
            return self.subsector_code
        elif self.level == NAICSLevel.SUBSECTOR:
            return self.sector_code
        return None

    def get_hierarchy_descriptions(self) -> Dict[str, str]:
        """
        Returns a dictionary of hierarchy levels to their descriptions.

        This helps users understand the classification context.
        """
        return {
            "sector": f"Sector {self.sector_code}" if self.sector_code else None,
            "subsector": f"Subsector {self.subsector_code}" if self.subsector_code else None,
            "industry_group": f"Industry Group {self.industry_group_code}" if self.industry_group_code else None,
            "naics_industry": f"NAICS Industry {self.naics_industry_code}" if self.naics_industry_code else None,
            "national_industry": f"National Industry {self.node_code}" if self.level == NAICSLevel.NATIONAL_INDUSTRY else None
        }


@dataclass
class CrossReference:
    """
    Represents a cross-reference from NAICS descriptions.

    Cross-references are CRITICAL for accurate classification.
    They explicitly tell you what activities are excluded from a code
    and where to classify them instead.
    """

    source_code: str  # The code containing the cross-reference
    reference_type: str  # 'excludes', 'see_also', 'includes'
    reference_text: str  # The original text
    target_code: Optional[str] = None  # Parsed target code if available
    excluded_activity: Optional[str] = None  # What activity is excluded

    def __str__(self) -> str:
        if self.target_code:
            return f"{self.reference_type}: {self.excluded_activity} → {self.target_code}"
        return f"{self.reference_type}: {self.reference_text}"


@dataclass
class IndexTerm:
    """
    Official NAICS index term (from the 20,398 official terms).

    These are the terms that NAICS officially associates with each code.
    """

    term_id: int
    naics_code: str
    index_term: str
    term_normalized: Optional[str] = None  # Lowercase for search

    def __str__(self) -> str:
        return f"{self.index_term} → {self.naics_code}"


@dataclass
class SICCrosswalk:
    """
    SIC to NAICS crosswalk entry for legacy system integration.
    """

    sic_code: str
    naics_code: str
    relationship_type: Optional[str] = None  # 'direct', 'partial', 'split'

    def __str__(self) -> str:
        return f"SIC {self.sic_code} → NAICS {self.naics_code}"
