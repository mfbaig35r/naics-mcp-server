"""
Cross-reference parsing and lookup for NAICS codes.

Cross-references are CRITICAL for accurate NAICS classification.
They tell you what activities are excluded from a code and where
to classify them instead.

Example from NAICS description:
    "Cross-References. Establishments primarily engaged in--
    - dog and cat food manufacturing--are classified in Industry 311111;
    - prepared animal feeds for cattle, hogs, etc.--are classified in Industry 311119."
"""

import logging
import re

from ..models.naics_models import CrossReference

logger = logging.getLogger(__name__)


class CrossReferenceParser:
    """
    Parses cross-references from NAICS description text.

    This extracts structured data from the cross-reference sections
    of NAICS code descriptions.
    """

    def __init__(self):
        """Initialize the parser with regex patterns."""

        # Pattern to find cross-reference sections
        self.cross_ref_section_pattern = re.compile(
            r"Cross-References?\s*\.?\s*(.*?)(?=\n\n|$)", re.IGNORECASE | re.DOTALL
        )

        # Pattern to extract individual cross-references
        # Matches: "- activity description--are classified in Industry XXXXXX"
        self.cross_ref_entry_pattern = re.compile(
            r"-\s*(.+?)\s*[-–—]+\s*(?:are|is)\s+classified\s+in\s+"
            r"(?:Industry|U\.?S\.?\s*Industry|NAICS)?\s*(\d{5,6})",
            re.IGNORECASE,
        )

        # Alternative pattern for simpler format
        # Matches: "See Industry XXXXXX"
        self.see_also_pattern = re.compile(r"See\s+(?:Industry|NAICS)?\s*(\d{5,6})", re.IGNORECASE)

        # Pattern for "except" clauses
        self.except_pattern = re.compile(
            r"except\s+(?:those|establishments)\s+(?:engaged\s+in\s+)?(.+?)(?:\s*[-–—]+\s*(?:are|is)\s+classified|\s*$)",
            re.IGNORECASE,
        )

    def parse_description(self, source_code: str, description: str) -> list[CrossReference]:
        """
        Parse cross-references from a NAICS description.

        Args:
            source_code: The NAICS code this description belongs to
            description: The full description text

        Returns:
            List of CrossReference objects
        """
        if not description:
            return []

        cross_refs = []

        # Find cross-reference section
        section_match = self.cross_ref_section_pattern.search(description)
        if section_match:
            section_text = section_match.group(1)

            # Extract individual entries
            for match in self.cross_ref_entry_pattern.finditer(section_text):
                excluded_activity = match.group(1).strip()
                target_code = match.group(2).strip()

                cross_refs.append(
                    CrossReference(
                        source_code=source_code,
                        reference_type="excludes",
                        reference_text=match.group(0).strip(),
                        target_code=target_code,
                        excluded_activity=excluded_activity,
                    )
                )

        # Also check for "See also" references throughout the text
        for match in self.see_also_pattern.finditer(description):
            target_code = match.group(1).strip()
            # Avoid duplicates
            if not any(cr.target_code == target_code for cr in cross_refs):
                cross_refs.append(
                    CrossReference(
                        source_code=source_code,
                        reference_type="see_also",
                        reference_text=match.group(0).strip(),
                        target_code=target_code,
                    )
                )

        return cross_refs

    def parse_excludes_only(self, description: str) -> list[tuple[str, str]]:
        """
        Extract just the excluded activities and target codes.

        Simpler method that returns tuples of (activity, target_code).

        Args:
            description: The description text

        Returns:
            List of (excluded_activity, target_code) tuples
        """
        excludes = []

        section_match = self.cross_ref_section_pattern.search(description or "")
        if section_match:
            section_text = section_match.group(1)

            for match in self.cross_ref_entry_pattern.finditer(section_text):
                excludes.append((match.group(1).strip(), match.group(2).strip()))

        return excludes


class CrossReferenceService:
    """
    Service for cross-reference operations.

    Provides methods to check if a query matches exclusion criteria
    and to find where excluded activities should be classified.
    """

    def __init__(self, database):
        """
        Initialize with database connection.

        Args:
            database: NAICSDatabase instance
        """
        self.database = database
        self.parser = CrossReferenceParser()

    async def check_exclusions(self, query: str, naics_code: str) -> list[dict[str, str]]:
        """
        Check if a query matches any exclusion criteria for a code.

        Args:
            query: The search query or business description
            naics_code: The NAICS code to check

        Returns:
            List of matching exclusions with target codes
        """
        cross_refs = await self.database.get_cross_references(naics_code)

        matches = []
        query_lower = query.lower()

        for cr in cross_refs:
            if cr.reference_type == "excludes" and cr.excluded_activity:
                # Check if query matches the excluded activity
                excluded_words = set(cr.excluded_activity.lower().split())
                query_words = set(query_lower.split())

                # Calculate word overlap
                overlap = excluded_words & query_words
                if len(overlap) >= 2 or any(
                    word in query_lower for word in excluded_words if len(word) > 5
                ):
                    matches.append(
                        {
                            "excluded_activity": cr.excluded_activity,
                            "target_code": cr.target_code,
                            "warning": f"This activity may be better classified under {cr.target_code}",
                        }
                    )

        return matches

    async def check_exclusions_batch(
        self, query: str, naics_codes: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        """
        Check exclusions for multiple NAICS codes in a single batch operation.

        This is much more efficient than calling check_exclusions() in a loop
        (avoids N+1 query pattern).

        Args:
            query: The search query or business description
            naics_codes: List of NAICS codes to check

        Returns:
            Dict mapping naics_code to list of matching exclusions
        """
        if not naics_codes:
            return {}

        # Batch fetch all cross-references in one query
        all_cross_refs = await self.database.get_cross_references_batch(naics_codes)

        query_lower = query.lower()
        query_words = set(query_lower.split())

        results: dict[str, list[dict[str, str]]] = {}
        for naics_code in naics_codes:
            cross_refs = all_cross_refs.get(naics_code, [])
            matches = []

            for cr in cross_refs:
                if cr.reference_type == "excludes" and cr.excluded_activity:
                    # Check if query matches the excluded activity
                    excluded_words = set(cr.excluded_activity.lower().split())

                    # Calculate word overlap using pre-computed query_words
                    overlap = excluded_words & query_words
                    if len(overlap) >= 2 or any(
                        word in query_lower for word in excluded_words if len(word) > 5
                    ):
                        matches.append(
                            {
                                "excluded_activity": cr.excluded_activity,
                                "target_code": cr.target_code,
                                "warning": f"This activity may be better classified under {cr.target_code}",
                            }
                        )

            results[naics_code] = matches

        return results

    async def find_correct_classification(
        self, activity_description: str, limit: int = 5
    ) -> list[dict[str, str]]:
        """
        Search cross-references to find where an activity should be classified.

        Args:
            activity_description: Description of the business activity
            limit: Maximum results

        Returns:
            List of potential classifications from cross-references
        """
        # Search cross-references for matching activities
        matching_refs = await self.database.search_cross_references(
            activity_description, limit=limit
        )

        results = []
        for ref in matching_refs:
            if ref.target_code:
                # Get the target code's title
                target_code = await self.database.get_by_code(ref.target_code)
                target_title = target_code.title if target_code else "Unknown"

                results.append(
                    {
                        "activity": ref.excluded_activity or ref.reference_text,
                        "recommended_code": ref.target_code,
                        "recommended_title": target_title,
                        "excluded_from": ref.source_code,
                        "reference_text": ref.reference_text,
                    }
                )

        return results

    async def get_exclusion_warnings(self, naics_code: str) -> list[str]:
        """
        Get human-readable exclusion warnings for a code.

        Args:
            naics_code: The NAICS code

        Returns:
            List of warning strings
        """
        cross_refs = await self.database.get_cross_references(naics_code)

        warnings = []
        for cr in cross_refs:
            if cr.reference_type == "excludes":
                if cr.excluded_activity and cr.target_code:
                    warnings.append(
                        f"Note: {cr.excluded_activity} is classified under {cr.target_code}, not this code"
                    )
                elif cr.reference_text:
                    warnings.append(f"Note: {cr.reference_text}")

        return warnings

    def calculate_exclusion_penalty(
        self, query: str, cross_refs: list[CrossReference], base_penalty: float = 0.7
    ) -> float:
        """
        Calculate confidence penalty based on exclusion matches.

        Args:
            query: The search query
            cross_refs: List of cross-references for the code
            base_penalty: Penalty multiplier for matches (default 0.7)

        Returns:
            Penalty factor (1.0 = no penalty, 0.7 = 30% penalty)
        """
        if not cross_refs:
            return 1.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for cr in cross_refs:
            if cr.reference_type == "excludes" and cr.excluded_activity:
                excluded_words = set(cr.excluded_activity.lower().split())

                # Check for significant overlap
                overlap = excluded_words & query_words
                overlap_ratio = len(overlap) / max(len(excluded_words), 1)

                if overlap_ratio > 0.3:  # More than 30% word overlap
                    return base_penalty

        return 1.0
