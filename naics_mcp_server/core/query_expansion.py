"""
Query expansion for improved search recall.

This module expands user queries with synonyms, abbreviations,
and industry-specific terms to improve search coverage.

NAICS-specific vocabulary focuses on industry classification concepts.
"""

import re
from typing import Any

from ..models.search_models import QueryTerms


class QueryExpander:
    """
    Expands queries with related terms to improve search recall.

    This is not magic—it's a clear process of finding synonyms,
    industry terms, and common variations that users might not know.
    """

    def __init__(self):
        """Initialize with NAICS domain-specific knowledge."""

        # Industry-specific synonyms for NAICS classification
        self.industry_synonyms = {
            # Manufacturing
            "manufacturing": ["production", "fabrication", "making", "processing"],
            "factory": ["plant", "mill", "workshop", "facility"],
            "assembly": ["construction", "building", "putting together"],
            # Retail/Wholesale
            "retail": ["selling", "store", "shop", "merchant"],
            "wholesale": ["distribution", "distributor", "bulk sales"],
            "store": ["shop", "outlet", "retailer", "merchant"],
            # Services
            "service": ["services", "provider", "consulting"],
            "consulting": ["advisory", "professional services", "consultancy"],
            "repair": ["maintenance", "service", "fixing"],
            # Construction
            "construction": ["building", "contractor", "development"],
            "contractor": ["builder", "construction company"],
            # Healthcare
            "healthcare": ["medical", "health services", "health care"],
            "hospital": ["medical center", "health facility", "clinic"],
            "doctor": ["physician", "medical practitioner", "healthcare provider"],
            # Finance
            "bank": ["banking", "financial institution", "depository"],
            "insurance": ["insurer", "underwriting", "coverage"],
            "investment": ["investing", "securities", "asset management"],
            # Technology
            "software": ["programming", "application", "computer program"],
            "technology": ["tech", "IT", "information technology"],
            "computer": ["computing", "IT", "data processing"],
            # Food/Agriculture
            "restaurant": ["food service", "eatery", "dining"],
            "farm": ["farming", "agriculture", "agricultural"],
            "food": ["foodstuff", "provisions", "edible"],
            # Transportation
            "trucking": ["freight", "hauling", "transportation"],
            "shipping": ["freight", "cargo", "transport"],
            "airline": ["air carrier", "aviation", "air transport"],
            # Real Estate
            "real estate": ["property", "realty", "real property"],
            "rental": ["leasing", "letting", "renting"],
            # Professional Services
            "lawyer": ["attorney", "legal services", "law firm"],
            "accountant": ["accounting", "CPA", "bookkeeping"],
            "engineer": ["engineering", "technical services"],
        }

        # Industry abbreviations and their expansions
        self.abbreviation_expansions = {
            # Business abbreviations
            "mfg": "manufacturing",
            "mfr": "manufacturer",
            "dist": "distribution",
            "svcs": "services",
            "svc": "service",
            "mgmt": "management",
            "admin": "administration",
            "corp": "corporation",
            "inc": "incorporated",
            "llc": "limited liability company",
            # Industry abbreviations
            "it": "information technology",
            "hr": "human resources",
            "r&d": "research and development",
            "hvac": "heating ventilation air conditioning",
            "oem": "original equipment manufacturer",
            "b2b": "business to business",
            "b2c": "business to consumer",
            # Healthcare
            "rx": "pharmaceutical",
            "md": "medical doctor",
            "dds": "dental",
            # Real estate
            "reit": "real estate investment trust",
            # Food
            "f&b": "food and beverage",
            "qsr": "quick service restaurant",
        }

        # Related concepts for common business categories
        self.related_concepts = {
            "grocery": ["supermarket", "food retail", "convenience store"],
            "restaurant": ["food service", "dining", "catering", "eatery"],
            "hotel": ["lodging", "hospitality", "accommodation", "motel"],
            "salon": ["beauty", "spa", "personal care", "barbershop"],
            "gym": ["fitness", "health club", "recreation", "athletic"],
            "daycare": ["childcare", "child care", "preschool", "nursery"],
            "auto": ["automotive", "vehicle", "car", "automobile"],
            "pharmacy": ["drugstore", "pharmaceutical", "prescription"],
            "clinic": ["medical office", "healthcare", "doctor office"],
            "warehouse": ["storage", "distribution center", "fulfillment"],
        }

    async def expand(self, query: str) -> QueryTerms:
        """
        Expand a query into related terms.

        Returns a structured set of terms with their purposes:
        - Original: the user's exact input
        - Synonyms: alternative words with same meaning
        - Expanded: spelled-out abbreviations
        - Related: conceptually connected terms

        Args:
            query: The user's search query

        Returns:
            QueryTerms object with expanded terms
        """
        terms = QueryTerms(original=query)

        # Clean and tokenize the query
        tokens = self._tokenize(query.lower())

        # Find synonyms for each token
        synonyms = set()
        for token in tokens:
            if token in self.industry_synonyms:
                synonyms.update(self.industry_synonyms[token])
        terms.synonyms = list(synonyms)

        # Expand abbreviations
        expanded = set()
        for token in tokens:
            if token in self.abbreviation_expansions:
                expanded.add(self.abbreviation_expansions[token])

        # Also check if the full query is an abbreviation
        if query.lower() in self.abbreviation_expansions:
            expanded.add(self.abbreviation_expansions[query.lower()])

        terms.expanded = list(expanded)

        # Find related concepts
        related = set()
        for token in tokens:
            if token in self.related_concepts:
                related.update(self.related_concepts[token])
        terms.related = list(related)

        return terms

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into meaningful words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, keep alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Filter out very short tokens (likely not meaningful)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def add_domain_knowledge(self, category: str, terms: dict[str, list[str]]) -> None:
        """
        Add domain-specific knowledge to the expander.

        This allows customization for specific industries or use cases.

        Args:
            category: Category of terms (synonyms, abbreviations, related)
            terms: Dictionary of term mappings
        """
        if category == "synonyms":
            self.industry_synonyms.update(terms)
        elif category == "abbreviations":
            self.abbreviation_expansions.update(terms)
        elif category == "related":
            self.related_concepts.update(terms)
        else:
            raise ValueError(f"Unknown category: {category}")


class SmartQueryParser:
    """
    Parses queries to understand user intent.

    This helps identify what kind of search would be most effective.
    """

    def __init__(self):
        """Initialize parser patterns."""

        # Patterns that suggest exact match preference
        self.exact_patterns = [
            r'^".*"$',  # Quoted phrases
            r"^\d{2,6}$",  # NAICS codes (2-6 digits)
            r"^naics[\d\s]+$",  # Explicit NAICS codes
            r"^sector\s+\d+$",  # Sector references
        ]

        # Patterns that suggest semantic search
        self.semantic_patterns = [
            r"\b(like|similar to|type of|kind of)\b",
            r"\b(business that|company that|industry for)\b",
            r"\b(engaged in|specializing in|focused on)\b",
            r"\?$",  # Questions
        ]

        # Patterns that indicate business description
        self.business_patterns = [
            r"\b(we|our company|my business|i run)\b",
            r"\b(provides?|offers?|sells?|makes?|produces?)\b",
        ]

    def parse_intent(self, query: str) -> dict[str, Any]:
        """
        Parse query to understand search intent.

        Args:
            query: User's search query

        Returns:
            Dictionary with parsed intent information
        """
        intent = {
            "original_query": query,
            "suggests_exact": False,
            "suggests_semantic": False,
            "is_business_description": False,
            "has_code": False,
            "is_question": query.strip().endswith("?"),
        }

        query_lower = query.lower()

        # Check for exact match patterns
        for pattern in self.exact_patterns:
            if re.search(pattern, query_lower):
                intent["suggests_exact"] = True
                break

        # Check for semantic patterns
        for pattern in self.semantic_patterns:
            if re.search(pattern, query_lower):
                intent["suggests_semantic"] = True
                break

        # Check for business description patterns
        for pattern in self.business_patterns:
            if re.search(pattern, query_lower):
                intent["is_business_description"] = True
                break

        # Check for NAICS code patterns
        if re.search(r"\b\d{2,6}\b", query):
            intent["has_code"] = True

        # Default to semantic if no clear signal
        if not intent["suggests_exact"] and not intent["suggests_semantic"]:
            intent["suggests_semantic"] = True

        return intent
