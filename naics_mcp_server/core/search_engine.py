"""
Core search engine for NAICS codes.

This module orchestrates all search operations with clear delegation
and purposeful composition of search strategies.

NAICS-specific features:
- 5-level hierarchy support
- Index term boosting
- Cross-reference integration
- NAICS-specific confidence scoring
"""

import hashlib
import logging
import time
from typing import Any

from ..config import SearchConfig
from ..core.cross_reference import CrossReferenceService
from ..core.database import NAICSDatabase
from ..core.embeddings import EmbeddingCache, TextEmbedder
from ..core.errors import DatabaseError, EmbeddingError
from ..core.query_expansion import QueryExpander, SmartQueryParser
from ..observability.metrics import (
    record_cache_hit,
    record_cache_miss,
    record_crossref_lookup,
    record_search_fallback,
    update_cache_stats,
)
from ..models.naics_models import CrossReference, IndexTerm, NAICSCode, NAICSLevel
from ..models.search_models import (
    ConfidenceScore,
    NAICSMatch,
    QueryMetadata,
    QueryTerms,
    SearchResults,
    SearchStrategy,
)

logger = logging.getLogger(__name__)


def generate_search_guidance(results: SearchResults) -> list[str]:
    """
    Generate observational guidance based on search results.

    This provides neutral, factual observations about the results
    without making quality judgments or assuming thresholds.

    Args:
        results: The search results to analyze

    Returns:
        List of observational statements about the results
    """
    observations = []

    if not results.matches:
        observations.append("No matches found - consider alternative terminology or broader terms")
        return observations

    # Describe the top result confidence
    top_confidence = results.matches[0].confidence.overall
    observations.append(f"Best match confidence: {top_confidence:.1%}")

    # Describe relative confidence patterns
    if len(results.matches) >= 2:
        confidence_gap = (
            results.matches[0].confidence.overall - results.matches[1].confidence.overall
        )

        if confidence_gap > 0.3:
            observations.append("Top result significantly stronger than alternatives")
        elif confidence_gap < 0.05:
            observations.append("Multiple similar-strength matches found")

        # Overall spread of results
        if len(results.matches) >= 3:
            confidences = [m.confidence.overall for m in results.matches[:5]]
            spread = max(confidences) - min(confidences)

            if spread < 0.1:
                observations.append("Results have similar confidence scores")
            elif spread > 0.5:
                observations.append("Wide variation in match confidence")

    # Note index term matches
    index_matches = sum(1 for m in results.matches[:10] if m.matched_index_terms)
    if index_matches > 0:
        observations.append(f"{index_matches} result(s) match official NAICS index terms")

    # Note exclusion warnings
    exclusions = results.get_exclusion_warnings()
    if exclusions:
        observations.append(
            f"WARNING: {len(exclusions)} result(s) have exclusion warnings - review carefully"
        )

    # Note if query was expanded
    if results.query_metadata.was_expanded:
        observations.append("Search included expanded/related terms")

    # Note hierarchical distribution
    levels = [m.code.level.value for m in results.matches[:10]]
    unique_levels = set(levels)
    if len(unique_levels) == 1:
        observations.append(f"All results at {levels[0]} level")
    elif "national_industry" in unique_levels and "sector" in unique_levels:
        observations.append("Results span from broad sectors to specific industries")

    return observations


class ConfidenceCalculator:
    """
    Calculates transparent confidence scores for search results.

    NAICS-specific formula:
    overall = (
        0.40 * semantic_score +
        0.20 * lexical_score +
        0.15 * index_term_match +
        0.15 * specificity_preference +
        0.10 * cross_ref_relevance
    )
    """

    def calculate(
        self,
        semantic_score: float,
        lexical_score: float,
        index_term_score: float = 0.0,
        specificity_score: float = 0.5,
        cross_ref_factor: float = 1.0,
    ) -> ConfidenceScore:
        """
        Calculate confidence score from multiple factors.

        Args:
            semantic_score: Semantic similarity (0-1)
            lexical_score: Text matching score (0-1)
            index_term_score: Official index term match (0-1)
            specificity_score: Level preference (0-1, 1=most specific)
            cross_ref_factor: Cross-reference penalty/boost (0.7-1.2)

        Returns:
            Comprehensive confidence score
        """
        # NAICS-specific weighted combination
        overall = (
            0.40 * semantic_score
            + 0.20 * lexical_score
            + 0.15 * index_term_score
            + 0.15 * specificity_score
            + 0.10 * max(0, min(1, cross_ref_factor))
        )

        # Apply cross-reference penalty if < 1.0
        if cross_ref_factor < 1.0:
            overall *= cross_ref_factor

        return ConfidenceScore(
            semantic=semantic_score,
            lexical=lexical_score,
            index_term=index_term_score,
            specificity=specificity_score,
            cross_ref=cross_ref_factor - 1.0,  # -0.3 to +0.2 range
            overall=min(1.0, max(0.0, overall)),
        )


class SearchCache:
    """
    Simple in-memory cache for search results.
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the search cache.

        Args:
            maxsize: Maximum number of cached queries
            ttl_seconds: Time to live for cached results (default 1 hour)
        """
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, query: str, strategy: str, limit: int, min_confidence: float) -> str:
        """Generate a unique cache key for the search parameters."""
        key_str = f"{query}:{strategy}:{limit}:{min_confidence}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self, query: str, strategy: str, limit: int, min_confidence: float
    ) -> SearchResults | None:
        """Retrieve cached results if available and not expired."""
        key = self._get_cache_key(query, strategy, limit, min_confidence)

        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                record_cache_hit("search")
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                del self.cache[key]

        self.misses += 1
        record_cache_miss("search")
        return None

    def put(
        self, query: str, strategy: str, limit: int, min_confidence: float, results: SearchResults
    ) -> None:
        """Store search results in cache."""
        if len(self.cache) >= self.maxsize:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        key = self._get_cache_key(query, strategy, limit, min_confidence)
        self.cache[key] = (results, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class NAICSSearchEngine:
    """
    A semantic search engine for NAICS codes.

    This class orchestrates search operations with clear delegation:
    - Semantic search through embeddings
    - Lexical search through text matching
    - Index term search for official terms
    - Hybrid scoring through weighted combination
    - Cross-reference integration for accuracy
    """

    def __init__(
        self, database: NAICSDatabase, embedder: TextEmbedder, config: SearchConfig = None
    ):
        """
        Initialize with explicit dependencies.

        Args:
            database: Database connection
            embedder: Text embedding model
            config: Search configuration
        """
        self.database = database
        self.embedder = embedder
        self.config = config or SearchConfig()

        self.query_expander = QueryExpander()
        self.query_parser = SmartQueryParser()
        self.confidence_calculator = ConfidenceCalculator()
        self.cross_ref_service = CrossReferenceService(database)
        self.embedding_cache = EmbeddingCache(max_size=1000)
        self.search_cache = SearchCache(maxsize=100, ttl_seconds=3600)

        # Track if embeddings are initialized
        self.embeddings_ready = False

    async def initialize_embeddings(self, force_rebuild: bool = False) -> dict[str, Any]:
        """
        Initialize or verify embeddings are ready.

        Args:
            force_rebuild: Force regeneration of all embeddings

        Returns:
            Status information about initialization
        """
        logger.info("Initializing NAICS embeddings...")
        start_time = time.time()

        try:
            # Check if embeddings table exists and has data
            tables = self.database.connection.execute("SHOW TABLES").fetchall()
            has_embeddings_table = any("naics_embeddings" in str(t) for t in tables)

            if not has_embeddings_table or force_rebuild:
                logger.info("Creating/rebuilding embeddings...")
                result = await self._generate_all_embeddings()

                self.embeddings_ready = True

                return {
                    "status": "initialized",
                    "embeddings_generated": result["count"],
                    "time_seconds": time.time() - start_time,
                    "action": "created" if not has_embeddings_table else "rebuilt",
                }
            else:
                # Verify embeddings are populated
                count = self.database.connection.execute(
                    "SELECT COUNT(*) FROM naics_embeddings"
                ).fetchone()[0]

                if count == 0:
                    result = await self._generate_all_embeddings()
                    self.embeddings_ready = True

                    return {
                        "status": "initialized",
                        "embeddings_generated": result["count"],
                        "time_seconds": time.time() - start_time,
                        "action": "populated",
                    }
                else:
                    self.embeddings_ready = True
                    return {
                        "status": "ready",
                        "embeddings_count": count,
                        "time_seconds": time.time() - start_time,
                        "action": "verified",
                    }

        except EmbeddingError as e:
            logger.error(f"Embedding error during initialization: {e}")
            return {
                "status": "error",
                "error": e.message,
                "error_category": e.category.value,
                "retryable": e.retryable,
                "time_seconds": time.time() - start_time,
            }
        except DatabaseError as e:
            logger.error(f"Database error during embedding initialization: {e}")
            return {
                "status": "error",
                "error": e.message,
                "error_category": e.category.value,
                "retryable": e.retryable,
                "time_seconds": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_category": "permanent",
                "retryable": False,
                "time_seconds": time.time() - start_time,
            }

    async def _generate_all_embeddings(self) -> dict[str, Any]:
        """
        Generate embeddings for all NAICS codes.

        Returns:
            Generation statistics
        """
        # Load the model if not already loaded
        if not self.embedder.model:
            self.embedder.load_model()

        # Get all codes with embedding text
        codes = self.database.connection.execute("""
            SELECT node_code, raw_embedding_text, title, description
            FROM naics_nodes
            WHERE raw_embedding_text IS NOT NULL
            AND raw_embedding_text != ''
        """).fetchall()

        if not codes:
            # Build embedding text if not pre-computed
            codes = self.database.connection.execute("""
                SELECT node_code, title || ' ' || COALESCE(description, ''), title, description
                FROM naics_nodes
            """).fetchall()

        logger.info(f"Generating embeddings for {len(codes)} NAICS codes...")

        # Process in batches
        batch_size = self.config.batch_size
        total_processed = 0

        for i in range(0, len(codes), batch_size):
            batch = codes[i : i + batch_size]

            # Extract texts for embedding
            texts = [row[1] if row[1] else f"{row[2]} {row[3] or ''}" for row in batch]

            # Generate embeddings
            embeddings = self.embedder.embed_batch(
                texts, batch_size=batch_size, normalize=self.config.normalize_embeddings
            )

            # Prepare batch data for insertion
            batch_data = []
            for j, (node_code, text, title, desc) in enumerate(batch):
                embedding_list = embeddings[j].tolist()
                batch_data.append([node_code, embedding_list, texts[j]])

            # Batch insert into database
            self.database.connection.executemany(
                """
                INSERT OR REPLACE INTO naics_embeddings
                (node_code, embedding, embedding_text)
                VALUES (?, ?, ?)
            """,
                batch_data,
            )

            total_processed += len(batch)

            if total_processed % 200 == 0:
                progress = (total_processed / len(codes)) * 100
                logger.info(
                    f"Generating embeddings: {progress:.1f}% complete ({total_processed}/{len(codes)})..."
                )

        logger.info(f"Successfully generated {total_processed} embeddings")

        return {"count": total_processed}

    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 10,
        min_confidence: float = None,
        include_cross_refs: bool = True,
    ) -> SearchResults:
        """
        Search for NAICS codes using the specified strategy.

        Args:
            query: Search query
            strategy: Search strategy to use
            limit: Maximum results
            min_confidence: Minimum confidence threshold
            include_cross_refs: Whether to check cross-references

        Returns:
            Search results with confidence scores
        """
        start_time = time.time()

        if min_confidence is None:
            min_confidence = self.config.min_confidence

        # Check cache first
        cached_result = self.search_cache.get(query, strategy.value, limit, min_confidence)
        if cached_result is not None:
            cached_result.query_metadata.processing_time_ms = int((time.time() - start_time) * 1000)
            return cached_result

        # Smart fallback: Check if embeddings are available
        if strategy in [SearchStrategy.HYBRID, SearchStrategy.SEMANTIC]:
            if not self.embeddings_ready:
                logger.warning(
                    f"Embeddings not ready for {strategy.value} search, falling back to lexical search"
                )
                strategy = SearchStrategy.LEXICAL

        # Parse query intent
        intent = self.query_parser.parse_intent(query)

        # Expand query if enabled
        expanded_terms = QueryTerms(original=query)
        if self.config.enable_query_expansion and not intent["has_code"]:
            expanded_terms = await self.query_expander.expand(query)

        # Search index terms first
        index_term_matches = await self._search_index_terms(query)
        index_term_codes = {it.naics_code for it in index_term_matches}

        # Execute search based on strategy with graceful degradation
        fallback_used = None
        try:
            if strategy == SearchStrategy.HYBRID:
                matches = await self._hybrid_search(query, expanded_terms, index_term_codes)
            elif strategy == SearchStrategy.SEMANTIC:
                matches = await self._semantic_search(query, expanded_terms)
            else:
                matches = await self._lexical_search(query, expanded_terms)
        except EmbeddingError as e:
            logger.warning(
                f"Embedding error during {strategy.value} search, falling back to lexical: {e}"
            )
            fallback_used = "lexical"
            record_search_fallback(strategy.value, "lexical")
            try:
                matches = await self._lexical_search(query, expanded_terms)
                strategy = SearchStrategy.LEXICAL
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                matches = []
        except DatabaseError as e:
            if e.retryable:
                logger.warning(f"Transient database error during search: {e}")
                # Could implement retry here
            logger.error(f"Database error during {strategy.value} search: {e}")
            matches = []
        except Exception as e:
            logger.error(f"Search failed with {strategy.value}, falling back to lexical: {e}")
            fallback_used = "lexical"
            record_search_fallback(strategy.value, "lexical")
            try:
                matches = await self._lexical_search(query, expanded_terms)
                strategy = SearchStrategy.LEXICAL
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                matches = []

        # Enhance matches with index term info
        for match in matches:
            if match.code.node_code in index_term_codes:
                relevant_terms = [
                    it for it in index_term_matches if it.naics_code == match.code.node_code
                ]
                match.matched_index_terms = [it.index_term for it in relevant_terms[:3]]
                # Boost confidence for index term matches
                match.confidence.index_term = min(1.0, len(relevant_terms) * 0.3)

        # Check cross-references if enabled
        cross_refs_checked = 0
        total_exclusions_found = 0
        if include_cross_refs and self.config.enable_cross_references:
            for match in matches[:20]:  # Only check top 20
                exclusions = await self.cross_ref_service.check_exclusions(
                    query, match.code.node_code
                )
                if exclusions:
                    match.exclusion_warnings = [e["warning"] for e in exclusions]
                    match.relevant_cross_refs = [
                        CrossReference(
                            source_code=match.code.node_code,
                            reference_type="excludes",
                            reference_text=e["warning"],
                            target_code=e["target_code"],
                            excluded_activity=e["excluded_activity"],
                        )
                        for e in exclusions
                    ]
                    total_exclusions_found += len(exclusions)
                cross_refs_checked += 1
            # Record cross-reference metrics
            record_crossref_lookup(total_exclusions_found)

        # Filter by minimum confidence
        matches = [m for m in matches if m.confidence.overall >= min_confidence]

        # Sort and limit results
        matches = sorted(matches, key=lambda x: x.confidence.overall, reverse=True)[:limit]

        # Add rank
        for i, match in enumerate(matches):
            match.rank = i + 1

        # Build metadata
        metadata = QueryMetadata(
            original_query=query,
            expanded_terms=expanded_terms,
            strategy_used=strategy.value,
            was_expanded=expanded_terms.was_expanded(),
            processing_time_ms=int((time.time() - start_time) * 1000),
            total_candidates_considered=len(matches),
            index_terms_searched=len(index_term_matches),
            cross_refs_checked=cross_refs_checked,
            fallback_used=fallback_used,
        )

        results = SearchResults(matches=matches, query_metadata=metadata)

        # Cache the results
        self.search_cache.put(query, strategy.value, limit, min_confidence, results)

        return results

    async def _search_index_terms(self, query: str) -> list[IndexTerm]:
        """Search the official NAICS index terms."""
        return await self.database.search_index_terms(query, limit=50)

    async def _semantic_search(self, query: str, expanded_terms: QueryTerms) -> list[NAICSMatch]:
        """
        Perform semantic search using embeddings.
        """
        # Check cache first
        cached = self.embedding_cache.get(query)
        if cached is not None:
            query_embedding = cached
        else:
            query_embedding = self.embedder.embed_text(
                query, normalize=self.config.normalize_embeddings
            )
            self.embedding_cache.put(query, query_embedding)

        # Search using DuckDB's vector similarity
        results = self.database.connection.execute(
            """
            SELECT
                n.*,
                array_cosine_similarity(e.embedding, ?::FLOAT[384]) as similarity
            FROM naics_nodes n
            JOIN naics_embeddings e ON n.node_code = e.node_code
            WHERE array_cosine_similarity(e.embedding, ?::FLOAT[384]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
        """,
            [query_embedding.tolist(), query_embedding.tolist(), 0.3, self.config.max_candidates],
        ).fetchall()

        # Convert to matches
        matches = []
        for row in results:
            code = self.database._row_to_naics_code(row[:11])
            similarity = row[11]

            match = NAICSMatch(
                code=code,
                confidence=self.confidence_calculator.calculate(
                    semantic_score=similarity,
                    lexical_score=0.0,
                    specificity_score=self._calculate_specificity_score(code.level),
                ),
                match_reasons=["Semantic similarity"],
                embedding_similarity=similarity,
                text_similarity=0.0,
                hierarchy_path=code.get_hierarchy_path(),
                rank=0,
            )
            matches.append(match)

        return matches

    async def _lexical_search(self, query: str, expanded_terms: QueryTerms) -> list[NAICSMatch]:
        """
        Perform lexical search using text matching.
        """
        search_terms = expanded_terms.all_terms()
        results = await self.database.search_by_text(search_terms, self.config.max_candidates)

        matches = []
        for result in results:
            code = NAICSCode(
                node_code=result["node_code"],
                level=NAICSLevel(result["level"]),
                title=result["title"],
                description=result.get("description"),
                sector_code=result.get("sector_code"),
                subsector_code=result.get("subsector_code"),
                industry_group_code=result.get("industry_group_code"),
                naics_industry_code=result.get("naics_industry_code"),
                raw_embedding_text=result.get("raw_embedding_text"),
            )

            lexical_score = self._calculate_lexical_score(
                query.lower(), f"{code.title} {code.description or ''}".lower()
            )

            match = NAICSMatch(
                code=code,
                confidence=self.confidence_calculator.calculate(
                    semantic_score=0.0,
                    lexical_score=lexical_score,
                    specificity_score=self._calculate_specificity_score(code.level),
                ),
                match_reasons=["Text match"],
                embedding_similarity=0.0,
                text_similarity=lexical_score,
                hierarchy_path=code.get_hierarchy_path(),
                rank=0,
            )
            matches.append(match)

        return matches

    async def _hybrid_search(
        self, query: str, expanded_terms: QueryTerms, index_term_codes: set
    ) -> list[NAICSMatch]:
        """
        Perform hybrid search combining semantic and lexical.
        """
        semantic_matches = await self._semantic_search(query, expanded_terms)
        lexical_matches = await self._lexical_search(query, expanded_terms)

        # Combine results
        combined = {}

        # Add semantic matches
        for match in semantic_matches:
            combined[match.code.node_code] = match

        # Merge lexical matches
        for lex_match in lexical_matches:
            node_code = lex_match.code.node_code

            if node_code in combined:
                sem_match = combined[node_code]
                # Recalculate with both scores
                index_term_score = 0.5 if node_code in index_term_codes else 0.0
                sem_match.confidence = self.confidence_calculator.calculate(
                    semantic_score=sem_match.embedding_similarity,
                    lexical_score=lex_match.text_similarity,
                    index_term_score=index_term_score,
                    specificity_score=self._calculate_specificity_score(sem_match.code.level),
                )
                sem_match.text_similarity = lex_match.text_similarity
                sem_match.match_reasons.append("Text match")
            else:
                combined[node_code] = lex_match

        # Sort by combined confidence
        matches = sorted(combined.values(), key=lambda x: x.confidence.overall, reverse=True)

        return matches

    def _calculate_lexical_score(self, query: str, text: str) -> float:
        """Calculate simple lexical similarity score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words & text_words
        union = query_words | text_words

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_specificity_score(self, level: NAICSLevel) -> float:
        """
        Calculate specificity preference score.

        More specific codes (6-digit) get higher scores.
        """
        scores = {
            NAICSLevel.NATIONAL_INDUSTRY: 1.0,
            NAICSLevel.NAICS_INDUSTRY: 0.8,
            NAICSLevel.INDUSTRY_GROUP: 0.6,
            NAICSLevel.SUBSECTOR: 0.4,
            NAICSLevel.SECTOR: 0.2,
        }
        return scores.get(level, 0.5)
