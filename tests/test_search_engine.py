"""
Unit tests for NAICSSearchEngine.

Tests search operations, confidence scoring, caching, and result generation.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from naics_mcp_server.core.search_engine import (
    NAICSSearchEngine, ConfidenceCalculator, SearchCache,
    generate_search_guidance
)
from naics_mcp_server.core.database import NAICSDatabase
from naics_mcp_server.core.embeddings import TextEmbedder
from naics_mcp_server.config import SearchConfig
from naics_mcp_server.models.naics_models import NAICSCode, NAICSLevel
from naics_mcp_server.models.search_models import (
    NAICSMatch, SearchResults, SearchStrategy,
    QueryTerms, QueryMetadata, ConfidenceScore
)


class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator."""

    @pytest.fixture
    def calculator(self):
        return ConfidenceCalculator()

    def test_calculate_basic_confidence(self, calculator):
        """Should calculate confidence from all factors."""
        confidence = calculator.calculate(
            semantic_score=0.8,
            lexical_score=0.6,
            index_term_score=0.5,
            specificity_score=1.0,
            cross_ref_factor=1.0
        )

        assert isinstance(confidence, ConfidenceScore)
        assert 0.0 <= confidence.overall <= 1.0
        assert confidence.semantic == 0.8
        assert confidence.lexical == 0.6
        assert confidence.index_term == 0.5
        assert confidence.specificity == 1.0

    def test_calculate_high_semantic_score(self, calculator):
        """High semantic score should result in high overall confidence."""
        confidence = calculator.calculate(
            semantic_score=0.95,
            lexical_score=0.0,
            index_term_score=0.0,
            specificity_score=1.0,
            cross_ref_factor=1.0
        )

        # Semantic weight is 0.40, so 0.95 * 0.40 = 0.38
        # Plus specificity 1.0 * 0.15 = 0.15, cross_ref 1.0 * 0.10 = 0.10
        assert confidence.overall >= 0.5

    def test_calculate_cross_ref_penalty(self, calculator):
        """Cross-reference factor < 1.0 should penalize score."""
        confidence_no_penalty = calculator.calculate(
            semantic_score=0.8,
            lexical_score=0.6,
            cross_ref_factor=1.0
        )

        confidence_with_penalty = calculator.calculate(
            semantic_score=0.8,
            lexical_score=0.6,
            cross_ref_factor=0.7  # 30% penalty
        )

        assert confidence_with_penalty.overall < confidence_no_penalty.overall

    def test_calculate_clamps_to_valid_range(self, calculator):
        """Overall score should always be between 0.0 and 1.0."""
        # Try to push overall very high
        confidence_high = calculator.calculate(
            semantic_score=1.0,
            lexical_score=1.0,
            index_term_score=1.0,
            specificity_score=1.0,
            cross_ref_factor=1.2  # Boost factor
        )
        assert confidence_high.overall <= 1.0

        # Try to push overall very low
        confidence_low = calculator.calculate(
            semantic_score=0.0,
            lexical_score=0.0,
            index_term_score=0.0,
            specificity_score=0.0,
            cross_ref_factor=0.0
        )
        assert confidence_low.overall >= 0.0

    def test_calculate_weighted_combination(self, calculator):
        """Should weight semantic > lexical > index_term = specificity > cross_ref."""
        # Only semantic (with neutral cross_ref to avoid penalty)
        semantic_only = calculator.calculate(
            semantic_score=1.0, lexical_score=0.0,
            index_term_score=0.0, specificity_score=0.0, cross_ref_factor=1.0
        )

        # Only lexical (with neutral cross_ref to avoid penalty)
        lexical_only = calculator.calculate(
            semantic_score=0.0, lexical_score=1.0,
            index_term_score=0.0, specificity_score=0.0, cross_ref_factor=1.0
        )

        # Semantic should have more impact (0.40 weight vs 0.20 weight)
        assert semantic_only.overall > lexical_only.overall


class TestSearchCache:
    """Tests for SearchCache."""

    @pytest.fixture
    def cache(self):
        return SearchCache(maxsize=10, ttl_seconds=60)

    @pytest.fixture
    def sample_results(self, sample_naics_code):
        """Create sample search results."""
        match = NAICSMatch(
            code=sample_naics_code,
            confidence=ConfidenceScore(
                semantic=0.8, lexical=0.6, index_term=0.0,
                specificity=1.0, cross_ref=0.0, overall=0.75
            ),
            match_reasons=["Test match"],
            rank=1
        )

        return SearchResults(
            matches=[match],
            query_metadata=QueryMetadata(
                original_query="test",
                expanded_terms=QueryTerms(original="test"),
                strategy_used="hybrid",
                was_expanded=False,
                processing_time_ms=100,
                total_candidates_considered=1
            )
        )

    def test_cache_put_and_get(self, cache, sample_results):
        """Should store and retrieve cached results."""
        cache.put("dog food", "hybrid", 10, 0.3, sample_results)

        cached = cache.get("dog food", "hybrid", 10, 0.3)

        assert cached is not None
        assert len(cached.matches) == 1

    def test_cache_miss_different_params(self, cache, sample_results):
        """Different parameters should result in cache miss."""
        cache.put("dog food", "hybrid", 10, 0.3, sample_results)

        # Different query
        assert cache.get("cat food", "hybrid", 10, 0.3) is None

        # Different strategy
        assert cache.get("dog food", "lexical", 10, 0.3) is None

        # Different limit
        assert cache.get("dog food", "hybrid", 5, 0.3) is None

        # Different min_confidence
        assert cache.get("dog food", "hybrid", 10, 0.5) is None

    def test_cache_respects_maxsize(self, sample_results):
        """Cache should evict oldest entries when maxsize is reached."""
        cache = SearchCache(maxsize=3, ttl_seconds=60)

        cache.put("query1", "hybrid", 10, 0.3, sample_results)
        cache.put("query2", "hybrid", 10, 0.3, sample_results)
        cache.put("query3", "hybrid", 10, 0.3, sample_results)

        # All three should be cached
        assert cache.get("query1", "hybrid", 10, 0.3) is not None
        assert cache.get("query2", "hybrid", 10, 0.3) is not None
        assert cache.get("query3", "hybrid", 10, 0.3) is not None

        # Adding fourth should evict oldest
        cache.put("query4", "hybrid", 10, 0.3, sample_results)

        # query1 should be evicted (oldest)
        assert cache.get("query1", "hybrid", 10, 0.3) is None
        assert cache.get("query4", "hybrid", 10, 0.3) is not None

    def test_cache_ttl_expiration(self, sample_results):
        """Cache entries should expire after TTL."""
        cache = SearchCache(maxsize=10, ttl_seconds=1)  # 1 second TTL

        cache.put("query", "hybrid", 10, 0.3, sample_results)

        # Should be cached immediately
        assert cache.get("query", "hybrid", 10, 0.3) is not None

        # Wait for TTL expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("query", "hybrid", 10, 0.3) is None

    def test_cache_stats(self, cache, sample_results):
        """Should track hit/miss statistics."""
        cache.put("query", "hybrid", 10, 0.3, sample_results)

        # Miss
        cache.get("nonexistent", "hybrid", 10, 0.3)

        # Hit
        cache.get("query", "hybrid", 10, 0.3)

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1
        assert stats["maxsize"] == 10

    def test_cache_clear(self, cache, sample_results):
        """Clear should remove all entries and reset stats."""
        cache.put("query1", "hybrid", 10, 0.3, sample_results)
        cache.put("query2", "hybrid", 10, 0.3, sample_results)
        cache.get("query1", "hybrid", 10, 0.3)  # hit

        cache.clear()

        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestGenerateSearchGuidance:
    """Tests for generate_search_guidance function."""

    @pytest.fixture
    def empty_results(self):
        """Create empty search results."""
        return SearchResults(
            matches=[],
            query_metadata=QueryMetadata(
                original_query="test",
                expanded_terms=QueryTerms(original="test"),
                strategy_used="hybrid",
                was_expanded=False,
                processing_time_ms=100,
                total_candidates_considered=0
            )
        )

    @pytest.fixture
    def single_match_results(self, sample_naics_code):
        """Create results with single high-confidence match."""
        match = NAICSMatch(
            code=sample_naics_code,
            confidence=ConfidenceScore(
                semantic=0.9, lexical=0.8, index_term=0.5,
                specificity=1.0, cross_ref=0.0, overall=0.85
            ),
            match_reasons=["Semantic similarity"],
            matched_index_terms=["Dog food manufacturing"],
            rank=1
        )

        return SearchResults(
            matches=[match],
            query_metadata=QueryMetadata(
                original_query="dog food",
                expanded_terms=QueryTerms(original="dog food"),
                strategy_used="hybrid",
                was_expanded=False,
                processing_time_ms=100,
                total_candidates_considered=1
            )
        )

    def test_guidance_for_no_results(self, empty_results):
        """Should provide guidance when no matches found."""
        guidance = generate_search_guidance(empty_results)

        assert len(guidance) > 0
        assert any("No matches" in g for g in guidance)

    def test_guidance_includes_top_confidence(self, single_match_results):
        """Should include top confidence in guidance."""
        guidance = generate_search_guidance(single_match_results)

        assert any("85" in g or "0.85" in g or "85%" in g for g in guidance)

    def test_guidance_notes_index_term_matches(self, single_match_results):
        """Should note when results match index terms."""
        guidance = generate_search_guidance(single_match_results)

        assert any("index term" in g.lower() for g in guidance)

    def test_guidance_for_multiple_similar_results(self, sample_naics_code):
        """Should note when multiple results have similar confidence."""
        matches = []
        for i, conf in enumerate([0.85, 0.84, 0.83]):
            match = NAICSMatch(
                code=sample_naics_code,
                confidence=ConfidenceScore(
                    semantic=conf, lexical=0.0, index_term=0.0,
                    specificity=1.0, cross_ref=0.0, overall=conf
                ),
                match_reasons=["Test"],
                rank=i + 1
            )
            matches.append(match)

        results = SearchResults(
            matches=matches,
            query_metadata=QueryMetadata(
                original_query="test",
                expanded_terms=QueryTerms(original="test"),
                strategy_used="hybrid",
                was_expanded=False,
                processing_time_ms=100,
                total_candidates_considered=3
            )
        )

        guidance = generate_search_guidance(results)

        # Should note similar-strength matches
        assert any("similar" in g.lower() for g in guidance)


class TestNAICSSearchEngine:
    """Tests for NAICSSearchEngine."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock TextEmbedder."""
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        embedder.embed_batch.return_value = np.random.rand(10, 384).astype(np.float32)
        return embedder

    @pytest.fixture
    def search_engine(self, populated_database, mock_embedder):
        """Create search engine with populated database."""
        config = SearchConfig()
        engine = NAICSSearchEngine(
            database=populated_database,
            embedder=mock_embedder,
            config=config
        )
        return engine

    def test_calculate_lexical_score_perfect_match(self, search_engine):
        """Perfect word overlap should give high score."""
        score = search_engine._calculate_lexical_score(
            "dog food manufacturing",
            "dog food manufacturing"
        )

        assert score == 1.0

    def test_calculate_lexical_score_partial_match(self, search_engine):
        """Partial word overlap should give moderate score."""
        score = search_engine._calculate_lexical_score(
            "dog food",
            "dog food manufacturing from ingredients"
        )

        assert 0.0 < score < 1.0

    def test_calculate_lexical_score_no_match(self, search_engine):
        """No word overlap should give zero score."""
        score = search_engine._calculate_lexical_score(
            "dog food",
            "beverage manufacturing"
        )

        assert score == 0.0

    def test_calculate_lexical_score_empty_query(self, search_engine):
        """Empty query should give zero score."""
        score = search_engine._calculate_lexical_score(
            "",
            "dog food manufacturing"
        )

        assert score == 0.0

    def test_calculate_specificity_score_national_industry(self, search_engine):
        """National industry (6-digit) should get highest specificity."""
        score = search_engine._calculate_specificity_score(NAICSLevel.NATIONAL_INDUSTRY)
        assert score == 1.0

    def test_calculate_specificity_score_sector(self, search_engine):
        """Sector (2-digit) should get lowest specificity."""
        score = search_engine._calculate_specificity_score(NAICSLevel.SECTOR)
        assert score == 0.2

    def test_calculate_specificity_score_hierarchy(self, search_engine):
        """Specificity should increase with code length."""
        sector_score = search_engine._calculate_specificity_score(NAICSLevel.SECTOR)
        subsector_score = search_engine._calculate_specificity_score(NAICSLevel.SUBSECTOR)
        industry_group_score = search_engine._calculate_specificity_score(NAICSLevel.INDUSTRY_GROUP)
        naics_industry_score = search_engine._calculate_specificity_score(NAICSLevel.NAICS_INDUSTRY)
        national_score = search_engine._calculate_specificity_score(NAICSLevel.NATIONAL_INDUSTRY)

        assert sector_score < subsector_score < industry_group_score < naics_industry_score < national_score

    @pytest.mark.asyncio
    async def test_lexical_search_finds_matches(self, search_engine):
        """Lexical search should find matching codes."""
        expanded_terms = QueryTerms(original="dog food")

        matches = await search_engine._lexical_search("dog food", expanded_terms)

        assert len(matches) > 0
        codes = [m.code.node_code for m in matches]
        assert "311111" in codes

    @pytest.mark.asyncio
    async def test_lexical_search_returns_naics_matches(self, search_engine):
        """Lexical search should return NAICSMatch objects."""
        expanded_terms = QueryTerms(original="manufacturing")

        matches = await search_engine._lexical_search("manufacturing", expanded_terms)

        assert all(isinstance(m, NAICSMatch) for m in matches)
        assert all(hasattr(m, 'confidence') for m in matches)
        assert all(hasattr(m, 'code') for m in matches)

    @pytest.mark.asyncio
    async def test_search_fallback_to_lexical(self, search_engine):
        """Search should fall back to lexical when embeddings not ready."""
        search_engine.embeddings_ready = False

        results = await search_engine.search(
            "dog food",
            strategy=SearchStrategy.SEMANTIC
        )

        # Should still return results via lexical fallback
        assert isinstance(results, SearchResults)
        assert results.query_metadata.strategy_used == "lexical"

    @pytest.mark.asyncio
    async def test_search_uses_cache(self, search_engine):
        """Subsequent identical searches should use cache."""
        search_engine.embeddings_ready = False

        # First search
        results1 = await search_engine.search(
            "dog food",
            strategy=SearchStrategy.LEXICAL
        )

        # Second identical search
        results2 = await search_engine.search(
            "dog food",
            strategy=SearchStrategy.LEXICAL
        )

        # Cache should have been used
        cache_stats = search_engine.search_cache.get_stats()
        assert cache_stats["hits"] >= 1

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, search_engine):
        """Search should respect limit parameter."""
        search_engine.embeddings_ready = False

        results = await search_engine.search(
            "manufacturing",
            strategy=SearchStrategy.LEXICAL,
            limit=3
        )

        assert len(results.matches) <= 3

    @pytest.mark.asyncio
    async def test_search_respects_min_confidence(self, search_engine):
        """Search should filter by min_confidence."""
        search_engine.embeddings_ready = False

        results = await search_engine.search(
            "manufacturing",
            strategy=SearchStrategy.LEXICAL,
            min_confidence=0.0  # Low threshold
        )

        low_threshold_count = len(results.matches)

        results = await search_engine.search(
            "manufacturing",
            strategy=SearchStrategy.LEXICAL,
            min_confidence=0.5  # Higher threshold
        )

        high_threshold_count = len(results.matches)

        assert high_threshold_count <= low_threshold_count

    @pytest.mark.asyncio
    async def test_search_results_are_ranked(self, search_engine):
        """Search results should be sorted by confidence and ranked."""
        search_engine.embeddings_ready = False

        results = await search_engine.search(
            "food manufacturing",
            strategy=SearchStrategy.LEXICAL,
            limit=5
        )

        if len(results.matches) >= 2:
            # Check descending confidence order
            confidences = [m.confidence.overall for m in results.matches]
            assert confidences == sorted(confidences, reverse=True)

            # Check ranks are sequential
            ranks = [m.rank for m in results.matches]
            assert ranks == list(range(1, len(ranks) + 1))

    @pytest.mark.asyncio
    async def test_search_includes_metadata(self, search_engine):
        """Search results should include query metadata."""
        search_engine.embeddings_ready = False

        results = await search_engine.search(
            "dog food",
            strategy=SearchStrategy.LEXICAL
        )

        assert results.query_metadata is not None
        assert results.query_metadata.original_query == "dog food"
        assert results.query_metadata.strategy_used == "lexical"
        assert results.query_metadata.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_search_index_terms(self, search_engine):
        """Should search official index terms."""
        terms = await search_engine._search_index_terms("dog food")

        assert len(terms) > 0
        assert any(t.naics_code == "311111" for t in terms)


class TestSearchEngineWithEmbeddings:
    """Tests for search engine with embeddings enabled."""

    @pytest.fixture
    def search_engine_with_embeddings(self, populated_database, mock_embedder):
        """Create search engine with embeddings ready."""
        config = SearchConfig()
        engine = NAICSSearchEngine(
            database=populated_database,
            embedder=mock_embedder,
            config=config
        )
        engine.embeddings_ready = True

        # Mock the embedding search to return some results
        return engine

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder that returns consistent embeddings."""
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        embedder.embed_batch.return_value = np.random.rand(10, 384).astype(np.float32)
        return embedder

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_strategies(self, search_engine_with_embeddings):
        """Hybrid search should combine semantic and lexical."""
        # This test validates the combination logic
        expanded_terms = QueryTerms(original="dog food")
        index_term_codes = {"311111"}

        # Patch the semantic search to avoid actual embedding operations
        with patch.object(
            search_engine_with_embeddings, '_semantic_search',
            new_callable=AsyncMock
        ) as mock_semantic:
            # Return a match from semantic search
            mock_semantic.return_value = []

            matches = await search_engine_with_embeddings._hybrid_search(
                "dog food", expanded_terms, index_term_codes
            )

            # Should have called semantic search
            mock_semantic.assert_called_once()


class TestSearchEngineEdgeCases:
    """Edge case tests for search engine."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock(spec=TextEmbedder)
        embedder.model = MagicMock()
        embedder.embed_text.return_value = np.random.rand(384).astype(np.float32)
        return embedder

    @pytest.fixture
    def search_engine(self, populated_database, mock_embedder):
        config = SearchConfig()
        engine = NAICSSearchEngine(
            database=populated_database,
            embedder=mock_embedder,
            config=config
        )
        engine.embeddings_ready = False
        return engine

    @pytest.mark.asyncio
    async def test_search_empty_query(self, search_engine):
        """Should handle empty query gracefully."""
        results = await search_engine.search(
            "",
            strategy=SearchStrategy.LEXICAL
        )

        assert isinstance(results, SearchResults)

    @pytest.mark.asyncio
    async def test_search_whitespace_query(self, search_engine):
        """Should handle whitespace-only query."""
        results = await search_engine.search(
            "   ",
            strategy=SearchStrategy.LEXICAL
        )

        assert isinstance(results, SearchResults)

    @pytest.mark.asyncio
    async def test_search_special_characters(self, search_engine):
        """Should handle special characters in query."""
        results = await search_engine.search(
            "dog & cat food (manufacturing)",
            strategy=SearchStrategy.LEXICAL
        )

        assert isinstance(results, SearchResults)

    @pytest.mark.asyncio
    async def test_search_unicode_query(self, search_engine):
        """Should handle Unicode characters in query."""
        results = await search_engine.search(
            "café manufacturing",
            strategy=SearchStrategy.LEXICAL
        )

        assert isinstance(results, SearchResults)

    @pytest.mark.asyncio
    async def test_search_very_long_query(self, search_engine):
        """Should handle very long queries."""
        long_query = "manufacturing " * 100  # Very long query

        results = await search_engine.search(
            long_query,
            strategy=SearchStrategy.LEXICAL
        )

        assert isinstance(results, SearchResults)
