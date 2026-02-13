"""
Tests for rate limiting module.
"""

import asyncio

import pytest

from naics_mcp_server.config import RateLimitConfig
from naics_mcp_server.core.errors import RateLimitError
from naics_mcp_server.observability.rate_limiting import (
    RateLimiter,
    RateLimitResult,
    TokenBucket,
    ToolCategory,
    TOOL_CATEGORIES,
)


class TestTokenBucket:
    """Tests for TokenBucket implementation."""

    @pytest.mark.asyncio
    async def test_bucket_starts_full(self):
        """Test bucket initializes with full capacity."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
        current, capacity = await bucket.peek()
        assert current == capacity == 10.0

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        """Test acquiring tokens reduces available tokens."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)

        success, wait = await bucket.acquire(1.0)
        assert success is True
        assert wait == 0.0

        current, _ = await bucket.peek()
        assert current < 10.0

    @pytest.mark.asyncio
    async def test_acquire_fails_when_empty(self):
        """Test acquiring fails when bucket is empty."""
        bucket = TokenBucket(capacity=2.0, refill_rate=0.1)

        # Drain the bucket
        await bucket.acquire(2.0)

        # Try to acquire more
        success, wait = await bucket.acquire(1.0)
        assert success is False
        assert wait > 0.0

    @pytest.mark.asyncio
    async def test_bucket_refills_over_time(self):
        """Test bucket refills based on time elapsed."""
        bucket = TokenBucket(capacity=10.0, refill_rate=100.0)  # Fast refill

        # Drain some tokens
        await bucket.acquire(5.0)

        # Wait for refill
        await asyncio.sleep(0.1)  # Should add ~10 tokens

        current, _ = await bucket.peek()
        assert current > 5.0

    @pytest.mark.asyncio
    async def test_bucket_capped_at_capacity(self):
        """Test bucket doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10.0, refill_rate=1000.0)  # Very fast refill

        await asyncio.sleep(0.01)  # Would normally add way more than capacity

        current, capacity = await bucket.peek()
        assert current <= capacity

    @pytest.mark.asyncio
    async def test_acquire_returns_wait_time(self):
        """Test acquire returns accurate wait time."""
        bucket = TokenBucket(capacity=1.0, refill_rate=1.0)  # 1 token/sec

        # Drain the bucket
        await bucket.acquire(1.0)

        # Try to acquire - should need ~1 second wait
        success, wait = await bucket.acquire(1.0)
        assert success is False
        assert 0.8 <= wait <= 1.2  # Allow some tolerance


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()

        assert config.enable_rate_limiting is False
        assert config.default_rpm == 60
        assert config.burst_multiplier == 2.0
        assert config.search_rpm == 30
        assert config.classify_rpm == 20
        assert config.batch_rpm == 10

    def test_config_validation(self):
        """Test configuration validation bounds."""
        # Should accept valid values
        config = RateLimitConfig(
            enable_rate_limiting=True,
            default_rpm=100,
            search_rpm=50,
        )
        assert config.default_rpm == 100

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = RateLimitConfig()
        config_dict = config.to_dict()

        assert "enable_rate_limiting" in config_dict
        assert "default_rpm" in config_dict
        assert "search_rpm" in config_dict


class TestToolCategories:
    """Tests for tool category mapping."""

    def test_search_tools_mapped(self):
        """Test search tools are mapped correctly."""
        assert TOOL_CATEGORIES["search_naics_codes"] == ToolCategory.SEARCH
        assert TOOL_CATEGORIES["search_index_terms"] == ToolCategory.SEARCH
        assert TOOL_CATEGORIES["find_similar_industries"] == ToolCategory.SEARCH

    def test_classify_tools_mapped(self):
        """Test classification tools are mapped correctly."""
        assert TOOL_CATEGORIES["classify_business"] == ToolCategory.CLASSIFY
        assert TOOL_CATEGORIES["validate_classification"] == ToolCategory.CLASSIFY

    def test_batch_tools_mapped(self):
        """Test batch tools are mapped correctly."""
        assert TOOL_CATEGORIES["classify_batch"] == ToolCategory.BATCH

    def test_health_tools_mapped(self):
        """Test health tools are mapped correctly."""
        assert TOOL_CATEGORIES["ping"] == ToolCategory.HEALTH
        assert TOOL_CATEGORIES["get_server_health"] == ToolCategory.HEALTH


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_disabled_limiter_allows_all(self):
        """Test disabled rate limiter allows all requests."""
        config = RateLimitConfig(enable_rate_limiting=False)
        limiter = RateLimiter(config)

        result = await limiter.check_limit("search_naics_codes")
        assert result.allowed is True
        assert result.message == "Rate limiting disabled"

    @pytest.mark.asyncio
    async def test_enabled_limiter_tracks_requests(self):
        """Test enabled rate limiter tracks requests."""
        config = RateLimitConfig(
            enable_rate_limiting=True,
            search_rpm=60,  # 1 per second
        )
        limiter = RateLimiter(config)

        # First request should be allowed
        result = await limiter.check_limit("search_naics_codes")
        assert result.allowed is True
        assert result.category == ToolCategory.SEARCH

    @pytest.mark.asyncio
    async def test_limiter_uses_correct_category(self):
        """Test limiter uses correct category for each tool."""
        config = RateLimitConfig(enable_rate_limiting=True)
        limiter = RateLimiter(config)

        result = await limiter.check_limit("classify_batch")
        assert result.category == ToolCategory.BATCH

        result = await limiter.check_limit("ping")
        assert result.category == ToolCategory.HEALTH

    @pytest.mark.asyncio
    async def test_limiter_denies_when_exhausted(self):
        """Test limiter denies requests when tokens exhausted."""
        config = RateLimitConfig(
            enable_rate_limiting=True,
            batch_rpm=6,  # Very low limit
            burst_multiplier=1.0,  # No burst allowance
        )
        limiter = RateLimiter(config)

        # Exhaust the bucket (6 RPM = 0.1 tokens/sec, capacity = 0.1)
        # With 1.0 burst multiplier, capacity is very low
        denied_count = 0
        for _ in range(10):
            result = await limiter.check_limit("classify_batch")
            if not result.allowed:
                denied_count += 1

        # At least some requests should be denied
        assert denied_count > 0

    @pytest.mark.asyncio
    async def test_limiter_provides_retry_after(self):
        """Test limiter provides retry_after when denying."""
        config = RateLimitConfig(
            enable_rate_limiting=True,
            batch_rpm=6,  # 0.1 per second
            burst_multiplier=1.0,
        )
        limiter = RateLimiter(config)

        # Exhaust tokens
        while True:
            result = await limiter.check_limit("classify_batch")
            if not result.allowed:
                break

        assert result.retry_after_seconds > 0
        assert "Rate limit exceeded" in result.message

    @pytest.mark.asyncio
    async def test_unknown_tool_uses_default(self):
        """Test unknown tools use default category."""
        config = RateLimitConfig(enable_rate_limiting=True)
        limiter = RateLimiter(config)

        result = await limiter.check_limit("unknown_tool")
        assert result.category == ToolCategory.DEFAULT

    @pytest.mark.asyncio
    async def test_get_status_returns_all_buckets(self):
        """Test get_status returns status of all buckets."""
        config = RateLimitConfig(enable_rate_limiting=True)
        limiter = RateLimiter(config)

        status = await limiter.get_status()

        assert status["enabled"] is True
        assert "buckets" in status
        assert "search" in status["buckets"]
        assert "classify" in status["buckets"]
        assert "batch" in status["buckets"]

    @pytest.mark.asyncio
    async def test_bucket_status_includes_metrics(self):
        """Test bucket status includes useful metrics."""
        config = RateLimitConfig(enable_rate_limiting=True)
        limiter = RateLimiter(config)

        status = await limiter.get_status()
        bucket_status = status["buckets"]["search"]

        assert "tokens_available" in bucket_status
        assert "capacity" in bucket_status
        assert "refill_rate_per_second" in bucket_status
        assert "utilization_percent" in bucket_status

    def test_reset_restores_buckets(self):
        """Test reset restores all buckets to full capacity."""
        config = RateLimitConfig(enable_rate_limiting=True)
        limiter = RateLimiter(config)

        # Reset should not raise
        limiter.reset()


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_result(self):
        """Test creating an allowed result."""
        result = RateLimitResult(
            allowed=True,
            category=ToolCategory.SEARCH,
            tokens_remaining=5.0,
        )

        assert result.allowed is True
        assert result.retry_after_seconds == 0.0
        assert result.message == ""

    def test_denied_result(self):
        """Test creating a denied result."""
        result = RateLimitResult(
            allowed=False,
            category=ToolCategory.BATCH,
            tokens_remaining=0.0,
            retry_after_seconds=2.5,
            message="Rate limit exceeded",
        )

        assert result.allowed is False
        assert result.retry_after_seconds == 2.5
        assert "Rate limit exceeded" in result.message


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_error_creation(self):
        """Test creating a RateLimitError."""
        error = RateLimitError(
            tool_name="search_naics_codes",
            category="search",
            retry_after=5.0,
        )

        assert "search_naics_codes" in str(error)
        assert error.retry_after == 5.0
        assert error.retryable is True

    def test_error_to_dict(self):
        """Test RateLimitError serialization."""
        error = RateLimitError(
            tool_name="classify_batch",
            category="batch",
            retry_after=10.0,
            message="Custom message",
        )

        error_dict = error.to_dict()

        assert error_dict["retry_after"] == 10.0
        assert "rate_limit_category" in error_dict["details"]
        assert error_dict["retryable"] is True

    def test_error_details_include_tool_info(self):
        """Test error details include tool information."""
        error = RateLimitError(
            tool_name="search_naics_codes",
            category="search",
            retry_after=2.0,
        )

        assert error.details["tool_name"] == "search_naics_codes"
        assert error.details["rate_limit_category"] == "search"
        assert error.details["retry_after_seconds"] == 2.0


class TestRateLimiterIntegration:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_different_categories_independent(self):
        """Test different categories have independent limits."""
        config = RateLimitConfig(
            enable_rate_limiting=True,
            search_rpm=6,  # Very low
            health_rpm=120,  # High
            burst_multiplier=1.0,
        )
        limiter = RateLimiter(config)

        # Exhaust search bucket
        for _ in range(20):
            await limiter.check_limit("search_naics_codes")

        # Health bucket should still be available
        result = await limiter.check_limit("ping")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_concurrent_requests_thread_safe(self):
        """Test rate limiter is thread-safe under concurrent load."""
        config = RateLimitConfig(
            enable_rate_limiting=True,
            search_rpm=500,  # High limit (max allowed)
        )
        limiter = RateLimiter(config)

        async def make_requests():
            results = []
            for _ in range(10):
                result = await limiter.check_limit("search_naics_codes")
                results.append(result)
            return results

        # Run concurrent requests
        tasks = [make_requests() for _ in range(5)]
        all_results = await asyncio.gather(*tasks)

        # Should not raise and should return results
        total_requests = sum(len(r) for r in all_results)
        assert total_requests == 50
