"""
Rate limiting for NAICS MCP Server tools.

Implements token bucket algorithm for controlling request rates.
Each tool category has its own bucket with configurable limits.

Token Bucket Algorithm:
- Each bucket has a maximum capacity (burst tokens)
- Tokens are added at a fixed rate (RPM / 60 = tokens per second)
- Each request consumes one token
- If no tokens available, request is rate limited
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any

from ..config import RateLimitConfig, get_rate_limit_config

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools for rate limiting purposes."""

    SEARCH = "search"
    CLASSIFY = "classify"
    BATCH = "batch"
    HIERARCHY = "hierarchy"
    ANALYTICS = "analytics"
    HEALTH = "health"
    WORKBOOK = "workbook"
    DEFAULT = "default"


# Mapping of tool names to their categories
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    # Search tools (high impact)
    "search_naics_codes": ToolCategory.SEARCH,
    "search_index_terms": ToolCategory.SEARCH,
    "find_similar_industries": ToolCategory.SEARCH,
    # Classification tools (high impact)
    "classify_business": ToolCategory.CLASSIFY,
    "validate_classification": ToolCategory.CLASSIFY,
    "get_cross_references": ToolCategory.CLASSIFY,
    # Batch tools (very high impact)
    "classify_batch": ToolCategory.BATCH,
    # Hierarchy tools (medium impact)
    "get_code_hierarchy": ToolCategory.HIERARCHY,
    "get_children": ToolCategory.HIERARCHY,
    "get_siblings": ToolCategory.HIERARCHY,
    # Analytics tools (medium impact)
    "get_sector_overview": ToolCategory.ANALYTICS,
    "compare_codes": ToolCategory.ANALYTICS,
    # Health tools (low impact)
    "ping": ToolCategory.HEALTH,
    "check_readiness": ToolCategory.HEALTH,
    "get_server_health": ToolCategory.HEALTH,
    "get_metrics": ToolCategory.HEALTH,
    # Workbook tools (low impact)
    "write_to_workbook": ToolCategory.WORKBOOK,
    "search_workbook": ToolCategory.WORKBOOK,
    "get_workbook_entry": ToolCategory.WORKBOOK,
    "get_workbook_template": ToolCategory.WORKBOOK,
}


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Implements the token bucket algorithm with burst support.
    Thread-safe using asyncio locks.
    """

    capacity: float  # Maximum tokens (burst limit)
    refill_rate: float  # Tokens added per second
    tokens: float = field(default=0.0, init=False)
    last_refill: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize with full bucket."""
        self.tokens = self.capacity

    async def acquire(self, tokens: float = 1.0) -> tuple[bool, float]:
        """
        Try to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            Tuple of (success, wait_time_seconds)
            - success: True if tokens were acquired
            - wait_time_seconds: Time until tokens available (0 if acquired)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill

            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0

            # Calculate wait time until enough tokens
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return False, wait_time

    async def peek(self) -> tuple[float, float]:
        """
        Check current token count without consuming.

        Returns:
            Tuple of (current_tokens, capacity)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            current = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            return current, self.capacity


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    category: ToolCategory
    tokens_remaining: float
    retry_after_seconds: float = 0.0
    message: str = ""


class RateLimiter:
    """
    Rate limiter using token buckets per tool category.

    Provides configurable rate limits for different tool categories.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter with configuration.

        Args:
            config: Rate limit configuration (uses defaults if None)
        """
        self.config = config or get_rate_limit_config()
        self._buckets: dict[ToolCategory, TokenBucket] = {}
        self._initialize_buckets()

    def _initialize_buckets(self) -> None:
        """Create token buckets for each category based on config."""
        category_rpms = {
            ToolCategory.SEARCH: self.config.search_rpm,
            ToolCategory.CLASSIFY: self.config.classify_rpm,
            ToolCategory.BATCH: self.config.batch_rpm,
            ToolCategory.HIERARCHY: self.config.hierarchy_rpm,
            ToolCategory.ANALYTICS: self.config.analytics_rpm,
            ToolCategory.HEALTH: self.config.health_rpm,
            ToolCategory.WORKBOOK: self.config.workbook_rpm,
            ToolCategory.DEFAULT: self.config.default_rpm,
        }

        for category, rpm in category_rpms.items():
            # Convert RPM to tokens per second
            refill_rate = rpm / 60.0
            # Burst capacity = base rate * multiplier
            capacity = rpm * self.config.burst_multiplier / 60.0
            self._buckets[category] = TokenBucket(capacity=capacity, refill_rate=refill_rate)

    def _get_category(self, tool_name: str) -> ToolCategory:
        """Get the category for a tool name."""
        return TOOL_CATEGORIES.get(tool_name, ToolCategory.DEFAULT)

    async def check_limit(self, tool_name: str) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.

        Args:
            tool_name: Name of the tool being called

        Returns:
            RateLimitResult with allowed status and metadata
        """
        if not self.config.enable_rate_limiting:
            category = self._get_category(tool_name)
            return RateLimitResult(
                allowed=True,
                category=category,
                tokens_remaining=float("inf"),
                message="Rate limiting disabled",
            )

        category = self._get_category(tool_name)
        bucket = self._buckets[category]

        allowed, wait_time = await bucket.acquire()
        current_tokens, capacity = await bucket.peek()

        if allowed:
            return RateLimitResult(
                allowed=True,
                category=category,
                tokens_remaining=current_tokens,
            )

        message = (
            f"Rate limit exceeded for {category.value} tools. Try again in {wait_time:.1f} seconds."
        )

        if self.config.log_rate_limit_hits:
            logger.warning(
                f"Rate limit hit: tool={tool_name}, category={category.value}, "
                f"retry_after={wait_time:.1f}s"
            )

        return RateLimitResult(
            allowed=False,
            category=category,
            tokens_remaining=0.0,
            retry_after_seconds=wait_time,
            message=message,
        )

    async def get_status(self) -> dict[str, Any]:
        """
        Get current status of all rate limit buckets.

        Returns:
            Dict with bucket status for each category
        """
        status = {
            "enabled": self.config.enable_rate_limiting,
            "buckets": {},
        }

        for category, bucket in self._buckets.items():
            current, capacity = await bucket.peek()
            status["buckets"][category.value] = {
                "tokens_available": round(current, 2),
                "capacity": round(capacity, 2),
                "refill_rate_per_second": round(bucket.refill_rate, 2),
                "utilization_percent": round((1 - current / capacity) * 100, 1)
                if capacity > 0
                else 0,
            }

        return status

    def reset(self) -> None:
        """Reset all buckets to full capacity."""
        self._initialize_buckets()


def rate_limited(
    category: ToolCategory | None = None,
    tool_name: str | None = None,
) -> Callable:
    """
    Decorator to apply rate limiting to a tool function.

    Can be used with explicit category or tool_name for auto-detection.

    Args:
        category: Explicit category to use
        tool_name: Tool name for category lookup (uses function name if None)

    Returns:
        Decorator function

    Example:
        @rate_limited(category=ToolCategory.SEARCH)
        async def search_naics_codes(request, ctx):
            ...

        @rate_limited()  # Uses function name for category lookup
        async def classify_business(request, ctx):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from ..core.errors import RateLimitError

            # Get rate limiter from context if available
            rate_limiter = None
            for arg in args:
                if hasattr(arg, "request_context"):
                    ctx = arg
                    if hasattr(ctx.request_context, "lifespan_context"):
                        app_ctx = ctx.request_context.lifespan_context
                        if hasattr(app_ctx, "rate_limiter"):
                            rate_limiter = app_ctx.rate_limiter
                    break

            if rate_limiter is None:
                # No rate limiter available, proceed without limiting
                return await func(*args, **kwargs)

            # Determine tool name for rate limiting
            effective_tool_name = tool_name or func.__name__

            # Check rate limit
            result = await rate_limiter.check_limit(effective_tool_name)

            if not result.allowed:
                raise RateLimitError(
                    tool_name=effective_tool_name,
                    category=result.category.value,
                    retry_after=result.retry_after_seconds,
                    message=result.message,
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators for each category
def search_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for search tools."""
    return rate_limited(category=ToolCategory.SEARCH)(func)


def classify_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for classification tools."""
    return rate_limited(category=ToolCategory.CLASSIFY)(func)


def batch_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for batch tools."""
    return rate_limited(category=ToolCategory.BATCH)(func)


def hierarchy_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for hierarchy tools."""
    return rate_limited(category=ToolCategory.HIERARCHY)(func)


def analytics_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for analytics tools."""
    return rate_limited(category=ToolCategory.ANALYTICS)(func)


def health_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for health tools."""
    return rate_limited(category=ToolCategory.HEALTH)(func)


def workbook_rate_limited(func: Callable) -> Callable:
    """Rate limit decorator for workbook tools."""
    return rate_limited(category=ToolCategory.WORKBOOK)(func)
