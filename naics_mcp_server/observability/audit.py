"""
Audit logging for search operations.

This module tracks search queries and results to help improve
the system over time through understanding usage patterns.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

from ..models.search_models import SearchResults, SearchStrategy

logger = logging.getLogger(__name__)


@dataclass
class SearchEvent:
    """
    Represents a single search event for audit logging.

    Captures complete context about a search operation.
    """

    # Core event data
    timestamp: datetime
    query: str
    strategy: SearchStrategy

    # Expanded query information
    expanded_terms: Optional[List[str]] = None
    was_expanded: bool = False

    # Results information
    results_count: int = 0
    top_result_code: Optional[str] = None
    top_result_confidence: Optional[float] = None

    # NAICS-specific metrics
    index_terms_matched: int = 0
    cross_refs_checked: int = 0
    exclusion_warnings: int = 0

    # Performance metrics
    duration_ms: int = 0

    # Optional context
    user_context: Optional[Dict[str, Any]] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None

    @classmethod
    def start(cls, query: str, strategy: str) -> "SearchEvent":
        """
        Create a new search event at the start of a search.

        Args:
            query: The search query
            strategy: The search strategy being used

        Returns:
            New SearchEvent instance
        """
        return cls(
            timestamp=datetime.utcnow(),
            query=query,
            strategy=SearchStrategy(strategy),
            duration_ms=int(time.time() * 1000)  # Start timer
        )

    def complete(self, results: SearchResults) -> None:
        """
        Mark the search as complete with results.

        Args:
            results: The search results
        """
        # Calculate duration
        self.duration_ms = int(time.time() * 1000) - self.duration_ms

        # Extract result information
        self.results_count = len(results.matches)

        if results.matches:
            top_match = results.matches[0]
            self.top_result_code = top_match.code.node_code
            self.top_result_confidence = top_match.confidence.overall

        # Query expansion info
        if results.query_metadata:
            self.was_expanded = results.query_metadata.was_expanded
            if self.was_expanded:
                self.expanded_terms = results.query_metadata.expanded_terms.all_terms()
            self.index_terms_matched = results.query_metadata.index_terms_searched
            self.cross_refs_checked = results.query_metadata.cross_refs_checked

        # Count exclusion warnings
        self.exclusion_warnings = len(results.get_exclusion_warnings())

        self.success = True

    def fail(self, error: str) -> None:
        """
        Mark the search as failed.

        Args:
            error: Error message
        """
        self.duration_ms = int(time.time() * 1000) - self.duration_ms
        self.success = False
        self.error_message = error

    def to_json(self) -> str:
        """Convert to JSON for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["strategy"] = self.strategy.value
        return json.dumps(data)


class SearchAuditLog:
    """
    Maintains a transparent audit trail of all search operations.

    This isn't surveillance—it's a tool for understanding usage patterns,
    improving the system, and helping users find what they need.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        retention_days: int = 90,
        enable_file_logging: bool = True
    ):
        """
        Initialize the audit log.

        Args:
            log_dir: Directory to store audit logs
            retention_days: How long to keep logs
            enable_file_logging: Whether to write to files
        """
        self.log_dir = log_dir or Path.home() / ".cache" / "naics-mcp-server" / "logs"
        self.retention_days = retention_days
        self.enable_file_logging = enable_file_logging

        # Create log directory if needed
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for recent events
        self.recent_events: List[SearchEvent] = []
        self.max_recent = 1000

    async def log_search(self, event: SearchEvent) -> None:
        """
        Log a search event.

        Args:
            event: The search event to log
        """
        # Add to recent events buffer
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent:
            self.recent_events.pop(0)

        # Write to file if enabled
        if self.enable_file_logging:
            await self._write_to_file(event)

        # Log summary to standard logger
        if event.success:
            logger.info(
                f"Search completed: query='{event.query[:50]}', "
                f"strategy={event.strategy.value}, "
                f"results={event.results_count}, "
                f"duration={event.duration_ms}ms"
            )
        else:
            logger.warning(
                f"Search failed: query='{event.query[:50]}', "
                f"error='{event.error_message}', "
                f"duration={event.duration_ms}ms"
            )

    async def _write_to_file(self, event: SearchEvent) -> None:
        """Write event to daily log file."""
        try:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"naics_searches_{date_str}.jsonl"

            with open(log_file, "a") as f:
                f.write(event.to_json() + "\n")

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    async def analyze_patterns(
        self,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze search patterns to improve the system.

        Args:
            timeframe_hours: Hours to look back

        Returns:
            Dictionary of analysis results
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
        recent = [e for e in self.recent_events if e.timestamp >= cutoff_time]

        if not recent:
            return {"message": "No recent searches to analyze"}

        analysis = {
            "timeframe_hours": timeframe_hours,
            "total_searches": len(recent),
            "success_rate": sum(1 for e in recent if e.success) / len(recent),
            "average_duration_ms": sum(e.duration_ms for e in recent) / len(recent),
        }

        # Find common queries
        query_counts: Dict[str, int] = {}
        for event in recent:
            query_key = event.query.lower().strip()
            query_counts[query_key] = query_counts.get(query_key, 0) + 1

        top_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        analysis["top_queries"] = [
            {"query": q, "count": c} for q, c in top_queries
        ]

        # Queries with low confidence results
        low_confidence = []
        for event in recent:
            if event.success and event.top_result_confidence:
                if event.top_result_confidence < 0.5:
                    low_confidence.append({
                        "query": event.query,
                        "confidence": event.top_result_confidence,
                        "result": event.top_result_code
                    })

        analysis["low_confidence_queries"] = low_confidence[:10]

        # Strategy usage
        strategy_counts: Dict[str, int] = {}
        for event in recent:
            strategy = event.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        analysis["strategy_usage"] = strategy_counts

        # NAICS-specific: exclusion warning frequency
        exclusion_events = [e for e in recent if e.exclusion_warnings > 0]
        analysis["exclusion_warning_rate"] = len(exclusion_events) / len(recent) if recent else 0

        # Slow queries
        slow_queries = [
            {"query": e.query, "duration_ms": e.duration_ms}
            for e in recent
            if e.duration_ms > 200  # NAICS-specific threshold
        ]
        analysis["slow_queries"] = sorted(
            slow_queries,
            key=lambda x: x["duration_ms"],
            reverse=True
        )[:10]

        return analysis

    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent searches for monitoring.

        Args:
            limit: Maximum number of searches to return

        Returns:
            List of recent search events
        """
        recent = self.recent_events[-limit:]
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "query": e.query,
                "strategy": e.strategy.value,
                "results": e.results_count,
                "confidence": e.top_result_confidence,
                "duration_ms": e.duration_ms,
                "exclusion_warnings": e.exclusion_warnings,
                "success": e.success
            }
            for e in reversed(recent)
        ]
