"""
Tests for Prometheus metrics module.
"""

from naics_mcp_server.observability.metrics import (
    Timer,
    get_metrics_text,
    initialize_metrics,
    record_cache_hit,
    record_cache_miss,
    record_crossref_lookup,
    record_search_fallback,
    record_search_metrics,
    reset_metrics,
    update_cache_stats,
    update_data_stats,
    update_health_status,
)


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def test_initialize_metrics(self):
        """Test metrics can be initialized."""
        reset_metrics()
        initialize_metrics(
            version="0.1.0",
            embedding_model="test-model",
            database_path="/tmp/test.db",
        )
        # Should not raise

    def test_initialize_metrics_idempotent(self):
        """Test metrics initialization is idempotent."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")
        initialize_metrics("0.1.0", "test", "/tmp/test.db")  # Should not raise


class TestMetricsOutput:
    """Tests for metrics output generation."""

    def test_get_metrics_text_returns_string(self):
        """Test get_metrics_text returns Prometheus format."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")
        output = get_metrics_text()
        assert isinstance(output, str)
        assert "naics_" in output

    def test_metrics_include_naics_prefixed_metrics(self):
        """Test metrics include NAICS-specific metrics."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")
        output = get_metrics_text()

        # Check for key metric names
        assert "naics_tool_requests_total" in output
        assert "naics_search_requests_total" in output
        assert "naics_cache_operations_total" in output
        assert "naics_health_status" in output


class TestSearchMetrics:
    """Tests for search metrics recording."""

    def test_record_search_metrics(self):
        """Test recording search metrics."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        record_search_metrics(
            strategy="hybrid",
            duration_seconds=0.123,
            results_count=5,
            top_confidence=0.85,
        )

        output = get_metrics_text()
        assert "naics_search_requests_total" in output

    def test_record_search_fallback(self):
        """Test recording search strategy fallback."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        record_search_fallback("semantic", "lexical")

        output = get_metrics_text()
        assert "naics_search_fallbacks_total" in output


class TestCacheMetrics:
    """Tests for cache metrics."""

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        record_cache_hit("search")

        output = get_metrics_text()
        assert "naics_cache_operations_total" in output

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        record_cache_miss("embedding")

        output = get_metrics_text()
        assert "naics_cache_operations_total" in output

    def test_update_cache_stats(self):
        """Test updating cache statistics."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        update_cache_stats("search", size=50, hit_rate=0.75)

        output = get_metrics_text()
        assert "naics_cache_size" in output
        assert "naics_cache_hit_rate" in output


class TestCrossRefMetrics:
    """Tests for cross-reference metrics."""

    def test_record_crossref_lookup(self):
        """Test recording cross-reference lookup."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        record_crossref_lookup(exclusions_found=2)

        output = get_metrics_text()
        assert "naics_crossref_lookups_total" in output


class TestHealthMetrics:
    """Tests for health metrics."""

    def test_update_health_status_healthy(self):
        """Test updating health status to healthy."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        update_health_status(
            status="healthy",
            components={"database": True, "embedder": True},
            uptime_seconds=100.0,
        )

        output = get_metrics_text()
        assert "naics_health_status" in output
        assert "naics_uptime_seconds" in output

    def test_update_health_status_degraded(self):
        """Test updating health status to degraded."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        update_health_status(status="degraded")

        # Should not raise


class TestDataStats:
    """Tests for data statistics metrics."""

    def test_update_data_stats(self):
        """Test updating data statistics."""
        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        update_data_stats(
            total_codes=2000,
            total_embeddings=2000,
            total_index_terms=20000,
            total_cross_references=5000,
        )

        output = get_metrics_text()
        assert "naics_data_stats" in output


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_records_duration(self):
        """Test Timer records duration."""
        from naics_mcp_server.observability.metrics import TOOL_DURATION

        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        with Timer(TOOL_DURATION, {"tool": "test_tool"}) as timer:
            # Simulate some work
            _ = sum(range(1000))

        assert timer.duration is not None
        assert timer.duration >= 0
        assert timer.elapsed_ms >= 0

    def test_timer_elapsed_during_execution(self):
        """Test Timer.elapsed_ms works during execution."""
        from naics_mcp_server.observability.metrics import TOOL_DURATION

        reset_metrics()
        initialize_metrics("0.1.0", "test", "/tmp/test.db")

        timer = Timer(TOOL_DURATION, {"tool": "test_tool"})
        timer.__enter__()

        # Check elapsed during execution
        elapsed = timer.elapsed_ms
        assert elapsed >= 0

        timer.__exit__(None, None, None)


class TestConfigIntegration:
    """Tests for metrics config integration."""

    def test_metrics_config_in_app_config(self):
        """Test MetricsConfig is part of AppConfig."""
        from naics_mcp_server.config import get_config, reset_config

        reset_config()
        config = get_config()

        assert hasattr(config, "metrics")
        assert config.metrics.enable_metrics is True
        assert config.metrics.metrics_port == 9090

    def test_metrics_config_to_dict(self):
        """Test MetricsConfig.to_dict() works."""
        from naics_mcp_server.config import get_metrics_config, reset_config

        reset_config()
        metrics_config = get_metrics_config()

        config_dict = metrics_config.to_dict()
        assert "enable_metrics" in config_dict
        assert "metrics_port" in config_dict
