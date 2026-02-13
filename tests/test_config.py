"""
Unit tests for configuration module.

Tests Pydantic-based configuration with validation and environment variable loading.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from naics_mcp_server.config import (
    AppConfig,
    SearchConfig,
    ServerConfig,
    get_config,
    get_search_config,
    get_server_config,
    reset_config,
)
from naics_mcp_server.core.errors import ConfigurationError


class TestSearchConfigDefaults:
    """Tests for SearchConfig default values."""

    def test_default_embedding_model(self):
        """Should use all-MiniLM-L6-v2 by default."""
        config = SearchConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_default_embedding_dimension(self):
        """Should use 384 dimensions for MiniLM model."""
        config = SearchConfig()
        assert config.embedding_dimension == 384

    def test_default_hybrid_weights(self):
        """Should favor semantic search by default."""
        config = SearchConfig()
        assert config.hybrid_weight_semantic == 0.7
        assert config.hybrid_weight_lexical == 0.3

    def test_default_min_confidence(self):
        """Should use 0.3 minimum confidence."""
        config = SearchConfig()
        assert config.min_confidence == 0.3

    def test_default_limit(self):
        """Should return 10 results by default."""
        config = SearchConfig()
        assert config.default_limit == 10

    def test_default_query_timeout(self):
        """Should timeout after 5 seconds."""
        config = SearchConfig()
        assert config.query_timeout_seconds == 5

    def test_default_features_enabled(self):
        """Should enable key features by default."""
        config = SearchConfig()
        assert config.enable_query_expansion is True
        assert config.enable_cross_references is True
        assert config.enable_audit_log is True
        assert config.explain_results is True
        assert config.include_hierarchy is True


class TestSearchConfigValidation:
    """Tests for SearchConfig field validation."""

    def test_semantic_weight_min_bound(self):
        """Semantic weight cannot be negative."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SearchConfig(hybrid_weight_semantic=-0.1)

    def test_semantic_weight_max_bound(self):
        """Semantic weight cannot exceed 1.0."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            SearchConfig(hybrid_weight_semantic=1.5)

    def test_min_confidence_bounds(self):
        """Min confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            SearchConfig(min_confidence=-0.1)
        with pytest.raises(ValueError):
            SearchConfig(min_confidence=1.5)

    def test_default_limit_min_bound(self):
        """Default limit must be at least 1."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            SearchConfig(default_limit=0)

    def test_default_limit_max_bound(self):
        """Default limit cannot exceed 100."""
        with pytest.raises(ValueError, match="less than or equal to 100"):
            SearchConfig(default_limit=200)

    def test_query_timeout_bounds(self):
        """Query timeout must be between 1 and 300 seconds."""
        with pytest.raises(ValueError):
            SearchConfig(query_timeout_seconds=0)
        with pytest.raises(ValueError):
            SearchConfig(query_timeout_seconds=500)

    def test_embedding_dimension_bounds(self):
        """Embedding dimension must be between 32 and 4096."""
        with pytest.raises(ValueError):
            SearchConfig(embedding_dimension=16)
        with pytest.raises(ValueError):
            SearchConfig(embedding_dimension=5000)

    def test_boost_index_terms_bounds(self):
        """Index term boost must be between 1.0 and 10.0."""
        with pytest.raises(ValueError):
            SearchConfig(boost_index_terms=0.5)
        with pytest.raises(ValueError):
            SearchConfig(boost_index_terms=15.0)

    def test_max_candidates_bounds(self):
        """Max candidates must be between 10 and 5000."""
        with pytest.raises(ValueError):
            SearchConfig(max_candidates=5)
        with pytest.raises(ValueError):
            SearchConfig(max_candidates=10000)

    def test_embedding_model_not_empty(self):
        """Embedding model name cannot be empty."""
        with pytest.raises(ValueError, match="at least 1"):
            SearchConfig(embedding_model="")


class TestSearchConfigWeightValidation:
    """Tests for hybrid weight auto-adjustment."""

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        config = SearchConfig()
        total = config.hybrid_weight_semantic + config.hybrid_weight_lexical
        assert abs(total - 1.0) < 0.01

    def test_weights_auto_adjusted(self):
        """Lexical weight should be auto-adjusted when semantic is set."""
        config = SearchConfig(hybrid_weight_semantic=0.6, hybrid_weight_lexical=0.6)
        # Should be adjusted to sum to 1.0
        total = config.hybrid_weight_semantic + config.hybrid_weight_lexical
        assert abs(total - 1.0) < 0.01


class TestSearchConfigDatabasePath:
    """Tests for database path resolution."""

    def test_explicit_database_path(self):
        """Explicit path should be used as-is."""
        config = SearchConfig(database_path=Path("/custom/path/db.duckdb"))
        assert config.database_path == Path("/custom/path/db.duckdb")

    def test_string_path_converted(self):
        """String paths should be converted to Path objects."""
        config = SearchConfig(database_path="/custom/path/db.duckdb")
        assert isinstance(config.database_path, Path)
        assert config.database_path == Path("/custom/path/db.duckdb")

    def test_default_path_resolved(self):
        """Default path should be resolved to a valid location."""
        config = SearchConfig()
        assert config.database_path is not None
        assert "naics.duckdb" in str(config.database_path)


class TestSearchConfigEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_database_path_from_env(self):
        """Should load database path from NAICS_DATABASE_PATH."""
        with patch.dict(os.environ, {"NAICS_DATABASE_PATH": "/env/path/db.duckdb"}):
            reset_config()
            config = SearchConfig()
            assert config.database_path == Path("/env/path/db.duckdb")

    def test_embedding_model_from_env(self):
        """Should load embedding model from NAICS_EMBEDDING_MODEL."""
        with patch.dict(os.environ, {"NAICS_EMBEDDING_MODEL": "custom-model"}):
            config = SearchConfig()
            assert config.embedding_model == "custom-model"

    def test_min_confidence_from_env(self):
        """Should load min confidence from NAICS_MIN_CONFIDENCE."""
        with patch.dict(os.environ, {"NAICS_MIN_CONFIDENCE": "0.5"}):
            config = SearchConfig()
            assert config.min_confidence == 0.5

    def test_semantic_weight_from_env(self):
        """Should load semantic weight from NAICS_HYBRID_WEIGHT_SEMANTIC."""
        with patch.dict(os.environ, {"NAICS_HYBRID_WEIGHT_SEMANTIC": "0.8"}):
            config = SearchConfig()
            assert config.hybrid_weight_semantic == 0.8

    def test_audit_disabled_from_env(self):
        """Should disable audit from NAICS_ENABLE_AUDIT_LOG."""
        with patch.dict(os.environ, {"NAICS_ENABLE_AUDIT_LOG": "false"}):
            config = SearchConfig()
            assert config.enable_audit_log is False


class TestSearchConfigToDict:
    """Tests for configuration serialization."""

    def test_to_dict_includes_key_fields(self):
        """to_dict should include important configuration."""
        config = SearchConfig(database_path=Path("/test/db.duckdb"))
        result = config.to_dict()

        assert "database_path" in result
        assert "embedding_model" in result
        assert "hybrid_weight_semantic" in result
        assert "min_confidence" in result
        assert "enable_query_expansion" in result

    def test_to_dict_path_is_string(self):
        """Database path should be serialized as string."""
        config = SearchConfig(database_path=Path("/test/db.duckdb"))
        result = config.to_dict()

        assert isinstance(result["database_path"], str)
        assert result["database_path"] == "/test/db.duckdb"


class TestServerConfigDefaults:
    """Tests for ServerConfig default values."""

    def test_default_name(self):
        """Should have NAICS Classification Assistant as default name."""
        config = ServerConfig()
        assert config.name == "NAICS Classification Assistant"

    def test_default_version(self):
        """Should have version 0.1.0."""
        config = ServerConfig()
        assert config.version == "0.1.0"

    def test_default_debug_disabled(self):
        """Debug should be disabled by default."""
        config = ServerConfig()
        assert config.debug is False

    def test_default_cors_enabled(self):
        """CORS should be enabled by default."""
        config = ServerConfig()
        assert config.enable_cors is True


class TestServerConfigValidation:
    """Tests for ServerConfig field validation."""

    def test_name_not_empty(self):
        """Server name cannot be empty."""
        with pytest.raises(ValueError, match="at least 1"):
            ServerConfig(name="")

    def test_version_semver_format(self):
        """Version must be in semver format."""
        # Valid versions
        ServerConfig(version="1.0.0")
        ServerConfig(version="0.1.0")
        ServerConfig(version="10.20.30")

        # Invalid version
        with pytest.raises(ValueError, match="pattern"):
            ServerConfig(version="invalid")


class TestServerConfigDebugEnv:
    """Tests for debug mode environment variable handling."""

    def test_debug_from_naics_debug(self):
        """Should enable debug from NAICS_DEBUG."""
        with patch.dict(os.environ, {"NAICS_DEBUG": "true"}, clear=False):
            config = ServerConfig()
            assert config.debug is True

    def test_debug_from_debug(self):
        """Should enable debug from DEBUG."""
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=False):
            config = ServerConfig()
            assert config.debug is True

    def test_debug_accepts_various_values(self):
        """Should accept true, 1, yes for debug."""
        for value in ["true", "True", "TRUE", "1", "yes", "Yes"]:
            with patch.dict(os.environ, {"NAICS_DEBUG": value}, clear=False):
                config = ServerConfig()
                assert config.debug is True, f"Failed for value: {value}"


class TestServerConfigToDict:
    """Tests for ServerConfig serialization."""

    def test_to_dict_includes_key_fields(self):
        """to_dict should include server configuration."""
        config = ServerConfig()
        result = config.to_dict()

        assert "name" in result
        assert "version" in result
        assert "debug" in result
        assert "enable_cors" in result


class TestAppConfig:
    """Tests for combined AppConfig."""

    def test_contains_search_config(self):
        """AppConfig should contain SearchConfig."""
        config = AppConfig()
        assert isinstance(config.search, SearchConfig)

    def test_contains_server_config(self):
        """AppConfig should contain ServerConfig."""
        config = AppConfig()
        assert isinstance(config.server, ServerConfig)

    def test_to_dict_combines_configs(self):
        """to_dict should include both search and server config."""
        config = AppConfig()
        result = config.to_dict()

        assert "search" in result
        assert "server" in result
        assert isinstance(result["search"], dict)
        assert isinstance(result["server"], dict)


class TestAppConfigStartupValidation:
    """Tests for startup validation."""

    def test_validates_database_path_set(self):
        """Should fail if database path is None."""
        config = AppConfig()
        # Force database_path to None
        config.search.database_path = None

        with pytest.raises(ConfigurationError, match="Database path"):
            config.validate_startup()

    def test_warns_about_debug_mode(self):
        """Should warn when debug mode is enabled."""
        config = AppConfig()
        config.server.debug = True

        warnings = config.validate_startup()
        assert any("debug" in w.lower() for w in warnings)

    def test_warns_about_disabled_audit(self):
        """Should warn when audit logging is disabled."""
        config = AppConfig()
        config.search.enable_audit_log = False

        warnings = config.validate_startup()
        assert any("audit" in w.lower() for w in warnings)


class TestConfigSingleton:
    """Tests for configuration singleton behavior."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_same_instance(self):
        """get_config should return the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """reset_config should clear the cached instance."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_get_search_config_convenience(self):
        """get_search_config should return search config."""
        search_config = get_search_config()
        app_config = get_config()
        assert search_config is app_config.search

    def test_get_server_config_convenience(self):
        """get_server_config should return server config."""
        server_config = get_server_config()
        app_config = get_config()
        assert server_config is app_config.server


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with from_env()."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_search_config_from_env(self):
        """SearchConfig.from_env() should work for compatibility."""
        config = SearchConfig.from_env()
        assert isinstance(config, SearchConfig)

    def test_server_config_from_env(self):
        """ServerConfig.from_env() should work for compatibility."""
        config = ServerConfig.from_env()
        assert isinstance(config, ServerConfig)

    def test_from_env_returns_singleton(self):
        """from_env() should return the same instance as singleton."""
        env_config = SearchConfig.from_env()
        singleton_config = get_search_config()
        # They should be equivalent (same values)
        assert env_config.embedding_model == singleton_config.embedding_model
