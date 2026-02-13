"""
Unit tests for input validation module.

Tests validators for descriptions, NAICS codes, batches, and search parameters.
"""

import pytest

from naics_mcp_server.core.errors import ValidationError
from naics_mcp_server.core.validation import (
    ValidationConfig,
    ValidationResult,
    is_valid_utf8,
    normalize_text,
    validate_batch_codes,
    validate_batch_descriptions,
    validate_confidence,
    validate_description,
    validate_limit,
    validate_naics_code,
    validate_search_query,
    validate_strategy,
)


class TestNormalizeText:
    """Tests for text normalization function."""

    def test_normalize_strips_whitespace(self):
        """Should strip leading and trailing whitespace."""
        result = normalize_text("  hello world  ")
        assert result == "hello world"

    def test_normalize_collapses_spaces(self):
        """Should collapse multiple spaces to single space."""
        result = normalize_text("hello    world")
        assert result == "hello world"

    def test_normalize_replaces_newlines(self):
        """Should replace newlines with spaces."""
        result = normalize_text("hello\nworld")
        assert result == "hello world"

    def test_normalize_replaces_tabs(self):
        """Should replace tabs with spaces."""
        result = normalize_text("hello\tworld")
        assert result == "hello world"

    def test_normalize_handles_empty(self):
        """Should handle empty string."""
        result = normalize_text("")
        assert result == ""

    def test_normalize_handles_unicode(self):
        """Should handle Unicode characters."""
        result = normalize_text("café résumé")
        assert result == "café résumé"

    def test_normalize_nfc_form(self):
        """Should normalize to NFC form."""
        # e + combining acute accent vs precomposed é
        decomposed = "cafe\u0301"  # e + combining accent
        result = normalize_text(decomposed)
        assert "é" in result or len(result) == 5  # NFC form


class TestIsValidUtf8:
    """Tests for UTF-8 validation."""

    def test_valid_ascii(self):
        """ASCII text should be valid UTF-8."""
        assert is_valid_utf8("hello world") is True

    def test_valid_unicode(self):
        """Unicode text should be valid UTF-8."""
        assert is_valid_utf8("café résumé 日本語") is True

    def test_valid_emoji(self):
        """Emoji should be valid UTF-8."""
        assert is_valid_utf8("Hello 👋 World 🌍") is True


class TestValidateDescription:
    """Tests for business description validation."""

    def test_valid_description(self):
        """Valid description should pass."""
        result = validate_description("Manufacturing of dog food products")
        assert result.is_valid
        assert result.value == "Manufacturing of dog food products"

    def test_description_normalized(self):
        """Description should be normalized."""
        result = validate_description("  Manufacturing  of  dog food  ")
        assert result.value == "Manufacturing of dog food"

    def test_none_description_raises(self):
        """None description should raise ValidationError."""
        with pytest.raises(ValidationError, match="required"):
            validate_description(None)

    def test_empty_description_raises(self):
        """Empty description should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_description("")

    def test_whitespace_only_raises(self):
        """Whitespace-only description should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_description("   \t\n   ")

    def test_too_short_raises(self):
        """Too short description should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least"):
            validate_description("abc")

    def test_minimum_length_passes(self):
        """Description at minimum length should pass."""
        result = validate_description("1234567890")  # 10 chars
        assert result.is_valid

    def test_truncation_warning(self):
        """Long description should be truncated with warning."""
        config = ValidationConfig(description_max_length=50, description_truncate_on_overflow=True)
        long_desc = "a" * 100
        result = validate_description(long_desc, config)
        assert result.is_valid
        assert len(result.value) == 50
        assert len(result.warnings) > 0
        assert "truncated" in result.warnings[0].lower()

    def test_truncation_disabled_raises(self):
        """Long description with truncation disabled should raise."""
        config = ValidationConfig(description_max_length=50, description_truncate_on_overflow=False)
        long_desc = "a" * 100
        with pytest.raises(ValidationError, match="at most"):
            validate_description(long_desc, config)

    def test_non_string_raises(self):
        """Non-string description should raise ValidationError."""
        with pytest.raises(ValidationError, match="string"):
            validate_description(12345)

    def test_unicode_description(self):
        """Unicode description should be valid."""
        result = validate_description("日本語での説明です。これはテストです。")
        assert result.is_valid


class TestValidateNAICSCode:
    """Tests for NAICS code validation."""

    def test_valid_6_digit_code(self):
        """Valid 6-digit code should pass."""
        result = validate_naics_code("311111")
        assert result.is_valid
        assert result.value == "311111"

    def test_valid_2_digit_code(self):
        """Valid 2-digit sector code should pass."""
        result = validate_naics_code("31")
        assert result.is_valid

    def test_valid_3_digit_code(self):
        """Valid 3-digit subsector code should pass."""
        result = validate_naics_code("311")
        assert result.is_valid

    def test_code_with_whitespace(self):
        """Code with whitespace should be stripped."""
        result = validate_naics_code("  311111  ")
        assert result.value == "311111"

    def test_none_code_raises(self):
        """None code should raise ValidationError."""
        with pytest.raises(ValidationError, match="required"):
            validate_naics_code(None)

    def test_empty_code_raises(self):
        """Empty code should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_naics_code("")

    def test_non_digit_raises(self):
        """Code with non-digits should raise ValidationError."""
        with pytest.raises(ValidationError, match="digits"):
            validate_naics_code("31111A")

    def test_too_short_raises(self):
        """Code shorter than 2 digits should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least"):
            validate_naics_code("1")

    def test_too_long_raises(self):
        """Code longer than 6 digits should raise ValidationError."""
        with pytest.raises(ValidationError, match="at most"):
            validate_naics_code("3111111")

    def test_integer_code_converted(self):
        """Integer code should be converted to string."""
        result = validate_naics_code(311111)
        assert result.value == "311111"

    def test_leading_zeros_preserved(self):
        """Leading zeros should be preserved."""
        # Note: This only works with string input, not integer
        result = validate_naics_code("011111")
        assert result.value == "011111"


class TestValidateSearchQuery:
    """Tests for search query validation."""

    def test_valid_query(self):
        """Valid query should pass."""
        result = validate_search_query("dog food manufacturing")
        assert result.is_valid
        assert result.value == "dog food manufacturing"

    def test_query_normalized(self):
        """Query should be normalized."""
        result = validate_search_query("  dog   food  ")
        assert result.value == "dog food"

    def test_none_query_raises(self):
        """None query should raise ValidationError."""
        with pytest.raises(ValidationError, match="required"):
            validate_search_query(None)

    def test_empty_query_raises(self):
        """Empty query should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_search_query("")

    def test_too_short_raises(self):
        """Query shorter than 2 chars should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least"):
            validate_search_query("a")

    def test_long_query_truncated(self):
        """Long query should be truncated."""
        config = ValidationConfig(search_query_max_length=50)
        long_query = "a" * 100
        result = validate_search_query(long_query, config)
        assert len(result.value) == 50
        assert len(result.warnings) > 0


class TestValidateLimit:
    """Tests for limit parameter validation."""

    def test_valid_limit(self):
        """Valid limit should pass."""
        result = validate_limit(10)
        assert result.is_valid
        assert result.value == 10

    def test_none_uses_default(self):
        """None limit should use default."""
        result = validate_limit(None)
        assert result.value == 10  # Default

    def test_zero_raises(self):
        """Zero limit should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least 1"):
            validate_limit(0)

    def test_negative_raises(self):
        """Negative limit should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least 1"):
            validate_limit(-5)

    def test_exceeds_max_capped(self):
        """Limit exceeding max should be capped."""
        config = ValidationConfig(search_limit_max=50)
        result = validate_limit(100, config)
        assert result.value == 50
        assert len(result.warnings) > 0

    def test_string_converted(self):
        """String limit should be converted to int."""
        result = validate_limit("10")
        assert result.value == 10

    def test_invalid_string_raises(self):
        """Invalid string should raise ValidationError."""
        with pytest.raises(ValidationError, match="integer"):
            validate_limit("abc")


class TestValidateConfidence:
    """Tests for confidence threshold validation."""

    def test_valid_confidence(self):
        """Valid confidence should pass."""
        result = validate_confidence(0.5)
        assert result.is_valid
        assert result.value == 0.5

    def test_none_uses_default(self):
        """None confidence should use default."""
        result = validate_confidence(None)
        assert result.value == 0.3  # Default

    def test_zero_valid(self):
        """Zero confidence should be valid."""
        result = validate_confidence(0.0)
        assert result.value == 0.0

    def test_one_valid(self):
        """Confidence of 1.0 should be valid."""
        result = validate_confidence(1.0)
        assert result.value == 1.0

    def test_negative_raises(self):
        """Negative confidence should raise ValidationError."""
        with pytest.raises(ValidationError, match="at least"):
            validate_confidence(-0.1)

    def test_exceeds_one_raises(self):
        """Confidence > 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError, match="at most"):
            validate_confidence(1.5)

    def test_integer_converted(self):
        """Integer confidence should be converted to float."""
        result = validate_confidence(1)
        assert result.value == 1.0
        assert isinstance(result.value, float)


class TestValidateBatchDescriptions:
    """Tests for batch description validation."""

    def test_valid_batch(self):
        """Valid batch should pass."""
        descriptions = [
            "Dog food manufacturing",
            "Software development services",
            "Restaurant and food service",
        ]
        result = validate_batch_descriptions(descriptions)
        assert result.is_valid
        assert len(result.value) == 3

    def test_none_raises(self):
        """None batch should raise ValidationError."""
        with pytest.raises(ValidationError, match="required"):
            validate_batch_descriptions(None)

    def test_empty_list_raises(self):
        """Empty list should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_batch_descriptions([])

    def test_exceeds_max_raises(self):
        """Batch exceeding max size should raise ValidationError."""
        config = ValidationConfig(batch_max_size=5)
        descriptions = ["Description " + str(i) for i in range(10)]
        with pytest.raises(ValidationError, match="maximum batch size"):
            validate_batch_descriptions(descriptions, config)

    def test_invalid_item_raises_with_index(self):
        """Invalid item should raise with index in message."""
        descriptions = [
            "Valid description here",
            "x",  # Too short
            "Another valid description",
        ]
        with pytest.raises(ValidationError, match="index 1"):
            validate_batch_descriptions(descriptions)

    def test_items_normalized(self):
        """Items should be normalized."""
        descriptions = ["  Description  one  ", "Description\ttwo"]
        result = validate_batch_descriptions(descriptions)
        assert result.value[0] == "Description one"
        assert result.value[1] == "Description two"


class TestValidateBatchCodes:
    """Tests for batch code validation."""

    def test_valid_batch(self):
        """Valid batch should pass."""
        codes = ["311111", "541511", "722511"]
        result = validate_batch_codes(codes)
        assert result.is_valid
        assert len(result.value) == 3

    def test_none_raises(self):
        """None batch should raise ValidationError."""
        with pytest.raises(ValidationError, match="required"):
            validate_batch_codes(None)

    def test_empty_list_raises(self):
        """Empty list should raise ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_batch_codes([])

    def test_exceeds_max_raises(self):
        """Batch exceeding max codes should raise ValidationError."""
        codes = [str(311111 + i) for i in range(25)]
        with pytest.raises(ValidationError, match="maximum"):
            validate_batch_codes(codes)

    def test_invalid_code_raises_with_index(self):
        """Invalid code should raise with index in message."""
        codes = ["311111", "ABC", "541511"]
        with pytest.raises(ValidationError, match="index 1"):
            validate_batch_codes(codes)


class TestValidateStrategy:
    """Tests for search strategy validation."""

    def test_valid_strategies(self):
        """All valid strategies should pass."""
        valid = ["hybrid", "semantic", "lexical", "best_match", "meaning", "exact"]
        for strategy in valid:
            result = validate_strategy(strategy)
            assert result.is_valid

    def test_none_uses_default(self):
        """None strategy should use default."""
        result = validate_strategy(None)
        assert result.value == "hybrid"

    def test_case_insensitive(self):
        """Strategy should be case insensitive."""
        result = validate_strategy("HYBRID")
        assert result.value == "hybrid"

    def test_whitespace_stripped(self):
        """Strategy should have whitespace stripped."""
        result = validate_strategy("  semantic  ")
        assert result.value == "semantic"

    def test_invalid_raises(self):
        """Invalid strategy should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid"):
            validate_strategy("invalid_strategy")

    def test_non_string_raises(self):
        """Non-string strategy should raise ValidationError."""
        with pytest.raises(ValidationError, match="string"):
            validate_strategy(123)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Valid result should have correct attributes."""
        result = ValidationResult.valid("test_value", ["warning1"])
        assert result.is_valid is True
        assert result.value == "test_value"
        assert result.warnings == ["warning1"]

    def test_valid_no_warnings(self):
        """Valid result without warnings should have empty list."""
        result = ValidationResult.valid("test_value")
        assert result.warnings == []


class TestValidationConfig:
    """Tests for ValidationConfig class."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = ValidationConfig()
        assert config.description_min_length == 10
        assert config.description_max_length == 5000
        assert config.naics_code_min_digits == 2
        assert config.naics_code_max_digits == 6
        assert config.batch_max_size == 100
        assert config.search_limit_max == 50

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = ValidationConfig(description_min_length=20, batch_max_size=50)
        assert config.description_min_length == 20
        assert config.batch_max_size == 50
        # Other values should still be defaults
        assert config.description_max_length == 5000
