"""
Input validation for NAICS MCP Server.

Provides validators for business descriptions, NAICS codes, and batch operations
with clear error messages and consistent behavior.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError

logger = logging.getLogger(__name__)


# --- Configuration ---


@dataclass
class ValidationConfig:
    """Configuration for input validation."""

    # Business description constraints
    description_min_length: int = 10
    description_max_length: int = 5000
    description_truncate_on_overflow: bool = True

    # NAICS code constraints
    naics_code_min_digits: int = 2
    naics_code_max_digits: int = 6

    # Batch operation constraints
    batch_max_size: int = 100
    batch_timeout_seconds: int = 60

    # Search constraints
    search_query_min_length: int = 2
    search_query_max_length: int = 500
    search_limit_max: int = 50
    search_limit_default: int = 10

    # Confidence constraints
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    confidence_default: float = 0.3


# Default configuration
DEFAULT_CONFIG = ValidationConfig()


# --- Validation Result ---


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    value: Any  # The validated (potentially transformed) value
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def valid(cls, value: Any, warnings: list[str] | None = None) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True, value=value, warnings=warnings or [])

    @classmethod
    def invalid(cls, message: str) -> "ValidationResult":
        """Create an invalid result (will raise ValidationError)."""
        raise ValidationError(message)


# --- Text Normalization ---


def normalize_text(text: str) -> str:
    """
    Normalize text to NFC form and clean up whitespace.

    Args:
        text: Input text

    Returns:
        Normalized text with consistent Unicode and whitespace
    """
    if not text:
        return ""

    # Normalize to NFC (composed form)
    normalized = unicodedata.normalize("NFC", text)

    # Replace various whitespace characters with regular space
    normalized = re.sub(r"[\t\r\n\f\v]+", " ", normalized)

    # Collapse multiple spaces
    normalized = re.sub(r" +", " ", normalized)

    # Strip leading/trailing whitespace
    return normalized.strip()


def is_valid_utf8(text: str) -> bool:
    """Check if text contains only valid UTF-8 characters."""
    try:
        text.encode("utf-8").decode("utf-8")
        return True
    except UnicodeError:
        return False


# --- Business Description Validator ---


def validate_description(
    description: str, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "description"
) -> ValidationResult:
    """
    Validate a business description.

    Args:
        description: The business description to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated description

    Raises:
        ValidationError: If validation fails
    """
    warnings = []

    # Check for None
    if description is None:
        raise ValidationError(message=f"{field_name} is required", field=field_name)

    # Check type
    if not isinstance(description, str):
        raise ValidationError(
            message=f"{field_name} must be a string",
            field=field_name,
            value=type(description).__name__,
        )

    # Normalize the text
    normalized = normalize_text(description)

    # Check for empty/whitespace only
    if not normalized:
        raise ValidationError(
            message=f"{field_name} cannot be empty or whitespace only", field=field_name
        )

    # Check UTF-8 validity
    if not is_valid_utf8(normalized):
        raise ValidationError(message=f"{field_name} contains invalid characters", field=field_name)

    # Check minimum length
    if len(normalized) < config.description_min_length:
        raise ValidationError(
            message=f"{field_name} must be at least {config.description_min_length} characters",
            field=field_name,
            value=normalized,
            constraints={
                "min_length": config.description_min_length,
                "actual_length": len(normalized),
            },
        )

    # Check maximum length
    if len(normalized) > config.description_max_length:
        if config.description_truncate_on_overflow:
            original_length = len(normalized)
            normalized = normalized[: config.description_max_length]
            warnings.append(
                f"{field_name} truncated from {original_length} to {config.description_max_length} characters"
            )
            logger.warning(
                f"Description truncated from {original_length} to {config.description_max_length} chars"
            )
        else:
            raise ValidationError(
                message=f"{field_name} must be at most {config.description_max_length} characters",
                field=field_name,
                constraints={
                    "max_length": config.description_max_length,
                    "actual_length": len(normalized),
                },
            )

    return ValidationResult.valid(normalized, warnings)


# --- NAICS Code Validator ---


def validate_naics_code(
    code: str, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "naics_code"
) -> ValidationResult:
    """
    Validate a NAICS code format.

    Args:
        code: The NAICS code to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated code

    Raises:
        ValidationError: If validation fails
    """
    # Check for None
    if code is None:
        raise ValidationError(message=f"{field_name} is required", field=field_name)

    # Check type
    if not isinstance(code, str):
        # Try to convert integers
        if isinstance(code, int):
            code = str(code)
        else:
            raise ValidationError(
                message=f"{field_name} must be a string or integer",
                field=field_name,
                value=type(code).__name__,
            )

    # Strip whitespace
    code = code.strip()

    # Check for empty
    if not code:
        raise ValidationError(message=f"{field_name} cannot be empty", field=field_name)

    # Check for digits only
    if not code.isdigit():
        raise ValidationError(
            message=f"{field_name} must contain only digits", field=field_name, value=code
        )

    # Check length (NAICS codes are 2-6 digits)
    if len(code) < config.naics_code_min_digits:
        raise ValidationError(
            message=f"{field_name} must be at least {config.naics_code_min_digits} digits",
            field=field_name,
            value=code,
            constraints={"min_digits": config.naics_code_min_digits},
        )

    if len(code) > config.naics_code_max_digits:
        raise ValidationError(
            message=f"{field_name} must be at most {config.naics_code_max_digits} digits",
            field=field_name,
            value=code,
            constraints={"max_digits": config.naics_code_max_digits},
        )

    return ValidationResult.valid(code)


async def validate_naics_code_exists(
    code: str, database, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "naics_code"
) -> ValidationResult:
    """
    Validate that a NAICS code exists in the database.

    Args:
        code: The NAICS code to validate
        database: NAICSDatabase instance
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated code

    Raises:
        ValidationError: If validation fails
    """
    # First validate format
    result = validate_naics_code(code, config, field_name)

    # Then check existence
    naics_code = await database.get_by_code(result.value)
    if naics_code is None:
        raise ValidationError(
            message=f"NAICS code '{result.value}' not found", field=field_name, value=result.value
        )

    return result


# --- Search Query Validator ---


def validate_search_query(
    query: str, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "query"
) -> ValidationResult:
    """
    Validate a search query.

    Args:
        query: The search query to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated query

    Raises:
        ValidationError: If validation fails
    """
    warnings = []

    # Check for None
    if query is None:
        raise ValidationError(message=f"{field_name} is required", field=field_name)

    # Check type
    if not isinstance(query, str):
        raise ValidationError(
            message=f"{field_name} must be a string", field=field_name, value=type(query).__name__
        )

    # Normalize
    normalized = normalize_text(query)

    # Check for empty
    if not normalized:
        raise ValidationError(message=f"{field_name} cannot be empty", field=field_name)

    # Check minimum length
    if len(normalized) < config.search_query_min_length:
        raise ValidationError(
            message=f"{field_name} must be at least {config.search_query_min_length} characters",
            field=field_name,
            value=normalized,
            constraints={"min_length": config.search_query_min_length},
        )

    # Truncate if too long
    if len(normalized) > config.search_query_max_length:
        original_length = len(normalized)
        normalized = normalized[: config.search_query_max_length]
        warnings.append(
            f"{field_name} truncated from {original_length} to {config.search_query_max_length} characters"
        )

    return ValidationResult.valid(normalized, warnings)


# --- Numeric Validators ---


def validate_limit(
    limit: int, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "limit"
) -> ValidationResult:
    """
    Validate a limit parameter.

    Args:
        limit: The limit value to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated limit

    Raises:
        ValidationError: If validation fails
    """
    # Handle None - use default
    if limit is None:
        return ValidationResult.valid(config.search_limit_default)

    # Check type
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            raise ValidationError(
                message=f"{field_name} must be an integer", field=field_name, value=str(limit)
            )

    # Check bounds
    if limit < 1:
        raise ValidationError(
            message=f"{field_name} must be at least 1",
            field=field_name,
            value=limit,
            constraints={"min": 1},
        )

    if limit > config.search_limit_max:
        # Cap at max instead of rejecting
        return ValidationResult.valid(
            config.search_limit_max,
            warnings=[f"{field_name} capped at maximum of {config.search_limit_max}"],
        )

    return ValidationResult.valid(limit)


def validate_confidence(
    confidence: float, config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "min_confidence"
) -> ValidationResult:
    """
    Validate a confidence threshold.

    Args:
        confidence: The confidence value to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated confidence

    Raises:
        ValidationError: If validation fails
    """
    # Handle None - use default
    if confidence is None:
        return ValidationResult.valid(config.confidence_default)

    # Check type
    if not isinstance(confidence, (int, float)):
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            raise ValidationError(
                message=f"{field_name} must be a number", field=field_name, value=str(confidence)
            )

    # Convert to float
    confidence = float(confidence)

    # Check bounds
    if confidence < config.confidence_min:
        raise ValidationError(
            message=f"{field_name} must be at least {config.confidence_min}",
            field=field_name,
            value=confidence,
            constraints={"min": config.confidence_min},
        )

    if confidence > config.confidence_max:
        raise ValidationError(
            message=f"{field_name} must be at most {config.confidence_max}",
            field=field_name,
            value=confidence,
            constraints={"max": config.confidence_max},
        )

    return ValidationResult.valid(confidence)


# --- Batch Validators ---


def validate_batch_descriptions(
    descriptions: list[str],
    config: ValidationConfig = DEFAULT_CONFIG,
    field_name: str = "descriptions",
) -> ValidationResult:
    """
    Validate a batch of descriptions.

    Args:
        descriptions: List of descriptions to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated descriptions

    Raises:
        ValidationError: If validation fails
    """
    warnings = []

    # Check for None
    if descriptions is None:
        raise ValidationError(message=f"{field_name} is required", field=field_name)

    # Check type
    if not isinstance(descriptions, list):
        raise ValidationError(
            message=f"{field_name} must be a list",
            field=field_name,
            value=type(descriptions).__name__,
        )

    # Check for empty list
    if len(descriptions) == 0:
        raise ValidationError(message=f"{field_name} cannot be empty", field=field_name)

    # Check batch size
    if len(descriptions) > config.batch_max_size:
        raise ValidationError(
            message=f"{field_name} exceeds maximum batch size of {config.batch_max_size}",
            field=field_name,
            constraints={"max_size": config.batch_max_size, "actual_size": len(descriptions)},
        )

    # Validate each description
    validated_descriptions = []
    for i, desc in enumerate(descriptions):
        try:
            result = validate_description(desc, config, field_name=f"{field_name}[{i}]")
            validated_descriptions.append(result.value)
            warnings.extend(result.warnings)
        except ValidationError as e:
            # Re-raise with index context
            raise ValidationError(
                message=f"Invalid item at index {i}: {e.message}",
                field=f"{field_name}[{i}]",
                value=desc[:50] if isinstance(desc, str) else str(desc)[:50],
                constraints=e.details.get("constraints"),
            )

    return ValidationResult.valid(validated_descriptions, warnings)


def validate_batch_codes(
    codes: list[str], config: ValidationConfig = DEFAULT_CONFIG, field_name: str = "codes"
) -> ValidationResult:
    """
    Validate a batch of NAICS codes.

    Args:
        codes: List of codes to validate
        config: Validation configuration
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated codes

    Raises:
        ValidationError: If validation fails
    """
    # Check for None
    if codes is None:
        raise ValidationError(message=f"{field_name} is required", field=field_name)

    # Check type
    if not isinstance(codes, list):
        raise ValidationError(
            message=f"{field_name} must be a list", field=field_name, value=type(codes).__name__
        )

    # Check for empty list
    if len(codes) == 0:
        raise ValidationError(message=f"{field_name} cannot be empty", field=field_name)

    # Check batch size (use a reasonable limit for code comparisons)
    max_codes = 20  # Reasonable limit for code comparison
    if len(codes) > max_codes:
        raise ValidationError(
            message=f"{field_name} exceeds maximum of {max_codes} codes",
            field=field_name,
            constraints={"max_size": max_codes, "actual_size": len(codes)},
        )

    # Validate each code
    validated_codes = []
    for i, code in enumerate(codes):
        try:
            result = validate_naics_code(code, config, field_name=f"{field_name}[{i}]")
            validated_codes.append(result.value)
        except ValidationError as e:
            raise ValidationError(
                message=f"Invalid code at index {i}: {e.message}",
                field=f"{field_name}[{i}]",
                value=str(code)[:20],
            )

    return ValidationResult.valid(validated_codes)


# --- Strategy Validator ---

VALID_STRATEGIES = {"hybrid", "semantic", "lexical", "best_match", "meaning", "exact"}


def validate_strategy(strategy: str, field_name: str = "strategy") -> ValidationResult:
    """
    Validate a search strategy.

    Args:
        strategy: The strategy to validate
        field_name: Name of the field for error messages

    Returns:
        ValidationResult with validated strategy

    Raises:
        ValidationError: If validation fails
    """
    # Handle None - use default
    if strategy is None:
        return ValidationResult.valid("hybrid")

    # Check type
    if not isinstance(strategy, str):
        raise ValidationError(
            message=f"{field_name} must be a string",
            field=field_name,
            value=type(strategy).__name__,
        )

    # Normalize
    strategy = strategy.lower().strip()

    # Check validity
    if strategy not in VALID_STRATEGIES:
        raise ValidationError(
            message=f"Invalid {field_name}: '{strategy}'. Must be one of: {', '.join(sorted(VALID_STRATEGIES))}",
            field=field_name,
            value=strategy,
            constraints={"valid_values": sorted(VALID_STRATEGIES)},
        )

    return ValidationResult.valid(strategy)
