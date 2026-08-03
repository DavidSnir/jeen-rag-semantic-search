"""Shared strict validation and normalization for embedding vectors."""

import math
from collections.abc import Sequence
from numbers import Real

from rag_app.config import EMBEDDING_DIMENSION
from rag_app.documents import EmbeddingVector
from rag_app.exceptions import (
    EmbeddingDimensionError,
    EmbeddingNormalizationError,
    InvalidGeminiResponseError,
)


def validate_and_normalize_embedding(
    values: object,
    *,
    chunk_index: int,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> EmbeddingVector:
    """Return a finite unit vector or fail without exposing chunk content."""
    if (
        values is None
        or isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
    ):
        raise InvalidGeminiResponseError(
            f"Gemini embedding values are missing for chunk {chunk_index}"
        )
    if not values:
        raise InvalidGeminiResponseError(
            f"Gemini returned an empty embedding for chunk {chunk_index}"
        )
    if len(values) != expected_dimension:
        raise EmbeddingDimensionError(
            f"Gemini returned {len(values)} values for chunk {chunk_index}; "
            f"expected {expected_dimension}"
        )

    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidGeminiResponseError(
                f"Gemini returned a non-numeric value for chunk {chunk_index}"
            )
        try:
            converted_value = float(value)
        except (OverflowError, ValueError) as error:
            raise InvalidGeminiResponseError(
                f"Gemini returned an invalid numeric value for chunk {chunk_index}"
            ) from error
        if not math.isfinite(converted_value):
            raise InvalidGeminiResponseError(
                f"Gemini returned a non-finite value for chunk {chunk_index}"
            )
        converted.append(converted_value)

    norm = math.hypot(*converted)
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbeddingNormalizationError(
            f"Gemini embedding for chunk {chunk_index} cannot be normalized"
        )

    normalized = tuple(value / norm for value in converted)
    if not all(math.isfinite(value) for value in normalized):
        raise EmbeddingNormalizationError(
            f"Gemini embedding for chunk {chunk_index} cannot be normalized"
        )
    if not math.isclose(math.hypot(*normalized), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise EmbeddingNormalizationError(
            f"Gemini embedding for chunk {chunk_index} cannot be normalized"
        )
    return normalized


__all__ = ["validate_and_normalize_embedding"]
