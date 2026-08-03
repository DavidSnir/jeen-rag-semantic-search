"""Complete semantic-search application service."""

import logging
import math
from collections.abc import Callable
from numbers import Real
from pathlib import Path, PureWindowsPath

from rag_app.config import DEFAULT_TOP_K, EMBEDDING_DIMENSION
from rag_app.database.repository import check_database_readiness
from rag_app.database.search import SemanticSearchRow, search_similar_chunks
from rag_app.documents import (
    ChunkingStrategy,
    EmbeddingVector,
    SearchMatch,
    SearchResponse,
)
from rag_app.embeddings import embed_query
from rag_app.exceptions import SearchPipelineError, SearchValidationError
from rag_app.processing.chunking import validate_chunking_strategy

logger = logging.getLogger(__name__)

_COSINE_RANGE_TOLERANCE = 1e-9


def validate_search_query(query: object) -> str:
    """Return query text with only surrounding whitespace removed."""
    if not isinstance(query, str):
        raise SearchValidationError("Search query must be a string.")
    canonical_query = query.strip()
    if not canonical_query:
        raise SearchValidationError("Search query must not be empty.")
    return canonical_query


def validate_top_k(top_k: object) -> int:
    """Require a positive non-boolean integer result count."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise SearchValidationError("Search top_k must be a positive integer.")
    return top_k


def search_documents(
    query: str,
    strategy: str | ChunkingStrategy,
    top_k: int = DEFAULT_TOP_K,
    *,
    query_validator: Callable[[object], str] = validate_search_query,
    strategy_validator: Callable[
        [str | ChunkingStrategy], ChunkingStrategy
    ] = validate_chunking_strategy,
    top_k_validator: Callable[[object], int] = validate_top_k,
    readiness_checker: Callable[[], object] = check_database_readiness,
    query_embedder: Callable[[str], EmbeddingVector] = embed_query,
    repository_search: Callable[
        [EmbeddingVector, ChunkingStrategy, int],
        tuple[SemanticSearchRow, ...],
    ] = search_similar_chunks,
) -> SearchResponse:
    """Validate, embed, retrieve, and return ordered immutable matches."""
    canonical_query = query_validator(query)
    canonical_strategy = strategy_validator(strategy)
    validated_top_k = top_k_validator(top_k)

    logger.info(
        "Starting semantic search strategy=%s top_k=%d query_chars=%d",
        canonical_strategy.value,
        validated_top_k,
        len(canonical_query),
    )
    readiness_checker()
    query_vector = query_embedder(canonical_query)
    _validate_query_vector(query_vector)
    rows = repository_search(query_vector, canonical_strategy, validated_top_k)
    matches = _validate_search_rows(rows, canonical_strategy, validated_top_k)
    logger.info(
        "Semantic search completed strategy=%s top_k=%d query_chars=%d results=%d",
        canonical_strategy.value,
        validated_top_k,
        len(canonical_query),
        len(matches),
    )
    return SearchResponse(
        query=canonical_query,
        chunking_strategy=canonical_strategy,
        top_k=validated_top_k,
        matches=matches,
    )


def _validate_query_vector(vector: object) -> None:
    if (
        not isinstance(vector, tuple)
        or len(vector) != EMBEDDING_DIMENSION
        or any(type(value) is not float or not math.isfinite(value) for value in vector)
        or not math.isclose(
            math.hypot(*vector), 1.0, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise SearchPipelineError(
            "The query embedding stage returned an invalid normalized vector."
        )


def _validate_search_rows(
    rows: object,
    strategy: ChunkingStrategy,
    top_k: int,
) -> tuple[SearchMatch, ...]:
    if not isinstance(rows, tuple):
        raise SearchPipelineError(
            "The search repository returned an invalid result collection."
        )
    if len(rows) > top_k:
        raise SearchPipelineError(
            "The search repository returned more results than requested."
        )

    matches: list[SearchMatch] = []
    previous_distance: float | None = None
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, SemanticSearchRow):
            raise SearchPipelineError(
                "The search repository returned an invalid result row."
            )
        distance = _validate_distance(row.distance)
        if (
            previous_distance is not None
            and distance < previous_distance - _COSINE_RANGE_TOLERANCE
        ):
            raise SearchPipelineError(
                "The search repository returned results out of distance order."
            )
        previous_distance = distance
        score = 1.0 - distance
        if not math.isfinite(score):
            raise SearchPipelineError(
                "The search repository returned an invalid similarity score."
            )
        source_type = _validate_source_type(row.source_type)
        source_file = _validate_source_file(row.source_file, source_type)
        chunk_index = _validate_chunk_index(row.chunk_index)
        page_number = _validate_page_number(row.page_number, source_type)
        if row.chunking_strategy != strategy.value:
            raise SearchPipelineError(
                "The search repository returned a result for another strategy."
            )
        if not isinstance(row.content, str) or not row.content.strip():
            raise SearchPipelineError(
                "The search repository returned a result without meaningful content."
            )
        matches.append(
            SearchMatch(
                rank=rank,
                content=row.content,
                source_file=source_file,
                source_type=source_type,
                chunk_index=chunk_index,
                chunking_strategy=strategy,
                page_number=page_number,
                distance=distance,
                score=score,
            )
        )
    return tuple(matches)


def _validate_distance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SearchPipelineError(
            "The search repository returned an invalid cosine distance."
        )
    try:
        distance = float(value)
    except (OverflowError, ValueError) as error:
        raise SearchPipelineError(
            "The search repository returned an invalid cosine distance."
        ) from error
    if (
        not math.isfinite(distance)
        or distance < -_COSINE_RANGE_TOLERANCE
        or distance > 2.0 + _COSINE_RANGE_TOLERANCE
    ):
        raise SearchPipelineError(
            "The search repository returned an invalid cosine distance."
        )
    return distance


def _validate_source_type(value: object) -> str:
    if not isinstance(value, str) or value not in ("PDF", "DOCX"):
        raise SearchPipelineError(
            "The search repository returned an unsupported source type."
        )
    return value  # type: ignore[return-value]


def _validate_source_file(value: object, source_type: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or value in {".", ".."}
        or Path(value).name != value
        or PureWindowsPath(value).name != value
    ):
        raise SearchPipelineError(
            "The search repository returned an unsafe source filename."
        )
    expected_suffix = ".pdf" if source_type == "PDF" else ".docx"
    if Path(value).suffix.lower() != expected_suffix:
        raise SearchPipelineError(
            "The search repository returned inconsistent source metadata."
        )
    return value


def _validate_chunk_index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SearchPipelineError(
            "The search repository returned an invalid chunk index."
        )
    return value


def _validate_page_number(value: object, source_type: str) -> int | None:
    if source_type == "PDF":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SearchPipelineError(
                "The search repository returned an invalid PDF page number."
            )
        return value
    if value is not None:
        raise SearchPipelineError(
            "The search repository returned a page number for a DOCX result."
        )
    return None


__all__ = ["search_documents", "validate_search_query", "validate_top_k"]
