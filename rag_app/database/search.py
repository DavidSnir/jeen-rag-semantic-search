"""Read-only PostgreSQL/pgvector semantic-search repository."""

import math
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector import Vector

from rag_app.config import EMBEDDING_DIMENSION
from rag_app.database.connection import open_database_connection
from rag_app.documents import ChunkingStrategy, EmbeddingVector
from rag_app.exceptions import RagAppError, SemanticSearchError

_SEMANTIC_SEARCH_SQL = """
    SELECT content,
           source_file,
           source_type,
           chunk_index,
           chunking_strategy,
           page_number,
           embedding <=> %s AS distance
    FROM public.chunks
    WHERE chunking_strategy = %s
    ORDER BY embedding <=> %s
    LIMIT %s
"""
_ENABLE_ITERATIVE_SCAN_SQL = "SET LOCAL hnsw.iterative_scan = strict_order"


@dataclass(frozen=True, slots=True)
class SemanticSearchRow:
    """Raw ordered database values needed by the search service."""

    content: object
    source_file: object
    source_type: object
    chunk_index: object
    chunking_strategy: object
    page_number: object
    distance: object


def search_similar_chunks(
    query_vector: EmbeddingVector,
    strategy: ChunkingStrategy,
    top_k: int,
) -> tuple[SemanticSearchRow, ...]:
    """Return strategy-scoped rows in ascending cosine-distance order."""
    _validate_repository_input(query_vector, strategy, top_k)
    vector = Vector(list(query_vector))

    with open_database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(_ENABLE_ITERATIVE_SCAN_SQL)
                rows = cursor.execute(
                    _SEMANTIC_SEARCH_SQL,
                    (vector, strategy.value, vector, top_k),
                ).fetchall()
            results = tuple(_map_row(row) for row in rows)
            connection.rollback()
            return results
        except RagAppError:
            _rollback_quietly(connection)
            raise
        except psycopg.Error as error:
            _rollback_quietly(connection)
            raise SemanticSearchError(
                "Semantic search query failed."
            ) from error


def _validate_repository_input(
    query_vector: object,
    strategy: object,
    top_k: object,
) -> None:
    if (
        not isinstance(query_vector, tuple)
        or len(query_vector) != EMBEDDING_DIMENSION
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in query_vector
        )
    ):
        raise SemanticSearchError(
            "Semantic search requires a valid 768-dimensional query vector."
        )
    norm = math.hypot(*query_vector)
    if not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise SemanticSearchError(
            "Semantic search requires a normalized query vector."
        )
    if not isinstance(strategy, ChunkingStrategy):
        raise SemanticSearchError(
            "Semantic search requires a canonical chunking strategy."
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise SemanticSearchError(
            "Semantic search requires a positive integer result count."
        )


def _map_row(row: Any) -> SemanticSearchRow:
    try:
        return SemanticSearchRow(*row)
    except (TypeError, ValueError) as error:
        raise SemanticSearchError(
            "Semantic search returned an invalid database row."
        ) from error


def _rollback_quietly(connection: Any) -> None:
    try:
        connection.rollback()
    except psycopg.Error:
        pass


__all__ = ["SemanticSearchRow", "search_similar_chunks"]
