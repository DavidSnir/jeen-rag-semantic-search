"""Stage 6 integration tests for PostgreSQL/pgvector semantic search."""

import hashlib
import math
import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from rag_app.database.connection import open_database_connection
from rag_app.database.repository import initialize_schema, persist_embedded_document
from rag_app.database.search import search_similar_chunks
from rag_app.documents import (
    Chunk,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
)
from rag_app.services.search import search_documents

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_URL"),
    reason="POSTGRES_URL is required for PostgreSQL integration tests",
)

_VECTOR_DIMENSION = 768


def _normalized_vector(*coordinates: tuple[int, float]) -> tuple[float, ...]:
    values = [0.0] * _VECTOR_DIMENSION
    for position, value in coordinates:
        values[position] = value
    norm = math.hypot(*values)
    return tuple(value / norm for value in values)


_QUERY_VECTOR = _normalized_vector((0, 1.0))
_RELEVANT_VECTOR = _normalized_vector((0, 0.8), (1, 0.6))
_ORTHOGONAL_VECTOR = _normalized_vector((1, 1.0))
_OPPOSING_VECTOR = _normalized_vector((0, -1.0))


@pytest.fixture(scope="module", autouse=True)
def _initialized_schema() -> None:
    initialize_schema()


@pytest.fixture
def unique_source_file():
    source_files: set[str] = set()

    def create(label: str, suffix: str = ".pdf") -> str:
        source_file = f"stage-6-{label}-{uuid4().hex}{suffix}"
        source_files.add(source_file)
        return source_file

    yield create

    if source_files:
        with open_database_connection() as connection:
            connection.execute(
                "DELETE FROM public.chunks WHERE source_file = ANY(%s)",
                (list(source_files),),
            )
            connection.commit()


def _persist_chunk(
    source_file: str,
    strategy: ChunkingStrategy,
    content: str,
    embedding: tuple[float, ...],
    *,
    page_number: int | None = None,
) -> None:
    source_type = "PDF" if Path(source_file).suffix.lower() == ".pdf" else "DOCX"
    document = EmbeddedDocument(
        source_file=source_file,
        source_type=source_type,
        chunking_strategy=strategy,
        chunks=(
            EmbeddedChunk(
                chunk=Chunk(
                    content=content,
                    chunk_index=0,
                    page_number=page_number,
                ),
                embedding=embedding,
            ),
        ),
    )
    document_hash = hashlib.sha256(source_file.encode("utf-8")).hexdigest()
    persist_embedded_document(document, document_hash)


def _strategy_row_count(strategy: ChunkingStrategy) -> int:
    with open_database_connection() as connection:
        count = connection.execute(
            "SELECT count(*) FROM public.chunks WHERE chunking_strategy = %s",
            (strategy.value,),
        ).fetchone()[0]
        connection.rollback()
    return count


def _total_row_count() -> int:
    with open_database_connection() as connection:
        count = connection.execute("SELECT count(*) FROM public.chunks").fetchone()[0]
        connection.rollback()
    return count


def _require_empty_strategy() -> ChunkingStrategy:
    for strategy in ChunkingStrategy:
        if _strategy_row_count(strategy) == 0:
            return strategy
    pytest.skip("exact ranking assertions require one strategy without existing rows")


def test_service_ranks_cosine_results_preserves_metadata_and_does_not_write(
    unique_source_file,
) -> None:
    strategy = _require_empty_strategy()
    aligned_pdf = unique_source_file("aligned", ".pdf")
    relevant_docx = unique_source_file("relevant", ".docx")
    orthogonal_pdf = unique_source_file("orthogonal", ".pdf")
    opposing_docx = unique_source_file("opposing", ".docx")

    _persist_chunk(
        aligned_pdf,
        strategy,
        "Exactly aligned semantic content",
        _QUERY_VECTOR,
        page_number=7,
    )
    _persist_chunk(
        relevant_docx,
        strategy,
        "Relevant but less aligned semantic content",
        _RELEVANT_VECTOR,
    )
    _persist_chunk(
        orthogonal_pdf,
        strategy,
        "Semantically orthogonal content",
        _ORTHOGONAL_VECTOR,
        page_number=3,
    )
    _persist_chunk(
        opposing_docx,
        strategy,
        "Irrelevant opposing semantic content",
        _OPPOSING_VECTOR,
    )
    row_count_before_search = _total_row_count()
    embedded_queries: list[str] = []

    def deterministic_query_embedder(query: str) -> tuple[float, ...]:
        embedded_queries.append(query)
        return _QUERY_VECTOR

    limited = search_documents(
        "  deterministic semantic query  ",
        strategy.value,
        top_k=3,
        query_embedder=deterministic_query_embedder,
    )
    complete = search_documents(
        "deterministic semantic query",
        strategy,
        top_k=4,
        query_embedder=deterministic_query_embedder,
    )

    assert len(limited.matches) == limited.top_k == 3
    assert [match.source_file for match in limited.matches] == [
        aligned_pdf,
        relevant_docx,
        orthogonal_pdf,
    ]
    assert len(complete.matches) == complete.top_k == 4
    assert complete.query == "deterministic semantic query"
    assert complete.chunking_strategy is strategy
    assert [match.source_file for match in complete.matches] == [
        aligned_pdf,
        relevant_docx,
        orthogonal_pdf,
        opposing_docx,
    ]
    assert [match.rank for match in complete.matches] == [1, 2, 3, 4]
    assert [match.distance for match in complete.matches] == pytest.approx(
        [0.0, 0.2, 1.0, 2.0], abs=1e-6
    )
    assert [match.score for match in complete.matches] == pytest.approx(
        [1.0, 0.8, 0.0, -1.0], abs=1e-6
    )
    assert [match.distance for match in complete.matches] == sorted(
        match.distance for match in complete.matches
    )
    assert [match.score for match in complete.matches] == sorted(
        (match.score for match in complete.matches), reverse=True
    )
    assert all(
        match.score == pytest.approx(1.0 - match.distance, abs=1e-9)
        for match in complete.matches
    )
    assert [
        (match.source_type, match.page_number, match.chunk_index)
        for match in complete.matches
    ] == [
        ("PDF", 7, 0),
        ("DOCX", None, 0),
        ("PDF", 3, 0),
        ("DOCX", None, 0),
    ]
    assert complete.matches[1].score > complete.matches[2].score
    assert complete.matches[2].score > complete.matches[3].score
    assert complete.matches[-1].content == "Irrelevant opposing semantic content"
    assert embedded_queries == [
        "deterministic semantic query",
        "deterministic semantic query",
    ]
    assert _total_row_count() == row_count_before_search


def test_repository_filters_every_canonical_strategy(unique_source_file) -> None:
    seeded_files: dict[ChunkingStrategy, str] = {}
    for strategy in ChunkingStrategy:
        source_file = unique_source_file(f"filter-{strategy.value}")
        seeded_files[strategy] = source_file
        _persist_chunk(
            source_file,
            strategy,
            f"Unique {strategy.value} strategy result",
            _QUERY_VECTOR,
            page_number=1,
        )

    all_seeded_files = set(seeded_files.values())
    for strategy, expected_source_file in seeded_files.items():
        strategy_row_count = _strategy_row_count(strategy)
        rows = search_similar_chunks(
            _QUERY_VECTOR,
            strategy,
            strategy_row_count,
        )

        assert len(rows) == strategy_row_count
        assert all(row.chunking_strategy == strategy.value for row in rows)
        returned_seeded_files = {
            row.source_file for row in rows if row.source_file in all_seeded_files
        }
        assert returned_seeded_files == {expected_source_file}


def test_service_returns_empty_matches_for_strategy_without_chunks() -> None:
    strategy = _require_empty_strategy()

    response = search_documents(
        "query with no indexed strategy",
        strategy.value,
        top_k=5,
        query_embedder=lambda _query: _QUERY_VECTOR,
    )

    assert response.chunking_strategy is strategy
    assert response.top_k == 5
    assert response.matches == ()
