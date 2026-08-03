"""End-to-end indexing integration tests using deterministic local embeddings."""

import hashlib
import shutil
from pathlib import Path

import pytest

from rag_app.database.connection import open_database_connection
from rag_app.documents import (
    ChunkedDocument,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    IndexingStatus,
)
from rag_app.services.indexing import index_document
from rag_app.services.search import search_documents

_VECTOR_DIMENSION = 768
_DOCX_FIXTURE = Path(__file__).parents[1] / "fixtures" / "docx" / "ordered-content.docx"
_EXPECTED_CONTENT = (
    "Before table",
    "A1 | B1",
    "Between tables",
    "Left | | Right",
    "After table",
)


def _coordinate_vector(position: int) -> tuple[float, ...]:
    """Return a normalized deterministic vector with one active coordinate."""
    values = [0.0] * _VECTOR_DIMENSION
    values[position] = 1.0
    return tuple(values)


def _embed_deterministically(document: ChunkedDocument) -> EmbeddedDocument:
    """Embed chunks without network or Gemini configuration dependencies."""
    return EmbeddedDocument(
        source_file=document.source_file,
        source_type=document.source_type,
        chunking_strategy=document.chunking_strategy,
        chunks=tuple(
            EmbeddedChunk(
                chunk=chunk,
                embedding=_coordinate_vector(chunk.chunk_index),
            )
            for chunk in document.chunks
        ),
    )


@pytest.mark.parametrize(
    "strategy",
    (ChunkingStrategy.sentence, ChunkingStrategy.paragraph),
)
def test_docx_full_indexing_pipeline_persists_and_retrieves_exact_chunks(
    tmp_path: Path,
    strategy: ChunkingStrategy,
) -> None:
    source_path = tmp_path / _DOCX_FIXTURE.name
    shutil.copy2(_DOCX_FIXTURE, source_path)
    expected_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    result = index_document(
        source_path,
        strategy.value,
        embedder=_embed_deterministically,
    )

    assert result.status is IndexingStatus.indexed
    assert result.source_file == "ordered-content.docx"
    assert result.chunking_strategy is strategy
    assert result.chunk_count == len(_EXPECTED_CONTENT)

    with open_database_connection() as connection:
        rows = connection.execute(
            """
            SELECT content, embedding, source_file, document_hash, source_type,
                   chunk_index, chunking_strategy, page_number
            FROM public.chunks
            ORDER BY chunk_index
            """
        ).fetchall()
        connection.rollback()

    assert [row[0] for row in rows] == list(_EXPECTED_CONTENT)
    assert [row[2:] for row in rows] == [
        (
            "ordered-content.docx",
            expected_hash,
            "DOCX",
            chunk_index,
            strategy.value,
            None,
        )
        for chunk_index in range(len(_EXPECTED_CONTENT))
    ]
    stored_vectors = [tuple(row[1].to_list()) for row in rows]
    assert all(len(vector) == _VECTOR_DIMENSION for vector in stored_vectors)
    assert stored_vectors == [
        _coordinate_vector(chunk_index) for chunk_index in range(len(_EXPECTED_CONTENT))
    ]

    response = search_documents(
        "first ordered content",
        strategy,
        top_k=1,
        query_embedder=lambda _query: _coordinate_vector(0),
    )

    assert len(response.matches) == 1
    assert response.matches[0].content == _EXPECTED_CONTENT[0]
    assert response.matches[0].source_file == "ordered-content.docx"
    assert response.matches[0].source_type == "DOCX"
    assert response.matches[0].chunk_index == 0
    assert response.matches[0].chunking_strategy is strategy
    assert response.matches[0].page_number is None
    assert response.matches[0].distance == pytest.approx(0.0, abs=1e-9)
