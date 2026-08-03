"""Stage 5 integration tests for transactional PostgreSQL chunk persistence."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

import rag_app.database.repository as repository
from rag_app.database.connection import open_database_connection
from rag_app.database.repository import (
    _document_lock_key,
    get_indexed_document_state,
    initialize_schema,
    persist_embedded_document,
)
from rag_app.documents import (
    Chunk,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    IndexingStatus,
)
from rag_app.exceptions import PersistenceError

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_URL"),
    reason="POSTGRES_URL is required for PostgreSQL integration tests",
)

_VECTOR_DIMENSION = 768


@pytest.fixture(scope="module", autouse=True)
def _initialized_schema() -> None:
    initialize_schema()


@pytest.fixture
def unique_source_file():
    source_files: set[str] = set()

    def create(label: str, suffix: str = ".pdf") -> str:
        source_file = f"stage-5-{label}-{uuid4().hex}{suffix}"
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


def _normalized_vector(position: int) -> tuple[float, ...]:
    values = [0.0] * _VECTOR_DIMENSION
    values[position % _VECTOR_DIMENSION] = 1.0
    return tuple(values)


def _embedded_document(
    source_file: str,
    strategy: ChunkingStrategy,
    contents: tuple[str, ...],
    *,
    vector_offset: int = 0,
) -> EmbeddedDocument:
    source_type = "PDF" if Path(source_file).suffix.lower() == ".pdf" else "DOCX"
    return EmbeddedDocument(
        source_file=source_file,
        source_type=source_type,
        chunking_strategy=strategy,
        chunks=tuple(
            EmbeddedChunk(
                chunk=Chunk(
                    content=content,
                    chunk_index=index,
                    page_number=index + 1 if source_type == "PDF" else None,
                ),
                embedding=_normalized_vector(vector_offset + index),
            )
            for index, content in enumerate(contents)
        ),
    )


def _stored_rows(source_file: str, strategy: ChunkingStrategy):
    with open_database_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, content, embedding, source_file, document_hash,
                   source_type, chunk_index, chunking_strategy, page_number,
                   created_at
            FROM public.chunks
            WHERE source_file = %s AND chunking_strategy = %s
            ORDER BY chunk_index
            """,
            (source_file, strategy.value),
        ).fetchall()
        connection.rollback()
    return rows


def _assert_rows_match(
    rows, document: EmbeddedDocument, document_hash: str
) -> None:
    assert len(rows) == len(document.chunks)
    assert [row[6] for row in rows] == list(range(len(document.chunks)))
    assert len({row[0] for row in rows}) == len(rows)

    for row, embedded_chunk in zip(rows, document.chunks, strict=True):
        chunk = embedded_chunk.chunk
        assert isinstance(row[0], int) and row[0] > 0
        assert row[1] == chunk.content
        stored_embedding = tuple(row[2].to_list())
        assert len(stored_embedding) == _VECTOR_DIMENSION
        assert stored_embedding == embedded_chunk.embedding
        assert row[3] == document.source_file
        assert row[4] == document_hash
        assert row[5] == document.source_type
        assert row[6] == chunk.chunk_index
        assert row[7] == document.chunking_strategy.value
        assert row[8] == chunk.page_number
        assert isinstance(row[9], datetime)
        assert row[9].utcoffset() is not None


def test_multi_chunk_insert_retrieves_every_field_and_vector(
    unique_source_file,
) -> None:
    source_file = unique_source_file("multi-chunk")
    document_hash = "a" * 64
    document = _embedded_document(
        source_file,
        ChunkingStrategy.fixed,
        ("First persisted chunk", "Second persisted chunk", "Third persisted chunk"),
        vector_offset=11,
    )

    result = persist_embedded_document(document, document_hash)

    assert result.status is IndexingStatus.indexed
    assert result.chunk_count == 3
    _assert_rows_match(
        _stored_rows(source_file, ChunkingStrategy.fixed), document, document_hash
    )
    state = get_indexed_document_state(source_file, ChunkingStrategy.fixed)
    assert state.document_hashes == (document_hash,)
    assert state.total_chunk_count == 3
    assert state.chunk_count_for(document_hash) == 3


def test_exact_duplicate_is_skipped_without_touching_rows_or_timestamps(
    unique_source_file,
) -> None:
    source_file = unique_source_file("duplicate")
    document_hash = "b" * 64
    document = _embedded_document(
        source_file,
        ChunkingStrategy.sentence,
        ("Duplicate chunk one", "Duplicate chunk two"),
        vector_offset=21,
    )
    first_result = persist_embedded_document(document, document_hash)
    before = _stored_rows(source_file, ChunkingStrategy.sentence)

    duplicate_result = persist_embedded_document(document, document_hash)
    after = _stored_rows(source_file, ChunkingStrategy.sentence)

    assert first_result.status is IndexingStatus.indexed
    assert duplicate_result.status is IndexingStatus.skipped
    assert duplicate_result.chunk_count == len(document.chunks)
    assert len(after) == len(before) == 2
    assert [(row[0], row[9]) for row in after] == [
        (row[0], row[9]) for row in before
    ]
    assert after == before
    state = get_indexed_document_state(source_file, ChunkingStrategy.sentence)
    assert state.document_hashes == (document_hash,)
    assert state.total_chunk_count == 2


def test_changed_hash_atomically_replaces_the_previous_document(
    unique_source_file,
) -> None:
    source_file = unique_source_file("changed-hash")
    old_hash = "c" * 64
    new_hash = "d" * 64
    old_document = _embedded_document(
        source_file,
        ChunkingStrategy.paragraph,
        ("Old paragraph one", "Old paragraph two"),
        vector_offset=31,
    )
    new_document = _embedded_document(
        source_file,
        ChunkingStrategy.paragraph,
        ("New paragraph one", "New paragraph two", "New paragraph three"),
        vector_offset=41,
    )
    persist_embedded_document(old_document, old_hash)
    old_ids = {row[0] for row in _stored_rows(source_file, ChunkingStrategy.paragraph)}

    result = persist_embedded_document(new_document, new_hash)
    rows = _stored_rows(source_file, ChunkingStrategy.paragraph)

    assert result.status is IndexingStatus.replaced
    assert result.chunk_count == 3
    assert old_ids.isdisjoint({row[0] for row in rows})
    _assert_rows_match(rows, new_document, new_hash)
    state = get_indexed_document_state(source_file, ChunkingStrategy.paragraph)
    assert state.document_hashes == (new_hash,)
    assert state.chunk_count_for(old_hash) is None
    assert state.chunk_count_for(new_hash) == 3


def test_different_chunking_strategies_for_one_filename_coexist(
    unique_source_file,
) -> None:
    source_file = unique_source_file("strategies")
    documents = {
        ChunkingStrategy.fixed: (_embedded_document(
            source_file, ChunkingStrategy.fixed, ("Fixed one", "Fixed two")
        ), "1" * 64),
        ChunkingStrategy.sentence: (_embedded_document(
            source_file,
            ChunkingStrategy.sentence,
            ("Sentence one", "Sentence two", "Sentence three"),
            vector_offset=10,
        ), "2" * 64),
        ChunkingStrategy.paragraph: (_embedded_document(
            source_file,
            ChunkingStrategy.paragraph,
            ("Paragraph only",),
            vector_offset=20,
        ), "3" * 64),
    }

    for document, document_hash in documents.values():
        result = persist_embedded_document(document, document_hash)
        assert result.status is IndexingStatus.indexed

    for strategy, (document, document_hash) in documents.items():
        _assert_rows_match(_stored_rows(source_file, strategy), document, document_hash)
        state = get_indexed_document_state(source_file, strategy)
        assert state.document_hashes == (document_hash,)
        assert state.total_chunk_count == len(document.chunks)


def test_identical_hashes_under_different_filenames_coexist(
    unique_source_file,
) -> None:
    first_source = unique_source_file("same-hash-first")
    second_source = unique_source_file("same-hash-second")
    document_hash = "e" * 64
    first_document = _embedded_document(
        first_source, ChunkingStrategy.fixed, ("First file content",)
    )
    second_document = _embedded_document(
        second_source,
        ChunkingStrategy.fixed,
        ("Second file content one", "Second file content two"),
        vector_offset=50,
    )

    first_result = persist_embedded_document(first_document, document_hash)
    second_result = persist_embedded_document(second_document, document_hash)

    assert first_result.status is second_result.status is IndexingStatus.indexed
    _assert_rows_match(
        _stored_rows(first_source, ChunkingStrategy.fixed),
        first_document,
        document_hash,
    )
    _assert_rows_match(
        _stored_rows(second_source, ChunkingStrategy.fixed),
        second_document,
        document_hash,
    )
    assert get_indexed_document_state(
        first_source, ChunkingStrategy.fixed
    ).total_chunk_count == 1
    assert get_indexed_document_state(
        second_source, ChunkingStrategy.fixed
    ).total_chunk_count == 2


def test_insertion_failure_after_deletion_rolls_back_the_old_rows(
    unique_source_file, monkeypatch
) -> None:
    source_file = unique_source_file("rollback")
    old_hash = "f" * 64
    new_hash = "0" * 64
    strategy = ChunkingStrategy.sentence
    old_document = _embedded_document(
        source_file,
        strategy,
        ("Stable old chunk one", "Stable old chunk two"),
        vector_offset=60,
    )
    replacement = _embedded_document(
        source_file,
        strategy,
        ("Replacement chunk",),
        vector_offset=70,
    )
    persist_embedded_document(old_document, old_hash)
    before = _stored_rows(source_file, strategy)
    rows_seen_after_delete: list[int] = []

    def fail_insertion(cursor, _rows) -> None:
        remaining = cursor.execute(
            """
            SELECT count(*) FROM public.chunks
            WHERE source_file = %s AND chunking_strategy = %s
            """,
            (source_file, strategy.value),
        ).fetchone()[0]
        rows_seen_after_delete.append(remaining)
        raise RuntimeError("forced insertion failure")

    monkeypatch.setattr(repository, "_insert_rows", fail_insertion)

    with pytest.raises(PersistenceError):
        persist_embedded_document(replacement, new_hash)

    assert rows_seen_after_delete == [0]
    assert _stored_rows(source_file, strategy) == before
    state = get_indexed_document_state(source_file, strategy)
    assert state.document_hashes == (old_hash,)
    assert state.chunk_count_for(old_hash) == len(old_document.chunks)
    assert state.chunk_count_for(new_hash) is None


def test_concurrent_writes_leave_one_complete_document_version(
    unique_source_file,
) -> None:
    source_file = unique_source_file("concurrent")
    strategy = ChunkingStrategy.fixed
    first_hash = "4" * 64
    second_hash = "5" * 64
    first_document = _embedded_document(
        source_file,
        strategy,
        ("First A", "First B", "First C"),
        vector_offset=80,
    )
    second_document = _embedded_document(
        source_file,
        strategy,
        ("Second A", "Second B", "Second C", "Second D"),
        vector_offset=90,
    )
    barrier = threading.Barrier(2)

    def persist_concurrently(document, document_hash):
        barrier.wait(timeout=10)
        return persist_embedded_document(document, document_hash)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            persist_concurrently, first_document, first_hash
        )
        second_future = executor.submit(
            persist_concurrently, second_document, second_hash
        )
        results = (first_future.result(timeout=20), second_future.result(timeout=20))

    assert sorted(result.status.value for result in results) == ["indexed", "replaced"]
    rows = _stored_rows(source_file, strategy)
    stored_hashes = {row[4] for row in rows}
    assert len(stored_hashes) == 1
    final_hash = stored_hashes.pop()
    expected_document = (
        first_document if final_hash == first_hash else second_document
    )
    assert final_hash in {first_hash, second_hash}
    _assert_rows_match(rows, expected_document, final_hash)
    state = get_indexed_document_state(source_file, strategy)
    assert state.document_hashes == (final_hash,)
    assert state.total_chunk_count == len(expected_document.chunks)


def test_document_advisory_lock_does_not_block_an_unrelated_source(
    unique_source_file,
) -> None:
    locked_source = unique_source_file("locked-source")
    unrelated_source = unique_source_file("unrelated-source")
    strategy = ChunkingStrategy.paragraph
    document_hash = "6" * 64
    unrelated_document = _embedded_document(
        unrelated_source,
        strategy,
        ("Unrelated document chunk",),
        vector_offset=100,
    )
    locked_key = _document_lock_key(locked_source, strategy)
    unrelated_key = _document_lock_key(unrelated_source, strategy)
    assert locked_key != unrelated_key

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with open_database_connection() as lock_connection:
            lock_connection.execute(
                "SELECT pg_advisory_xact_lock(%s)", (locked_key,)
            )
            future = executor.submit(
                persist_embedded_document, unrelated_document, document_hash
            )
            try:
                result = future.result(timeout=10)
            finally:
                lock_connection.rollback()
    finally:
        executor.shutdown(wait=True)

    assert result.status is IndexingStatus.indexed
    _assert_rows_match(
        _stored_rows(unrelated_source, strategy), unrelated_document, document_hash
    )
