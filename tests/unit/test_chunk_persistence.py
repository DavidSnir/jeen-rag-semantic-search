"""Unit tests for Stage 5 repository validation and batch persistence."""

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import psycopg
import pytest
from pgvector import Vector

import rag_app.database.repository as repository
from rag_app.database.repository import (
    PersistenceResult,
    get_indexed_document_state,
    persist_embedded_document,
)
from rag_app.documents import (
    Chunk,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    IndexingStatus,
)
from rag_app.exceptions import PersistenceError, PersistenceVerificationError

HASH = "a" * 64
VECTOR = (1.0,) + (0.0,) * 767


def _document(
    *,
    source_file: str = "report.pdf",
    source_type: str = "PDF",
    strategy: ChunkingStrategy = ChunkingStrategy.fixed,
) -> EmbeddedDocument:
    page_number = 1 if source_type == "PDF" else None
    chunks = (
        EmbeddedChunk(Chunk("First", 0, page_number), VECTOR),
        EmbeddedChunk(Chunk("Second", 1, page_number), VECTOR),
    )
    return EmbeddedDocument(source_file, source_type, strategy, chunks)


class _CopyRecorder:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.rows: list[tuple[object, ...]] = []
        self.error = error

    def __enter__(self) -> "_CopyRecorder":
        if self.error:
            raise self.error
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def write_row(self, row: tuple[object, ...]) -> None:
        self.rows.append(row)


class _CopyCursor:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.copy_sql: str | None = None
        self.copy_recorder = _CopyRecorder(error=error)
        self.executemany_called = False

    def copy(self, sql: str) -> _CopyRecorder:
        self.copy_sql = sql
        return self.copy_recorder

    def executemany(self, sql: str, rows: object) -> None:
        self.executemany_called = True


class _FallbackCursor:
    copy = None

    def __init__(self) -> None:
        self.call: tuple[str, object] | None = None

    def executemany(self, sql: str, rows: object) -> None:
        self.call = (sql, rows)


def test_copy_is_default_and_receives_every_row_in_order() -> None:
    document = _document()
    rows = repository._build_persistence_rows(document, HASH)
    cursor = _CopyCursor()

    repository._insert_rows(cursor, rows)

    assert "COPY public.chunks" in cursor.copy_sql
    assert cursor.copy_recorder.rows == list(rows)
    assert cursor.executemany_called is False


def test_genuine_copy_unavailability_uses_executemany() -> None:
    rows = repository._build_persistence_rows(_document(), HASH)
    cursor = _FallbackCursor()

    repository._insert_rows(cursor, rows)

    assert cursor.call is not None
    assert "INSERT INTO public.chunks" in cursor.call[0]
    assert cursor.call[1] is rows


def test_copy_database_failure_never_invokes_fallback() -> None:
    rows = repository._build_persistence_rows(_document(), HASH)
    cursor = _CopyCursor(error=psycopg.IntegrityError("constraint failed"))

    with pytest.raises(psycopg.IntegrityError):
        repository._insert_rows(cursor, rows)

    assert cursor.executemany_called is False


def test_row_mapping_preserves_metadata_and_omits_created_at() -> None:
    document = _document(source_file="source.docx", source_type="DOCX")

    rows = repository._build_persistence_rows(document, HASH)

    assert len(rows) == 2
    assert rows[0][0] == "First"
    assert isinstance(rows[0][1], Vector)
    assert rows[0][2:] == (
        "source.docx",
        HASH,
        "DOCX",
        0,
        "fixed",
        None,
    )
    assert len(rows[0]) == 8


@pytest.mark.parametrize(
    ("document", "document_hash"),
    [
        (_document(source_file="../report.pdf"), HASH),
        (_document(), 123),
        (_document(), "A" * 64),
        (_document(), "short"),
        (replace(_document(), chunks=()), HASH),
        (
            replace(
                _document(),
                chunks=(EmbeddedChunk(Chunk("First", 1, 1), VECTOR),),
            ),
            HASH,
        ),
        (
            replace(
                _document(),
                chunks=(
                    EmbeddedChunk(Chunk("First", 0, 1), (0.0,) * 768),
                ),
            ),
            HASH,
        ),
    ],
)
def test_invalid_local_state_fails_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    document: EmbeddedDocument,
    document_hash: object,
) -> None:
    opened = False

    @contextmanager
    def open_connection() -> Iterator[object]:
        nonlocal opened
        opened = True
        yield object()

    monkeypatch.setattr(repository, "open_database_connection", open_connection)

    with pytest.raises(PersistenceError):
        persist_embedded_document(document, document_hash)  # type: ignore[arg-type]

    assert opened is False


class _StateCursor:
    def __init__(self, rows: list[tuple[str, int]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, object]] = []

    def __enter__(self) -> "_StateCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> "_StateCursor":
        self.calls.append((sql, params))
        return self

    def fetchall(self) -> list[tuple[str, int]]:
        return self.rows


class _StateConnection:
    def __init__(self, cursor: _StateCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _StateCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _use_connection(
    monkeypatch: pytest.MonkeyPatch, connection: _StateConnection
) -> None:
    @contextmanager
    def open_connection() -> Iterator[_StateConnection]:
        yield connection

    monkeypatch.setattr(repository, "open_database_connection", open_connection)


@pytest.mark.parametrize(
    ("rows", "hashes", "count"),
    [
        ([], (), 0),
        ([(HASH, 3)], (HASH,), 3),
        ([("b" * 64, 2)], ("b" * 64,), 2),
        ([(HASH, 3), ("b" * 64, 2)], (HASH, "b" * 64), 5),
    ],
)
def test_indexed_state_is_immutable_and_queries_are_parameterized(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[str, int]],
    hashes: tuple[str, ...],
    count: int,
) -> None:
    cursor = _StateCursor(rows)
    connection = _StateConnection(cursor)
    _use_connection(monkeypatch, connection)

    state = get_indexed_document_state("report.pdf", ChunkingStrategy.fixed)

    assert state.document_hashes == hashes
    assert state.total_chunk_count == count
    assert cursor.calls[0][1] == ("report.pdf", "fixed")
    assert connection.rollbacks == 1


class _WriteCursor(_StateCursor):
    def __init__(
        self,
        state_rows: list[tuple[str, int]],
        verification_row: tuple[object, ...],
        *,
        copy_error: Exception | None = None,
    ) -> None:
        super().__init__(state_rows)
        self.verification_row = verification_row
        self.copy_recorder = _CopyRecorder(error=copy_error)
        self.last_sql = ""

    def execute(self, sql: str, params: object = None) -> "_WriteCursor":
        self.last_sql = sql
        self.calls.append((sql, params))
        return self

    def fetchall(self) -> list[tuple[str, int]]:
        return self.rows

    def fetchone(self) -> tuple[object, ...]:
        return self.verification_row

    def copy(self, sql: str) -> _CopyRecorder:
        self.calls.append((sql, None))
        return self.copy_recorder


def test_replacement_delete_is_scoped_and_committed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _WriteCursor(
        [("b" * 64, 4)],
        ([HASH], 2, 0, 1, 2),
    )
    connection = _StateConnection(cursor)
    _use_connection(monkeypatch, connection)

    result = persist_embedded_document(_document(), HASH)

    assert result == PersistenceResult(IndexingStatus.replaced, 2)
    delete_calls = [call for call in cursor.calls if "DELETE FROM" in call[0]]
    assert len(delete_calls) == 1
    assert delete_calls[0][1] == ("report.pdf", "fixed")
    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_transaction_duplicate_commits_without_delete_or_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _WriteCursor([(HASH, 2)], (None,))
    connection = _StateConnection(cursor)
    _use_connection(monkeypatch, connection)

    result = persist_embedded_document(_document(), HASH)

    assert result == PersistenceResult(IndexingStatus.skipped, 2)
    assert not any("DELETE FROM" in sql for sql, _ in cursor.calls)
    assert not any("COPY public.chunks" in sql for sql, _ in cursor.calls)
    assert connection.commits == 1


def test_verification_failure_rolls_back_complete_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _WriteCursor([], ([HASH], 1, 0, 0, 1))
    connection = _StateConnection(cursor)
    _use_connection(monkeypatch, connection)

    with pytest.raises(PersistenceVerificationError):
        persist_embedded_document(_document(), HASH)

    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_copy_failure_rolls_back_and_is_safely_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = psycopg.IntegrityError("raw content and vector details")
    cursor = _WriteCursor(
        [("b" * 64, 2)],
        ([HASH], 2, 0, 1, 2),
        copy_error=original,
    )
    connection = _StateConnection(cursor)
    _use_connection(monkeypatch, connection)

    with pytest.raises(PersistenceError) as raised:
        persist_embedded_document(_document(), HASH)

    assert raised.value.__cause__ is original
    assert "raw content" not in str(raised.value)
    assert connection.commits == 0
    assert connection.rollbacks == 1
