"""Unit tests for the read-only semantic-search repository."""

from contextlib import contextmanager
from typing import Iterator

import psycopg
import pytest
from pgvector import Vector

import rag_app.database.search as search_repository
from rag_app.database.search import SemanticSearchRow, search_similar_chunks
from rag_app.documents import ChunkingStrategy
from rag_app.exceptions import SemanticSearchError

QUERY_VECTOR = (1.0,) + (0.0,) * 767
ALTERNATE_VECTOR = (0.6, 0.8) + (0.0,) * 766

FIXED_ROW = (
    "Closest fixed chunk",
    "fixed.pdf",
    "PDF",
    0,
    "fixed",
    1,
    0.05,
)


class _SearchCursor:
    def __init__(
        self,
        rows: list[object],
        *,
        execute_error: Exception | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.rows = rows
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.entries = 0
        self.exits = 0

    def __enter__(self) -> "_SearchCursor":
        self.entries += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.exits += 1
        return None

    def execute(
        self, sql: str, params: tuple[object, ...] | None = None
    ) -> "_SearchCursor":
        self.calls.append((sql, params))
        if self.execute_error is not None:
            raise self.execute_error
        return self

    def fetchall(self) -> list[object]:
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.rows


class _SearchConnection:
    def __init__(self, cursor: _SearchCursor) -> None:
        self._cursor = cursor
        self.cursor_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0

    def cursor(self) -> _SearchCursor:
        self.cursor_calls += 1
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closes += 1


class _ConnectionFactory:
    def __init__(self, connection: _SearchConnection) -> None:
        self.connection = connection
        self.entries = 0
        self.exits = 0

    @contextmanager
    def open(self) -> Iterator[_SearchConnection]:
        self.entries += 1
        try:
            yield self.connection
        finally:
            self.exits += 1
            self.connection.close()


def _use_connection(
    monkeypatch: pytest.MonkeyPatch,
    cursor: _SearchCursor,
) -> tuple[_SearchConnection, _ConnectionFactory]:
    connection = _SearchConnection(cursor)
    factory = _ConnectionFactory(connection)
    monkeypatch.setattr(search_repository, "open_database_connection", factory.open)
    return connection, factory


def test_query_has_static_parameterized_cosine_search_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = (
        "Requested paragraph chunk",
        "paragraph.pdf",
        "PDF",
        4,
        "paragraph",
        2,
        0.25,
    )
    cursor = _SearchCursor([row])
    _use_connection(monkeypatch, cursor)

    results = search_similar_chunks(
        ALTERNATE_VECTOR,
        ChunkingStrategy.paragraph,
        37,
    )

    assert len(cursor.calls) == 2
    setup_sql, setup_params = cursor.calls[0]
    assert " ".join(setup_sql.split()) == (
        "SET LOCAL hnsw.iterative_scan = strict_order"
    )
    assert setup_params is None
    sql, params = cursor.calls[1]
    normalized_sql = " ".join(sql.split())
    projection = normalized_sql.removeprefix("SELECT ").split(
        " FROM public.chunks", 1
    )[0]
    expressions = projection.split(", ")

    assert expressions == [
        "content",
        "source_file",
        "source_type",
        "chunk_index",
        "chunking_strategy",
        "page_number",
        "embedding <=> %s AS distance",
    ]
    assert set(expressions).isdisjoint(
        {"embedding", "document_hash", "id", "created_at"}
    )
    assert "FROM public.chunks" in normalized_sql
    assert "WHERE chunking_strategy = %s" in normalized_sql
    assert "ORDER BY embedding <=> %s" in normalized_sql
    assert "ORDER BY distance" not in normalized_sql
    assert " DESC" not in normalized_sql
    assert normalized_sql.endswith("LIMIT %s")
    assert normalized_sql.count("<=>") == 2
    assert normalized_sql.count("%s") == 4

    assert len(params) == 4
    assert isinstance(params[0], Vector)
    assert params[0] is params[2]
    assert params[0].to_list() == pytest.approx(ALTERNATE_VECTOR)
    assert params[1:] == ("paragraph", params[2], 37)
    assert "paragraph" not in sql
    assert "37" not in sql
    assert "0.6" not in sql
    assert "0.8" not in sql
    assert results == (SemanticSearchRow(*row),)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [FIXED_ROW],
        [
            FIXED_ROW,
            (
                "Middle fixed chunk",
                "fixed.docx",
                "DOCX",
                3,
                "fixed",
                None,
                0.4,
            ),
            (
                "Farthest fixed chunk",
                "other.pdf",
                "PDF",
                8,
                "fixed",
                9,
                1.2,
            ),
        ],
    ],
    ids=["empty", "one", "several"],
)
def test_returns_zero_to_fewer_than_top_k_rows_in_database_order(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[object],
) -> None:
    cursor = _SearchCursor(rows)
    _use_connection(monkeypatch, cursor)

    results = search_similar_chunks(QUERY_VECTOR, ChunkingStrategy.fixed, 5)

    assert len(results) == len(rows) < 5
    assert [result.content for result in results] == [row[0] for row in rows]  # type: ignore[index]
    assert [result.distance for result in results] == [row[6] for row in rows]  # type: ignore[index]
    assert cursor.calls[1][1][3] == 5


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_requests_and_returns_only_the_requested_strategy_rows(
    monkeypatch: pytest.MonkeyPatch,
    strategy: ChunkingStrategy,
) -> None:
    row = (
        f"{strategy.value} chunk",
        f"{strategy.value}.pdf",
        "PDF",
        0,
        strategy.value,
        1,
        0.1,
    )
    cursor = _SearchCursor([row])
    _use_connection(monkeypatch, cursor)

    results = search_similar_chunks(QUERY_VECTOR, strategy, 3)

    assert cursor.calls[1][1][1] == strategy.value
    assert [result.chunking_strategy for result in results] == [strategy.value]


def test_success_rolls_back_without_commit_and_closes_via_context_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _SearchCursor([FIXED_ROW])
    connection, factory = _use_connection(monkeypatch, cursor)

    search_similar_chunks(QUERY_VECTOR, ChunkingStrategy.fixed, 1)

    assert connection.cursor_calls == 1
    assert cursor.entries == cursor.exits == 1
    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert factory.entries == factory.exits == 1
    assert connection.closes == 1


@pytest.mark.parametrize("failure_stage", ["execute", "fetchall"])
def test_psycopg_query_failures_are_safely_converted_and_chained(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    original = psycopg.DataError(
        f"raw {failure_stage} database details with secret-marker"
    )
    cursor = _SearchCursor(
        [],
        execute_error=original if failure_stage == "execute" else None,
        fetch_error=original if failure_stage == "fetchall" else None,
    )
    connection, factory = _use_connection(monkeypatch, cursor)

    with pytest.raises(SemanticSearchError) as raised:
        search_similar_chunks(QUERY_VECTOR, ChunkingStrategy.fixed, 2)

    assert str(raised.value) == "Semantic search query failed."
    assert raised.value.__cause__ is original
    assert "secret-marker" not in str(raised.value)
    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert cursor.exits == 1
    assert factory.exits == 1
    assert connection.closes == 1


@pytest.mark.parametrize(
    "malformed_row",
    [
        ("only", "six", "database", "values", "are", "present"),
        ("eight", "database", "values", "are", "not", "valid", 0.2, "extra"),
        123,
    ],
    ids=["missing-value", "extra-value", "not-iterable"],
)
def test_malformed_database_rows_fail_safely_and_roll_back(
    monkeypatch: pytest.MonkeyPatch,
    malformed_row: object,
) -> None:
    cursor = _SearchCursor([malformed_row])
    connection, factory = _use_connection(monkeypatch, cursor)

    with pytest.raises(SemanticSearchError) as raised:
        search_similar_chunks(QUERY_VECTOR, ChunkingStrategy.fixed, 1)

    assert str(raised.value) == "Semantic search returned an invalid database row."
    assert isinstance(raised.value.__cause__, (TypeError, ValueError))
    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert factory.exits == 1
    assert connection.closes == 1


@pytest.mark.parametrize(
    ("query_vector", "strategy", "top_k"),
    [
        ((1.0,), ChunkingStrategy.fixed, 1),
        ([1.0] + [0.0] * 767, ChunkingStrategy.fixed, 1),
        ((1,) + (0.0,) * 767, ChunkingStrategy.fixed, 1),
        ((float("nan"),) + (0.0,) * 767, ChunkingStrategy.fixed, 1),
        ((0.5,) + (0.0,) * 767, ChunkingStrategy.fixed, 1),
        (QUERY_VECTOR, "fixed", 1),
        (QUERY_VECTOR, ChunkingStrategy.fixed, 0),
        (QUERY_VECTOR, ChunkingStrategy.fixed, True),
        (QUERY_VECTOR, ChunkingStrategy.fixed, 1.5),
    ],
    ids=[
        "wrong-dimension",
        "non-tuple-vector",
        "non-float-coordinate",
        "non-finite-coordinate",
        "unnormalized-vector",
        "non-canonical-strategy",
        "non-positive-top-k",
        "boolean-top-k",
        "non-integer-top-k",
    ],
)
def test_invalid_input_fails_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    query_vector: object,
    strategy: object,
    top_k: object,
) -> None:
    opened = False

    @contextmanager
    def open_connection() -> Iterator[object]:
        nonlocal opened
        opened = True
        yield object()

    monkeypatch.setattr(
        search_repository,
        "open_database_connection",
        open_connection,
    )

    with pytest.raises(SemanticSearchError):
        search_similar_chunks(query_vector, strategy, top_k)  # type: ignore[arg-type]

    assert opened is False
