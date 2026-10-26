"""Failure-path tests for schema initialization and readiness checks."""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
import pytest

import rag_app.database.repository as repository
from rag_app.exceptions import DatabaseOperationError, DatabaseSchemaError


class _Result:
    def __init__(self, row: object) -> None:
        self.row = row

    def fetchone(self) -> object:
        return self.row


class _Connection:
    def __init__(self, *, execute_error: Exception | None = None) -> None:
        self.execute_error = execute_error
        self.commits = 0
        self.rollbacks = 0

    def execute(self, sql: str) -> _Result:
        if self.execute_error is not None:
            raise self.execute_error
        return _Result((1,))

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _use_connection(monkeypatch: pytest.MonkeyPatch, connection: _Connection) -> None:
    @contextmanager
    def open_connection(
        *, register_vector_types_on_open: bool = True
    ) -> Iterator[_Connection]:
        yield connection

    monkeypatch.setattr(repository, "open_database_connection", open_connection)


def test_schema_mismatch_is_actionable_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    _use_connection(monkeypatch, connection)
    monkeypatch.setattr(
        repository,
        "_inspect_schema",
        lambda connection: (_ for _ in ()).throw(
            repository._SchemaMismatchError("raw incompatible schema details")
        ),
    )

    with pytest.raises(DatabaseSchemaError) as raised:
        repository.check_database_readiness()

    assert str(raised.value) == (
        "The database schema is not ready. Run the database initialization "
        "command and verify PostgreSQL and pgvector versions."
    )
    assert "raw incompatible" not in str(raised.value)
    assert isinstance(raised.value.__cause__, repository._SchemaMismatchError)
    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_readiness_psycopg_failure_is_safe_chained_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "postgresql://user:stage7-password@localhost/private"
    original = psycopg.OperationalError(f"raw readiness SQL {secret}")
    connection = _Connection(execute_error=original)
    _use_connection(monkeypatch, connection)

    with pytest.raises(DatabaseOperationError) as raised:
        repository.check_database_readiness()

    assert raised.value.__cause__ is original
    assert str(raised.value) == "Database readiness check failed."
    assert secret not in str(raised.value)
    assert "stage7-password" not in str(raised.value)
    assert connection.rollbacks == 1


def test_schema_initialization_failure_is_safe_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    _use_connection(monkeypatch, connection)
    monkeypatch.setattr(repository, "register_vector_types", lambda connection: None)
    monkeypatch.setattr(
        repository,
        "_inspect_schema",
        lambda connection: (_ for _ in ()).throw(
            repository._SchemaMismatchError("raw schema object details")
        ),
    )

    with pytest.raises(DatabaseSchemaError) as raised:
        repository.initialize_schema()

    assert str(raised.value) == (
        "Database schema initialization failed. Verify PostgreSQL and "
        "pgvector versions."
    )
    assert "raw schema object" not in str(raised.value)
    assert connection.rollbacks == 1
    assert connection.commits == 0


def test_unexpected_readiness_error_is_rolled_back_but_not_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    _use_connection(monkeypatch, connection)
    monkeypatch.setattr(
        repository,
        "_inspect_schema",
        lambda connection: (_ for _ in ()).throw(AssertionError("programming bug")),
    )

    with pytest.raises(AssertionError, match="programming bug"):
        repository.check_database_readiness()

    assert connection.rollbacks == 1
    assert connection.commits == 0
