"""Unit tests for database configuration, connection safety, and CLI errors."""

import logging
import time
from unittest.mock import MagicMock

import psycopg
import pytest
from typer.testing import CliRunner

import rag_app.cli as cli_module
import rag_app.config as config_module
import rag_app.database.connection as connection_module
from rag_app.config import DatabaseSettings, load_database_settings
from rag_app.database.connection import open_database_connection
from rag_app.exceptions import (
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a developer's local dotenv file from affecting unit tests."""
    monkeypatch.setattr(config_module, "load_dotenv", lambda: False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)


def test_missing_postgres_url(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="POSTGRES_URL") as raised:
        load_database_settings()

    assert raised.value.__cause__ is None


def test_malformed_postgres_url_is_rejected_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = "not-a-connection-string secret-password"
    monkeypatch.setenv("POSTGRES_URL", malformed)

    with pytest.raises(ConfigurationError) as raised:
        load_database_settings()

    assert isinstance(raised.value.__cause__, psycopg.ProgrammingError)
    assert malformed not in str(raised.value)
    assert "secret-password" not in str(raised.value)


def test_database_settings_repr_redacts_connection_url() -> None:
    settings = DatabaseSettings(
        postgres_url="postgresql://user:secret-password@localhost/database"
    )

    assert "secret-password" not in repr(settings)
    assert "postgresql://" not in repr(settings)


def test_psycopg_connection_failure_is_converted_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = "postgresql://user:secret-password@localhost:5432/database"
    original = psycopg.OperationalError(
        "connection rejected for password secret-password"
    )
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url=database_url),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(side_effect=original),
    )

    with pytest.raises(DatabaseConnectionError) as raised:
        with open_database_connection():
            pass

    assert raised.value.__cause__ is original
    assert "PostgreSQL is unavailable" in str(raised.value)
    assert database_url not in str(raised.value)
    assert "secret-password" not in str(raised.value)


def test_connection_registers_vector_types_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    connect = MagicMock(return_value=connection)
    register = MagicMock()
    settings = DatabaseSettings(postgres_url="postgresql://localhost/database")
    monkeypatch.setattr(connection_module, "load_database_settings", lambda: settings)
    monkeypatch.setattr(connection_module.psycopg, "connect", connect)
    monkeypatch.setattr(connection_module, "register_vector", register)

    with open_database_connection() as opened:
        assert opened is connection
        connection.close.assert_not_called()

    connect.assert_called_once_with(
        settings.postgres_url,
        connect_timeout=5,
        autocommit=False,
    )
    register.assert_called_once_with(connection)
    connection.commit.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_connection_closes_when_vector_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    original = psycopg.ProgrammingError("vector type missing; secret-password")
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url="postgresql://localhost/database"),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(return_value=connection),
    )
    monkeypatch.setattr(
        connection_module,
        "register_vector",
        MagicMock(side_effect=original),
    )

    with pytest.raises(DatabaseOperationError) as raised:
        with open_database_connection():
            pass

    assert raised.value.__cause__ is original
    assert "secret-password" not in str(raised.value)
    connection.rollback.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_connection_closes_after_caller_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = MagicMock()
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url="postgresql://localhost/database"),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(return_value=connection),
    )

    with pytest.raises(RuntimeError, match="caller failed"):
        with open_database_connection(register_vector_types_on_open=False):
            raise RuntimeError("caller failed")

    connection.close.assert_called_once_with()


def test_close_failure_without_primary_error_is_safe_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = psycopg.OperationalError("close exposed stage7-password")
    connection = MagicMock()
    connection.close.side_effect = original
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url="postgresql://localhost/database"),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(return_value=connection),
    )

    with pytest.raises(DatabaseOperationError) as raised:
        with open_database_connection(register_vector_types_on_open=False):
            pass

    assert raised.value.__cause__ is original
    assert str(raised.value) == "Database connection cleanup failed."
    assert "stage7-password" not in str(raised.value)


def test_close_failure_does_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    connection = MagicMock()
    connection.close.side_effect = psycopg.OperationalError(
        "close exposed stage7-password"
    )
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url="postgresql://localhost/database"),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(return_value=connection),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(
        RuntimeError, match="primary failure"
    ):
        with open_database_connection(register_vector_types_on_open=False):
            raise RuntimeError("primary failure")

    assert "category=close" in caplog.text
    assert "stage7-password" not in caplog.text


def test_outer_exception_context_does_not_hide_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = psycopg.OperationalError("close exposed stage7-password")
    connection = MagicMock()
    connection.close.side_effect = original
    monkeypatch.setattr(
        connection_module,
        "load_database_settings",
        lambda: DatabaseSettings(postgres_url="postgresql://localhost/database"),
    )
    monkeypatch.setattr(
        connection_module.psycopg,
        "connect",
        MagicMock(return_value=connection),
    )

    try:
        raise RuntimeError("outer handled error")
    except RuntimeError:
        with pytest.raises(DatabaseOperationError) as raised:
            with open_database_connection(register_vector_types_on_open=False):
                pass

    assert raised.value.__cause__ is original
    assert "stage7-password" not in str(raised.value)


def test_intentional_local_connection_failure_is_prompt_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "intentional-secret-password"
    database_url = f"postgresql://invalid:{password}@127.0.0.1:1/unreachable"
    monkeypatch.setenv("POSTGRES_URL", database_url)

    started_at = time.monotonic()
    with pytest.raises(DatabaseConnectionError) as raised:
        with open_database_connection(register_vector_types_on_open=False):
            pass
    elapsed = time.monotonic() - started_at

    assert elapsed < 7
    assert "PostgreSQL is unavailable" in str(raised.value)
    assert password not in str(raised.value)
    assert database_url not in str(raised.value)
    assert isinstance(raised.value.__cause__, psycopg.Error)


def test_cli_database_failure_has_no_traceback_or_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = psycopg.OperationalError("raw psycopg secret-password traceback")

    def fail_readiness() -> None:
        raise DatabaseConnectionError("Database connection failed.") from original

    monkeypatch.setattr(cli_module, "check_database_readiness", fail_readiness)

    result = CliRunner().invoke(cli_module.app, ["database-check"])

    assert result.exit_code == 1
    assert "Error: Database connection failed." in result.output
    assert "secret-password" not in result.output
    assert "Traceback" not in result.output
    assert "psycopg" not in result.output
