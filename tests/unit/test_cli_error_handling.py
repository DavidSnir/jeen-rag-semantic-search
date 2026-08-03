"""Cross-command tests for the public CLI error contract."""

import subprocess
import sys

import psycopg
import pytest
from click import unstyle
from typer.testing import CliRunner

import rag_app.cli as cli_module
from rag_app.database.repository import DatabaseStatus
from rag_app.exceptions import (
    DatabaseConnectionError,
    DatabaseSchemaError,
    GeminiRequestError,
)


RUNNER = CliRunner()
STATUS = DatabaseStatus("17.5", "0.8.2", "vector(768)")


@pytest.mark.parametrize(
    ("command", "attribute", "expected"),
    [
        (
            "database-init",
            "initialize_schema",
            "Database schema initialized (PostgreSQL 17.5, pgvector 0.8.2, "
            "vector(768)).\n",
        ),
        (
            "database-check",
            "check_database_readiness",
            "Database is ready (PostgreSQL 17.5, pgvector 0.8.2, vector(768)).\n",
        ),
    ],
)
def test_database_commands_succeed(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    attribute: str,
    expected: str,
) -> None:
    monkeypatch.setattr(cli_module, attribute, lambda: STATUS)

    result = RUNNER.invoke(cli_module.app, [command])

    assert result.exit_code == 0
    assert result.output == expected


@pytest.mark.parametrize(
    ("command", "attribute", "error_type", "message"),
    [
        (
            "database-init",
            "initialize_schema",
            DatabaseSchemaError,
            "Database schema initialization failed.",
        ),
        (
            "database-check",
            "check_database_readiness",
            DatabaseConnectionError,
            "PostgreSQL is unavailable. Check POSTGRES_URL and the database service.",
        ),
    ],
)
def test_database_command_failures_are_safe(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    attribute: str,
    error_type: type[Exception],
    message: str,
) -> None:
    secret = "postgresql://user:stage7-password@localhost/private"
    original = psycopg.OperationalError(f"raw database failure {secret}")

    def fail() -> None:
        raise error_type(message) from original

    monkeypatch.setattr(cli_module, attribute, fail)
    result = RUNNER.invoke(cli_module.app, [command])

    assert result.exit_code == 1
    assert result.output == f"Error: {message}\n"
    assert secret not in result.output
    assert "stage7-password" not in result.output
    assert "Traceback" not in result.output


def test_reset_requires_confirmation_and_remains_unavailable() -> None:
    unconfirmed = RUNNER.invoke(cli_module.app, ["reset"])
    confirmed = RUNNER.invoke(cli_module.app, ["reset", "--yes"])

    assert unconfirmed.exit_code == 2
    assert "Pass --yes to confirm reset" in unstyle(unconfirmed.output)
    assert confirmed.exit_code == 1
    assert confirmed.output == "Error: Index reset is not implemented.\n"
    assert "Traceback" not in confirmed.output


def test_chained_provider_failure_does_not_expose_sensitive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "stage7-synthetic-gemini-key"
    query = "stage7 sensitive query text"
    chunk = "stage7 sensitive chunk content"
    database_url = "postgresql://user:stage7-password@localhost/private"
    original = RuntimeError(f"{api_key} {query} {chunk} {database_url}")

    def fail(query: str, strategy: str, top_k: int) -> None:
        raise GeminiRequestError(
            "Gemini authentication failed. Check GEMINI_API_KEY."
        ) from original

    monkeypatch.setattr(cli_module, "search_documents", fail)
    result = RUNNER.invoke(
        cli_module.app,
        ["search", "--query", query, "--strategy", "fixed"],
    )

    assert result.exit_code == 1
    assert result.output == (
        "Error: Gemini authentication failed. Check GEMINI_API_KEY.\n"
    )
    for sensitive_value in (api_key, query, chunk, database_url, "stage7-password"):
        assert sensitive_value not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("error", [AssertionError("bug"), TypeError("bug")])
def test_unexpected_programming_errors_are_not_converted(error: Exception) -> None:
    def fail() -> None:
        raise error

    with pytest.raises(type(error), match="bug"):
        cli_module._run(fail)


def test_application_logging_is_quiet_until_explicitly_configured() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, rag_app; "
            "logging.getLogger('rag_app.synthetic').warning('unexpected warning')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
