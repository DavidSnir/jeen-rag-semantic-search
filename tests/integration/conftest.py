"""Shared safety and isolation fixtures for PostgreSQL integration tests."""

import os
from collections.abc import Iterator
from typing import Protocol

import pytest
from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict

from rag_app.database.connection import open_database_connection
from rag_app.database.repository import initialize_schema

_POSTGRES_URL_VARIABLE = "POSTGRES_URL"


class SourceFileFactory(Protocol):
    """Build deterministic, distinct source filenames within one test."""

    def __call__(self, label: str, suffix: str = ".pdf") -> str: ...


def _required_test_database_url() -> str:
    """Return the explicitly exported URL after enforcing test-database safety."""
    postgres_url = os.environ.get(_POSTGRES_URL_VARIABLE, "").strip()
    if not postgres_url:
        pytest.skip(
            "POSTGRES_URL is required for PostgreSQL integration tests",
            allow_module_level=True,
        )

    try:
        database_name = conninfo_to_dict(postgres_url).get("dbname", "")
    except ProgrammingError:
        pytest.fail("POSTGRES_URL is not a valid PostgreSQL connection string.")
    if not database_name.endswith("_test"):
        pytest.fail("Integration tests require a database name ending in '_test'.")
    return postgres_url


def _truncate_chunks() -> None:
    """Remove all integration-test rows and reset generated identifiers."""
    with open_database_connection() as connection:
        connection.execute("TRUNCATE TABLE public.chunks RESTART IDENTITY")
        connection.commit()


@pytest.fixture(scope="session", autouse=True)
def initialized_integration_database() -> Iterator[None]:
    """Validate the target and initialize its canonical schema once per run."""
    _required_test_database_url()
    initialize_schema()
    yield


@pytest.fixture(autouse=True)
def isolated_chunks(initialized_integration_database: None) -> Iterator[None]:
    """Give each integration test an empty chunks table and clean up afterward."""
    _truncate_chunks()
    try:
        yield
    finally:
        _truncate_chunks()


@pytest.fixture
def unique_source_file() -> SourceFileFactory:
    """Return a deterministic source-filename factory scoped to one clean test."""
    sequence = 0

    def create(label: str, suffix: str = ".pdf") -> str:
        nonlocal sequence
        sequence += 1
        return f"integration-{label}-{sequence}{suffix}"

    return create
