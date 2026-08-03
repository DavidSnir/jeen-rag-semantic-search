"""Shared Psycopg connection management and pgvector type registration."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection

from rag_app.config import load_database_settings
from rag_app.exceptions import DatabaseConnectionError, DatabaseOperationError

logger = logging.getLogger(__name__)


@contextmanager
def open_database_connection(
    *, register_vector_types_on_open: bool = True
) -> Iterator[Connection[Any]]:
    """Open and deterministically close one non-autocommit connection.

    Callers retain responsibility for committing their own transactions. Closing
    the context rolls back any transaction the caller leaves uncommitted.
    """
    settings = load_database_settings()
    try:
        connection = psycopg.connect(
            settings.postgres_url,
            connect_timeout=settings.connect_timeout_seconds,
            autocommit=False,
        )
    except psycopg.Error as error:
        raise DatabaseConnectionError(
            "PostgreSQL is unavailable. Check POSTGRES_URL and the database service."
        ) from error

    completed = False
    try:
        if register_vector_types_on_open:
            try:
                register_vector_types(connection)
                connection.commit()
            except DatabaseOperationError:
                _rollback_quietly(connection)
                raise
            except psycopg.Error as error:
                _rollback_quietly(connection)
                raise DatabaseOperationError(
                    "PostgreSQL connection setup failed. Verify pgvector is installed."
                ) from error
        yield connection
        completed = True
    finally:
        try:
            connection.close()
        except psycopg.Error as error:
            if completed:
                raise DatabaseOperationError(
                    "Database connection cleanup failed."
                ) from error
            logger.warning("Database connection cleanup failed category=close")


def register_vector_types(connection: Connection[Any]) -> None:
    """Register pgvector adapters on a connection after extension creation."""
    try:
        register_vector(connection)
    except psycopg.Error as error:
        raise DatabaseOperationError(
            "PostgreSQL vector support is unavailable. Verify pgvector is installed."
        ) from error


def _rollback_quietly(connection: Connection[Any]) -> None:
    try:
        connection.rollback()
    except psycopg.Error:
        pass
