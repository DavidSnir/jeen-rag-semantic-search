"""Database schema initialization and infrastructure inspection boundary."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import Connection

from rag_app.database.connection import (
    open_database_connection,
    register_vector_types,
)
from rag_app.exceptions import (
    DatabaseOperationError,
    DatabaseSchemaError,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

EXPECTED_COLUMNS = {
    "id": ("bigint", True),
    "content": ("text", True),
    "embedding": ("vector(768)", True),
    "source_file": ("text", True),
    "document_hash": ("text", True),
    "source_type": ("text", True),
    "chunk_index": ("integer", True),
    "chunking_strategy": ("text", True),
    "page_number": ("integer", False),
    "created_at": ("timestamp with time zone", True),
}

EXPECTED_CONSTRAINTS = {
    "chunks_pkey": "p",
    "chunks_content_nonempty_check": "c",
    "chunks_document_hash_sha256_check": "c",
    "chunks_source_type_check": "c",
    "chunks_chunk_index_nonnegative_check": "c",
    "chunks_chunking_strategy_check": "c",
    "chunks_page_number_positive_check": "c",
    "chunks_docx_page_number_null_check": "c",
    "chunks_document_version_strategy_chunk_key": "u",
}

RELATIONAL_INDEXES = {
    "idx_chunks_source_file_chunking_strategy": (
        "source_file",
        "chunking_strategy",
    ),
    "idx_chunks_document_hash_chunking_strategy": (
        "document_hash",
        "chunking_strategy",
    ),
}
HNSW_INDEX = "idx_chunks_embedding_hnsw_cosine"
EXPECTED_POSTGRESQL_MAJOR = 17
EXPECTED_PGVECTOR_VERSION = "0.8.2"

EXPECTED_CONSTRAINT_DEFINITIONS = {
    "chunks_pkey": "PRIMARY KEY (id)",
    "chunks_content_nonempty_check": "CHECK ((char_length(content) > 0))",
    "chunks_document_hash_sha256_check": (
        "CHECK ((document_hash ~ '^[0-9A-Fa-f]{64}$'::text))"
    ),
    "chunks_source_type_check": (
        "CHECK ((source_type = ANY (ARRAY['PDF'::text, 'DOCX'::text])))"
    ),
    "chunks_chunk_index_nonnegative_check": "CHECK ((chunk_index >= 0))",
    "chunks_chunking_strategy_check": (
        "CHECK ((chunking_strategy = ANY (ARRAY['fixed'::text, "
        "'sentence'::text, 'paragraph'::text])))"
    ),
    "chunks_page_number_positive_check": (
        "CHECK (((page_number IS NULL) OR (page_number > 0)))"
    ),
    "chunks_docx_page_number_null_check": (
        "CHECK (((source_type <> 'DOCX'::text) OR (page_number IS NULL)))"
    ),
    "chunks_document_version_strategy_chunk_key": (
        "UNIQUE (source_file, document_hash, chunking_strategy, chunk_index)"
    ),
}


@dataclass(frozen=True)
class DatabaseStatus:
    """Non-sensitive details returned by a successful readiness check."""

    postgresql_version: str
    pgvector_version: str
    embedding_type: str


class _SchemaMismatchError(Exception):
    """Internal schema mismatch detail safe to include in an outer message."""


def initialize_schema() -> DatabaseStatus:
    """Apply the canonical schema atomically and validate the result."""
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise DatabaseSchemaError(
            "Database schema initialization failed because the canonical schema "
            "could not be read."
        ) from error

    with open_database_connection(register_vector_types_on_open=False) as connection:
        try:
            connection.execute(schema)
            register_vector_types(connection)
            status = _inspect_schema(connection)
            connection.commit()
            return status
        except _SchemaMismatchError as error:
            _rollback_quietly(connection)
            raise DatabaseSchemaError(
                f"Database schema initialization failed: {error}"
            ) from error
        except (psycopg.Error, DatabaseOperationError) as error:
            _rollback_quietly(connection)
            raise DatabaseSchemaError(
                "Database schema initialization failed."
            ) from error


def check_database_readiness() -> DatabaseStatus:
    """Verify connectivity and every required Stage 1 database object."""
    with open_database_connection(register_vector_types_on_open=False) as connection:
        try:
            if connection.execute("SELECT 1").fetchone() != (1,):
                raise _SchemaMismatchError("PostgreSQL did not answer a basic query")
            status = _inspect_schema(connection)
            register_vector_types(connection)
            connection.rollback()
            return status
        except _SchemaMismatchError as error:
            _rollback_quietly(connection)
            raise DatabaseSchemaError(
                f"Database schema validation failed: {error}"
            ) from error
        except DatabaseOperationError:
            _rollback_quietly(connection)
            raise
        except psycopg.Error as error:
            _rollback_quietly(connection)
            raise DatabaseOperationError(
                "Database readiness check failed."
            ) from error


def _inspect_schema(connection: Connection[Any]) -> DatabaseStatus:
    postgresql_version, server_version_num = connection.execute(
        """
        SELECT current_setting('server_version'),
               current_setting('server_version_num')::integer
        """
    ).fetchone()
    if server_version_num // 10000 != EXPECTED_POSTGRESQL_MAJOR:
        raise _SchemaMismatchError(
            f"PostgreSQL {EXPECTED_POSTGRESQL_MAJOR} is required"
        )
    extension_row = connection.execute(
        "SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if extension_row is None:
        raise _SchemaMismatchError("the vector extension is not installed")
    if extension_row[0] != EXPECTED_PGVECTOR_VERSION:
        raise _SchemaMismatchError(
            f"pgvector {EXPECTED_PGVECTOR_VERSION} is required"
        )

    relation_row = connection.execute(
        """
        SELECT relation.relkind
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relname = 'chunks'
        """
    ).fetchone()
    if relation_row != ("r",):
        raise _SchemaMismatchError("public.chunks is not an ordinary table")

    column_rows = connection.execute(
        """
        SELECT
            attribute.attname,
            pg_catalog.format_type(attribute.atttypid, attribute.atttypmod),
            attribute.attnotnull,
            attribute.attidentity,
            pg_catalog.pg_get_expr(default_value.adbin, default_value.adrelid)
        FROM pg_catalog.pg_attribute AS attribute
        JOIN pg_catalog.pg_class AS relation
          ON relation.oid = attribute.attrelid
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        LEFT JOIN pg_catalog.pg_attrdef AS default_value
          ON default_value.adrelid = relation.oid
         AND default_value.adnum = attribute.attnum
        WHERE namespace.nspname = 'public'
          AND relation.relname = 'chunks'
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
        ORDER BY attribute.attnum
        """
    ).fetchall()
    columns = {
        name: (data_type, not_null, identity, default)
        for name, data_type, not_null, identity, default in column_rows
    }
    if set(columns) != set(EXPECTED_COLUMNS):
        raise _SchemaMismatchError("public.chunks has unexpected columns")
    for name, (expected_type, expected_not_null) in EXPECTED_COLUMNS.items():
        data_type, not_null, _, _ = columns[name]
        if data_type != expected_type or not_null is not expected_not_null:
            raise _SchemaMismatchError(
                f"public.chunks column {name} has an incompatible definition"
            )
    if columns["id"][2] != "a":
        raise _SchemaMismatchError("public.chunks.id is not an always identity column")
    created_at_default = columns["created_at"][3]
    if created_at_default is None or "CURRENT_TIMESTAMP" not in created_at_default:
        raise _SchemaMismatchError(
            "public.chunks.created_at has no PostgreSQL timestamp default"
        )

    constraint_rows = connection.execute(
        """
        SELECT constraint_record.conname, constraint_record.contype,
               constraint_record.convalidated,
               pg_catalog.pg_get_constraintdef(constraint_record.oid)
        FROM pg_catalog.pg_constraint AS constraint_record
        JOIN pg_catalog.pg_class AS relation
          ON relation.oid = constraint_record.conrelid
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relname = 'chunks'
        """
    ).fetchall()
    constraints = {
        name: (constraint_type, is_validated, definition)
        for name, constraint_type, is_validated, definition in constraint_rows
    }
    for name, constraint_type in EXPECTED_CONSTRAINTS.items():
        if (
            name not in constraints
            or constraints[name][0] != constraint_type
            or constraints[name][1] is not True
        ):
            raise _SchemaMismatchError(f"required constraint {name} is missing")
        actual_definition = _normalize_sql(constraints[name][2])
        expected_definition = _normalize_sql(EXPECTED_CONSTRAINT_DEFINITIONS[name])
        if actual_definition != expected_definition:
            raise _SchemaMismatchError(f"required constraint {name} is incompatible")

    index_rows = connection.execute(
        """
        SELECT index_relation.relname,
               access_method.amname,
               pg_catalog.pg_get_indexdef(index_record.indexrelid),
               index_record.indisvalid,
               index_record.indisready,
               index_record.indpred IS NULL,
               index_record.indexprs IS NULL,
               index_relation.reloptions
        FROM pg_catalog.pg_index AS index_record
        JOIN pg_catalog.pg_class AS table_relation
          ON table_relation.oid = index_record.indrelid
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = table_relation.relnamespace
        JOIN pg_catalog.pg_class AS index_relation
          ON index_relation.oid = index_record.indexrelid
        JOIN pg_catalog.pg_am AS access_method
          ON access_method.oid = index_relation.relam
        WHERE namespace.nspname = 'public'
          AND table_relation.relname = 'chunks'
        """
    ).fetchall()
    indexes = {
        name: (
            access_method,
            definition,
            is_valid,
            is_ready,
            is_not_partial,
            has_no_expressions,
            options,
        )
        for (
            name,
            access_method,
            definition,
            is_valid,
            is_ready,
            is_not_partial,
            has_no_expressions,
            options,
        ) in index_rows
    }
    for name, columns_in_index in RELATIONAL_INDEXES.items():
        if name not in indexes:
            raise _SchemaMismatchError(f"required index {name} is missing")
        (
            access_method,
            raw_definition,
            is_valid,
            is_ready,
            is_not_partial,
            has_no_expressions,
            options,
        ) = indexes[name]
        if (
            access_method != "btree"
            or not is_valid
            or not is_ready
            or not is_not_partial
            or not has_no_expressions
            or options is not None
        ):
            raise _SchemaMismatchError(f"required index {name} is incompatible")
        definition = _normalize_sql(raw_definition)
        expected_columns = f"({','.join(columns_in_index)})"
        if expected_columns not in definition:
            raise _SchemaMismatchError(f"required index {name} is incompatible")

    if HNSW_INDEX not in indexes:
        raise _SchemaMismatchError(f"required index {HNSW_INDEX} is missing")
    (
        access_method,
        raw_definition,
        is_valid,
        is_ready,
        is_not_partial,
        has_no_expressions,
        options,
    ) = indexes[HNSW_INDEX]
    if (
        access_method != "hnsw"
        or not is_valid
        or not is_ready
        or not is_not_partial
        or not has_no_expressions
        or options is not None
    ):
        raise _SchemaMismatchError("the HNSW cosine index is incompatible")
    hnsw_definition = _normalize_sql(raw_definition)
    if "usinghnsw(embeddingvector_cosine_ops)" not in hnsw_definition:
        raise _SchemaMismatchError("the HNSW cosine index is incompatible")

    return DatabaseStatus(
        postgresql_version=postgresql_version,
        pgvector_version=extension_row[0],
        embedding_type=columns["embedding"][0],
    )


def _normalize_sql(value: str) -> str:
    return re.sub(r"[\s\"]+", "", value).lower()


def _rollback_quietly(connection: Connection[Any]) -> None:
    try:
        connection.rollback()
    except psycopg.Error:
        pass
