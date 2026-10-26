"""Integration verification for PostgreSQL 17 and pgvector infrastructure."""

import psycopg
import pytest
from pgvector import Vector

from rag_app.database.connection import open_database_connection
from rag_app.database.repository import (
    check_database_readiness,
    initialize_schema,
)

EXPECTED_COLUMNS = {
    "id": ("bigint", True, "a"),
    "content": ("text", True, ""),
    "embedding": ("vector(768)", True, ""),
    "source_file": ("text", True, ""),
    "document_hash": ("text", True, ""),
    "source_type": ("text", True, ""),
    "chunk_index": ("integer", True, ""),
    "chunking_strategy": ("text", True, ""),
    "page_number": ("integer", False, ""),
    "created_at": ("timestamp with time zone", True, ""),
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


def test_real_database_connection_is_closed_after_context_exit() -> None:
    with open_database_connection() as connection:
        assert connection.closed is False

    assert connection.closed is True


def test_schema_initialization_is_idempotent_and_ready() -> None:
    first_status = initialize_schema()
    second_status = initialize_schema()
    readiness_status = check_database_readiness()

    assert first_status == second_status == readiness_status
    assert first_status.postgresql_version.split(".", maxsplit=1)[0] == "17"
    assert first_status.pgvector_version == "0.8.2"
    assert first_status.embedding_type == "vector(768)"


def test_extension_table_and_complete_column_structure() -> None:
    with open_database_connection() as connection:
        extension = connection.execute(
            "SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'"
        ).fetchone()
        relation_kind = connection.execute(
            """
            SELECT relation.relkind
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'chunks'
            """
        ).fetchone()
        rows = connection.execute(
            """
            SELECT attribute.attname,
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
        connection.rollback()

    columns = {
        name: (data_type, not_null, identity)
        for name, data_type, not_null, identity, _ in rows
    }
    defaults = {name: default for name, *_, default in rows}
    assert extension == ("0.8.2",)
    assert relation_kind == ("r",)
    assert columns == EXPECTED_COLUMNS
    assert defaults["created_at"] is not None
    assert "CURRENT_TIMESTAMP" in defaults["created_at"]


def test_required_constraints_and_indexes_exist() -> None:
    with open_database_connection() as connection:
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
        connection.rollback()

    constraints = {
        name: (constraint_type, is_validated, definition)
        for name, constraint_type, is_validated, definition in constraint_rows
    }
    for name, constraint_type in EXPECTED_CONSTRAINTS.items():
        assert constraints[name][0] == constraint_type
        assert constraints[name][1] is True
    unique_definition = constraints["chunks_document_version_strategy_chunk_key"][2]
    assert "source_file, document_hash, chunking_strategy, chunk_index" in (
        unique_definition
    )

    indexes = {
        name: (
            access_method,
            definition.lower(),
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
    source_index = indexes["idx_chunks_source_file_chunking_strategy"]
    hash_index = indexes["idx_chunks_document_hash_chunking_strategy"]
    vector_index = indexes["idx_chunks_embedding_hnsw_cosine"]
    assert source_index[0] == "btree"
    assert source_index[2:] == (True, True, True, True, None)
    assert "(source_file, chunking_strategy)" in source_index[1]
    assert hash_index[0] == "btree"
    assert hash_index[2:] == (True, True, True, True, None)
    assert "(document_hash, chunking_strategy)" in hash_index[1]
    assert vector_index[0] == "hnsw"
    assert vector_index[2:] == (True, True, True, True, None)
    assert "using hnsw" in vector_index[1]
    assert "embedding vector_cosine_ops" in vector_index[1]
    assert "ivfflat" not in vector_index[1]


def test_vector_dimension_enforcement_and_rollback() -> None:
    source_file = "stage-1-integration-test.pdf"

    with open_database_connection() as connection:
        inserted = connection.execute(
            """
            INSERT INTO public.chunks (
                content,
                embedding,
                source_file,
                document_hash,
                source_type,
                chunk_index,
                chunking_strategy,
                page_number
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                "Infrastructure verification chunk",
                Vector([0.0] * 768),
                source_file,
                "a" * 64,
                "PDF",
                0,
                "fixed",
                1,
            ),
        ).fetchone()
        assert isinstance(inserted[0], int)
        connection.rollback()

    with open_database_connection() as connection:
        with pytest.raises(psycopg.Error):
            connection.execute(
                """
                INSERT INTO public.chunks (
                    content,
                    embedding,
                    source_file,
                    document_hash,
                    source_type,
                    chunk_index,
                    chunking_strategy
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "Invalid vector dimension",
                    Vector([0.0] * 767),
                    source_file,
                    "b" * 64,
                    "DOCX",
                    0,
                    "sentence",
                ),
            )
        connection.rollback()

    with open_database_connection() as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM public.chunks WHERE source_file = %s",
            (source_file,),
        ).fetchone()[0]
        connection.rollback()

    assert remaining == 0
