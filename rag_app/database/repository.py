"""Database schema inspection and transactional chunk persistence boundary."""

import hashlib
import logging
import math
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import psycopg
from pgvector import Vector
from psycopg import Connection, Cursor

from rag_app.config import EMBEDDING_DIMENSION
from rag_app.database.connection import (
    open_database_connection,
    register_vector_types,
)
from rag_app.documents import (
    Chunk,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    IndexingStatus,
    SourceType,
)
from rag_app.exceptions import (
    DatabaseOperationError,
    DatabaseSchemaError,
    DuplicateStateInconsistencyError,
    PersistenceError,
    PersistenceVerificationError,
    RagAppError,
)
from rag_app.processing.chunking import MAX_CHUNK_SIZE

logger = logging.getLogger(__name__)

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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_COPY_CHUNKS_SQL = """
    COPY public.chunks (
        content,
        embedding,
        source_file,
        document_hash,
        source_type,
        chunk_index,
        chunking_strategy,
        page_number
    ) FROM STDIN
"""

_INSERT_CHUNK_SQL = """
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
"""

_PersistenceRow = tuple[object, object, str, str, str, int, str, int | None]

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


@dataclass(frozen=True, slots=True)
class IndexedDocumentVersion:
    """One stored hash and its complete current chunk count."""

    document_hash: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class IndexedDocumentState:
    """Immutable stored state for one filename and chunking strategy."""

    versions: tuple[IndexedDocumentVersion, ...]

    @property
    def document_hashes(self) -> tuple[str, ...]:
        """Return stored document hashes in deterministic order."""
        return tuple(version.document_hash for version in self.versions)

    @property
    def total_chunk_count(self) -> int:
        """Return the chunk count across all stored versions."""
        return sum(version.chunk_count for version in self.versions)

    def chunk_count_for(self, document_hash: str) -> int | None:
        """Return the stored chunk count for a hash, if present."""
        for version in self.versions:
            if version.document_hash == document_hash:
                return version.chunk_count
        return None


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    """Outcome decided by the authoritative write transaction."""

    status: IndexingStatus
    chunk_count: int


class _SchemaMismatchError(Exception):
    """Internal schema mismatch detail safe to include in an outer message."""


def initialize_schema() -> DatabaseStatus:
    """Apply, validate, and commit the canonical schema as one transaction."""
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
                "Database schema initialization failed. Verify PostgreSQL and "
                "pgvector versions."
            ) from error
        except (psycopg.Error, DatabaseOperationError) as error:
            _rollback_quietly(connection)
            raise DatabaseSchemaError(
                "Database schema initialization failed."
            ) from error
        except Exception:
            _rollback_quietly(connection)
            raise


def check_database_readiness() -> DatabaseStatus:
    """Verify required database objects, then roll back the inspection."""
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
                "The database schema is not ready. Run the database initialization "
                "command and verify PostgreSQL and pgvector versions."
            ) from error
        except DatabaseOperationError:
            _rollback_quietly(connection)
            raise
        except psycopg.Error as error:
            _rollback_quietly(connection)
            raise DatabaseOperationError("Database readiness check failed.") from error
        except Exception:
            _rollback_quietly(connection)
            raise


def get_indexed_document_state(
    source_file: str, strategy: ChunkingStrategy
) -> IndexedDocumentState:
    """Read one document's stored versions, then roll back the transaction."""
    _validate_identity(source_file, strategy)

    with open_database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                state = _read_indexed_document_state(cursor, source_file, strategy)
            connection.rollback()
            return state
        except RagAppError:
            _rollback_quietly(connection)
            raise
        except psycopg.Error as error:
            _rollback_quietly(connection)
            raise PersistenceError(
                "Indexed document state could not be read."
            ) from error
        except Exception:
            _rollback_quietly(connection)
            raise


def persist_embedded_document(
    document: EmbeddedDocument, document_hash: str
) -> PersistenceResult:
    """Commit one complete insert or replacement, rolling back on failure."""
    _validate_embedded_document(document, document_hash)
    rows = _build_persistence_rows(document, document_hash)
    expected_count = len(document.chunks)
    if len(rows) != expected_count:
        raise PersistenceVerificationError(
            "Persistence row preparation did not preserve every document chunk."
        )

    with open_database_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (
                        _document_lock_key(
                            document.source_file, document.chunking_strategy
                        ),
                    ),
                )
                state = _read_indexed_document_state(
                    cursor, document.source_file, document.chunking_strategy
                )
                existing_count = state.chunk_count_for(document_hash)
                if len(state.versions) == 1 and existing_count is not None:
                    connection.commit()
                    return PersistenceResult(
                        status=IndexingStatus.skipped,
                        chunk_count=existing_count,
                    )

                status = (
                    IndexingStatus.replaced
                    if state.versions
                    else IndexingStatus.indexed
                )
                if state.versions:
                    cursor.execute(
                        """
                        DELETE FROM public.chunks
                        WHERE source_file = %s AND chunking_strategy = %s
                        """,
                        (
                            document.source_file,
                            document.chunking_strategy.value,
                        ),
                    )

                _insert_rows(cursor, rows)
                _verify_persisted_document(
                    cursor,
                    document.source_file,
                    document.chunking_strategy,
                    document_hash,
                    expected_count,
                )

            connection.commit()
            return PersistenceResult(status=status, chunk_count=expected_count)
        except RagAppError:
            _rollback_quietly(connection)
            raise
        except psycopg.Error as error:
            _rollback_quietly(connection)
            raise PersistenceError("Document persistence failed.") from error
        except Exception:
            _rollback_quietly(connection)
            raise


def _validate_identity(source_file: str, strategy: ChunkingStrategy) -> None:
    if not _is_safe_basename(source_file):
        raise PersistenceError("Persistence requires a safe source filename.")
    if not isinstance(strategy, ChunkingStrategy):
        raise PersistenceError("Persistence requires a canonical chunking strategy.")


def _validate_embedded_document(document: EmbeddedDocument, document_hash: str) -> None:
    if not isinstance(document, EmbeddedDocument):
        raise PersistenceError("Persistence input must be an EmbeddedDocument.")
    _validate_identity(document.source_file, document.chunking_strategy)
    if not isinstance(document_hash, str) or not _SHA256_PATTERN.fullmatch(
        document_hash
    ):
        raise PersistenceError(
            "Persistence requires a lowercase SHA-256 document hash."
        )
    if document.source_type not in {"PDF", "DOCX"}:
        raise PersistenceError("Persistence input has an unsupported source type.")

    expected_extension = ".pdf" if document.source_type == "PDF" else ".docx"
    if Path(document.source_file).suffix.lower() != expected_extension:
        raise PersistenceError(
            "Persistence source filename and source type are inconsistent."
        )
    if not isinstance(document.chunks, tuple) or not document.chunks:
        raise PersistenceError(
            "Persistence input must contain at least one embedded chunk."
        )

    for expected_index, embedded_chunk in enumerate(document.chunks):
        _validate_embedded_chunk(embedded_chunk, expected_index, document.source_type)


def _validate_embedded_chunk(
    embedded_chunk: EmbeddedChunk, expected_index: int, source_type: SourceType
) -> None:
    if not isinstance(embedded_chunk, EmbeddedChunk) or not isinstance(
        embedded_chunk.chunk, Chunk
    ):
        raise PersistenceError(
            f"Persistence input contains an invalid chunk at index {expected_index}."
        )

    chunk = embedded_chunk.chunk
    if (
        not isinstance(chunk.chunk_index, int)
        or isinstance(chunk.chunk_index, bool)
        or chunk.chunk_index != expected_index
    ):
        raise PersistenceError(
            "Persistence chunk indexes must be zero-based, continuous, and ordered."
        )
    if (
        not isinstance(chunk.content, str)
        or not chunk.content.strip()
        or len(chunk.content) > MAX_CHUNK_SIZE
    ):
        raise PersistenceError(
            f"Persistence chunk {expected_index} has invalid content."
        )

    if source_type == "PDF":
        if (
            not isinstance(chunk.page_number, int)
            or isinstance(chunk.page_number, bool)
            or chunk.page_number < 1
        ):
            raise PersistenceError(
                f"Persistence PDF chunk {expected_index} has an invalid page number."
            )
    elif chunk.page_number is not None:
        raise PersistenceError(
            f"Persistence DOCX chunk {expected_index} must not have a page number."
        )

    vector = embedded_chunk.embedding
    if not isinstance(vector, tuple) or len(vector) != EMBEDDING_DIMENSION:
        raise PersistenceError(
            f"Persistence embedding {expected_index} must contain "
            f"{EMBEDDING_DIMENSION} values."
        )
    if any(type(value) is not float or not math.isfinite(value) for value in vector):
        raise PersistenceError(
            f"Persistence embedding {expected_index} contains an invalid value."
        )
    norm = math.hypot(*vector)
    if norm == 0.0 or not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise PersistenceError(
            f"Persistence embedding {expected_index} is not normalized."
        )


def _is_safe_basename(source_file: object) -> bool:
    return (
        isinstance(source_file, str)
        and bool(source_file.strip())
        and not any(
            ord(character) < 32 or ord(character) == 127 for character in source_file
        )
        and source_file not in {".", ".."}
        and Path(source_file).name == source_file
        and PureWindowsPath(source_file).name == source_file
    )


def _build_persistence_rows(
    document: EmbeddedDocument, document_hash: str
) -> tuple[_PersistenceRow, ...]:
    return tuple(
        (
            embedded_chunk.chunk.content,
            Vector(list(embedded_chunk.embedding)),
            document.source_file,
            document_hash,
            document.source_type,
            embedded_chunk.chunk.chunk_index,
            document.chunking_strategy.value,
            embedded_chunk.chunk.page_number,
        )
        for embedded_chunk in document.chunks
    )


def _read_indexed_document_state(
    cursor: Cursor[tuple[object, ...]],
    source_file: str,
    strategy: ChunkingStrategy,
) -> IndexedDocumentState:
    rows = cursor.execute(
        """
        SELECT document_hash, count(*)
        FROM public.chunks
        WHERE source_file = %s AND chunking_strategy = %s
        GROUP BY document_hash
        ORDER BY document_hash
        """,
        (source_file, strategy.value),
    ).fetchall()
    versions: list[IndexedDocumentVersion] = []
    for row in rows:
        try:
            document_hash, chunk_count = row
        except (TypeError, ValueError) as error:
            raise DuplicateStateInconsistencyError(
                "Stored document-version state is inconsistent."
            ) from error
        if (
            not isinstance(document_hash, str)
            or _SHA256_PATTERN.fullmatch(document_hash) is None
            or not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count < 1
        ):
            raise DuplicateStateInconsistencyError(
                "Stored document-version state is inconsistent."
            )
        versions.append(
            IndexedDocumentVersion(
                document_hash=document_hash,
                chunk_count=chunk_count,
            )
        )

    state = IndexedDocumentState(versions=tuple(versions))
    if len(state.versions) > 1:
        logger.warning(
            "Multiple document versions detected source_file=%s strategy=%s "
            "versions=%d chunks=%d; replacement will restore the invariant",
            source_file,
            strategy.value,
            len(state.versions),
            state.total_chunk_count,
        )
    return state


def _insert_rows(
    cursor: Cursor[tuple[object, ...]], rows: tuple[_PersistenceRow, ...]
) -> None:
    copy_method = getattr(cursor, "copy", None)
    if not callable(copy_method):
        cursor.executemany(_INSERT_CHUNK_SQL, rows)
        return

    with copy_method(_COPY_CHUNKS_SQL) as copy:
        for row in rows:
            copy.write_row(row)


def _verify_persisted_document(
    cursor: Cursor[tuple[object, ...]],
    source_file: str,
    strategy: ChunkingStrategy,
    document_hash: str,
    expected_count: int,
) -> None:
    row = cursor.execute(
        """
        SELECT array_agg(DISTINCT document_hash ORDER BY document_hash),
               count(*),
               min(chunk_index),
               max(chunk_index),
               count(DISTINCT chunk_index)
        FROM public.chunks
        WHERE source_file = %s AND chunking_strategy = %s
        """,
        (source_file, strategy.value),
    ).fetchone()
    expected = ([document_hash], expected_count, 0, expected_count - 1, expected_count)
    if row != expected:
        raise PersistenceVerificationError(
            f"Persisted document verification failed for '{source_file}'."
        )


def _document_lock_key(source_file: str, strategy: ChunkingStrategy) -> int:
    identity = f"{len(source_file)}:{source_file}:{strategy.value}".encode()
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _inspect_schema(
    connection: Connection[tuple[object, ...]],
) -> DatabaseStatus:
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
        raise _SchemaMismatchError(f"pgvector {EXPECTED_PGVECTOR_VERSION} is required")

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


def _rollback_quietly(connection: Connection[tuple[object, ...]]) -> None:
    with suppress(psycopg.Error):
        connection.rollback()
