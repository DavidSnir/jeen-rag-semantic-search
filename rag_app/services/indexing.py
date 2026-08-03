"""Complete document-indexing application service."""

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from rag_app.database.repository import (
    IndexedDocumentState,
    PersistenceResult,
    check_database_readiness,
    get_indexed_document_state,
    persist_embedded_document,
)
from rag_app.documents import (
    ChunkedDocument,
    ChunkingStrategy,
    EmbeddedDocument,
    ExtractedDocument,
    IndexingResult,
    IndexingStatus,
    SourceType,
)
from rag_app.embeddings import embed_document
from rag_app.exceptions import FeatureUnavailableError, IndexingPipelineError
from rag_app.extractors import extract_document, validate_document_path
from rag_app.processing.chunking import chunk_document, validate_chunking_strategy
from rag_app.processing.hashing import calculate_document_hash


def index_document(
    file: Path,
    strategy: str,
    *,
    path_validator: Callable[[str | Path | None], tuple[Path, SourceType]] = (
        validate_document_path
    ),
    strategy_validator: Callable[
        [str | ChunkingStrategy], ChunkingStrategy
    ] = validate_chunking_strategy,
    readiness_checker: Callable[[], object] = check_database_readiness,
    hashing_function: Callable[[Path], str] = calculate_document_hash,
    state_reader: Callable[
        [str, ChunkingStrategy], IndexedDocumentState
    ] = get_indexed_document_state,
    extractor: Callable[[str | Path | None], ExtractedDocument] = extract_document,
    chunker: Callable[
        [ExtractedDocument, str | ChunkingStrategy], ChunkedDocument
    ] = chunk_document,
    embedder: Callable[[ChunkedDocument], EmbeddedDocument] = embed_document,
    persister: Callable[
        [EmbeddedDocument, str], PersistenceResult
    ] = persist_embedded_document,
    monotonic: Callable[[], float] = time.monotonic,
) -> IndexingResult:
    """Run every indexing stage and return one immutable safe result."""
    started_at = monotonic()
    source_path, source_type = path_validator(file)
    canonical_strategy = strategy_validator(strategy)
    source_file = source_path.name

    readiness_checker()
    document_hash = hashing_function(source_path)
    if not isinstance(document_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", document_hash
    ) is None:
        raise IndexingPipelineError(
            "The hashing stage returned an invalid document hash."
        )
    state = state_reader(source_file, canonical_strategy)
    if not isinstance(state, IndexedDocumentState):
        raise IndexingPipelineError(
            "The duplicate preflight returned invalid document state."
        )
    existing_count = state.chunk_count_for(document_hash)
    if len(state.versions) == 1 and existing_count is not None:
        return _indexing_result(
            IndexingStatus.skipped,
            source_file,
            canonical_strategy,
            existing_count,
            started_at,
            monotonic,
        )

    extracted = extractor(source_path)
    _validate_stage_metadata(
        "extraction", extracted, ExtractedDocument, source_file, source_type, None
    )
    chunked = chunker(extracted, canonical_strategy)
    _validate_stage_metadata(
        "chunking",
        chunked,
        ChunkedDocument,
        source_file,
        source_type,
        canonical_strategy,
    )
    embedded = embedder(chunked)
    _validate_stage_metadata(
        "embedding",
        embedded,
        EmbeddedDocument,
        source_file,
        source_type,
        canonical_strategy,
    )
    persistence_result = persister(embedded, document_hash)
    if (
        not isinstance(persistence_result, PersistenceResult)
        or not isinstance(persistence_result.status, IndexingStatus)
        or not isinstance(persistence_result.chunk_count, int)
        or isinstance(persistence_result.chunk_count, bool)
        or persistence_result.chunk_count < 1
    ):
        raise IndexingPipelineError(
            "The persistence stage returned an invalid indexing result."
        )

    return _indexing_result(
        persistence_result.status,
        source_file,
        canonical_strategy,
        persistence_result.chunk_count,
        started_at,
        monotonic,
    )


def reset_index() -> NoReturn:
    """Keep reset unavailable until its dedicated implementation stage."""
    raise FeatureUnavailableError("Index reset is not implemented")


def _validate_stage_metadata(
    stage: str,
    document: object,
    expected_type: type[object],
    source_file: str,
    source_type: SourceType,
    strategy: ChunkingStrategy | None,
) -> None:
    if (
        not isinstance(document, expected_type)
        or getattr(document, "source_file", None) != source_file
        or getattr(document, "source_type", None) != source_type
        or strategy is not None
        and getattr(document, "chunking_strategy", None) is not strategy
    ):
        raise IndexingPipelineError(
            f"The {stage} stage returned inconsistent document metadata."
        )


def _indexing_result(
    status: IndexingStatus,
    source_file: str,
    strategy: ChunkingStrategy,
    chunk_count: int,
    started_at: float,
    monotonic: Callable[[], float],
) -> IndexingResult:
    elapsed_seconds = max(0.0, float(monotonic() - started_at))
    return IndexingResult(
        status=status,
        source_file=source_file,
        chunking_strategy=strategy,
        chunk_count=chunk_count,
        elapsed_seconds=elapsed_seconds,
    )
