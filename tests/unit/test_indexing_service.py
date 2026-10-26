"""Unit tests for Stage 5 indexing orchestration."""

from pathlib import Path

import pytest

from rag_app.database.repository import (
    IndexedDocumentState,
    IndexedDocumentVersion,
    PersistenceResult,
)
from rag_app.documents import (
    Chunk,
    ChunkedDocument,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    ExtractedDocument,
    ExtractedTextUnit,
    IndexingStatus,
)
from rag_app.exceptions import (
    ChunkGenerationError,
    DocumentExtractionError,
    EmptyDocumentError,
    FeatureUnavailableError,
    GeminiRequestError,
    IndexingPipelineError,
    PersistenceError,
)
from rag_app.services.indexing import index_document, reset_index

HASH = "a" * 64
VECTOR = (1.0,) + (0.0,) * 767
FIXTURES = Path(__file__).parents[1] / "fixtures"


def _documents() -> tuple[ExtractedDocument, ChunkedDocument, EmbeddedDocument]:
    extracted = ExtractedDocument(
        source_file="report.pdf",
        source_type="PDF",
        units=(ExtractedTextUnit("Text", 0, 1),),
    )
    chunked = ChunkedDocument(
        source_file="report.pdf",
        source_type="PDF",
        chunking_strategy=ChunkingStrategy.fixed,
        chunks=(Chunk("Text", 0, 1),),
    )
    embedded = EmbeddedDocument(
        source_file="report.pdf",
        source_type="PDF",
        chunking_strategy=ChunkingStrategy.fixed,
        chunks=(EmbeddedChunk(chunked.chunks[0], VECTOR),),
    )
    return extracted, chunked, embedded


def _invoke(
    *,
    state: IndexedDocumentState | None = None,
    extractor_error: Exception | None = None,
    chunker_error: Exception | None = None,
    embedder_error: Exception | None = None,
    persister_error: Exception | None = None,
    readiness_error: Exception | None = None,
    state_error: Exception | None = None,
    persistence_status: IndexingStatus = IndexingStatus.indexed,
    recorded_calls: list[str] | None = None,
    hash_value: object = HASH,
) -> tuple[object, list[str]]:
    extracted, chunked, embedded = _documents()
    calls = recorded_calls if recorded_calls is not None else []

    def validator(file: object) -> tuple[Path, str]:
        calls.append("validate-path")
        return Path("/private/input/report.pdf"), "PDF"

    def validate_strategy(value: object) -> ChunkingStrategy:
        calls.append("validate-strategy")
        return ChunkingStrategy.fixed

    def ready() -> None:
        calls.append("database-ready")
        if readiness_error:
            raise readiness_error

    def hash_file(path: Path) -> str:
        calls.append("hash")
        return hash_value  # type: ignore[return-value]

    def read_state(
        source_file: str, strategy: ChunkingStrategy
    ) -> IndexedDocumentState:
        calls.append("preflight")
        if state_error:
            raise state_error
        assert source_file == "report.pdf"
        assert strategy is ChunkingStrategy.fixed
        return state or IndexedDocumentState(())

    def extract(path: object) -> ExtractedDocument:
        calls.append("extract")
        if extractor_error:
            raise extractor_error
        return extracted

    def chunk(document: object, strategy: object) -> ChunkedDocument:
        calls.append("chunk")
        if chunker_error:
            raise chunker_error
        return chunked

    def embed(document: object) -> EmbeddedDocument:
        calls.append("embed")
        if embedder_error:
            raise embedder_error
        return embedded

    def persist(document: object, document_hash: str) -> PersistenceResult:
        calls.append("persist")
        assert document is embedded
        assert document_hash == HASH
        if persister_error:
            raise persister_error
        return PersistenceResult(persistence_status, 1)

    clock = iter((10.0, 10.25))
    result = index_document(
        Path("/private/input/report.pdf"),
        "FIXED",
        path_validator=validator,
        strategy_validator=validate_strategy,
        readiness_checker=ready,
        hashing_function=hash_file,
        state_reader=read_state,
        extractor=extract,
        chunker=chunk,
        embedder=embed,
        persister=persist,
        monotonic=lambda: next(clock),
    )
    return result, calls


def test_new_document_executes_every_stage_in_order() -> None:
    result, calls = _invoke()

    assert calls == [
        "validate-path",
        "validate-strategy",
        "database-ready",
        "hash",
        "preflight",
        "extract",
        "chunk",
        "embed",
        "persist",
    ]
    assert result.status is IndexingStatus.indexed
    assert result.source_file == "report.pdf"
    assert result.chunking_strategy is ChunkingStrategy.fixed
    assert result.chunk_count == 1
    assert result.elapsed_seconds == pytest.approx(0.25)
    assert "/private" not in result.source_file


def test_exact_duplicate_skips_every_expensive_or_write_stage() -> None:
    state = IndexedDocumentState((IndexedDocumentVersion(HASH, 7),))
    result, calls = _invoke(state=state)

    assert result.status is IndexingStatus.skipped
    assert result.chunk_count == 7
    assert calls == [
        "validate-path",
        "validate-strategy",
        "database-ready",
        "hash",
        "preflight",
    ]


def test_conflicting_hash_runs_complete_replacement_pipeline() -> None:
    state = IndexedDocumentState((IndexedDocumentVersion("b" * 64, 2),))
    result, calls = _invoke(state=state, persistence_status=IndexingStatus.replaced)

    assert result.status is IndexingStatus.replaced
    assert calls[-4:] == ["extract", "chunk", "embed", "persist"]


def test_invalid_hashing_output_stops_before_preflight() -> None:
    calls: list[str] = []

    with pytest.raises(IndexingPipelineError, match="hashing stage"):
        _invoke(hash_value="INVALID", recorded_calls=calls)

    assert calls[-1] == "hash"


def test_transaction_time_duplicate_is_returned_as_skip() -> None:
    result, calls = _invoke(persistence_status=IndexingStatus.skipped)

    assert calls[-1] == "persist"
    assert result.status is IndexingStatus.skipped


@pytest.mark.parametrize(
    ("keyword", "error", "forbidden_calls"),
    [
        (
            "extractor_error",
            DocumentExtractionError("failed"),
            {"chunk", "embed", "persist"},
        ),
        ("chunker_error", ChunkGenerationError("failed"), {"embed", "persist"}),
        ("embedder_error", GeminiRequestError("failed"), {"persist"}),
        ("persister_error", PersistenceError("failed"), set()),
    ],
)
def test_stage_failure_prevents_later_work(
    keyword: str, error: Exception, forbidden_calls: set[str]
) -> None:
    calls: list[str] = []
    with pytest.raises(type(error)):
        _invoke(**{keyword: error}, recorded_calls=calls)

    assert forbidden_calls.isdisjoint(calls)


@pytest.mark.parametrize(
    ("keyword", "expected_calls"),
    [
        ("readiness_error", ["validate-path", "validate-strategy", "database-ready"]),
        (
            "state_error",
            [
                "validate-path",
                "validate-strategy",
                "database-ready",
                "hash",
                "preflight",
            ],
        ),
    ],
)
def test_database_preflight_failures_prevent_gemini_and_persistence(
    keyword: str, expected_calls: list[str]
) -> None:
    calls: list[str] = []
    error = PersistenceError("The database schema is not ready.")

    with pytest.raises(PersistenceError):
        _invoke(**{keyword: error}, recorded_calls=calls)

    assert calls == expected_calls
    assert "embed" not in calls
    assert "persist" not in calls


def test_empty_document_prevents_gemini_and_persistence() -> None:
    calls: list[str] = []
    path = FIXTURES / "pdf" / "empty.pdf"

    with pytest.raises(EmptyDocumentError, match="does not contain extractable text"):
        index_document(
            path,
            "fixed",
            readiness_checker=lambda: calls.append("readiness"),
            hashing_function=lambda source_path: HASH,
            state_reader=lambda source_file, strategy: IndexedDocumentState(()),
            embedder=lambda document: calls.append("embed"),  # type: ignore[arg-type,return-value]
            persister=lambda document, document_hash: calls.append("persist"),  # type: ignore[arg-type,return-value]
        )

    assert calls == ["readiness"]


def test_reset_is_typed_and_unavailable() -> None:
    with pytest.raises(FeatureUnavailableError, match="not implemented"):
        reset_index()


def test_service_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    result, _ = _invoke()

    assert result.elapsed_seconds >= 0
    assert capsys.readouterr() == ("", "")
