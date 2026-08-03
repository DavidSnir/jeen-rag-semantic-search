"""Unit tests for ordered, validated Gemini document embeddings."""

import logging
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors, types

import rag_app.embeddings.gemini as gemini_module
from rag_app.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EmbeddingSettings,
)
from rag_app.documents import Chunk, ChunkedDocument, ChunkingStrategy
from rag_app.embeddings import embed_document
from rag_app.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingNormalizationError,
    GeminiRequestError,
    InvalidEmbeddingInputError,
    InvalidGeminiResponseError,
    RagAppError,
)

SYNTHETIC_KEY = "stage4-recognizable-synthetic-key"
SENSITIVE_CONTENT = "private chunk content must stay out of provider errors"


class FakeModels:
    def __init__(self, response: object = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def embed_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: object = None, error: Exception | None = None):
        self.models = FakeModels(response=response, error=error)


class OwnedFakeClient(FakeClient):
    def __init__(self, response: object = None):
        super().__init__(response=response)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def block_real_gemini_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_client_creation(**kwargs: object) -> None:
        raise AssertionError("Automated tests must not construct a real Gemini client")

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)


@pytest.fixture
def settings() -> EmbeddingSettings:
    return EmbeddingSettings(gemini_api_key=SYNTHETIC_KEY)


def _vector(first: float = 1.0, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]


def _response(*vectors: object) -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


def _document(
    contents: tuple[str, ...] = ("First chunk.",),
    *,
    source_file: str = "source.docx",
    source_type: str = "DOCX",
    strategy: Any = ChunkingStrategy.fixed,
    page_numbers: tuple[int | None, ...] | None = None,
) -> ChunkedDocument:
    if page_numbers is None:
        page_numbers = tuple(
            index + 1 if source_type == "PDF" else None
            for index in range(len(contents))
        )
    return ChunkedDocument(
        source_file=source_file,
        source_type=source_type,  # type: ignore[arg-type]
        chunking_strategy=strategy,
        chunks=tuple(
            Chunk(content=content, chunk_index=index, page_number=page_numbers[index])
            for index, content in enumerate(contents)
        ),
    )


def test_embedding_errors_use_application_hierarchy() -> None:
    assert issubclass(InvalidEmbeddingInputError, EmbeddingError)
    assert issubclass(GeminiRequestError, EmbeddingError)
    assert issubclass(InvalidGeminiResponseError, EmbeddingError)
    assert issubclass(EmbeddingDimensionError, InvalidGeminiResponseError)
    assert issubclass(EmbeddingNormalizationError, EmbeddingError)
    assert issubclass(EmbeddingError, RagAppError)
    assert issubclass(EmbeddingConfigurationError, RagAppError)


def test_one_chunk_uses_one_multi_content_request(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector()))
    document = _document(("Only actual chunk text.",))

    result = embed_document(document, settings, client=client)

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call["model"] == DEFAULT_EMBEDDING_MODEL
    assert call["contents"] == ["Only actual chunk text."]
    config = call["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == "RETRIEVAL_DOCUMENT"
    assert config.title == "source.docx"
    assert config.output_dimensionality == EMBEDDING_DIMENSION
    assert len(result.chunks) == 1


def test_multiple_chunks_preserve_request_and_response_order(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector(3.0), _vector(0.0, -4.0)))
    document = _document(("First.", "Second."))

    result = embed_document(document, settings, client=client)

    assert len(client.models.calls) == 1
    assert client.models.calls[0]["contents"] == ["First.", "Second."]
    assert result.chunks[0].chunk is document.chunks[0]
    assert result.chunks[1].chunk is document.chunks[1]
    assert result.chunks[0].embedding[0] == pytest.approx(1.0)
    assert result.chunks[1].embedding[1] == pytest.approx(-1.0)


def test_production_path_creates_developer_api_client_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OwnedFakeClient(_response(_vector()))
    received_valid_configuration = False

    def create_client(*, api_key: str, vertexai: bool) -> OwnedFakeClient:
        nonlocal received_valid_configuration
        received_valid_configuration = (
            api_key == SYNTHETIC_KEY and vertexai is False
        )
        return client

    monkeypatch.setattr(gemini_module.genai, "Client", create_client)
    settings = EmbeddingSettings(gemini_api_key=f"  {SYNTHETIC_KEY}  ")

    result = embed_document(_document(), settings)

    assert received_valid_configuration
    assert len(result.chunks) == 1
    assert client.close_calls == 1


def test_client_creation_failure_is_safe_and_chained(
    monkeypatch: pytest.MonkeyPatch, settings: EmbeddingSettings
) -> None:
    original = ValueError(f"invalid client configuration {SYNTHETIC_KEY}")

    def fail_client_creation(**kwargs: object) -> None:
        raise original

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)

    with pytest.raises(EmbeddingConfigurationError) as raised:
        embed_document(_document(), settings)

    assert raised.value.__cause__ is original
    assert SYNTHETIC_KEY not in str(raised.value)


def test_request_uses_filename_only_and_does_not_serialize_metadata(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector()))
    document = _document(
        ("Actual PDF text only.",),
        source_file="/private/local/path/source.pdf",
        source_type="PDF",
        strategy=ChunkingStrategy.paragraph,
        page_numbers=(17,),
    )

    embed_document(document, settings, client=client)

    call = client.models.calls[0]
    assert call["contents"] == ["Actual PDF text only."]
    assert call["config"].title == "source.pdf"  # type: ignore[union-attr]
    request_repr = repr(call)
    assert "/private/local/path" not in request_repr
    assert "page_number" not in request_repr
    assert "chunk_index" not in request_repr
    assert SYNTHETIC_KEY not in request_repr


def test_request_strips_windows_style_source_path(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector()))
    document = _document(
        ("Actual PDF text only.",),
        source_file=r"C:\private\local\source.pdf",
        source_type="PDF",
        page_numbers=(1,),
    )

    embed_document(document, settings, client=client)

    call = client.models.calls[0]
    assert call["config"].title == "source.pdf"  # type: ignore[union-attr]
    assert "private" not in repr(call)


@pytest.mark.parametrize(
    "document",
    [
        ChunkedDocument(
            source_file="empty.docx",
            source_type="DOCX",
            chunking_strategy=ChunkingStrategy.fixed,
            chunks=(),
        ),
        _document(("",)),
        _document((" \n\t ",)),
        _document(("x" * 2_001,)),
        _document(("valid",), source_type="PDF", page_numbers=(None,)),
        _document(("valid",), source_type="PDF", page_numbers=(0,)),
        _document(("valid",), source_type="DOCX", page_numbers=(1,)),
        _document(("valid",), source_type="TXT"),
        _document(("valid",), strategy="fixed"),
    ],
)
def test_invalid_document_is_rejected_before_client_invocation(
    document: ChunkedDocument, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(_vector()))

    with pytest.raises(InvalidEmbeddingInputError):
        embed_document(document, settings, client=client)

    assert client.models.calls == []


def test_non_string_chunk_is_rejected_before_request(
    settings: EmbeddingSettings,
) -> None:
    document = _document(("valid",))
    invalid_chunk = Chunk(content=123, chunk_index=0, page_number=None)  # type: ignore[arg-type]
    invalid_document = ChunkedDocument(
        source_file=document.source_file,
        source_type=document.source_type,
        chunking_strategy=document.chunking_strategy,
        chunks=(invalid_chunk,),
    )
    client = FakeClient(_response(_vector()))

    with pytest.raises(InvalidEmbeddingInputError):
        embed_document(invalid_document, settings, client=client)

    assert client.models.calls == []


def test_mutable_chunk_collection_is_rejected_before_request(
    settings: EmbeddingSettings,
) -> None:
    document = _document()
    mutable_document = ChunkedDocument(
        source_file=document.source_file,
        source_type=document.source_type,
        chunking_strategy=document.chunking_strategy,
        chunks=list(document.chunks),  # type: ignore[arg-type]
    )
    client = FakeClient(_response(_vector()))

    with pytest.raises(InvalidEmbeddingInputError, match="immutable tuple"):
        embed_document(mutable_document, settings, client=client)

    assert client.models.calls == []


@pytest.mark.parametrize("indexes", [(1,), (0, 2), (1, 0)])
def test_invalid_chunk_indexes_are_rejected_before_request(
    indexes: tuple[int, ...], settings: EmbeddingSettings
) -> None:
    document = ChunkedDocument(
        source_file="source.docx",
        source_type="DOCX",
        chunking_strategy=ChunkingStrategy.fixed,
        chunks=tuple(
            Chunk(content=f"Chunk {position}", chunk_index=index, page_number=None)
            for position, index in enumerate(indexes)
        ),
    )
    client = FakeClient(_response(*(_vector() for _ in indexes)))

    with pytest.raises(InvalidEmbeddingInputError, match="zero-based"):
        embed_document(document, settings, client=client)

    assert client.models.calls == []


@pytest.mark.parametrize(
    "invalid_settings",
    [
        EmbeddingSettings(gemini_api_key=""),
        EmbeddingSettings(gemini_api_key="  "),
        EmbeddingSettings(
            gemini_api_key=SYNTHETIC_KEY,
            embedding_model="text-embedding-004",
        ),
        EmbeddingSettings(
            gemini_api_key=SYNTHETIC_KEY,
            embedding_dimension=767,
        ),
    ],
)
def test_invalid_settings_are_rejected_before_request(
    invalid_settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector()))

    with pytest.raises(EmbeddingConfigurationError):
        embed_document(_document(), invalid_settings, client=client)

    assert client.models.calls == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "no embedding response"),
        (SimpleNamespace(), "does not contain embeddings"),
        (SimpleNamespace(embeddings=None), "null embeddings"),
        (SimpleNamespace(embeddings=[]), "no embeddings"),
        (SimpleNamespace(embeddings="not-a-collection"), "invalid embeddings collection"),
    ],
)
def test_missing_embedding_response_data_is_rejected(
    response: object, message: str, settings: EmbeddingSettings
) -> None:
    client = FakeClient(response)

    with pytest.raises(InvalidGeminiResponseError, match=message):
        embed_document(_document(), settings, client=client)


@pytest.mark.parametrize("returned_count", [1, 3])
def test_response_count_must_equal_chunk_count(
    returned_count: int, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(*(_vector() for _ in range(returned_count))))

    with pytest.raises(InvalidGeminiResponseError, match="expected 2"):
        embed_document(_document(("One", "Two")), settings, client=client)


@pytest.mark.parametrize("values", [None, [], "not-a-vector"])
def test_missing_or_empty_vector_values_are_rejected(
    values: object, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(values))

    with pytest.raises(InvalidGeminiResponseError):
        embed_document(_document(), settings, client=client)


def test_missing_vector_values_field_is_rejected(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(SimpleNamespace(embeddings=[SimpleNamespace()]))

    with pytest.raises(InvalidGeminiResponseError, match="values are missing"):
        embed_document(_document(), settings, client=client)


@pytest.mark.parametrize("dimension", [767, 769])
def test_wrong_vector_dimension_identifies_chunk(
    dimension: int, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response([1.0] * dimension))

    with pytest.raises(EmbeddingDimensionError) as raised:
        embed_document(_document(), settings, client=client)

    assert str(dimension) in str(raised.value)
    assert "chunk 0" in str(raised.value)
    assert SENSITIVE_CONTENT not in str(raised.value)


@pytest.mark.parametrize("invalid_value", ["1.0", None, True, math.nan, math.inf, -math.inf])
def test_invalid_vector_values_are_rejected(
    invalid_value: object, settings: EmbeddingSettings
) -> None:
    vector: list[object] = _vector()
    vector[17] = invalid_value
    client = FakeClient(_response(vector))

    with pytest.raises(InvalidGeminiResponseError):
        embed_document(_document(), settings, client=client)


def test_zero_vector_is_rejected(settings: EmbeddingSettings) -> None:
    client = FakeClient(_response([0.0] * EMBEDDING_DIMENSION))

    with pytest.raises(EmbeddingNormalizationError, match="chunk 0"):
        embed_document(_document(), settings, client=client)


def test_valid_vector_is_immutable_normalized_and_preserves_direction(
    settings: EmbeddingSettings,
) -> None:
    raw_vector = _vector(3.0, -4.0)
    original = raw_vector.copy()
    client = FakeClient(_response(raw_vector))

    result = embed_document(_document(), settings, client=client)
    vector = result.chunks[0].embedding

    assert isinstance(vector, tuple)
    assert len(vector) == EMBEDDING_DIMENSION
    assert math.hypot(*vector) == pytest.approx(1.0)
    assert vector[0] == pytest.approx(0.6)
    assert vector[1] == pytest.approx(-0.8)
    assert raw_vector == original
    with pytest.raises(TypeError):
        vector[0] = 2.0  # type: ignore[index]


def test_every_vector_is_normalized_independently(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector(2.0), _vector(0.0, -7.0)))

    result = embed_document(_document(("One", "Two")), settings, client=client)

    assert [math.hypot(*chunk.embedding) for chunk in result.chunks] == pytest.approx(
        [1.0, 1.0]
    )
    assert result.chunks[0].embedding[0] == pytest.approx(1.0)
    assert result.chunks[1].embedding[1] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("source_type", "page_numbers"),
    [("PDF", (2, 4)), ("DOCX", (None, None))],
)
def test_embedded_document_preserves_metadata_without_mutating_input(
    source_type: str,
    page_numbers: tuple[int | None, ...],
    settings: EmbeddingSettings,
) -> None:
    source_file = "pages.pdf" if source_type == "PDF" else "document.docx"
    document = _document(
        ("First.", "Second."),
        source_file=source_file,
        source_type=source_type,
        strategy=ChunkingStrategy.sentence,
        page_numbers=page_numbers,
    )
    original = _document(
        ("First.", "Second."),
        source_file=source_file,
        source_type=source_type,
        strategy=ChunkingStrategy.sentence,
        page_numbers=page_numbers,
    )
    client = FakeClient(_response(_vector(), _vector(0.0, 1.0)))

    result = embed_document(document, settings, client=client)

    assert result.source_file == source_file
    assert result.source_type == source_type
    assert result.chunking_strategy is ChunkingStrategy.sentence
    assert [item.chunk.content for item in result.chunks] == ["First.", "Second."]
    assert [item.chunk.chunk_index for item in result.chunks] == [0, 1]
    assert [item.chunk.page_number for item in result.chunks] == list(page_numbers)
    assert document == original
    with pytest.raises(FrozenInstanceError):
        result.source_file = "replacement"  # type: ignore[misc]


def test_one_invalid_vector_returns_no_partial_document(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector(), [1.0] * 767))

    with pytest.raises(EmbeddingDimensionError, match="chunk 1"):
        embed_document(_document(("First", "Second")), settings, client=client)

    assert len(client.models.calls) == 1


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "authentication failed"),
        (403, "access was denied"),
        (404, "model was not found"),
        (429, "quota or rate limit"),
        (400, "rejected the embedding request"),
        (503, "temporarily unavailable"),
    ],
)
def test_sdk_api_errors_are_safe_chained_and_not_retried(
    status_code: int,
    message: str,
    settings: EmbeddingSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = errors.APIError(
        status_code,
        {
            "error": {
                "code": status_code,
                "status": "SYNTHETIC",
                "message": f"{SYNTHETIC_KEY}: {SENSITIVE_CONTENT}",
            }
        },
    )
    client = FakeClient(error=original)

    with caplog.at_level(logging.INFO), pytest.raises(GeminiRequestError) as raised:
        embed_document(_document((SENSITIVE_CONTENT,)), settings, client=client)

    assert raised.value.__cause__ is original
    assert message in str(raised.value)
    assert len(client.models.calls) == 1
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_CONTENT not in public_output


def test_transport_error_is_converted_safely(
    settings: EmbeddingSettings,
) -> None:
    original = httpx.ConnectError(
        f"connection URL included {SYNTHETIC_KEY} and {SENSITIVE_CONTENT}"
    )
    client = FakeClient(error=original)

    with pytest.raises(GeminiRequestError) as raised:
        embed_document(_document((SENSITIVE_CONTENT,)), settings, client=client)

    assert raised.value.__cause__ is original
    assert SYNTHETIC_KEY not in str(raised.value)
    assert SENSITIVE_CONTENT not in str(raised.value)
    assert len(client.models.calls) == 1


def test_sdk_protocol_value_error_is_converted_safely(
    settings: EmbeddingSettings,
) -> None:
    original = ValueError(f"invalid raw response {SYNTHETIC_KEY}")
    client = FakeClient(error=original)

    with pytest.raises(GeminiRequestError) as raised:
        embed_document(_document(), settings, client=client)

    assert raised.value.__cause__ is original
    assert SYNTHETIC_KEY not in str(raised.value)
    assert len(client.models.calls) == 1


def test_invalid_response_logs_only_safe_failure_metadata(
    settings: EmbeddingSettings, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(_response([1.0] * 767))

    with caplog.at_level(logging.INFO), pytest.raises(EmbeddingDimensionError):
        embed_document(_document((SENSITIVE_CONTENT,)), settings, client=client)

    assert "category=dimension-mismatch" in caplog.text
    assert SYNTHETIC_KEY not in caplog.text
    assert SENSITIVE_CONTENT not in caplog.text


def test_returned_models_do_not_contain_api_key(
    settings: EmbeddingSettings,
) -> None:
    result = embed_document(
        _document(), settings, client=FakeClient(_response(_vector()))
    )

    assert SYNTHETIC_KEY not in repr(result)
