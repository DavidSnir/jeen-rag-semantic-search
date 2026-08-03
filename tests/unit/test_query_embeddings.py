"""Unit tests for canonical, validated Gemini query embeddings."""

import logging
import math
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

import rag_app.embeddings.gemini as gemini_module
from rag_app.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EmbeddingSettings,
)
from rag_app.embeddings import embed_query
from rag_app.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingNormalizationError,
    GeminiRequestError,
    InvalidEmbeddingInputError,
    InvalidGeminiResponseError,
)

SYNTHETIC_KEY = "stage6-recognizable-synthetic-key"
SENSITIVE_QUERY = "private search query must stay out of logs and errors"


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
    def __init__(
        self,
        response: object = None,
        error: Exception | None = None,
        close_error: Exception | None = None,
    ):
        super().__init__(response=response, error=error)
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def block_real_gemini_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_client_creation(**kwargs: object) -> None:
        raise AssertionError("Query tests must not construct a real Gemini client")

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)


@pytest.fixture
def settings() -> EmbeddingSettings:
    return EmbeddingSettings(gemini_api_key=SYNTHETIC_KEY)


def _vector(
    first: float = 1.0,
    second: float = 0.0,
    *,
    dimension: int = EMBEDDING_DIMENSION,
) -> list[float]:
    values = [first, second, *([0.0] * (EMBEDDING_DIMENSION - 2))]
    return values[:dimension] if dimension <= EMBEDDING_DIMENSION else values + [0.0]


def _response(*vectors: object) -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


@pytest.mark.parametrize(
    ("query", "canonical_query"),
    [
        ("semantic search", "semantic search"),
        (" \t padded query \r\n", "padded query"),
        ("x", "x"),
        ("MiXeD CaSe", "MiXeD CaSe"),
        ("naïve café 東京", "naïve café 東京"),
        ('What is "RAG"?! #1', 'What is "RAG"?! #1'),
    ],
)
def test_valid_queries_are_sent_once_as_canonical_text(
    query: str, canonical_query: str, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(_vector()))

    result = embed_query(query, settings, client=client)

    assert len(client.models.calls) == 1
    assert client.models.calls[0]["contents"] == [canonical_query]
    assert len(result) == EMBEDDING_DIMENSION


def test_request_uses_exact_query_embedding_contract_and_no_title(
    settings: EmbeddingSettings, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(_response(_vector()))

    with caplog.at_level(logging.INFO):
        embed_query(f"  {SENSITIVE_QUERY}\n", settings, client=client)

    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert set(call) == {"model", "contents", "config"}
    assert call["model"] == DEFAULT_EMBEDDING_MODEL
    assert call["contents"] == [SENSITIVE_QUERY]
    config = call["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == EMBEDDING_DIMENSION
    assert config.title is None
    assert SYNTHETIC_KEY not in repr(call)
    assert SYNTHETIC_KEY not in caplog.text
    assert SENSITIVE_QUERY not in caplog.text


@pytest.mark.parametrize("query", [None, 7, True, b"query", "", " \t\r\n "])
def test_invalid_query_is_rejected_before_client_invocation(
    query: object, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(_vector()))

    with pytest.raises(InvalidEmbeddingInputError):
        embed_query(query, settings, client=client)  # type: ignore[arg-type]

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
def test_invalid_settings_are_rejected_before_query_request(
    invalid_settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector()))

    with pytest.raises(EmbeddingConfigurationError):
        embed_query("valid query", invalid_settings, client=client)

    assert client.models.calls == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (None, "no embedding response"),
        (SimpleNamespace(), "does not contain embeddings"),
        (SimpleNamespace(embeddings=None), "null embeddings"),
        (SimpleNamespace(embeddings=[]), "no embeddings"),
        (SimpleNamespace(embeddings="not-a-collection"), "invalid embeddings"),
        (SimpleNamespace(embeddings={}), "invalid embeddings"),
    ],
)
def test_missing_or_malformed_embedding_response_is_rejected(
    response: object, message: str, settings: EmbeddingSettings
) -> None:
    client = FakeClient(response)

    with pytest.raises(InvalidGeminiResponseError, match=message):
        embed_query("valid query", settings, client=client)


def test_query_response_must_contain_exactly_one_embedding(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response(_vector(), _vector(0.0, 1.0)))

    with pytest.raises(InvalidGeminiResponseError, match="2 embeddings; expected 1"):
        embed_query("valid query", settings, client=client)


@pytest.mark.parametrize("values", [None, [], "not-a-vector", object()])
def test_missing_or_malformed_query_vector_is_rejected(
    values: object, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(values))

    with pytest.raises(InvalidGeminiResponseError, match="query"):
        embed_query("valid query", settings, client=client)


def test_missing_query_vector_values_field_is_rejected(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(SimpleNamespace(embeddings=[SimpleNamespace()]))

    with pytest.raises(InvalidGeminiResponseError, match="values are missing for query"):
        embed_query("valid query", settings, client=client)


@pytest.mark.parametrize(
    ("dimension", "is_valid"),
    [(767, False), (768, True), (769, False)],
)
def test_query_vector_requires_exactly_768_dimensions(
    dimension: int, is_valid: bool, settings: EmbeddingSettings
) -> None:
    client = FakeClient(_response(_vector(dimension=dimension)))

    if is_valid:
        result = embed_query("valid query", settings, client=client)
        assert len(result) == EMBEDDING_DIMENSION
        return

    with pytest.raises(EmbeddingDimensionError) as raised:
        embed_query("valid query", settings, client=client)

    assert str(dimension) in str(raised.value)
    assert "query" in str(raised.value)


@pytest.mark.parametrize(
    "invalid_value",
    ["1.0", object(), True, None, math.nan, math.inf, -math.inf],
)
def test_invalid_query_vector_values_are_rejected(
    invalid_value: object, settings: EmbeddingSettings
) -> None:
    vector: list[object] = _vector()
    vector[17] = invalid_value
    client = FakeClient(_response(vector))

    with pytest.raises(InvalidGeminiResponseError, match="query"):
        embed_query("valid query", settings, client=client)


def test_zero_query_vector_cannot_be_normalized(
    settings: EmbeddingSettings,
) -> None:
    client = FakeClient(_response([0.0] * EMBEDDING_DIMENSION))

    with pytest.raises(EmbeddingNormalizationError, match="query"):
        embed_query("valid query", settings, client=client)


def test_query_vector_is_normalized_immutable_and_does_not_mutate_response(
    settings: EmbeddingSettings,
) -> None:
    raw_vector = _vector(3.0, -4.0)
    original = raw_vector.copy()
    client = FakeClient(_response(raw_vector))

    result = embed_query("valid query", settings, client=client)

    assert isinstance(result, tuple)
    assert len(result) == EMBEDDING_DIMENSION
    assert math.hypot(*result) == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(-0.8)
    assert raw_vector == original
    with pytest.raises(TypeError):
        result[0] = 2.0  # type: ignore[index]


def test_owned_client_uses_sanitized_key_and_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OwnedFakeClient(_response(_vector()))
    creation_calls: list[dict[str, object]] = []

    def create_client(*, api_key: str, vertexai: bool) -> OwnedFakeClient:
        creation_calls.append({"api_key": api_key, "vertexai": vertexai})
        return client

    monkeypatch.setattr(gemini_module.genai, "Client", create_client)
    settings = EmbeddingSettings(gemini_api_key=f"  {SYNTHETIC_KEY}  ")

    result = embed_query("  owned query  ", settings)

    assert creation_calls == [{"api_key": SYNTHETIC_KEY, "vertexai": False}]
    assert client.models.calls[0]["contents"] == ["owned query"]
    assert len(result) == EMBEDDING_DIMENSION
    assert client.close_calls == 1


def test_owned_client_is_closed_after_request_failure(
    monkeypatch: pytest.MonkeyPatch, settings: EmbeddingSettings
) -> None:
    original = ValueError(f"bad provider payload {SYNTHETIC_KEY} {SENSITIVE_QUERY}")
    client = OwnedFakeClient(error=original)
    monkeypatch.setattr(gemini_module.genai, "Client", lambda **kwargs: client)

    with pytest.raises(GeminiRequestError) as raised:
        embed_query(SENSITIVE_QUERY, settings)

    assert raised.value.__cause__ is original
    assert len(client.models.calls) == 1
    assert client.close_calls == 1


def test_injected_client_is_not_closed(settings: EmbeddingSettings) -> None:
    client = OwnedFakeClient(_response(_vector()))

    embed_query("valid query", settings, client=client)

    assert client.close_calls == 0


def test_cleanup_failure_does_not_expose_secrets_or_hide_result(
    monkeypatch: pytest.MonkeyPatch,
    settings: EmbeddingSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = OwnedFakeClient(
        _response(_vector()),
        close_error=httpx.TransportError(
            f"cleanup {SYNTHETIC_KEY} {SENSITIVE_QUERY}"
        ),
    )
    monkeypatch.setattr(gemini_module.genai, "Client", lambda **kwargs: client)

    with caplog.at_level(logging.INFO):
        result = embed_query(SENSITIVE_QUERY, settings)

    assert len(result) == EMBEDDING_DIMENSION
    assert client.close_calls == 1
    assert "client cleanup failed" in caplog.text
    assert SYNTHETIC_KEY not in caplog.text
    assert SENSITIVE_QUERY not in caplog.text


def test_client_creation_failure_is_safe_and_chained(
    monkeypatch: pytest.MonkeyPatch,
    settings: EmbeddingSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = ValueError(f"invalid {SYNTHETIC_KEY} for {SENSITIVE_QUERY}")
    creation_calls = 0

    def fail_client_creation(**kwargs: object) -> None:
        nonlocal creation_calls
        creation_calls += 1
        raise original

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)

    with caplog.at_level(logging.INFO), pytest.raises(
        EmbeddingConfigurationError
    ) as raised:
        embed_query(SENSITIVE_QUERY, settings)

    assert raised.value.__cause__ is original
    assert creation_calls == 1
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output


def test_unexpected_client_creation_error_is_not_converted(
    monkeypatch: pytest.MonkeyPatch, settings: EmbeddingSettings
) -> None:
    def fail_client_creation(**kwargs: object) -> None:
        raise TypeError("programming bug")

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)

    with pytest.raises(TypeError, match="programming bug"):
        embed_query("valid query", settings)


def test_unexpected_client_cleanup_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, settings: EmbeddingSettings
) -> None:
    client = OwnedFakeClient(
        _response(_vector()), close_error=AssertionError("programming bug")
    )
    monkeypatch.setattr(gemini_module.genai, "Client", lambda **kwargs: client)

    with pytest.raises(AssertionError, match="programming bug"):
        embed_query("valid query", settings)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "authentication failed"),
        (403, "permission was denied"),
        (404, "model was not found"),
        (429, "quota or rate limit"),
        (400, "rejected the embedding request"),
        (503, "temporarily unavailable"),
        (700, "embedding generation failed"),
    ],
)
def test_sdk_api_failures_are_safe_chained_and_not_retried(
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
                "message": f"{SYNTHETIC_KEY}: {SENSITIVE_QUERY}",
            }
        },
    )
    client = FakeClient(error=original)

    with caplog.at_level(logging.INFO), pytest.raises(GeminiRequestError) as raised:
        embed_query(SENSITIVE_QUERY, settings, client=client)

    assert raised.value.__cause__ is original
    assert message in str(raised.value)
    assert len(client.models.calls) == 1
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output


def test_malformed_sdk_status_code_is_handled_safely(
    settings: EmbeddingSettings, caplog: pytest.LogCaptureFixture
) -> None:
    malformed_code = f"invalid-{SYNTHETIC_KEY}"
    original = errors.APIError(
        malformed_code,  # type: ignore[arg-type]
        {"error": {"code": malformed_code, "message": SENSITIVE_QUERY}},
    )
    client = FakeClient(error=original)

    with caplog.at_level(logging.INFO), pytest.raises(GeminiRequestError) as raised:
        embed_query(SENSITIVE_QUERY, settings, client=client)

    assert raised.value.__cause__ is original
    assert str(raised.value) == "Gemini embedding generation failed."
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output
    assert "status_code=unknown" in caplog.text


def test_api_key_invalid_reason_is_classified_as_authentication_failure(
    settings: EmbeddingSettings, caplog: pytest.LogCaptureFixture
) -> None:
    original = errors.APIError(
        400,
        {
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": f"invalid {SYNTHETIC_KEY} for {SENSITIVE_QUERY}",
                "details": [
                    {
                        "reason": "API_KEY_INVALID",
                        "metadata": {"key": SYNTHETIC_KEY},
                    }
                ],
            }
        },
    )
    client = FakeClient(error=original)

    with caplog.at_level(logging.INFO), pytest.raises(GeminiRequestError) as raised:
        embed_query(SENSITIVE_QUERY, settings, client=client)

    assert raised.value.__cause__ is original
    assert str(raised.value) == (
        "Gemini authentication failed. Check GEMINI_API_KEY."
    )
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output
    assert "category=authentication" in caplog.text


@pytest.mark.parametrize(
    "original",
    [
        httpx.ConnectError(f"connection exposed {SYNTHETIC_KEY} {SENSITIVE_QUERY}"),
        ValueError(f"protocol exposed {SYNTHETIC_KEY} {SENSITIVE_QUERY}"),
    ],
    ids=["transport", "protocol"],
)
def test_transport_and_protocol_failures_are_safe_chained_and_not_retried(
    original: Exception,
    settings: EmbeddingSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient(error=original)

    with caplog.at_level(logging.INFO), pytest.raises(GeminiRequestError) as raised:
        embed_query(SENSITIVE_QUERY, settings, client=client)

    assert raised.value.__cause__ is original
    assert str(raised.value) == (
        "Gemini embedding request failed. Check network connectivity."
    )
    assert len(client.models.calls) == 1
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output


def test_invalid_query_embedding_logs_only_safe_validation_metadata(
    settings: EmbeddingSettings, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(_response(_vector(dimension=767)))

    with caplog.at_level(logging.INFO), pytest.raises(EmbeddingDimensionError) as raised:
        embed_query(SENSITIVE_QUERY, settings, client=client)

    assert "category=dimension-mismatch" in caplog.text
    public_output = str(raised.value) + caplog.text
    assert SYNTHETIC_KEY not in public_output
    assert SENSITIVE_QUERY not in public_output
