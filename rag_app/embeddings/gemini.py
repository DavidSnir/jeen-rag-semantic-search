"""Synchronous Gemini document-embedding boundary."""

import logging
from collections.abc import Sequence
from pathlib import PureWindowsPath
from typing import Protocol

import httpx
from google import genai
from google.genai import errors, types

from rag_app.config import (
    EmbeddingSettings,
    load_embedding_settings,
    validate_embedding_settings,
)
from rag_app.documents import (
    Chunk,
    ChunkedDocument,
    ChunkingStrategy,
    EmbeddedChunk,
    EmbeddedDocument,
    EmbeddingVector,
)
from rag_app.embeddings.vectors import validate_and_normalize_embedding
from rag_app.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingNormalizationError,
    GeminiRequestError,
    InvalidEmbeddingInputError,
    InvalidGeminiResponseError,
)
from rag_app.processing.chunking import MAX_CHUNK_SIZE

logger = logging.getLogger(__name__)

_RETRIEVAL_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
_RETRIEVAL_QUERY_TASK = "RETRIEVAL_QUERY"
_MISSING = object()


class _EmbeddingModels(Protocol):
    def embed_content(
        self, *, model: str, contents: list[str], config: types.EmbedContentConfig
    ) -> object: ...


class _GeminiClient(Protocol):
    models: _EmbeddingModels


def embed_document(
    document: ChunkedDocument,
    settings: EmbeddingSettings | None = None,
    *,
    client: _GeminiClient | None = None,
) -> EmbeddedDocument:
    """Embed every ordered chunk in one request and return no partial result."""
    title = _validate_document(document)
    validated_settings = validate_embedding_settings(
        settings if settings is not None else load_embedding_settings()
    )
    contents = [chunk.content for chunk in document.chunks]

    if client is not None:
        return _request_embeddings(
            client, document, validated_settings, contents, title
        )

    owned_client = _create_client(validated_settings)

    try:
        return _request_embeddings(
            owned_client, document, validated_settings, contents, title
        )
    finally:
        _close_client(owned_client, validated_settings)


def embed_query(
    query: str,
    settings: EmbeddingSettings | None = None,
    *,
    client: _GeminiClient | None = None,
) -> EmbeddingVector:
    """Embed one canonical search query using retrieval-query semantics."""
    canonical_query = _validate_query(query)
    validated_settings = validate_embedding_settings(
        settings if settings is not None else load_embedding_settings()
    )

    if client is not None:
        return _request_query_embedding(
            client, canonical_query, validated_settings
        )

    owned_client = _create_client(validated_settings)
    try:
        return _request_query_embedding(
            owned_client, canonical_query, validated_settings
        )
    finally:
        _close_client(owned_client, validated_settings)


def _create_client(settings: EmbeddingSettings) -> _GeminiClient:
    try:
        return genai.Client(api_key=settings.gemini_api_key, vertexai=False)
    except (errors.APIError, ValueError, httpx.TransportError) as error:
        raise EmbeddingConfigurationError(
            "The Gemini embedding client could not be configured"
        ) from error


def _close_client(client: _GeminiClient, settings: EmbeddingSettings) -> None:
    try:
        close = getattr(client, "close")
        close()
    except (errors.APIError, ValueError, httpx.TransportError):
        logger.warning(
            "Gemini embedding client cleanup failed model=%s",
            settings.embedding_model,
        )


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise InvalidEmbeddingInputError("Embedding query must be a string")
    canonical_query = query.strip()
    if not canonical_query:
        raise InvalidEmbeddingInputError("Embedding query must not be empty")
    return canonical_query


def _validate_document(document: ChunkedDocument) -> str:
    if not isinstance(document, ChunkedDocument):
        raise InvalidEmbeddingInputError(
            "Embedding input must be a ChunkedDocument"
        )
    if document.source_type not in {"PDF", "DOCX"}:
        raise InvalidEmbeddingInputError(
            "Embedding input has an unsupported source type"
        )
    if not isinstance(document.chunking_strategy, ChunkingStrategy):
        raise InvalidEmbeddingInputError(
            "Embedding input has an unsupported chunking strategy"
        )
    if (
        not isinstance(document.source_file, str)
        or not document.source_file.strip()
    ):
        raise InvalidEmbeddingInputError("Embedding input must have a source filename")
    title = PureWindowsPath(document.source_file).name
    if not title:
        raise InvalidEmbeddingInputError("Embedding input must have a source filename")
    if not isinstance(document.chunks, tuple):
        raise InvalidEmbeddingInputError(
            "Chunked document chunks must be an ordered immutable tuple"
        )
    if not document.chunks:
        raise InvalidEmbeddingInputError(
            "Chunked document must contain at least one chunk"
        )

    for expected_index, chunk in enumerate(document.chunks):
        _validate_chunk(chunk, expected_index, document.source_type)
    return title


def _validate_chunk(chunk: Chunk, expected_index: int, source_type: str) -> None:
    if not isinstance(chunk, Chunk):
        raise InvalidEmbeddingInputError(
            f"Embedding input contains an invalid chunk at index {expected_index}"
        )
    if (
        not isinstance(chunk.chunk_index, int)
        or isinstance(chunk.chunk_index, bool)
        or chunk.chunk_index != expected_index
    ):
        raise InvalidEmbeddingInputError(
            "Chunk indexes must be zero-based, continuous, and in order"
        )
    if not isinstance(chunk.content, str) or not chunk.content.strip():
        raise InvalidEmbeddingInputError(
            f"Chunk {expected_index} must contain meaningful text"
        )
    if len(chunk.content) > MAX_CHUNK_SIZE:
        raise InvalidEmbeddingInputError(
            f"Chunk {expected_index} exceeds the {MAX_CHUNK_SIZE}-character limit"
        )

    if source_type == "PDF":
        if (
            not isinstance(chunk.page_number, int)
            or isinstance(chunk.page_number, bool)
            or chunk.page_number < 1
        ):
            raise InvalidEmbeddingInputError(
                f"PDF chunk {expected_index} must have a positive page number"
            )
    elif chunk.page_number is not None:
        raise InvalidEmbeddingInputError(
            f"DOCX chunk {expected_index} must not have a page number"
        )


def _request_embeddings(
    client: _GeminiClient,
    document: ChunkedDocument,
    settings: EmbeddingSettings,
    contents: list[str],
    title: str,
) -> EmbeddedDocument:
    logger.info(
        "Requesting Gemini document embeddings model=%s dimension=%d chunks=%d "
        "source_file=%s strategy=%s",
        settings.embedding_model,
        settings.embedding_dimension,
        len(contents),
        title,
        document.chunking_strategy.value,
    )
    request_config = types.EmbedContentConfig(
        task_type=_RETRIEVAL_DOCUMENT_TASK,
        title=title,
        output_dimensionality=settings.embedding_dimension,
    )

    normalized_vectors = _request_vectors(
        client,
        settings,
        contents,
        request_config,
    )

    embedded_chunks = tuple(
        EmbeddedChunk(chunk=document.chunks[index], embedding=vector)
        for index, vector in enumerate(normalized_vectors)
    )

    logger.info(
        "Gemini document embeddings validated model=%s vectors=%d dimension=%d",
        settings.embedding_model,
        len(embedded_chunks),
        settings.embedding_dimension,
    )
    return EmbeddedDocument(
        source_file=document.source_file,
        source_type=document.source_type,
        chunking_strategy=document.chunking_strategy,
        chunks=embedded_chunks,
    )


def _request_query_embedding(
    client: _GeminiClient,
    query: str,
    settings: EmbeddingSettings,
) -> EmbeddingVector:
    logger.info(
        "Requesting Gemini query embedding model=%s dimension=%d query_chars=%d",
        settings.embedding_model,
        settings.embedding_dimension,
        len(query),
    )
    request_config = types.EmbedContentConfig(
        task_type=_RETRIEVAL_QUERY_TASK,
        output_dimensionality=settings.embedding_dimension,
    )
    vectors = _request_vectors(
        client,
        settings,
        [query],
        request_config,
        value_label="query",
    )
    logger.info(
        "Gemini query embedding validated model=%s vectors=1 dimension=%d",
        settings.embedding_model,
        settings.embedding_dimension,
    )
    return vectors[0]


def _request_vectors(
    client: _GeminiClient,
    settings: EmbeddingSettings,
    contents: list[str],
    request_config: types.EmbedContentConfig,
    *,
    value_label: str | None = None,
) -> tuple[EmbeddingVector, ...]:

    try:
        response = client.models.embed_content(
            model=settings.embedding_model,
            contents=contents,
            config=request_config,
        )
    except errors.APIError as error:
        status_code = _normalize_status_code(error.code)
        authentication_failed = _is_api_key_error(error)
        category = (
            "authentication"
            if authentication_failed
            else _api_error_category(status_code)
        )
        logger.warning(
            "Gemini embedding request failed category=%s status_code=%s "
            "model=%s chunks=%d",
            category,
            status_code if status_code is not None else "unknown",
            settings.embedding_model,
            len(contents),
        )
        message_status = 401 if authentication_failed else status_code
        raise GeminiRequestError(_api_error_message(message_status)) from error
    except (ValueError, httpx.TransportError) as error:
        logger.warning(
            "Gemini embedding request failed category=transport-or-protocol "
            "model=%s chunks=%d",
            settings.embedding_model,
            len(contents),
        )
        raise GeminiRequestError(
            "Gemini embedding request failed. Check network connectivity."
        ) from error

    try:
        embeddings = _response_embeddings(response, len(contents))
        normalized_vectors = tuple(
            validate_and_normalize_embedding(
                getattr(embedding, "values", _MISSING),
                chunk_index=chunk_index,
                expected_dimension=settings.embedding_dimension,
                value_label=value_label,
            )
            for chunk_index, embedding in enumerate(embeddings)
        )
    except EmbeddingDimensionError:
        _log_validation_failure(settings, len(contents), "dimension-mismatch")
        raise
    except EmbeddingNormalizationError:
        _log_validation_failure(settings, len(contents), "normalization")
        raise
    except InvalidGeminiResponseError:
        _log_validation_failure(settings, len(contents), "invalid-response")
        raise
    return normalized_vectors


def _response_embeddings(response: object, expected_count: int) -> Sequence[object]:
    if response is None:
        raise InvalidGeminiResponseError("Gemini returned no embedding response")
    embeddings = getattr(response, "embeddings", _MISSING)
    if embeddings is _MISSING:
        raise InvalidGeminiResponseError(
            "Gemini response does not contain embeddings"
        )
    if embeddings is None:
        raise InvalidGeminiResponseError("Gemini returned null embeddings")
    if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence):
        raise InvalidGeminiResponseError(
            "Gemini returned an invalid embeddings collection"
        )
    if not embeddings:
        raise InvalidGeminiResponseError("Gemini returned no embeddings")
    if len(embeddings) != expected_count:
        raise InvalidGeminiResponseError(
            f"Gemini returned {len(embeddings)} embeddings; expected {expected_count}"
        )
    return embeddings


def _log_validation_failure(
    settings: EmbeddingSettings, chunk_count: int, category: str
) -> None:
    logger.warning(
        "Gemini embedding validation failed category=%s model=%s chunks=%d",
        category,
        settings.embedding_model,
        chunk_count,
    )


def _normalize_status_code(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _is_api_key_error(error: errors.APIError) -> bool:
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return False
    error_details = details.get("error", details)
    if not isinstance(error_details, dict):
        return False
    provider_details = error_details.get("details", ())
    if not isinstance(provider_details, list):
        return False
    return any(
        isinstance(detail, dict) and detail.get("reason") == "API_KEY_INVALID"
        for detail in provider_details
    )


def _api_error_category(status_code: int | None) -> str:
    if status_code is None:
        return "provider-failure"
    if status_code == 401:
        return "authentication"
    if status_code == 403:
        return "access-denied"
    if status_code == 404:
        return "model-not-found"
    if status_code == 429:
        return "quota-or-rate-limit"
    if 400 <= status_code < 500:
        return "rejected-request"
    if 500 <= status_code < 600:
        return "temporarily-unavailable"
    return "provider-failure"


def _api_error_message(status_code: int | None) -> str:
    messages = {
        401: "Gemini authentication failed. Check GEMINI_API_KEY.",
        403: "Gemini permission was denied. Check API access for GEMINI_API_KEY.",
        404: "The configured Gemini embedding model was not found.",
        429: "Gemini quota or rate limit was exceeded.",
    }
    if status_code in messages:
        return messages[status_code]
    if status_code is None:
        return "Gemini embedding generation failed."
    if 400 <= status_code < 500:
        return "Gemini rejected the embedding request."
    if 500 <= status_code < 600:
        return "Gemini is temporarily unavailable."
    return "Gemini embedding generation failed."


__all__ = ["embed_document", "embed_query"]
