"""Deterministic Gemini test doubles and embedding response builders."""

from types import SimpleNamespace

from google.genai import types

from rag_app.config import EMBEDDING_DIMENSION


class FakeModels:
    """Record embedding requests and return or raise a configured result."""

    def __init__(
        self,
        response: object = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def embed_content(
        self,
        *,
        model: str,
        contents: list[str],
        config: types.EmbedContentConfig,
    ) -> object:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    """Minimal injectable client implementing the embedding client protocol."""

    def __init__(
        self,
        response: object = None,
        error: Exception | None = None,
    ) -> None:
        self.models = FakeModels(response=response, error=error)


class OwnedFakeClient(FakeClient):
    """Fake client that records and can fail application-owned cleanup."""

    def __init__(
        self,
        response: object = None,
        error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        super().__init__(response=response, error=error)
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def embedding_vector(
    first: float = 1.0,
    second: float = 0.0,
    *,
    dimension: int = EMBEDDING_DIMENSION,
) -> list[float]:
    """Build a deterministic vector of any non-negative test dimension."""
    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    values = [0.0] * dimension
    if dimension > 0:
        values[0] = first
    if dimension > 1:
        values[1] = second
    return values


def embedding_response(*vectors: object) -> SimpleNamespace:
    """Build the response shape returned by Gemini's embedding API."""
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


__all__ = [
    "FakeClient",
    "FakeModels",
    "OwnedFakeClient",
    "embedding_response",
    "embedding_vector",
]
