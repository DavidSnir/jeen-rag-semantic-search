"""Reusable test support objects."""

from tests.support.gemini import (
    FakeClient,
    FakeModels,
    OwnedFakeClient,
    embedding_response,
    embedding_vector,
)

__all__ = [
    "FakeClient",
    "FakeModels",
    "OwnedFakeClient",
    "embedding_response",
    "embedding_vector",
]
