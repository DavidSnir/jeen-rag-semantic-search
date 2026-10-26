"""Regression checks for approved Stage 0 application constants."""

from rag_app.config import DEFAULT_TOP_K, EMBEDDING_DIMENSION


def test_approved_application_constants() -> None:
    assert DEFAULT_TOP_K == 5
    assert EMBEDDING_DIMENSION == 768
