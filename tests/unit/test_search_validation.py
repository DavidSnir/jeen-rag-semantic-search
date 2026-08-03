"""Unit tests for application-level semantic-search validation."""

import pytest

from rag_app.config import DEFAULT_TOP_K
from rag_app.documents import ChunkingStrategy
from rag_app.exceptions import InvalidChunkingStrategyError, SearchValidationError
from rag_app.processing.chunking import validate_chunking_strategy
from rag_app.services.search import validate_search_query, validate_top_k


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("semantic search", "semantic search"),
        ("  trimmed query\n", "trimmed query"),
        ("x", "x"),
        ("MiXeD Case", "MiXeD Case"),
        ("What is RAG?!", "What is RAG?!"),
        ("internal   whitespace\tstays", "internal   whitespace\tstays"),
        ("東京 café", "東京 café"),
    ],
)
def test_query_validation_preserves_meaningful_text(
    query: str, expected: str
) -> None:
    assert validate_search_query(query) == expected


@pytest.mark.parametrize("query", [None, 1, True, b"query", "", " \t\r\n "])
def test_invalid_queries_raise_clear_search_validation_error(query: object) -> None:
    with pytest.raises(SearchValidationError):
        validate_search_query(query)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_shared_strategy_validator_returns_canonical_enum(
    strategy: ChunkingStrategy,
) -> None:
    assert validate_chunking_strategy(strategy.value) is strategy


def test_shared_strategy_validator_rejects_unsupported_alias() -> None:
    with pytest.raises(InvalidChunkingStrategyError):
        validate_chunking_strategy("windows")


def test_default_top_k_is_five() -> None:
    assert DEFAULT_TOP_K == 5
    assert validate_top_k(DEFAULT_TOP_K) == 5


@pytest.mark.parametrize("top_k", [1, 2, 50, 10_000])
def test_positive_integer_top_k_is_accepted(top_k: int) -> None:
    assert validate_top_k(top_k) == top_k


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.0, "5", None])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(SearchValidationError, match="positive integer"):
        validate_top_k(top_k)
