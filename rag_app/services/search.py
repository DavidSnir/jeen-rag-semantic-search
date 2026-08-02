"""Semantic-search service boundary."""

from typing import NoReturn

from rag_app.exceptions import FeatureUnavailableError


def search_documents(query: str, strategy: str, top_k: int) -> NoReturn:
    """Search indexed content when semantic search is implemented."""
    raise FeatureUnavailableError(
        "Semantic search is not implemented in Stage 0 "
        f"for strategy '{strategy}' and top_k {top_k}"
    )
