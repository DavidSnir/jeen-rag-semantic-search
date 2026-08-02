"""Document-indexing service boundary."""

from pathlib import Path
from typing import NoReturn

from rag_app.exceptions import FeatureUnavailableError


def index_document(file: Path, strategy: str) -> NoReturn:
    """Index a document when the indexing pipeline is implemented."""
    raise FeatureUnavailableError(
        f"Document indexing is not implemented in Stage 0 for file '{file.name}' "
        f"with strategy '{strategy}'"
    )


def reset_index() -> NoReturn:
    """Reset indexed content when database operations are implemented."""
    raise FeatureUnavailableError("Index reset is not implemented in Stage 0")
