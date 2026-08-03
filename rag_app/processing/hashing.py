"""Deterministic binary document hashing boundary."""

import hashlib
from pathlib import Path

from rag_app.exceptions import DocumentHashingError

_HASH_BLOCK_SIZE = 1024 * 1024


def calculate_document_hash(path: Path) -> str:
    """Return the SHA-256 digest of the exact source bytes."""
    filename = Path(path).name or "the supplied document"
    digest = hashlib.sha256()

    try:
        with Path(path).open("rb") as source:
            while block := source.read(_HASH_BLOCK_SIZE):
                digest.update(block)
    except OSError as error:
        raise DocumentHashingError(
            f"Document hash calculation failed for '{filename}'"
        ) from error

    return digest.hexdigest()


__all__ = ["calculate_document_hash"]
