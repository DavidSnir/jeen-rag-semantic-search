"""Public embedding boundary and immutable result representations."""

from rag_app.documents import EmbeddedChunk, EmbeddedDocument, EmbeddingVector
from rag_app.embeddings.gemini import embed_document, embed_query

__all__ = [
    "EmbeddedChunk",
    "EmbeddedDocument",
    "EmbeddingVector",
    "embed_document",
    "embed_query",
]
