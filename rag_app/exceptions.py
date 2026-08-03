"""Application-specific exceptions safe for presentation by the CLI."""


class RagAppError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(RagAppError):
    """Raised when required runtime configuration is invalid or missing."""


class EmbeddingConfigurationError(ConfigurationError):
    """Raised when Gemini embedding configuration is invalid or missing."""


class FeatureUnavailableError(RagAppError):
    """Raised when a later-stage application feature is invoked."""


class DocumentError(RagAppError):
    """Base class for expected document-ingestion failures."""


class InvalidDocumentPathError(DocumentError):
    """Raised when a document path is missing, invalid, or unreadable."""


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when a document does not have a supported extension."""


class DocumentExtractionError(DocumentError):
    """Raised when a supported document cannot be parsed safely."""


class EmptyDocumentError(DocumentError):
    """Raised when extraction and cleaning produce no meaningful text."""


class ChunkingError(RagAppError):
    """Base class for expected document-chunking failures."""


class InvalidChunkingStrategyError(ChunkingError):
    """Raised when a requested chunking strategy is not supported."""


class InvalidChunkingInputError(ChunkingError):
    """Raised when an extracted document cannot be chunked safely."""


class ChunkGenerationError(ChunkingError):
    """Raised when chunking cannot produce usable bounded content."""


class EmbeddingError(RagAppError):
    """Base class for expected embedding failures."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when a chunked document cannot be embedded safely."""


class GeminiRequestError(EmbeddingError):
    """Raised when Gemini cannot complete an embedding request."""


class InvalidGeminiResponseError(EmbeddingError):
    """Raised when Gemini returns missing or invalid embedding data."""


class EmbeddingDimensionError(InvalidGeminiResponseError):
    """Raised when a returned embedding has an unexpected dimension."""


class EmbeddingNormalizationError(EmbeddingError):
    """Raised when an embedding cannot be normalized to unit length."""


class DatabaseError(RagAppError):
    """Base class for expected database failures."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a PostgreSQL connection cannot be established."""


class DatabaseSchemaError(DatabaseError):
    """Raised when database schema initialization or validation fails."""


class DatabaseOperationError(DatabaseError):
    """Raised when an established connection cannot complete an operation."""
