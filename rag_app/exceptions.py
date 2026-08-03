"""Application-specific exceptions safe for presentation by the CLI."""


class RagAppError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(RagAppError):
    """Raised when required runtime configuration is invalid or missing."""


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


class DatabaseError(RagAppError):
    """Base class for expected database failures."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a PostgreSQL connection cannot be established."""


class DatabaseSchemaError(DatabaseError):
    """Raised when database schema initialization or validation fails."""


class DatabaseOperationError(DatabaseError):
    """Raised when an established connection cannot complete an operation."""
