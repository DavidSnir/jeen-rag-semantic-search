"""Application-specific exceptions safe for presentation by the CLI."""


class RagAppError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(RagAppError):
    """Raised when required runtime configuration is invalid or missing."""


class FeatureUnavailableError(RagAppError):
    """Raised when a later-stage application feature is invoked."""


class DatabaseError(RagAppError):
    """Base class for expected database failures."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a PostgreSQL connection cannot be established."""


class DatabaseSchemaError(DatabaseError):
    """Raised when database schema initialization or validation fails."""


class DatabaseOperationError(DatabaseError):
    """Raised when an established connection cannot complete an operation."""
