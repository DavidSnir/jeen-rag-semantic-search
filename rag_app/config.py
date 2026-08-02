"""Runtime configuration loading and application constants."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict

from rag_app.exceptions import ConfigurationError

DEFAULT_TOP_K = 5
EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_LOG_LEVEL = "INFO"
DATABASE_CONNECT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    gemini_api_key: str = field(repr=False)
    postgres_url: str = field(repr=False)
    embedding_model: str
    embedding_dimension: int
    log_level: str


@dataclass(frozen=True)
class DatabaseSettings:
    """Validated settings needed for database-only operations."""

    postgres_url: str = field(repr=False)
    connect_timeout_seconds: int = DATABASE_CONNECT_TIMEOUT_SECONDS


def load_settings() -> Settings:
    """Load and validate settings without exposing secret values."""
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    postgres_url = os.getenv("POSTGRES_URL", "").strip()
    missing = [
        name
        for name, value in (
            ("GEMINI_API_KEY", gemini_api_key),
            ("POSTGRES_URL", postgres_url),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            f"Missing required configuration variable(s): {', '.join(missing)}"
        )

    _validate_postgres_url(postgres_url)

    embedding_dimension = _positive_integer_setting(
        "EMBEDDING_DIMENSION", EMBEDDING_DIMENSION
    )
    if embedding_dimension != EMBEDDING_DIMENSION:
        raise ConfigurationError(
            f"EMBEDDING_DIMENSION must be {EMBEDDING_DIMENSION}"
        )

    return Settings(
        gemini_api_key=gemini_api_key,
        postgres_url=postgres_url,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ).strip()
        or DEFAULT_EMBEDDING_MODEL,
        embedding_dimension=embedding_dimension,
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip()
        or DEFAULT_LOG_LEVEL,
    )


def load_database_settings() -> DatabaseSettings:
    """Load database settings without requiring unrelated service credentials."""
    load_dotenv()

    postgres_url = os.getenv("POSTGRES_URL", "").strip()
    if not postgres_url:
        raise ConfigurationError(
            "Missing required configuration variable: POSTGRES_URL"
        )

    _validate_postgres_url(postgres_url)
    return DatabaseSettings(postgres_url=postgres_url)


def _validate_postgres_url(postgres_url: str) -> None:
    try:
        conninfo_to_dict(postgres_url)
    except ProgrammingError as error:
        raise ConfigurationError(
            "POSTGRES_URL must be a valid PostgreSQL connection string"
        ) from error


def _positive_integer_setting(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed
