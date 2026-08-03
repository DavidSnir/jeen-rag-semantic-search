"""Unit tests for embedding-only configuration and secret handling."""

import pytest

import rag_app.config as config_module
from rag_app.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EmbeddingSettings,
    load_database_settings,
    load_embedding_settings,
)
from rag_app.exceptions import EmbeddingConfigurationError

SYNTHETIC_KEY = "unit-test-placeholder"


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "load_dotenv", lambda: False)
    for name in (
        "GEMINI_API_KEY",
        "POSTGRES_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_embedding_settings_load_key_without_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", f"  {SYNTHETIC_KEY}  ")

    settings = load_embedding_settings()

    assert settings.gemini_api_key == SYNTHETIC_KEY
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimension == EMBEDDING_DIMENSION


@pytest.mark.parametrize("value", [None, "", "   \t "])
def test_embedding_settings_reject_missing_or_blank_key(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is not None:
        monkeypatch.setenv("GEMINI_API_KEY", value)

    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_API_KEY"):
        load_embedding_settings()


def test_embedding_settings_accept_approved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    assert load_embedding_settings().embedding_model == DEFAULT_EMBEDDING_MODEL


def test_embedding_settings_reject_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-004")

    with pytest.raises(EmbeddingConfigurationError, match=DEFAULT_EMBEDDING_MODEL):
        load_embedding_settings()


@pytest.mark.parametrize("value", ["not-an-integer", "0", "-1", "767", "769"])
def test_embedding_settings_reject_invalid_dimension(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("EMBEDDING_DIMENSION", value)

    with pytest.raises(EmbeddingConfigurationError, match="EMBEDDING_DIMENSION"):
        load_embedding_settings()


def test_embedding_settings_accept_exact_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SYNTHETIC_KEY)
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    assert load_embedding_settings().embedding_dimension == 768


def test_embedding_settings_repr_excludes_key() -> None:
    settings = EmbeddingSettings(gemini_api_key=SYNTHETIC_KEY)

    assert SYNTHETIC_KEY not in repr(settings)
    assert "gemini_api_key" not in repr(settings)


def test_database_settings_do_not_require_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgresql://localhost/rag_app")

    settings = load_database_settings()

    assert settings.postgres_url == "postgresql://localhost/rag_app"
