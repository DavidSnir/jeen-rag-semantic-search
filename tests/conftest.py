"""Suite-wide test isolation from the Gemini service."""

from typing import NoReturn

import pytest


@pytest.fixture(autouse=True)
def block_real_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient credentials and real Gemini clients out of every test."""
    import rag_app.embeddings.gemini as gemini_module

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fail_client_creation(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Tests must not construct a real Gemini client")

    monkeypatch.setattr(gemini_module.genai, "Client", fail_client_creation)
