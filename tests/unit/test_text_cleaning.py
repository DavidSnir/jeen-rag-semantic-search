"""Tests for deterministic shared text cleaning."""

import pytest

from rag_app.processing.cleaning import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("alpha   beta", "alpha beta"),
        ("alpha\t\t beta", "alpha beta"),
        ("alpha\r\nbeta\rgamma", "alpha\nbeta\ngamma"),
        ("alpha\n\n\n\n beta", "alpha\n\nbeta"),
        ("\n\n alpha \n\n", "alpha"),
        ("  alpha  \n\t beta \t", "alpha\nbeta"),
        ("alpha\x00\x07beta", "alphabeta"),
        ("alpha\u200bbeta\u2060gamma", "alphabetagamma"),
        ("Hello, world! (yes): 100%.", "Hello, world! (yes): 100%."),
        ("café Ελληνικά 日本語", "café Ελληνικά 日本語"),
        ("x² + y² = 25; π ≈ 3.14", "x² + y² = 25; π ≈ 3.14"),
    ],
)
def test_clean_text_applies_approved_rules(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_clean_text_is_idempotent() -> None:
    cleaned = clean_text("  First\t line\r\n\r\n\r\nSecond\u200b line  ")

    assert clean_text(cleaned) == cleaned


def test_clean_text_preserves_meaningful_line_and_paragraph_separation() -> None:
    assert clean_text("first line\nsecond line\n\nnext paragraph") == (
        "first line\nsecond line\n\nnext paragraph"
    )


def test_clean_text_preserves_meaningful_unicode_joiners() -> None:
    assert clean_text("👩\u200d💻") == "👩\u200d💻"
