"""Deterministic text cleaning shared by all document extractors."""

import re
import unicodedata

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_REMOVED_CATEGORIES = {"Cc", "Cf"}
_MEANINGFUL_FORMAT_CHARACTERS = {"\u200c", "\u200d"}


def clean_text(text: str) -> str:
    """Normalize extracted text without changing its meaningful content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"}
        or character in _MEANINGFUL_FORMAT_CHARACTERS
        or unicodedata.category(character) not in _REMOVED_CATEGORIES
    )

    cleaned_lines: list[str] = []
    previous_was_blank = False
    for line in normalized.split("\n"):
        cleaned_line = _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        if not cleaned_line:
            if cleaned_lines and not previous_was_blank:
                cleaned_lines.append("")
            previous_was_blank = True
            continue

        cleaned_lines.append(cleaned_line)
        previous_was_blank = False

    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)
