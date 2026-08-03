"""Shared document representations independent of parser libraries."""

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["PDF", "DOCX"]


@dataclass(frozen=True, slots=True)
class ExtractedTextUnit:
    """One ordered piece of cleaned text from a source document."""

    text: str
    position: int
    page_number: int | None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Cleaned source content ready for later chunk construction."""

    source_file: str
    source_type: SourceType
    units: tuple[ExtractedTextUnit, ...]
