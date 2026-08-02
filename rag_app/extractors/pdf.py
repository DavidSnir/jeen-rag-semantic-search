"""Page-aware PDF text extraction using pypdf."""

from collections import Counter
from pathlib import Path
import re

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag_app.exceptions import DocumentExtractionError

_MAX_REGION_LINES = 2
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+")


def extract_pdf(path: Path) -> list[tuple[str, int]]:
    """Extract raw page text while preserving physical PDF page numbers."""
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise DocumentExtractionError(
                f"PDF file '{path.name}' is password-protected and is not supported"
            )
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except DocumentExtractionError:
        raise
    except (
        PyPdfError,
        OSError,
        ValueError,
        KeyError,
        EOFError,
        NotImplementedError,
    ) as error:
        raise DocumentExtractionError(
            f"PDF text extraction failed for '{path.name}'; "
            "the file may be malformed or unsupported"
        ) from error

    page_texts = _remove_repeated_margins(page_texts)
    return [(text, page_number) for page_number, text in enumerate(page_texts, 1)]


def _remove_repeated_margins(page_texts: list[str]) -> list[str]:
    """Remove recurring lines only from conservative page-edge regions."""
    if len(page_texts) < 2:
        return page_texts

    pages: list[tuple[list[str], list[int], list[int], int]] = []
    header_counts: Counter[str] = Counter()
    exact_footer_counts: Counter[str] = Counter()
    numbered_footer_counts: Counter[str] = Counter()

    for page_number, text in enumerate(page_texts, 1):
        lines = text.splitlines()
        header_indices, footer_indices = _region_indices(lines)
        pages.append((lines, header_indices, footer_indices, page_number))
        header_counts.update(
            {_comparison_key(lines[index]) for index in header_indices}
        )
        exact_footer_counts.update(
            {_comparison_key(lines[index]) for index in footer_indices}
        )
        numbered_footer_counts.update(
            {
                template
                for index in footer_indices
                if (template := _page_number_footer_template(
                    lines[index], page_number
                ))
                is not None
            }
        )

    minimum_occurrences = max(2, (len(page_texts) + 1) // 2)
    repeated_headers = {
        key for key, count in header_counts.items() if count >= minimum_occurrences
    }
    repeated_exact_footers = {
        key
        for key, count in exact_footer_counts.items()
        if count >= minimum_occurrences
    }
    repeated_numbered_footers = {
        key
        for key, count in numbered_footer_counts.items()
        if count >= minimum_occurrences
    }

    cleaned_pages: list[str] = []
    for lines, header_indices, footer_indices, page_number in pages:
        removed_indices = {
            index
            for index in header_indices
            if _comparison_key(lines[index]) in repeated_headers
        }
        removed_indices.update(
            index
            for index in footer_indices
            if _comparison_key(lines[index]) in repeated_exact_footers
            or _page_number_footer_template(lines[index], page_number)
            in repeated_numbered_footers
        )
        cleaned_pages.append(
            "\n".join(
                line for index, line in enumerate(lines) if index not in removed_indices
            )
        )

    return cleaned_pages


def _region_indices(lines: list[str]) -> tuple[list[int], list[int]]:
    nonempty_indices = [
        index for index, line in enumerate(lines) if _comparison_key(line)
    ]
    region_size = min(_MAX_REGION_LINES, (len(nonempty_indices) - 1) // 2)
    if region_size < 1:
        return [], []
    return nonempty_indices[:region_size], nonempty_indices[-region_size:]


def _comparison_key(line: str) -> str:
    return _WHITESPACE.sub(" ", line).strip()


def _page_number_footer_template(line: str, page_number: int) -> str | None:
    key = _comparison_key(line)
    for match in _NUMBER.finditer(key):
        if int(match.group()) == page_number:
            return f"{key[:match.start()]}<page-number>{key[match.end():]}"
    return None
