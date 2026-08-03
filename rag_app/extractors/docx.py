"""Ordered DOCX body extraction using python-docx."""

from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import OpcError
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError

from rag_app.exceptions import DocumentExtractionError
from rag_app.processing.cleaning import clean_text

_CELL_SEPARATOR = " | "


def extract_docx(path: Path) -> list[tuple[str, None]]:
    """Extract paragraphs and table rows in their body order."""
    try:
        document = Document(path)
        extracted: list[tuple[str, None]] = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                extracted.append((block.text, None))
                continue
            if isinstance(block, Table):
                extracted.extend(_extract_table(block))
        return extracted
    except (
        OpcError,
        BadZipFile,
        XMLSyntaxError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        raise DocumentExtractionError(
            f"DOCX text extraction failed for '{path.name}'; "
            "the file may be malformed or unsupported"
        ) from error


def _extract_table(table: Table) -> list[tuple[str, None]]:
    rows: list[tuple[str, None]] = []
    for row in table.rows:
        cells = [_extract_cell(cell) for cell in row.cells]
        if any(cells):
            rows.append((_CELL_SEPARATOR.join(cells), None))
    return rows


def _extract_cell(cell: _Cell) -> str:
    parts: list[str] = []
    for block in cell.iter_inner_content():
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, Table):
            parts.extend(text for text, _ in _extract_table(block))
    return clean_text(" ".join(parts)).replace("\n", " ")
