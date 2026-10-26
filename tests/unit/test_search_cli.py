"""CLI tests for ranked semantic-search output and exit semantics."""

import pytest
from typer.testing import CliRunner

import rag_app.cli as cli_module
from rag_app.documents import (
    ChunkingStrategy,
    SearchMatch,
    SearchResponse,
)
from rag_app.exceptions import GeminiRequestError


def _response(
    strategy: ChunkingStrategy = ChunkingStrategy.fixed,
    top_k: int = 5,
    *,
    matches: tuple[SearchMatch, ...] | None = None,
) -> SearchResponse:
    if matches is None:
        matches = (
            SearchMatch(
                rank=1,
                content="Complete PDF chunk content.\nSecond line remains intact.",
                source_file="paper.pdf",
                source_type="PDF",
                chunk_index=4,
                chunking_strategy=strategy,
                page_number=7,
                distance=0.123456,
                score=0.876544,
            ),
            SearchMatch(
                rank=2,
                content="Complete DOCX chunk content.",
                source_file="notes.docx",
                source_type="DOCX",
                chunk_index=0,
                chunking_strategy=strategy,
                page_number=None,
                distance=0.5,
                score=0.5,
            ),
        )
    return SearchResponse("query", strategy, top_k, matches)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_ranked_output_default_top_k_and_every_strategy(
    monkeypatch: pytest.MonkeyPatch, strategy: ChunkingStrategy
) -> None:
    received: list[tuple[str, str, int]] = []

    def search(query: str, selected_strategy: str, top_k: int) -> SearchResponse:
        received.append((query, selected_strategy, top_k))
        return _response(strategy, top_k)

    monkeypatch.setattr(cli_module, "search_documents", search)
    result = CliRunner().invoke(
        cli_module.app,
        ["search", "--query", "  Mixed Query?!  ", "--strategy", strategy.value],
    )

    assert result.exit_code == 0
    assert received == [("Mixed Query?!", strategy.value, 5)]
    assert "Result 1 | score=0.8765" in result.output
    assert "source=paper.pdf | type=PDF" in result.output
    assert f"strategy={strategy.value} | chunk=4 | page=7" in result.output
    assert "Complete PDF chunk content.\nSecond line remains intact." in result.output
    assert "Result 2 | score=0.5000" in result.output
    assert "source=notes.docx | type=DOCX" in result.output
    assert "chunk=0 | page=n/a" in result.output
    assert "Complete DOCX chunk content." in result.output
    assert "embedding" not in result.output.lower()
    assert "document_hash" not in result.output
    assert "postgresql://" not in result.output


def test_explicit_top_k_is_passed_to_service(monkeypatch: pytest.MonkeyPatch) -> None:
    received_top_k = 0

    def search(query: str, strategy: str, top_k: int) -> SearchResponse:
        nonlocal received_top_k
        received_top_k = top_k
        return _response(top_k=top_k, matches=())

    monkeypatch.setattr(cli_module, "search_documents", search)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "search",
            "--query",
            "query",
            "--strategy",
            "fixed",
            "--top-k",
            "12",
        ],
    )

    assert result.exit_code == 0
    assert received_top_k == 12


def test_empty_results_are_successful_and_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "search_documents",
        lambda query, strategy, top_k: _response(
            ChunkingStrategy.sentence, top_k, matches=()
        ),
    )
    result = CliRunner().invoke(
        cli_module.app,
        ["search", "--query", "query", "--strategy", "sentence"],
    )

    assert result.exit_code == 0
    assert result.output == "No indexed results found for strategy=sentence.\n"


def test_expected_runtime_failure_uses_exit_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(query: str, strategy: str, top_k: int) -> None:
        raise GeminiRequestError("Gemini authentication failed")

    monkeypatch.setattr(cli_module, "search_documents", fail)
    result = CliRunner().invoke(
        cli_module.app,
        ["search", "--query", "query", "--strategy", "paragraph"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: Gemini authentication failed\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["search", "--strategy", "fixed"],
        ["search", "--query", "", "--strategy", "fixed"],
        ["search", "--query", "   ", "--strategy", "fixed"],
        ["search", "--query", "query", "--strategy", "invalid"],
        ["search", "--query", "query", "--strategy", "fixed", "--top-k", "0"],
        ["search", "--query", "query", "--strategy", "fixed", "--top-k", "x"],
    ],
)
def test_usage_failures_keep_exit_two(arguments: list[str]) -> None:
    result = CliRunner().invoke(cli_module.app, arguments)
    assert result.exit_code == 2
