"""Unit tests for complete semantic-search service orchestration."""

import logging
import math
from dataclasses import FrozenInstanceError

import pytest

from rag_app.database.search import SemanticSearchRow
from rag_app.documents import ChunkingStrategy, SearchResponse
from rag_app.exceptions import (
    GeminiRequestError,
    SearchPipelineError,
    SearchValidationError,
)
from rag_app.services.search import search_documents

VECTOR = (1.0,) + (0.0,) * 767
SENSITIVE_QUERY = "private query text must not be logged"


def _row(
    *,
    content: object = "Complete chunk content",
    source_file: object = "report.pdf",
    source_type: object = "PDF",
    chunk_index: object = 4,
    strategy: object = "fixed",
    page_number: object = 7,
    distance: object = 0.25,
) -> SemanticSearchRow:
    return SemanticSearchRow(
        content,
        source_file,
        source_type,
        chunk_index,
        strategy,
        page_number,
        distance,
    )


def test_complete_stages_execute_in_order_and_build_immutable_response() -> None:
    events: list[object] = []

    def query_validator(value: object) -> str:
        events.append(("query", value))
        return "Canonical Query?!"

    def strategy_validator(value: object) -> ChunkingStrategy:
        events.append(("strategy", value))
        return ChunkingStrategy.fixed

    def top_k_validator(value: object) -> int:
        events.append(("top_k", value))
        return 2

    def readiness() -> None:
        events.append("readiness")

    def embed(query: str) -> tuple[float, ...]:
        events.append(("embed", query))
        return VECTOR

    rows = (
        _row(distance=0.1, source_file="first.pdf", chunk_index=0, page_number=1),
        _row(
            distance=0.6,
            source_file="second.docx",
            source_type="DOCX",
            chunk_index=9,
            page_number=None,
            content="Second complete chunk",
        ),
    )

    def repository(vector, strategy, top_k):
        events.append(("repository", vector, strategy, top_k))
        return rows

    response = search_documents(
        " raw query ",
        "fixed",
        9,
        query_validator=query_validator,
        strategy_validator=strategy_validator,
        top_k_validator=top_k_validator,
        readiness_checker=readiness,
        query_embedder=embed,
        repository_search=repository,
    )

    assert events == [
        ("query", " raw query "),
        ("strategy", "fixed"),
        ("top_k", 9),
        "readiness",
        ("embed", "Canonical Query?!"),
        ("repository", VECTOR, ChunkingStrategy.fixed, 2),
    ]
    assert isinstance(response, SearchResponse)
    assert response.query == "Canonical Query?!"
    assert response.chunking_strategy is ChunkingStrategy.fixed
    assert response.top_k == 2
    assert [match.rank for match in response.matches] == [1, 2]
    assert [match.distance for match in response.matches] == [0.1, 0.6]
    assert [match.score for match in response.matches] == pytest.approx([0.9, 0.4])
    assert response.matches[0].source_file == "first.pdf"
    assert response.matches[0].page_number == 1
    assert response.matches[1].source_file == "second.docx"
    assert response.matches[1].page_number is None
    assert response.matches[1].chunk_index == 9
    with pytest.raises(FrozenInstanceError):
        response.top_k = 7  # type: ignore[misc]


def test_argument_validation_runs_before_readiness_or_embedding() -> None:
    called: list[str] = []

    with pytest.raises(SearchValidationError):
        search_documents(
            "   ",
            "fixed",
            readiness_checker=lambda: called.append("readiness"),
            query_embedder=lambda query: called.append("embedding") or VECTOR,
        )

    assert called == []


def test_readiness_runs_before_embedding_and_failure_stops_search() -> None:
    called: list[str] = []

    def fail_readiness() -> None:
        called.append("readiness")
        raise SearchPipelineError("Database not ready.")

    with pytest.raises(SearchPipelineError, match="not ready"):
        search_documents(
            "query",
            "fixed",
            readiness_checker=fail_readiness,
            query_embedder=lambda query: called.append("embedding") or VECTOR,
            repository_search=lambda *args: called.append("repository") or (),
        )

    assert called == ["readiness"]


def test_embedding_failure_prevents_repository_search() -> None:
    repository_called = False

    def fail_embedding(query: str) -> tuple[float, ...]:
        raise GeminiRequestError("Gemini is unavailable")

    def repository(*args):
        nonlocal repository_called
        repository_called = True
        return ()

    with pytest.raises(GeminiRequestError):
        search_documents(
            "query",
            "sentence",
            readiness_checker=lambda: None,
            query_embedder=fail_embedding,
            repository_search=repository,
        )

    assert repository_called is False


def test_repository_failure_returns_no_successful_response() -> None:
    def fail_repository(*args):
        raise SearchPipelineError("Invalid repository state.")

    with pytest.raises(SearchPipelineError):
        search_documents(
            "query",
            "fixed",
            readiness_checker=lambda: None,
            query_embedder=lambda query: VECTOR,
            repository_search=fail_repository,
        )


def test_empty_repository_result_returns_successful_empty_response() -> None:
    response = search_documents(
        "  no results  ",
        ChunkingStrategy.paragraph,
        readiness_checker=lambda: None,
        query_embedder=lambda query: VECTOR,
        repository_search=lambda vector, strategy, top_k: (),
    )

    assert response.query == "no results"
    assert response.chunking_strategy is ChunkingStrategy.paragraph
    assert response.top_k == 5
    assert response.matches == ()


def test_service_prints_nothing_and_does_not_log_query(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        search_documents(
            SENSITIVE_QUERY,
            "fixed",
            readiness_checker=lambda: None,
            query_embedder=lambda query: VECTOR,
            repository_search=lambda vector, strategy, top_k: (),
        )

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert SENSITIVE_QUERY not in caplog.text
    assert "query_chars=" in caplog.text


@pytest.mark.parametrize(
    "invalid_vector",
    [
        [1.0] + [0.0] * 767,
        (1.0,),
        (1,) + (0.0,) * 767,
        (0.0,) * 768,
        (math.nan,) + (0.0,) * 767,
    ],
)
def test_invalid_embedding_stage_state_is_rejected(invalid_vector: object) -> None:
    repository_called = False

    def repository(*args):
        nonlocal repository_called
        repository_called = True
        return ()

    with pytest.raises(SearchPipelineError, match="query embedding stage"):
        search_documents(
            "query",
            "fixed",
            readiness_checker=lambda: None,
            query_embedder=lambda query: invalid_vector,  # type: ignore[return-value]
            repository_search=repository,
        )
    assert repository_called is False


@pytest.mark.parametrize(
    "rows",
    [
        [],
        (_row(),) * 6,
        (object(),),
        (_row(content="   "),),
        (_row(source_file="../report.pdf"),),
        (_row(source_file="spoof\nreport.pdf"),),
        (_row(source_file="\x1b[31mreport.pdf"),),
        (_row(source_file="report.docx"),),
        (_row(source_type="TXT"),),
        (_row(source_type=[]),),
        (_row(chunk_index=-1),),
        (_row(chunk_index=True),),
        (_row(strategy="sentence"),),
        (_row(page_number=0),),
        (_row(source_file="report.docx", source_type="DOCX", page_number=1),),
        (_row(distance=None),),
        (_row(distance=True),),
        (_row(distance=math.nan),),
        (_row(distance=-0.1),),
        (_row(distance=2.1),),
        (_row(distance=0.7), _row(distance=0.2)),
    ],
)
def test_invalid_repository_responses_are_rejected(rows: object) -> None:
    with pytest.raises(SearchPipelineError):
        search_documents(
            "query",
            "fixed",
            readiness_checker=lambda: None,
            query_embedder=lambda query: VECTOR,
            repository_search=lambda vector, strategy, top_k: rows,  # type: ignore[return-value]
        )


def test_distance_tolerance_is_not_clamped_and_score_is_exact() -> None:
    distance = -5e-10
    response = search_documents(
        "query",
        "fixed",
        readiness_checker=lambda: None,
        query_embedder=lambda query: VECTOR,
        repository_search=lambda vector, strategy, top_k: (_row(distance=distance),),
    )

    assert response.matches[0].distance == distance
    assert response.matches[0].score == 1.0 - distance
