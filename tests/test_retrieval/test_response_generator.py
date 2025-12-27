"""Tests for response generator (Task 4.6) with numbered citations."""

from unittest.mock import patch

import pytest

from src.retrieval.models import HybridChunk, HybridRetrievalResult, RetrievalStrategy
from src.retrieval.response_generator import ResponseGenerator
from src.utils.config import Config


@pytest.fixture
def config() -> Config:
    return Config.from_yaml()


@pytest.fixture
def response_generator(config: Config) -> ResponseGenerator:
    return ResponseGenerator(config=config)


@pytest.fixture
def sample_retrieval_result() -> HybridRetrievalResult:
    chunks = [
        HybridChunk(
            chunk_id="chunk_1",
            document_id="doc_alpha",
            content="Starship is a fully reusable spacecraft.",
            level=3,
            final_score=0.9,
            rank=1,
            source="vector",
            metadata={"filename": "starship_specs.pdf"},
        ),
        HybridChunk(
            chunk_id="chunk_2",
            document_id="doc_beta",
            content="The Super Heavy booster is the first stage.",
            level=3,
            final_score=0.8,
            rank=2,
            source="vector",
            metadata={"filename": "booster_manual.pdf"},
        ),
    ]
    return HybridRetrievalResult(
        query_id="query_1",
        query_text="What is Starship?",
        strategy_used=RetrievalStrategy.VECTOR_ONLY,
        chunks=chunks,
        total_results=2,
        retrieval_time_ms=100.0,
    )


def test_format_chunks(response_generator, sample_retrieval_result):
    """Test that chunks are formatted with numbered headers."""
    formatted = response_generator._format_chunks(sample_retrieval_result.chunks)
    assert "--- Chunk [1] (ID: chunk_1, Doc: doc_alpha) ---" in formatted
    assert "--- Chunk [2] (ID: chunk_2, Doc: doc_beta) ---" in formatted
    assert "Starship is a fully reusable spacecraft." in formatted


@patch("src.retrieval.response_generator.ResponseGenerator._call_llm")
def test_generate_with_numbered_citations(
    mock_call_llm, response_generator, sample_retrieval_result
):
    """Test that citations [1], [2] are correctly extracted and Sources section appended."""
    # Mock LLM response with numbered citations
    mock_call_llm.return_value = "Starship is reusable [1]. It uses the Super Heavy booster [2]."

    response = response_generator.generate("What is Starship?", sample_retrieval_result)

    assert "[1]" in response.answer
    assert "[2]" in response.answer
    assert "### Sources" in response.answer
    assert "1. starship_specs.pdf (Chunk: chunk_1)" in response.answer
    assert "2. booster_manual.pdf (Chunk: chunk_2)" in response.answer
    assert response.chunks_used == ["chunk_1", "chunk_2"]


@patch("src.retrieval.response_generator.ResponseGenerator._call_llm")
def test_generate_with_partial_citations(
    mock_call_llm, response_generator, sample_retrieval_result
):
    """Test when only some chunks are cited."""
    mock_call_llm.return_value = "Starship is reusable [1]."

    response = response_generator.generate("What is Starship?", sample_retrieval_result)

    assert "[1]" in response.answer
    assert "[2]" not in response.answer
    assert "### Sources" in response.answer
    assert "1. starship_specs.pdf (Chunk: chunk_1)" in response.answer
    assert "2. booster_manual.pdf" not in response.answer
    assert response.chunks_used == ["chunk_1"]


@patch("src.retrieval.response_generator.ResponseGenerator._call_llm")
def test_generate_with_document_title(mock_call_llm, response_generator, sample_retrieval_result):
    """Test that document_title is preferred over filename/ID."""
    sample_retrieval_result.chunks[0].metadata["document_title"] = "Starship System Overview"
    mock_call_llm.return_value = "Starship is reusable [1]."

    response = response_generator.generate("What is Starship?", sample_retrieval_result)

    assert "1. Starship System Overview (Chunk: chunk_1)" in response.answer


@patch("src.retrieval.response_generator.ResponseGenerator._call_llm")
def test_generate_no_citations(mock_call_llm, response_generator, sample_retrieval_result):
    """Test when no citations are found in the answer."""
    mock_call_llm.return_value = "I don't know."

    response = response_generator.generate("What is Starship?", sample_retrieval_result)

    assert "### Sources" not in response.answer
    assert response.chunks_used == []
