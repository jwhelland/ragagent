"""Tests for ResearchAgent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.models import (
    GeneratedResponse,
    HybridChunk,
    HybridRetrievalResult,
    RetrievalStrategy,
)
from src.retrieval.query_parser import ParsedQuery, QueryIntent
from src.retrieval.research_agent import ResearchAgent


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.llm.resolve.return_value = MagicMock(
        provider="openai", model="gpt-4-test", retry_attempts=1
    )
    return config


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    return retriever


@pytest.fixture
def mock_response_generator():
    generator = MagicMock()
    generator.prompts = {
        "sufficiency_check": {
            "system": "sys",
            "user_template": "user {query_text} {context_summary}",
        },
        "sub_query_generation": {
            "system": "sys",
            "user_template": "user {query_text} {missing_info_list}",
        },
        "synthesis": {"system": "sys", "user_template": "user"},
    }
    return generator


@pytest.fixture
def mock_query_parser():
    parser = MagicMock()
    parser.parse.return_value = ParsedQuery(
        query_id="test",
        original_text="test query",
        normalized_text="test query",
        intent=QueryIntent.SEMANTIC,
        intent_confidence=1.0,
    )
    return parser


@pytest.fixture
def research_agent(mock_config, mock_retriever, mock_response_generator, mock_query_parser):
    return ResearchAgent(
        config=mock_config,
        retriever=mock_retriever,
        response_generator=mock_response_generator,
        query_parser=mock_query_parser,
    )


def test_research_initial_sufficiency(research_agent, mock_retriever):
    """Test research when initial information is sufficient."""
    # Mock retrieval result
    chunk = HybridChunk(
        chunk_id="c1",
        document_id="d1",
        content="content",
        level=1,
        final_score=0.9,
        rank=1,
        source="vector",
    )
    mock_retriever.retrieve.return_value = HybridRetrievalResult(
        query_id="q1",
        query_text="test",
        strategy_used=RetrievalStrategy.HYBRID_PARALLEL,
        chunks=[chunk],
        total_results=1,
        retrieval_time_ms=10,
    )

    # Mock LLM calls
    with patch.object(research_agent, "_call_llm") as mock_call:
        # Sufficiency check returns True
        mock_call.side_effect = [
            json.dumps({"is_sufficient": True, "missing_information": [], "reasoning": "ok"}),
        ]

        # Mock synthesis
        research_agent.response_generator.generate.return_value = GeneratedResponse(
            answer="Final Answer", query_id="q1", chunks_used=["c1"]
        )

        result = research_agent.research("test query")

        assert result.final_answer.answer == "Final Answer"
        assert len(result.steps) == 1
        assert result.steps[0].sufficiency_check.is_sufficient is True
        assert mock_retriever.retrieve.call_count == 1  # Only initial retrieval


def test_research_iterative(research_agent, mock_retriever, mock_query_parser):
    """Test research with one iteration of refinement."""
    # Mock chunks
    chunk1 = HybridChunk(
        chunk_id="c1",
        document_id="d1",
        content="content1",
        level=1,
        final_score=0.9,
        rank=1,
        source="vector",
    )
    chunk2 = HybridChunk(
        chunk_id="c2",
        document_id="d2",
        content="content2",
        level=1,
        final_score=0.9,
        rank=1,
        source="vector",
    )

    # Setup retriever to return different results
    # 1. Initial retrieval
    # 2. Sub-query retrieval
    mock_retriever.retrieve.side_effect = [
        HybridRetrievalResult(
            query_id="q1",
            query_text="test",
            strategy_used=RetrievalStrategy.HYBRID_PARALLEL,
            chunks=[chunk1],
            total_results=1,
            retrieval_time_ms=10,
        ),
        HybridRetrievalResult(
            query_id="q2",
            query_text="sub",
            strategy_used=RetrievalStrategy.HYBRID_PARALLEL,
            chunks=[chunk2],
            total_results=1,
            retrieval_time_ms=10,
        ),
    ]

    # Mock LLM calls
    with patch.object(research_agent, "_call_llm") as mock_call:
        mock_call.side_effect = [
            # 1. Sufficiency check (False)
            json.dumps(
                {
                    "is_sufficient": False,
                    "missing_information": ["missing info"],
                    "reasoning": "bad",
                }
            ),
            # 2. Sub-query generation
            json.dumps({"sub_queries": ["sub query"]}),
            # 3. Sufficiency check (True)
            json.dumps({"is_sufficient": True, "missing_information": [], "reasoning": "good"}),
        ]

        # Mock synthesis
        research_agent.response_generator.generate.return_value = GeneratedResponse(
            answer="Final Answer", query_id="q1", chunks_used=["c1", "c2"]
        )

        result = research_agent.research("test query")

        assert len(result.steps) == 2
        assert len(result.accumulated_context) == 2  # c1 and c2
        assert mock_retriever.retrieve.call_count == 2
        assert result.steps[0].sufficiency_check.is_sufficient is False
        assert result.steps[1].sufficiency_check.is_sufficient is True
