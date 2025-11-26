from unittest.mock import Mock

import pytest

from ragagent.agent.service import AgentResponse, RetrievalAgent
from ragagent.agent.verification import SectionReport, VerificationReport
from ragagent.retrieval.models import (
    ContextBundle,
    ContextChunk,
    RetrievedChunk,
    VectorQueryResult,
)


@pytest.fixture
def mock_vector_retriever():
    """Mock VectorRetriever."""
    retriever = Mock()
    retriever.retrieve.return_value = VectorQueryResult(
        query="test question",
        embedding=[0.1] * 1024,
        chunks=[
            RetrievedChunk(
                chunk_id="v1",
                doc_id="doc1",
                text="Vector result 1",
                score=0.9,
                page=1,
                origin="vector",
            ),
            RetrievedChunk(
                chunk_id="v2",
                doc_id="doc1",
                text="Vector result 2",
                score=0.8,
                page=2,
                origin="vector",
            ),
        ],
    )
    return retriever


@pytest.fixture
def mock_graph_retriever():
    """Mock GraphRetriever."""
    retriever = Mock()
    retriever.expand.return_value = [
        RetrievedChunk(
            chunk_id="g1",
            doc_id="doc2",
            text="Graph result 1",
            score=0.5,
            page=3,
            origin="graph",
        ),
    ]
    return retriever


@pytest.fixture
def mock_context_assembler():
    """Mock ContextAssembler."""
    assembler = Mock()
    chunk1 = ContextChunk(
        tag="S1",
        chunk=RetrievedChunk(
            chunk_id="v1",
            doc_id="doc1",
            text="Vector result 1",
            score=0.9,
            page=1,
            origin="vector",
        ),
        snippet="Vector result 1",
    )
    chunk2 = ContextChunk(
        tag="S2",
        chunk=RetrievedChunk(
            chunk_id="g1",
            doc_id="doc2",
            text="Graph result 1",
            score=0.5,
            page=3,
            origin="graph",
        ),
        snippet="Graph result 1",
    )
    assembler.build.return_value = ContextBundle(
        chunks=[chunk1, chunk2],
        formatted="[S1] Doc doc1 (p.1)\nVector result 1\n\n[S2] Doc doc2 (p.3)\nGraph result 1",
        citation_map={
            "S1": {
                "chunk_id": "v1",
                "doc_id": "doc1",
                "page": 1,
                "table_id": None,
                "origin": "vector",
            },
            "S2": {
                "chunk_id": "g1",
                "doc_id": "doc2",
                "page": 3,
                "table_id": None,
                "origin": "graph",
            },
        },
    )
    return assembler


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    completion = Mock()
    completion.choices = [Mock()]
    completion.choices[0].message.content = "This is the answer citing [S1] and [S2]."
    client.chat.completions.create.return_value = completion
    return client


@pytest.fixture
def mock_verifier():
    """Mock AnswerVerifier."""
    verifier = Mock()
    verifier.verify.return_value = VerificationReport(
        overall_status="supported",
        cited_tags=["S1", "S2"],
        missing_tags=[],
        uncited_tags=[],
        sections=[
            SectionReport(
                tag="S1",
                status="supported",
                overlap_ratio=0.95,
                overlap_tokens=10,
            ),
            SectionReport(
                tag="S2",
                status="supported",
                overlap_ratio=0.90,
                overlap_tokens=8,
            ),
        ],
    )
    return verifier


@pytest.fixture
def agent(
    mock_vector_retriever,
    mock_graph_retriever,
    mock_context_assembler,
    mock_openai_client,
    mock_verifier,
):
    """RetrievalAgent with all mocked dependencies."""
    return RetrievalAgent(
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        context_assembler=mock_context_assembler,
        openai_client=mock_openai_client,
        model="gpt-4o-mini",
        temperature=0.2,
        max_graph_results=4,
        verifier=mock_verifier,
    )


def test_run_calls_vector_retrieval(agent, mock_vector_retriever):
    """run() calls vector retriever with question."""
    agent.run("What is the answer?", top_k=8)

    mock_vector_retriever.retrieve.assert_called_once_with(
        query="What is the answer?", top_k=8
    )


def test_run_calls_graph_expansion(agent, mock_graph_retriever, mock_vector_retriever):
    """run() calls graph retriever with vector results."""
    vector_result = mock_vector_retriever.retrieve.return_value

    agent.run("What is the answer?", top_k=8)

    mock_graph_retriever.expand.assert_called_once()
    call_args = mock_graph_retriever.expand.call_args
    assert call_args[0][0] == "What is the answer?"
    assert call_args[0][1] == vector_result
    assert call_args.kwargs["max_results"] == 4


def test_run_builds_context(
    agent, mock_context_assembler, mock_vector_retriever, mock_graph_retriever
):
    """run() builds context from vector and graph results."""
    vector_chunks = mock_vector_retriever.retrieve.return_value.chunks
    graph_chunks = mock_graph_retriever.expand.return_value

    agent.run("What is the answer?", top_k=8)

    mock_context_assembler.build.assert_called_once_with(vector_chunks, graph_chunks)


def test_run_calls_llm_with_context(agent, mock_openai_client, mock_context_assembler):
    """run() calls OpenAI with system prompt, user prompt, and context."""
    agent.run("What is the answer?", top_k=8)

    mock_openai_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.2

    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is the answer?" in messages[1]["content"]


def test_run_verifies_answer(agent, mock_verifier, mock_context_assembler):
    """run() verifies answer against context."""
    result = agent.run("What is the answer?", top_k=8)

    mock_verifier.verify.assert_called_once()
    call_args = mock_verifier.verify.call_args
    assert "This is the answer citing [S1] and [S2]" in call_args[0][0]
    assert call_args[0][1] == mock_context_assembler.build.return_value


def test_run_returns_agent_response(agent):
    """run() returns AgentResponse with all fields populated."""
    result = agent.run("What is the answer?", top_k=8)

    assert isinstance(result, AgentResponse)
    assert result.answer == "This is the answer citing [S1] and [S2]."
    assert "S1" in result.citations
    assert "S2" in result.citations
    assert result.context is not None
    assert result.verification.overall_status == "supported"
    assert result.raw_model_response is not None


def test_run_raises_on_no_vector_results(agent, mock_vector_retriever):
    """run() raises ValueError when no vector results found."""
    mock_vector_retriever.retrieve.return_value.chunks = []

    with pytest.raises(ValueError, match="no_vector_results"):
        agent.run("What is the answer?", top_k=8)


def test_run_accepts_history(agent, mock_openai_client):
    """run() includes conversation history in LLM call."""
    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    agent.run("Follow-up question?", top_k=8, history=history)

    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Previous question"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Previous answer"
    assert messages[3]["role"] == "user"


def test_run_accepts_guidance(agent, mock_openai_client):
    """run() includes guidance in user prompt."""
    agent.run(
        "What is the answer?",
        top_k=8,
        guidance="Be concise and focus on key points.",
    )

    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    user_message = messages[-1]["content"]
    assert "Be concise and focus on key points." in user_message


def test_run_creates_default_verifier_if_none(
    mock_vector_retriever,
    mock_graph_retriever,
    mock_context_assembler,
    mock_openai_client,
):
    """RetrievalAgent creates default AnswerVerifier when none provided."""
    agent = RetrievalAgent(
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        context_assembler=mock_context_assembler,
        openai_client=mock_openai_client,
    )

    assert agent._verifier is not None


def test_normalize_history_filters_invalid_roles(agent):
    """_normalize_history() filters out invalid roles."""
    history = [
        {"role": "user", "content": "Valid user"},
        {"role": "system", "content": "Invalid system"},
        {"role": "assistant", "content": "Valid assistant"},
        {"role": "unknown", "content": "Invalid role"},
    ]

    result = agent._normalize_history(history)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_normalize_history_filters_empty_content(agent):
    """_normalize_history() filters out entries with empty content."""
    history = [
        {"role": "user", "content": "Valid"},
        {"role": "user", "content": ""},
        {"role": "assistant"},
        {"role": "assistant", "content": "Also valid"},
    ]

    result = agent._normalize_history(history)

    assert len(result) == 2
    assert result[0]["content"] == "Valid"
    assert result[1]["content"] == "Also valid"


def test_normalize_history_filters_missing_role(agent):
    """_normalize_history() filters out entries without role."""
    history = [
        {"content": "No role"},
        {"role": "user", "content": "Has role"},
    ]

    result = agent._normalize_history(history)

    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_normalize_history_returns_empty_for_invalid_input(agent):
    """_normalize_history() returns empty list for all-invalid input."""
    history = [
        {"role": "system", "content": "System message"},
        {"role": "unknown", "content": "Unknown role"},
    ]

    result = agent._normalize_history(history)

    assert result == []


def test_call_model_constructs_messages_correctly(agent, mock_openai_client):
    """_call_model() constructs messages in correct order."""
    history = [{"role": "user", "content": "Previous"}]

    agent._call_model(history, "New question")

    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Previous"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "New question"


def test_call_model_uses_configured_model_and_temp(agent, mock_openai_client):
    """_call_model() uses configured model and temperature."""
    agent._call_model([], "Question")

    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.2


def test_run_handles_empty_answer(agent, mock_openai_client):
    """run() handles empty or whitespace-only LLM response."""
    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].message.content = "  \n  "

    result = agent.run("What is the answer?", top_k=8)

    assert result.answer == ""


def test_run_strips_answer_whitespace(agent, mock_openai_client):
    """run() strips leading/trailing whitespace from answer."""
    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].message.content = "\n  Answer with whitespace  \n"

    result = agent.run("What is the answer?", top_k=8)

    assert result.answer == "Answer with whitespace"


def test_run_handles_none_answer(agent, mock_openai_client):
    """run() handles None content from LLM."""
    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].message.content = None

    result = agent.run("What is the answer?", top_k=8)

    assert result.answer == ""


def test_run_max_graph_results_configurable(
    mock_vector_retriever,
    mock_graph_retriever,
    mock_context_assembler,
    mock_openai_client,
    mock_verifier,
):
    """RetrievalAgent respects max_graph_results configuration."""
    agent = RetrievalAgent(
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        context_assembler=mock_context_assembler,
        openai_client=mock_openai_client,
        verifier=mock_verifier,
        max_graph_results=10,
    )

    agent.run("Question", top_k=8)

    call_kwargs = mock_graph_retriever.expand.call_args.kwargs
    assert call_kwargs["max_results"] == 10
