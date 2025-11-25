from unittest.mock import Mock

import pytest
from qdrant_client.http.models import ScoredPoint

from ragagent.retrieval.vector import VectorRetriever


@pytest.fixture
def mock_store():
    """Mock QdrantStore."""
    return Mock()


@pytest.fixture
def mock_embedder():
    """Mock EmbeddingsClient."""
    embedder = Mock()
    embedder.embed.return_value = [[0.1] * 1024]
    return embedder


@pytest.fixture
def retriever(mock_store, mock_embedder):
    """VectorRetriever with mocked dependencies."""
    return VectorRetriever(mock_store, mock_embedder, top_k=5)


def test_retrieve_calls_embedder_and_store(retriever, mock_embedder, mock_store):
    """retrieve() embeds query and searches vector store."""
    mock_store.search.return_value = []

    retriever.retrieve("test query")

    mock_embedder.embed.assert_called_once_with(["test query"])
    mock_store.search.assert_called_once()


def test_retrieve_uses_default_top_k(retriever, mock_store):
    """retrieve() uses default top_k when not overridden."""
    mock_store.search.return_value = []

    retriever.retrieve("query")

    call_kwargs = mock_store.search.call_args.kwargs
    assert call_kwargs["top_k"] == 5


def test_retrieve_overrides_top_k(retriever, mock_store):
    """retrieve() accepts top_k override."""
    mock_store.search.return_value = []

    retriever.retrieve("query", top_k=10)

    call_kwargs = mock_store.search.call_args.kwargs
    assert call_kwargs["top_k"] == 10


def test_retrieve_passes_score_threshold(mock_store, mock_embedder):
    """retrieve() passes score_threshold to store."""
    retriever = VectorRetriever(mock_store, mock_embedder, score_threshold=0.7)
    mock_store.search.return_value = []

    retriever.retrieve("query")

    call_kwargs = mock_store.search.call_args.kwargs
    assert call_kwargs["score_threshold"] == 0.7


def test_retrieve_converts_points_to_chunks(retriever, mock_store):
    """retrieve() converts ScoredPoints to RetrievedChunks."""
    mock_store.search.return_value = [
        ScoredPoint(
            id="chunk_1",
            score=0.95,
            payload={
                "text": "First result",
                "doc_id": "doc1",
                "page": 5,
                "table_id": "t1",
                "source_path": "/path/doc.pdf",
                "entities": ["Entity1"],
                "sha256": "abc123",
            },
            version=1,
        ),
        ScoredPoint(
            id="chunk_2",
            score=0.85,
            payload={
                "text": "Second result",
                "doc_id": "doc2",
                "page": 10,
            },
            version=1,
        ),
    ]

    result = retriever.retrieve("query")

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk_1"
    assert result.chunks[0].text == "First result"
    assert result.chunks[0].doc_id == "doc1"
    assert result.chunks[0].page == 5
    assert result.chunks[0].table_id == "t1"
    assert result.chunks[0].score == 0.95
    assert result.chunks[0].origin == "vector"
    assert result.chunks[0].entities == ["Entity1"]
    assert result.chunks[0].metadata["rank"] == 1
    assert result.chunks[0].metadata["sha256"] == "abc123"


def test_retrieve_handles_missing_payload_fields(retriever, mock_store):
    """retrieve() handles missing optional payload fields gracefully."""
    mock_store.search.return_value = [
        ScoredPoint(
            id="chunk_1",
            score=0.9,
            payload={"text": "Minimal payload"},
            version=1,
        ),
    ]

    result = retriever.retrieve("query")

    chunk = result.chunks[0]
    assert chunk.doc_id == "unknown"
    assert chunk.page is None
    assert chunk.table_id is None
    assert chunk.source_path is None
    assert chunk.entities == []


def test_retrieve_skips_chunks_without_text(retriever, mock_store):
    """retrieve() skips points with missing text field."""
    mock_store.search.return_value = [
        ScoredPoint(id="chunk_1", score=0.9, payload={"text": "Valid"}, version=1),
        ScoredPoint(
            id="chunk_2", score=0.8, payload={"doc_id": "doc2"}, version=1
        ),  # No text
        ScoredPoint(id="chunk_3", score=0.7, payload={"text": "Also valid"}, version=1),
    ]

    result = retriever.retrieve("query")

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk_1"
    assert result.chunks[1].chunk_id == "chunk_3"


def test_retrieve_handles_empty_payload(retriever, mock_store):
    """retrieve() handles points with null payload."""
    mock_store.search.return_value = [
        ScoredPoint(id="chunk_1", score=0.9, payload=None, version=1),
    ]

    result = retriever.retrieve("query")

    assert len(result.chunks) == 0


def test_retrieve_returns_query_and_embedding(retriever, mock_embedder, mock_store):
    """retrieve() returns VectorQueryResult with query and embedding."""
    embedding = [0.1, 0.2, 0.3]
    mock_embedder.embed.return_value = [embedding]
    mock_store.search.return_value = []

    result = retriever.retrieve("test query")

    assert result.query == "test query"
    assert result.embedding == embedding


def test_retrieve_assigns_rank_metadata(retriever, mock_store):
    """retrieve() assigns rank starting from 1."""
    mock_store.search.return_value = [
        ScoredPoint(id="c1", score=0.9, payload={"text": "First"}, version=1),
        ScoredPoint(id="c2", score=0.8, payload={"text": "Second"}, version=1),
        ScoredPoint(id="c3", score=0.7, payload={"text": "Third"}, version=1),
    ]

    result = retriever.retrieve("query")

    assert result.chunks[0].metadata["rank"] == 1
    assert result.chunks[1].metadata["rank"] == 2
    assert result.chunks[2].metadata["rank"] == 3


def test_embed_query_calls_embedder(retriever, mock_embedder):
    """_embed_query() calls embedder with query list."""
    result = retriever._embed_query("test query")

    mock_embedder.embed.assert_called_once_with(["test query"])
    assert result == [0.1] * 1024


def test_embed_query_raises_on_empty_result(retriever, mock_embedder):
    """_embed_query() raises RuntimeError if embedder returns empty."""
    mock_embedder.embed.return_value = []

    with pytest.raises(RuntimeError, match="embeddings_service_returned_empty_vector"):
        retriever._embed_query("query")
