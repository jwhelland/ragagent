from unittest.mock import Mock, call

import pytest
from qdrant_client.http.models import Filter, PointStruct, ScoredPoint

from ragagent.vectorstore.qdrant_store import QdrantStore, _point_id_from_chunk_id


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Mock QdrantClient."""
    mock_client = Mock()
    mock_qdrant_class = Mock(return_value=mock_client)
    monkeypatch.setattr(
        "ragagent.vectorstore.qdrant_store.QdrantClient", mock_qdrant_class
    )
    return mock_client


@pytest.fixture
def store(mock_qdrant_client):
    """QdrantStore instance with mocked client."""
    return QdrantStore(
        url="http://localhost:6333",
        api_key=None,
        collection="test_collection",
        vector_size=1024,
        distance="Cosine",
    )


def test_init_creates_qdrant_client(mock_qdrant_client):
    """__init__ creates QdrantClient with correct parameters."""
    store = QdrantStore(
        url="http://localhost:6333",
        api_key="test_key",
        collection="test_coll",
        vector_size=1024,
        distance="Cosine",
    )

    assert store.client is not None
    assert store.collection == "test_coll"
    assert store.vector_size == 1024
    assert store.distance == "Cosine"


def test_upsert_creates_point_structs(store, mock_qdrant_client):
    """upsert() converts chunks to PointStructs and calls client.upsert."""
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "page": 5,
            "sha256": "hash1",
            "source_path": "/path/doc.pdf",
            "table_id": "t1",
            "text": "Chunk text",
            "entities": ["Entity1"],
            "keyphrases": ["phrase1"],
        },
    ]
    vectors = [[0.1] * 1024]

    store.upsert(chunks, vectors)

    mock_qdrant_client.upsert.assert_called_once()
    call_args = mock_qdrant_client.upsert.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"

    points = call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == _point_id_from_chunk_id("c1")
    assert points[0].vector == [0.1] * 1024
    assert points[0].payload["doc_id"] == "doc1"
    assert points[0].payload["page"] == 5
    assert points[0].payload["text"] == "Chunk text"
    assert points[0].payload["table_id"] == "t1"
    assert points[0].payload["entities"] == ["Entity1"]
    assert points[0].payload["keyphrases"] == ["phrase1"]


def test_upsert_handles_missing_optional_fields(store, mock_qdrant_client):
    """upsert() handles chunks with missing optional fields."""
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "page": None,
            "sha256": "hash1",
            "source_path": "/path/doc.pdf",
        },
    ]
    vectors = [[0.1] * 1024]

    store.upsert(chunks, vectors)

    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    payload = points[0].payload
    assert payload["table_id"] is None
    assert payload["text"] is None
    assert payload["entities"] == []
    assert payload["keyphrases"] == []


def test_upsert_multiple_chunks(store, mock_qdrant_client):
    """upsert() handles multiple chunks in batch."""
    chunks = [
        {
            "chunk_id": f"c{i}",
            "doc_id": f"doc{i}",
            "page": i,
            "sha256": f"hash{i}",
            "source_path": f"/path/doc{i}.pdf",
            "text": f"Text {i}",
        }
        for i in range(3)
    ]
    vectors = [[0.1 * i] * 1024 for i in range(3)]

    store.upsert(chunks, vectors)

    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    assert points[0].id == _point_id_from_chunk_id("c0")
    assert points[1].id == _point_id_from_chunk_id("c1")
    assert points[2].id == _point_id_from_chunk_id("c2")


def test_search_calls_client_with_vector(store, mock_qdrant_client):
    """search() calls client.search with correct parameters."""
    mock_qdrant_client.search.return_value = []
    vector = [0.5] * 1024

    store.search(vector, top_k=10)

    mock_qdrant_client.search.assert_called_once()
    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    assert call_kwargs["query_vector"] == vector
    assert call_kwargs["limit"] == 10
    assert call_kwargs["with_payload"] is True
    assert call_kwargs["with_vectors"] is False


def test_search_default_top_k(store, mock_qdrant_client):
    """search() uses default top_k=8."""
    mock_qdrant_client.search.return_value = []
    vector = [0.5] * 1024

    store.search(vector)

    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert call_kwargs["limit"] == 8


def test_search_with_filters(store, mock_qdrant_client):
    """search() passes query_filter when provided."""
    mock_qdrant_client.search.return_value = []
    vector = [0.5] * 1024
    filters = Filter()

    store.search(vector, filters=filters)

    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert call_kwargs["query_filter"] == filters


def test_search_with_score_threshold(store, mock_qdrant_client):
    """search() passes score_threshold when provided."""
    mock_qdrant_client.search.return_value = []
    vector = [0.5] * 1024

    store.search(vector, score_threshold=0.7)

    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert call_kwargs["score_threshold"] == 0.7


def test_search_without_score_threshold(store, mock_qdrant_client):
    """search() omits score_threshold when None."""
    mock_qdrant_client.search.return_value = []
    vector = [0.5] * 1024

    store.search(vector, score_threshold=None)

    call_kwargs = mock_qdrant_client.search.call_args.kwargs
    assert "score_threshold" not in call_kwargs


def test_search_returns_scored_points(store, mock_qdrant_client):
    """search() returns list of ScoredPoints from client."""
    expected_points = [
        ScoredPoint(id="c1", score=0.9, payload={"text": "Result"}, version=1),
    ]
    mock_qdrant_client.search.return_value = expected_points
    vector = [0.5] * 1024

    result = store.search(vector)

    assert result == expected_points


def test_fetch_by_ids_calls_retrieve(store, mock_qdrant_client):
    """fetch_by_ids() calls client.retrieve with ids."""
    mock_qdrant_client.retrieve.return_value = []
    ids = ["c1", "c2", "c3"]

    store.fetch_by_ids(ids)

    call_kwargs = mock_qdrant_client.retrieve.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_collection"
    # Client sees internal numeric IDs derived from chunk_ids
    numeric_ids = call_kwargs["ids"]
    expected_ids = [
        _point_id_from_chunk_id("c1"),
        _point_id_from_chunk_id("c2"),
        _point_id_from_chunk_id("c3"),
    ]
    assert numeric_ids == expected_ids
    assert call_kwargs["with_payload"] is True
    assert call_kwargs["with_vectors"] is False


def test_fetch_by_ids_returns_payload_dict(store, mock_qdrant_client):
    """fetch_by_ids() returns dict mapping id to payload."""

    class MockRecord:
        def __init__(self, id, payload):
            self.id = id
            self.payload = payload

    mock_qdrant_client.retrieve.return_value = [
        MockRecord(id=_point_id_from_chunk_id("c1"), payload={"text": "First"}),
        MockRecord(id=_point_id_from_chunk_id("c2"), payload={"text": "Second"}),
    ]

    result = store.fetch_by_ids(["c1", "c2"])

    assert result == {
        "c1": {"text": "First"},
        "c2": {"text": "Second"},
    }


def test_fetch_by_ids_handles_empty_ids(store, mock_qdrant_client):
    """fetch_by_ids() returns empty dict for empty ids list."""
    result = store.fetch_by_ids([])

    assert result == {}
    mock_qdrant_client.retrieve.assert_not_called()


def test_fetch_by_ids_handles_null_payload(store, mock_qdrant_client):
    """fetch_by_ids() handles records with null payload."""

    class MockRecord:
        def __init__(self, id, payload):
            self.id = id
            self.payload = payload

    mock_qdrant_client.retrieve.return_value = [
        MockRecord(id=_point_id_from_chunk_id("c1"), payload=None),
    ]

    result = store.fetch_by_ids(["c1"])

    assert result == {"c1": {}}
