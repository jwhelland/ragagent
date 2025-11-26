from unittest.mock import Mock, patch

import pytest

from ragagent.retrieval.graph import GraphRetriever
from ragagent.retrieval.models import RetrievedChunk, VectorQueryResult


@pytest.fixture(autouse=True)
def mock_extract_entities():
    """Auto-mock extract_entities_and_phrases to avoid spaCy dependency."""
    with patch("ragagent.retrieval.graph.extract_entities_and_phrases") as mock:
        mock.return_value = ([], [])  # Default: no entities or phrases
        yield mock


@pytest.fixture
def mock_graph_store():
    """Mock GraphStore."""
    store = Mock()
    store.get_related_sections.return_value = []
    store.search_sections_by_entities.return_value = []
    return store


@pytest.fixture
def mock_vector_store():
    """Mock QdrantStore."""
    store = Mock()
    store.fetch_by_ids.return_value = {}
    return store


@pytest.fixture
def retriever(mock_graph_store, mock_vector_store):
    """GraphRetriever with mocked dependencies."""
    return GraphRetriever(
        mock_graph_store,
        mock_vector_store,
        seed_limit=4,
        related_per_seed=2,
        fallback_entity_terms=5,
    )


@pytest.fixture
def vector_result():
    """Sample VectorQueryResult with seed chunks."""
    return VectorQueryResult(
        query="test query",
        embedding=[0.1] * 1024,
        chunks=[
            RetrievedChunk(
                chunk_id=f"seed_{i}",
                doc_id="doc1",
                text=f"Seed {i}",
                score=0.9 - i * 0.1,
            )
            for i in range(6)
        ],
    )


def test_expand_uses_seed_limit(retriever, mock_graph_store, vector_result):
    """expand() limits seeds to seed_limit."""
    retriever.expand("query", vector_result)

    call_args = mock_graph_store.get_related_sections.call_args
    seed_ids = call_args[0][0]
    assert len(seed_ids) == 4
    assert set(seed_ids) == {"seed_0", "seed_1", "seed_2", "seed_3"}


def test_expand_calls_get_related_sections(retriever, mock_graph_store, vector_result):
    """expand() calls get_related_sections with seed IDs."""
    retriever.expand("query", vector_result)

    mock_graph_store.get_related_sections.assert_called_once()
    call_kwargs = mock_graph_store.get_related_sections.call_args.kwargs
    assert call_kwargs["limit_per_seed"] == 2


def test_expand_hydrates_related_rows(
    retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() hydrates graph results with vector store payloads."""
    mock_graph_store.get_related_sections.return_value = [
        {"chunk_id": "related_1", "doc_id": "doc2", "page": 5, "entities": ["Entity1"]},
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "related_1": {"text": "Related text", "doc_id": "doc2", "page": 5},
    }

    result = retriever.expand("query", vector_result, max_results=4)

    assert len(result) == 1
    assert result[0].chunk_id == "related_1"
    assert result[0].text == "Related text"
    assert result[0].origin == "graph"


def test_expand_skips_seed_chunks(
    retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() filters out chunks already in seed set."""
    mock_graph_store.get_related_sections.return_value = [
        {"chunk_id": "seed_0", "doc_id": "doc1"},  # Already a seed
        {"chunk_id": "related_1", "doc_id": "doc2", "entities": []},
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "related_1": {"text": "New text", "doc_id": "doc2"},
    }

    result = retriever.expand("query", vector_result, max_results=4)

    assert len(result) == 1
    assert result[0].chunk_id == "related_1"


def test_expand_uses_fallback_when_few_results(
    mock_extract_entities, retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() searches by entities when related sections are insufficient."""
    mock_graph_store.get_related_sections.return_value = [
        {"chunk_id": "related_1", "doc_id": "doc2", "entities": []},
    ]
    mock_extract_entities.return_value = (["Entity1", "Entity2"], ["phrase1"])
    mock_graph_store.search_sections_by_entities.return_value = [
        {"chunk_id": "entity_result", "doc_id": "doc3", "entities": ["Entity1"]},
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "related_1": {"text": "Text1", "doc_id": "doc2"},
        "entity_result": {"text": "Text2", "doc_id": "doc3"},
    }

    result = retriever.expand("query", vector_result, max_results=4)

    mock_extract_entities.assert_called_once_with("query")
    mock_graph_store.search_sections_by_entities.assert_called_once()
    call_args = mock_graph_store.search_sections_by_entities.call_args
    assert call_args[0][0] == ["Entity1", "Entity2"]
    assert len(result) == 2


def test_expand_skips_fallback_when_enough_results(
    mock_extract_entities, retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() skips entity fallback when enough related sections found."""
    mock_graph_store.get_related_sections.return_value = [
        {"chunk_id": f"related_{i}", "doc_id": "doc2", "entities": []} for i in range(5)
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        f"related_{i}": {"text": f"Text{i}", "doc_id": "doc2"} for i in range(5)
    }

    retriever.expand("query", vector_result, max_results=4)

    mock_extract_entities.assert_not_called()
    mock_graph_store.search_sections_by_entities.assert_not_called()


def test_expand_uses_phrases_as_fallback(
    mock_extract_entities, retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() uses phrases when no entities found."""
    mock_graph_store.get_related_sections.return_value = []
    mock_extract_entities.return_value = ([], ["phrase1", "phrase2", "phrase3"])
    mock_graph_store.search_sections_by_entities.return_value = []

    retriever.expand("query", vector_result, max_results=4)

    call_args = mock_graph_store.search_sections_by_entities.call_args
    assert call_args[0][0] == ["phrase1", "phrase2", "phrase3"]


def test_expand_limits_fallback_terms(
    mock_extract_entities, retriever, mock_graph_store, mock_vector_store, vector_result
):
    """expand() limits fallback search terms to fallback_entity_terms."""
    mock_graph_store.get_related_sections.return_value = []
    mock_extract_entities.return_value = (
        ["E1", "E2", "E3", "E4", "E5", "E6", "E7"],
        [],
    )
    mock_graph_store.search_sections_by_entities.return_value = []

    retriever.expand("query", vector_result, max_results=4)

    call_args = mock_graph_store.search_sections_by_entities.call_args
    assert len(call_args[0][0]) == 5  # fallback_entity_terms=5


def test_hydrate_rows_fetches_text_from_vector_store(retriever, mock_vector_store):
    """_hydrate_rows() fetches text payloads from vector store."""
    rows = [{"chunk_id": "c1", "doc_id": "doc1"}]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Content", "doc_id": "doc1"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    mock_vector_store.fetch_by_ids.assert_called_once_with(["c1"])
    assert len(result) == 1
    assert result[0].text == "Content"


def test_hydrate_rows_skips_seen_chunk_ids(retriever, mock_vector_store):
    """_hydrate_rows() skips chunks already in seen set."""
    rows = [
        {"chunk_id": "c1", "doc_id": "doc1"},
        {"chunk_id": "c2", "doc_id": "doc2"},
    ]
    seen_ids = {"c1"}
    mock_vector_store.fetch_by_ids.return_value = {
        "c2": {"text": "Text2", "doc_id": "doc2"},
    }

    result = retriever._hydrate_rows(rows, seen_ids, max_results=5)

    assert len(result) == 1
    assert result[0].chunk_id == "c2"


def test_hydrate_rows_skips_missing_text(retriever, mock_vector_store):
    """_hydrate_rows() skips chunks without text payload."""
    rows = [
        {"chunk_id": "c1", "doc_id": "doc1"},
        {"chunk_id": "c2", "doc_id": "doc2"},
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"doc_id": "doc1"},  # No text
        "c2": {"text": "Valid text", "doc_id": "doc2"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert len(result) == 1
    assert result[0].chunk_id == "c2"


def test_hydrate_rows_respects_max_results(retriever, mock_vector_store):
    """_hydrate_rows() limits output to max_results."""
    rows = [{"chunk_id": f"c{i}", "doc_id": "doc1"} for i in range(10)]
    mock_vector_store.fetch_by_ids.return_value = {
        f"c{i}": {"text": f"Text{i}", "doc_id": "doc1"} for i in range(10)
    }

    result = retriever._hydrate_rows(rows, set(), max_results=3)

    assert len(result) == 3


def test_hydrate_rows_assigns_graph_origin(retriever, mock_vector_store):
    """_hydrate_rows() sets origin to 'graph'."""
    rows = [{"chunk_id": "c1", "doc_id": "doc1"}]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text", "doc_id": "doc1"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].origin == "graph"


def test_hydrate_rows_calculates_score_from_entities(retriever, mock_vector_store):
    """_hydrate_rows() calculates score based on entity count."""
    rows = [
        {"chunk_id": "c1", "doc_id": "doc1", "entities": ["E1", "E2", "E3"]},
        {"chunk_id": "c2", "doc_id": "doc1", "entities": []},
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text1", "doc_id": "doc1"},
        "c2": {"text": "Text2", "doc_id": "doc1"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].score == 0.35 + 0.05 * 3  # 3 entities
    assert result[1].score == 0.35  # No entities


def test_hydrate_rows_caps_entity_score_at_five(retriever, mock_vector_store):
    """_hydrate_rows() caps entity score contribution at 5 entities."""
    rows = [
        {"chunk_id": "c1", "doc_id": "doc1", "entities": [f"E{i}" for i in range(10)]}
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text", "doc_id": "doc1"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].score == 0.35 + 0.05 * 5  # Capped at 5


def test_hydrate_rows_includes_seed_chunk_id(retriever, mock_vector_store):
    """_hydrate_rows() includes seed_chunk_id from row."""
    rows = [{"chunk_id": "c1", "doc_id": "doc1", "seed_chunk_id": "seed_x"}]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text", "doc_id": "doc1"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].seed_chunk_id == "seed_x"


def test_hydrate_rows_sets_graph_rank_metadata(retriever, mock_vector_store):
    """_hydrate_rows() assigns graph_rank metadata starting from 1."""
    rows = [{"chunk_id": f"c{i}", "doc_id": "doc1"} for i in range(3)]
    mock_vector_store.fetch_by_ids.return_value = {
        f"c{i}": {"text": f"Text{i}", "doc_id": "doc1"} for i in range(3)
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].metadata["graph_rank"] == 1
    assert result[1].metadata["graph_rank"] == 2
    assert result[2].metadata["graph_rank"] == 3


def test_hydrate_rows_handles_missing_doc_id(retriever, mock_vector_store):
    """_hydrate_rows() falls back to 'unknown' for missing doc_id."""
    rows = [{"chunk_id": "c1"}]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text"},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].doc_id == "unknown"


def test_hydrate_rows_prefers_row_entities_over_payload(retriever, mock_vector_store):
    """_hydrate_rows() prefers entities from row over payload."""
    rows = [{"chunk_id": "c1", "doc_id": "doc1", "entities": ["RowEntity"]}]
    mock_vector_store.fetch_by_ids.return_value = {
        "c1": {"text": "Text", "doc_id": "doc1", "entities": ["PayloadEntity"]},
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    assert result[0].entities == ["RowEntity"]


def test_hydrate_rows_stops_at_double_max_results(retriever, mock_vector_store):
    """_hydrate_rows() stops processing rows at max_results * 2."""
    rows = [{"chunk_id": f"c{i}", "doc_id": "doc1"} for i in range(100)]
    mock_vector_store.fetch_by_ids.return_value = {
        f"c{i}": {"text": f"Text{i}", "doc_id": "doc1"} for i in range(100)
    }

    result = retriever._hydrate_rows(rows, set(), max_results=5)

    # Should process up to 10 rows (max_results * 2), but return only 5
    mock_vector_store.fetch_by_ids.assert_called_once()
    fetch_ids = mock_vector_store.fetch_by_ids.call_args[0][0]
    assert len(fetch_ids) <= 10


def test_expand_uses_entity_relations_when_enabled(
    mock_extract_entities, mock_graph_store, mock_vector_store, vector_result
):
    """expand() can prioritize entity-relation graph edges."""
    # Enable entity-relation expansion
    retriever = GraphRetriever(
        mock_graph_store,
        mock_vector_store,
        seed_limit=4,
        related_per_seed=2,
        fallback_entity_terms=5,
        use_entity_relations=True,
        relation_types=["MENTIONS", "CO_OCCURS"],
    )

    # Relation-based rows plus plain co-mentions
    mock_graph_store.get_sections_via_entity_relations.return_value = [
        {
            "chunk_id": "rel_1",
            "doc_id": "doc_rel",
            "entities": ["EntityX"],
            "relation_types": ["MENTIONS"],
        }
    ]
    mock_graph_store.get_related_sections.return_value = [
        {"chunk_id": "co_1", "doc_id": "doc_co", "entities": []}
    ]
    mock_vector_store.fetch_by_ids.return_value = {
        "rel_1": {"text": "Rel text", "doc_id": "doc_rel"},
        "co_1": {"text": "Co text", "doc_id": "doc_co"},
    }

    result = retriever.expand("query", vector_result, max_results=4)

    # Relation-based API is invoked with seed ids and relation_types
    call_args = mock_graph_store.get_sections_via_entity_relations.call_args
    seed_ids = set(call_args[0][0])
    assert seed_ids == {f"seed_{i}" for i in range(4)}
    assert call_args.kwargs["relation_types"] == ["MENTIONS", "CO_OCCURS"]

    # Relation-based row is prioritized and relation_types propagated to metadata
    assert result[0].chunk_id == "rel_1"
    assert result[0].metadata.get("relation_types") == ["MENTIONS"]
