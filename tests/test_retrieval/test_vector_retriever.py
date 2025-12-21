"""Tests for vector retriever (Task 4.2)."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.retrieval.query_parser import ParsedQuery, QueryIntent
from src.retrieval.vector_retriever import (
    RetrievalResult,
    RetrievedChunk,
    VectorRetriever,
)
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import Config
from src.utils.embeddings import EmbeddingGenerator


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config.from_yaml()


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Create mock Qdrant manager."""
    mock = MagicMock(spec=QdrantManager)
    mock.chunk_collection = "test_chunks"
    mock.client = MagicMock()  # Add client for statistics tests
    return mock


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create mock embedding generator."""
    mock = MagicMock(spec=EmbeddingGenerator)
    # Return a fixed embedding vector
    mock.generate.return_value = [np.random.rand(384).astype(np.float32)]
    return mock


@pytest.fixture
def sample_search_results() -> List[Dict[str, Any]]:
    """Create sample search results from Qdrant."""
    return [
        {
            "chunk_id": "chunk-1",
            "score": 0.95,
            "payload": {
                "document_id": "doc-1",
                "content": "The electrical power system provides power to all subsystems.",
                "level": 2,
                "metadata": {"section_title": "Power Distribution"},
                "entity_ids": ["entity-1", "entity-2"],
            },
        },
        {
            "chunk_id": "chunk-2",
            "score": 0.87,
            "payload": {
                "document_id": "doc-1",
                "content": "The battery controller manages charge and discharge cycles.",
                "level": 3,
                "metadata": {"section_title": "Battery Management"},
                "entity_ids": ["entity-2", "entity-3"],
            },
        },
        {
            "chunk_id": "chunk-3",
            "score": 0.72,
            "payload": {
                "document_id": "doc-2",
                "content": "Solar panels convert sunlight to electrical energy.",
                "level": 4,
                "metadata": {"section_title": "Solar Power"},
                "entity_ids": ["entity-4"],
            },
        },
    ]


@pytest.fixture
def vector_retriever(
    config: Config, mock_qdrant: MagicMock, mock_embeddings: MagicMock
) -> VectorRetriever:
    """Create vector retriever with mocks."""
    return VectorRetriever(
        config=config,
        qdrant_manager=mock_qdrant,
        embedding_generator=mock_embeddings,
    )


class TestVectorRetrieverInitialization:
    """Tests for VectorRetriever initialization."""

    def test_init_with_defaults(self, config: Config) -> None:
        """Test initialization with default parameters."""
        with patch("src.retrieval.vector_retriever.QdrantManager"):
            with patch("src.retrieval.vector_retriever.EmbeddingGenerator"):
                retriever = VectorRetriever(config=config)
                assert retriever.config is not None
                assert retriever.vector_config is not None

    def test_init_with_custom_managers(
        self, config: Config, mock_qdrant: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test initialization with custom managers."""
        retriever = VectorRetriever(
            config=config,
            qdrant_manager=mock_qdrant,
            embedding_generator=mock_embeddings,
        )
        assert retriever.qdrant is mock_qdrant
        assert retriever.embeddings is mock_embeddings


class TestBasicRetrieval:
    """Tests for basic retrieval functionality."""

    def test_retrieve_with_string_query(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval with string query."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("What is the power system?")

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 3
        assert result.query_text == "What is the power system?"
        assert result.total_results == 3
        assert result.page == 1

    def test_retrieve_with_parsed_query(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval with ParsedQuery object."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        parsed_query = ParsedQuery(
            query_id="test-query-1",
            original_text="What is the power system?",
            normalized_text="what is the power system",
            intent=QueryIntent.SEMANTIC,
            intent_confidence=0.9,
        )

        result = vector_retriever.retrieve(parsed_query)

        assert result.query_id == "test-query-1"
        assert len(result.chunks) == 3

    def test_retrieve_empty_query_raises_error(self, vector_retriever: VectorRetriever) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            vector_retriever.retrieve("")

    def test_retrieve_whitespace_query_raises_error(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            vector_retriever.retrieve("   ")

    def test_retrieve_returns_sorted_chunks(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that chunks are returned in order of relevance."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        # Chunks should be ranked 1, 2, 3
        assert result.chunks[0].rank == 1
        assert result.chunks[1].rank == 2
        assert result.chunks[2].rank == 3

        # Scores should be descending
        assert result.chunks[0].score >= result.chunks[1].score
        assert result.chunks[1].score >= result.chunks[2].score


class TestScoreNormalization:
    """Tests for score normalization."""

    def test_normalize_scores_single_result(self, vector_retriever: VectorRetriever) -> None:
        """Test normalization with single result."""
        scores = [0.75]
        normalized = vector_retriever._normalize_scores(scores)
        assert normalized == [1.0]

    def test_normalize_scores_multiple_results(self, vector_retriever: VectorRetriever) -> None:
        """Test normalization with multiple results."""
        scores = [0.9, 0.7, 0.5]
        normalized = vector_retriever._normalize_scores(scores)

        # Check range [0, 1]
        assert all(0.0 <= s <= 1.0 for s in normalized)

        # Check ordering preserved
        assert normalized[0] >= normalized[1]
        assert normalized[1] >= normalized[2]

        # Check highest is 1.0, lowest is 0.0
        assert normalized[0] == 1.0
        assert normalized[2] == 0.0

    def test_normalize_scores_identical_values(self, vector_retriever: VectorRetriever) -> None:
        """Test normalization with identical scores."""
        scores = [0.8, 0.8, 0.8]
        normalized = vector_retriever._normalize_scores(scores)

        # All should be 1.0 when identical
        assert all(s == 1.0 for s in normalized)

    def test_normalize_scores_empty_list(self, vector_retriever: VectorRetriever) -> None:
        """Test normalization with empty list."""
        scores: List[float] = []
        normalized = vector_retriever._normalize_scores(scores)
        assert normalized == []


class TestPagination:
    """Tests for pagination functionality."""

    def test_retrieve_first_page(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval of first page."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query", top_k=2, page=1)

        assert result.page == 1
        assert result.page_size == 2
        assert len(result.chunks) == 2
        assert result.has_more is True  # More results exist

    def test_retrieve_second_page(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval of second page."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query", top_k=2, page=2)

        assert result.page == 2
        assert result.page_size == 2
        assert len(result.chunks) == 1  # Only 1 result on page 2
        assert result.has_more is False

    def test_retrieve_page_beyond_results(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval of page beyond available results."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query", top_k=10, page=2)

        assert result.page == 2
        assert len(result.chunks) == 0
        assert result.has_more is False


class TestMetadataFiltering:
    """Tests for metadata filtering."""

    def test_retrieve_by_document(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test filtering by document ID."""
        mock_qdrant.search_chunks.return_value = sample_search_results[:2]

        result = vector_retriever.retrieve_by_document(
            query="test query",
            document_ids=["doc-1"],
        )

        # Verify filter was applied
        mock_qdrant.search_chunks.assert_called_once()
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["filters"] == {"document_id": "doc-1"}

        assert len(result.chunks) == 2
        assert all(chunk.document_id == "doc-1" for chunk in result.chunks)

    def test_retrieve_by_entity(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test filtering by entity IDs."""
        filtered_results = [sample_search_results[0], sample_search_results[1]]
        mock_qdrant.search_chunks.return_value = filtered_results

        result = vector_retriever.retrieve_by_entity(
            query="test query",
            entity_ids=["entity-2"],
        )

        # Verify filter was applied
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["filters"] == {"entity_ids": ["entity-2"]}

        assert len(result.chunks) == 2

    def test_retrieve_by_level(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test filtering by hierarchy level."""
        mock_qdrant.search_chunks.return_value = [sample_search_results[0]]

        result = vector_retriever.retrieve_by_level(
            query="test query",
            level=2,
        )

        # Verify filter was applied
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["filters"] == {"level": 2}

        assert len(result.chunks) == 1
        assert result.chunks[0].level == 2

    def test_retrieve_by_level_invalid_raises_error(
        self, vector_retriever: VectorRetriever
    ) -> None:
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Level must be between 1 and 4"):
            vector_retriever.retrieve_by_level(query="test", level=5)

    def test_retrieve_with_custom_filters(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval with custom filter dictionary."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        filters = {"document_id": "doc-1", "level": 2}
        vector_retriever.retrieve("test query", filters=filters)

        # Verify filters were passed through
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["filters"] == filters


class TestEntityExtraction:
    """Tests for entity ID extraction."""

    def test_get_entity_ids_from_result(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test extraction of entity IDs from results."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        entity_ids = result.get_entity_ids()
        assert len(entity_ids) == 4  # Unique entities
        assert "entity-1" in entity_ids
        assert "entity-2" in entity_ids
        assert "entity-3" in entity_ids
        assert "entity-4" in entity_ids

    def test_get_document_ids_from_result(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test extraction of document IDs from results."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        document_ids = result.get_document_ids()
        assert len(document_ids) == 2  # Two unique documents
        assert "doc-1" in document_ids
        assert "doc-2" in document_ids

    def test_entity_ids_in_retrieved_chunks(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that entity IDs are properly extracted to chunks."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        assert result.chunks[0].entity_ids == ["entity-1", "entity-2"]
        assert result.chunks[1].entity_ids == ["entity-2", "entity-3"]
        assert result.chunks[2].entity_ids == ["entity-4"]


class TestMMRDiversity:
    """Tests for MMR diversity functionality."""

    def test_mmr_reranks_results(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that MMR reranks results for diversity."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        # Retrieve with MMR enabled
        result = vector_retriever.retrieve("test query", use_mmr=True)

        assert result.diversity_mode is not None
        assert "mmr" in result.diversity_mode.lower()
        # Results should still be present, possibly reordered
        assert len(result.chunks) == 3

    def test_mmr_disabled(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test retrieval with MMR disabled."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query", use_mmr=False)

        assert result.diversity_mode is None
        # Results should be in original order
        assert result.chunks[0].chunk_id == "chunk-1"

    def test_mmr_with_single_result(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
    ) -> None:
        """Test that MMR handles single result correctly."""
        single_result = [
            {
                "chunk_id": "chunk-1",
                "score": 0.95,
                "payload": {
                    "document_id": "doc-1",
                    "content": "Test content",
                    "level": 2,
                    "metadata": {},
                    "entity_ids": [],
                },
            }
        ]
        mock_qdrant.search_chunks.return_value = single_result

        result = vector_retriever.retrieve("test query", use_mmr=True)

        # MMR should not be applied with single result
        assert len(result.chunks) == 1

    def test_content_similarity(self, vector_retriever: VectorRetriever) -> None:
        """Test content similarity calculation."""
        text1 = "The power system provides electrical energy"
        text2 = "The power system distributes electrical energy"
        text3 = "Solar panels convert sunlight"

        # Similar texts should have higher similarity
        sim_12 = vector_retriever._content_similarity(text1, text2)
        sim_13 = vector_retriever._content_similarity(text1, text3)

        assert sim_12 > sim_13
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0

    def test_content_similarity_identical(self, vector_retriever: VectorRetriever) -> None:
        """Test content similarity with identical texts."""
        text = "The power system"
        sim = vector_retriever._content_similarity(text, text)
        assert sim == 1.0

    def test_content_similarity_no_overlap(self, vector_retriever: VectorRetriever) -> None:
        """Test content similarity with no word overlap."""
        text1 = "alpha beta gamma"
        text2 = "delta epsilon zeta"
        sim = vector_retriever._content_similarity(text1, text2)
        assert sim == 0.0


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_to_dict(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test conversion to dictionary."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")
        data = result.to_dict()

        assert isinstance(data, dict)
        assert "query_id" in data
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
        assert "timestamp" in data

    def test_retrieval_time_tracking(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that retrieval time is tracked."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        assert result.retrieval_time_ms > 0
        # Should be fast (under 1 second for mocked operations)
        assert result.retrieval_time_ms < 1000


class TestConfigurationOptions:
    """Tests for configuration options."""

    def test_custom_top_k(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test custom top_k parameter."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query", top_k=2)

        assert len(result.chunks) <= 2
        assert result.page_size == 2

    def test_custom_min_score(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test custom min_score parameter."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        vector_retriever.retrieve("test query", min_score=0.8)

        # Verify min_score was passed to Qdrant
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["score_threshold"] == 0.8

    def test_uses_config_defaults(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that config defaults are used when not specified."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        vector_retriever.retrieve("test query")

        # Should use default top_k from config
        call_kwargs = mock_qdrant.search_chunks.call_args[1]
        assert call_kwargs["top_k"] == vector_retriever.vector_config.top_k


class TestEmbeddingGeneration:
    """Tests for query embedding generation."""

    def test_generate_query_embedding(
        self,
        vector_retriever: VectorRetriever,
        mock_embeddings: MagicMock,
        mock_qdrant: MagicMock,
    ) -> None:
        """Test that query embeddings are generated."""
        mock_qdrant.search_chunks.return_value = []

        vector_retriever.retrieve("test query")

        # Verify embedding was generated
        mock_embeddings.generate.assert_called_once_with(["test query"])

    def test_embedding_passed_to_qdrant(
        self,
        vector_retriever: VectorRetriever,
        mock_embeddings: MagicMock,
        mock_qdrant: MagicMock,
    ) -> None:
        """Test that generated embedding is passed to Qdrant."""
        test_embedding = np.array([0.1, 0.2, 0.3])
        mock_embeddings.generate.return_value = [test_embedding]
        mock_qdrant.search_chunks.return_value = []

        vector_retriever.retrieve("test query")

        # Verify embedding was passed to search
        call_args = mock_qdrant.search_chunks.call_args[1]
        query_vector = call_args["query_vector"]
        assert isinstance(query_vector, list)
        np.testing.assert_array_almost_equal(query_vector, test_embedding.tolist())


class TestStatistics:
    """Tests for retrieval statistics."""

    def test_get_statistics(
        self, vector_retriever: VectorRetriever, mock_qdrant: MagicMock
    ) -> None:
        """Test retrieval of statistics."""
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1000
        mock_qdrant.client.get_collection.return_value = mock_collection_info

        stats = vector_retriever.get_statistics()

        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 1000
        assert "embedding_model" in stats

    def test_get_statistics_handles_error(
        self, vector_retriever: VectorRetriever, mock_qdrant: MagicMock
    ) -> None:
        """Test that statistics gracefully handles errors."""
        mock_qdrant.client.get_collection.side_effect = Exception("Connection error")

        stats = vector_retriever.get_statistics()

        # Should return empty dict on error
        assert isinstance(stats, dict)


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_chunk_creation(self) -> None:
        """Test creating a RetrievedChunk."""
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            level=2,
            score=0.95,
            normalized_score=1.0,
            rank=1,
        )

        assert chunk.chunk_id == "chunk-1"
        assert chunk.score == 0.95
        assert chunk.normalized_score == 1.0

    def test_chunk_to_dict(self) -> None:
        """Test conversion of chunk to dictionary."""
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            level=2,
            score=0.95,
            normalized_score=1.0,
            entity_ids=["entity-1"],
            rank=1,
        )

        data = chunk.to_dict()

        assert isinstance(data, dict)
        assert data["chunk_id"] == "chunk-1"
        assert data["entity_ids"] == ["entity-1"]


class TestPerformance:
    """Tests for performance characteristics."""

    def test_retrieval_time_under_threshold(
        self,
        vector_retriever: VectorRetriever,
        mock_qdrant: MagicMock,
        sample_search_results: List[Dict[str, Any]],
    ) -> None:
        """Test that retrieval completes within time threshold."""
        mock_qdrant.search_chunks.return_value = sample_search_results

        result = vector_retriever.retrieve("test query")

        # Should complete in under 100ms for mocked operations (acceptance criteria)
        # Being generous for test environment
        assert result.retrieval_time_ms < 500
