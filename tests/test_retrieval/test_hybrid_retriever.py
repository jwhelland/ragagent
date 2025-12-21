"""Tests for hybrid retriever (Task 4.4)."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.retrieval.graph_retriever import (
    GraphPath,
    GraphRetrievalResult,
    GraphRetriever,
    ResolvedEntity,
    TraversalStrategy,
)
from src.retrieval.hybrid_retriever import (
    HybridChunk,
    HybridRetrievalResult,
    HybridRetriever,
    RetrievalStrategy,
)
from src.retrieval.query_parser import EntityMention, ParsedQuery, QueryIntent
from src.retrieval.vector_retriever import RetrievalResult, RetrievedChunk, VectorRetriever
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import EntityType, RelationshipType
from src.utils.config import Config


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config.from_yaml()


@pytest.fixture
def mock_neo4j() -> Mock:
    """Create mock Neo4j manager."""
    neo4j = Mock(spec=Neo4jManager)
    neo4j.execute_cypher.return_value = []
    neo4j.get_statistics.return_value = {}
    return neo4j


@pytest.fixture
def mock_vector_retriever() -> Mock:
    """Create mock vector retriever."""
    retriever = Mock(spec=VectorRetriever)
    retriever.get_statistics.return_value = {}
    return retriever


@pytest.fixture
def mock_graph_retriever() -> Mock:
    """Create mock graph retriever."""
    retriever = Mock(spec=GraphRetriever)
    retriever.get_statistics.return_value = {}
    return retriever


@pytest.fixture
def hybrid_retriever(
    config: Config,
    mock_neo4j: Mock,
    mock_vector_retriever: Mock,
    mock_graph_retriever: Mock,
) -> HybridRetriever:
    """Create hybrid retriever with mock dependencies."""
    return HybridRetriever(
        config=config,
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        neo4j_manager=mock_neo4j,
    )


@pytest.fixture
def sample_parsed_query() -> ParsedQuery:
    """Create a sample parsed query for testing."""
    return ParsedQuery(
        query_id="test_query_hybrid_123",
        original_text="What are the components of the power system?",
        normalized_text="what are the components of the power system",
        intent=QueryIntent.STRUCTURAL,
        intent_confidence=0.9,
        entity_mentions=[
            EntityMention(
                text="power system",
                normalized="power_system",
                entity_type=EntityType.SYSTEM,
                start_char=30,
                end_char=42,
                confidence=0.9,
            )
        ],
        relationship_types=[RelationshipType.CONTAINS],
        constraints=[],
        expanded_terms={},
        keywords=["components", "power", "system"],
        requires_graph_traversal=True,
        max_depth=3,
        timestamp=datetime.now(),
        metadata={},
    )


@pytest.fixture
def sample_semantic_query() -> ParsedQuery:
    """Create a semantic query for strategy selection testing."""
    return ParsedQuery(
        query_id="test_query_semantic_123",
        original_text="What is thermal management?",
        normalized_text="what is thermal management",
        intent=QueryIntent.SEMANTIC,
        intent_confidence=0.9,
        entity_mentions=[],
        relationship_types=[],
        constraints=[],
        expanded_terms={},
        keywords=["thermal", "management"],
        requires_graph_traversal=False,
        max_depth=None,
        timestamp=datetime.now(),
        metadata={},
    )


@pytest.fixture
def sample_vector_result() -> RetrievalResult:
    """Create sample vector retrieval result."""
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_v_{i}",
            document_id="doc_001",
            content=f"Vector chunk content {i}",
            level=3,
            score=0.9 - (i * 0.1),
            normalized_score=1.0 - (i * 0.1),
            metadata={"source": "vector"},
            entity_ids=[f"entity_{i}"],
            rank=i + 1,
        )
        for i in range(5)
    ]

    return RetrievalResult(
        query_id="test_query_hybrid_123",
        query_text="test query",
        chunks=chunks,
        total_results=5,
        page=1,
        page_size=20,
        has_more=False,
        retrieval_time_ms=50.0,
        filters_applied={},
        diversity_mode=None,
    )


@pytest.fixture
def sample_graph_result() -> GraphRetrievalResult:
    """Create sample graph retrieval result."""
    resolved_entities = [
        ResolvedEntity(
            entity_id="entity_eps_001",
            canonical_name="power_system",
            entity_type=EntityType.SYSTEM,
            mention_text="power system",
            confidence=0.9,
            match_method="exact",
        )
    ]

    paths = [
        GraphPath(
            start_entity_id="entity_eps_001",
            end_entity_id=f"entity_comp_{i}",
            nodes=[{"id": "entity_eps_001"}, {"id": f"entity_comp_{i}"}],
            relationships=[{"type": "CONTAINS"}],
            length=1,
            score=0.85 - (i * 0.05),
            confidence=0.8,
            chunk_ids={f"chunk_g_{i}", f"chunk_g_{i+1}"},
        )
        for i in range(3)
    ]

    # Collect all chunk IDs
    chunk_ids = set()
    for path in paths:
        chunk_ids.update(path.chunk_ids)

    return GraphRetrievalResult(
        query_id="test_query_hybrid_123",
        query_text="test query",
        resolved_entities=resolved_entities,
        paths=paths,
        chunk_ids=chunk_ids,
        entity_ids={"entity_eps_001", "entity_comp_0", "entity_comp_1", "entity_comp_2"},
        strategy_used=TraversalStrategy.MULTI_HOP,
        max_depth=3,
        retrieval_time_ms=75.0,
    )


class TestHybridRetrieverInitialization:
    """Tests for HybridRetriever initialization."""

    def test_init_with_config(
        self,
        config: Config,
        mock_neo4j: Mock,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
    ) -> None:
        """Test initialization with configuration."""
        retriever = HybridRetriever(
            config=config,
            vector_retriever=mock_vector_retriever,
            graph_retriever=mock_graph_retriever,
            neo4j_manager=mock_neo4j,
        )
        assert retriever.config is not None
        assert retriever.neo4j is mock_neo4j
        assert retriever.vector_retriever is mock_vector_retriever
        assert retriever.graph_retriever is mock_graph_retriever
        assert retriever.hybrid_config is not None
        assert retriever.reranking_config is not None

    def test_init_with_defaults(self, mock_neo4j: Mock) -> None:
        """Test initialization with default configuration."""
        retriever = HybridRetriever(neo4j_manager=mock_neo4j)
        assert retriever.config is not None
        assert retriever.vector_retriever is not None
        assert retriever.graph_retriever is not None
        assert retriever.hybrid_config.parallel_execution is True


class TestStrategySelection:
    """Tests for automatic strategy selection based on query intent."""

    def test_select_strategy_semantic_no_entities(
        self, hybrid_retriever: HybridRetriever, sample_semantic_query: ParsedQuery
    ) -> None:
        """Test strategy selection for semantic query without entities."""
        strategy = hybrid_retriever._select_strategy(sample_semantic_query)
        assert strategy == RetrievalStrategy.VECTOR_ONLY

    def test_select_strategy_structural_with_entities(
        self, hybrid_retriever: HybridRetriever, sample_parsed_query: ParsedQuery
    ) -> None:
        """Test strategy selection for structural query with entities."""
        strategy = hybrid_retriever._select_strategy(sample_parsed_query)
        # Should select graph-first or hybrid for structural queries with entities
        assert strategy in [RetrievalStrategy.GRAPH_FIRST, RetrievalStrategy.HYBRID_PARALLEL]

    def test_select_strategy_procedural(self, hybrid_retriever: HybridRetriever) -> None:
        """Test strategy selection for procedural query."""
        query = ParsedQuery(
            query_id="test_procedural",
            original_text="How to perform battery health check?",
            normalized_text="how to perform battery health check",
            intent=QueryIntent.PROCEDURAL,
            intent_confidence=0.9,
            entity_mentions=[],
            relationship_types=[RelationshipType.PRECEDES],
            constraints=[],
            expanded_terms={},
            keywords=["perform", "battery", "health", "check"],
            requires_graph_traversal=True,
            max_depth=3,
            timestamp=datetime.now(),
            metadata={},
        )
        strategy = hybrid_retriever._select_strategy(query)
        assert strategy == RetrievalStrategy.GRAPH_FIRST

    def test_select_strategy_hybrid_intent(self, hybrid_retriever: HybridRetriever) -> None:
        """Test strategy selection for hybrid query intent."""
        query = ParsedQuery(
            query_id="test_hybrid",
            original_text="What is the battery and how is it connected?",
            normalized_text="what is the battery and how is it connected",
            intent=QueryIntent.HYBRID,
            intent_confidence=0.8,
            entity_mentions=[],
            relationship_types=[],
            constraints=[],
            expanded_terms={},
            keywords=["battery", "connected"],
            requires_graph_traversal=True,
            max_depth=3,
            timestamp=datetime.now(),
            metadata={},
        )
        strategy = hybrid_retriever._select_strategy(query)
        assert strategy == RetrievalStrategy.HYBRID_PARALLEL


class TestVectorOnlyRetrieval:
    """Tests for vector-only retrieval strategy."""

    def test_retrieve_vector_only(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        sample_semantic_query: ParsedQuery,
        sample_vector_result: RetrievalResult,
    ) -> None:
        """Test vector-only retrieval."""
        mock_vector_retriever.retrieve.return_value = sample_vector_result

        result = hybrid_retriever._retrieve_vector_only(
            sample_semantic_query, top_k=5, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.VECTOR_ONLY
        assert len(result.chunks) == 5
        assert result.vector_success is True
        assert result.graph_success is True  # Not attempted
        assert result.vector_results == 5
        assert result.graph_results == 0
        assert all(chunk.source == "vector" for chunk in result.chunks)

    def test_retrieve_vector_only_failure(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        sample_semantic_query: ParsedQuery,
    ) -> None:
        """Test vector-only retrieval with failure."""
        mock_vector_retriever.retrieve.side_effect = Exception("Vector retrieval failed")

        result = hybrid_retriever._retrieve_vector_only(
            sample_semantic_query, top_k=5, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.VECTOR_ONLY
        assert len(result.chunks) == 0
        assert result.vector_success is False


class TestGraphOnlyRetrieval:
    """Tests for graph-only retrieval strategy."""

    def test_retrieve_graph_only(
        self,
        hybrid_retriever: HybridRetriever,
        mock_graph_retriever: Mock,
        mock_neo4j: Mock,
        sample_parsed_query: ParsedQuery,
        sample_graph_result: GraphRetrievalResult,
    ) -> None:
        """Test graph-only retrieval."""
        mock_graph_retriever.retrieve.return_value = sample_graph_result

        # Mock chunk fetching
        mock_chunks = [
            {
                "id": f"chunk_g_{i}",
                "document_id": "doc_001",
                "content": f"Graph chunk content {i}",
                "level": 3,
                "metadata": {},
                "entity_ids": [f"entity_{i}"],
            }
            for i in range(6)
        ]
        mock_neo4j.execute_cypher.return_value = [
            {
                "id": c["id"],
                "document_id": c["document_id"],
                "content": c["content"],
                "level": c["level"],
                "metadata": c["metadata"],
                "entity_ids": c["entity_ids"],
            }
            for c in mock_chunks
        ]

        result = hybrid_retriever._retrieve_graph_only(sample_parsed_query, top_k=5, timeout=10.0)

        assert result.strategy_used == RetrievalStrategy.GRAPH_ONLY
        assert result.vector_success is True  # Not attempted
        assert result.graph_success is True
        assert result.vector_results == 0
        assert len(result.graph_paths) > 0


class TestHybridParallelRetrieval:
    """Tests for parallel hybrid retrieval strategy."""

    def test_retrieve_hybrid_parallel_both_succeed(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
        mock_neo4j: Mock,
        sample_parsed_query: ParsedQuery,
        sample_vector_result: RetrievalResult,
        sample_graph_result: GraphRetrievalResult,
    ) -> None:
        """Test hybrid parallel retrieval when both succeed."""
        mock_vector_retriever.retrieve.return_value = sample_vector_result
        mock_graph_retriever.retrieve.return_value = sample_graph_result

        # Mock chunk fetching for graph results
        mock_chunks = [
            {
                "id": f"chunk_g_{i}",
                "document_id": "doc_001",
                "content": f"Graph chunk content {i}",
                "level": 3,
                "metadata": {},
                "entity_ids": [f"entity_{i}"],
            }
            for i in range(6)
        ]
        mock_neo4j.execute_cypher.return_value = [
            {
                "id": c["id"],
                "document_id": c["document_id"],
                "content": c["content"],
                "level": c["level"],
                "metadata": c["metadata"],
                "entity_ids": c["entity_ids"],
            }
            for c in mock_chunks
        ]

        result = hybrid_retriever._retrieve_hybrid_parallel(
            sample_parsed_query, top_k=10, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.HYBRID_PARALLEL
        assert result.vector_success is True
        assert result.graph_success is True
        assert result.vector_results > 0
        assert result.graph_results > 0
        assert len(result.chunks) > 0
        assert result.reranking_enabled is True

    def test_retrieve_hybrid_parallel_vector_only_succeeds(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
        sample_parsed_query: ParsedQuery,
        sample_vector_result: RetrievalResult,
    ) -> None:
        """Test hybrid parallel retrieval when only vector succeeds."""
        mock_vector_retriever.retrieve.return_value = sample_vector_result
        mock_graph_retriever.retrieve.side_effect = Exception("Graph retrieval failed")

        result = hybrid_retriever._retrieve_hybrid_parallel(
            sample_parsed_query, top_k=10, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.HYBRID_PARALLEL
        assert result.vector_success is True
        assert result.graph_success is False
        assert len(result.chunks) > 0  # Should have vector results

    def test_retrieve_hybrid_parallel_graph_only_succeeds(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
        mock_neo4j: Mock,
        sample_parsed_query: ParsedQuery,
        sample_graph_result: GraphRetrievalResult,
    ) -> None:
        """Test hybrid parallel retrieval when only graph succeeds."""
        mock_vector_retriever.retrieve.side_effect = Exception("Vector retrieval failed")
        mock_graph_retriever.retrieve.return_value = sample_graph_result

        # Mock chunk fetching
        mock_chunks = [
            {
                "id": f"chunk_g_{i}",
                "document_id": "doc_001",
                "content": f"Graph chunk content {i}",
                "level": 3,
                "metadata": {},
                "entity_ids": [f"entity_{i}"],
            }
            for i in range(6)
        ]
        mock_neo4j.execute_cypher.return_value = [
            {
                "id": c["id"],
                "document_id": c["document_id"],
                "content": c["content"],
                "level": c["level"],
                "metadata": c["metadata"],
                "entity_ids": c["entity_ids"],
            }
            for c in mock_chunks
        ]

        result = hybrid_retriever._retrieve_hybrid_parallel(
            sample_parsed_query, top_k=10, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.HYBRID_PARALLEL
        assert result.vector_success is False
        assert result.graph_success is True
        assert len(result.chunks) > 0  # Should have graph results

    def test_retrieve_hybrid_parallel_both_fail(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
        sample_parsed_query: ParsedQuery,
    ) -> None:
        """Test hybrid parallel retrieval when both fail."""
        mock_vector_retriever.retrieve.side_effect = Exception("Vector retrieval failed")
        mock_graph_retriever.retrieve.side_effect = Exception("Graph retrieval failed")

        result = hybrid_retriever._retrieve_hybrid_parallel(
            sample_parsed_query, top_k=10, timeout=10.0
        )

        assert result.strategy_used == RetrievalStrategy.HYBRID_PARALLEL
        assert result.vector_success is False
        assert result.graph_success is False
        assert len(result.chunks) == 0


class TestScoreFusion:
    """Tests for score fusion and reranking."""

    def test_apply_score_fusion_with_both_scores(self, hybrid_retriever: HybridRetriever) -> None:
        """Test score fusion with both vector and graph scores."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=0.9,
                graph_score=0.7,
                entity_coverage_score=0.8,
                confidence_score=0.85,
                diversity_score=0.0,
                final_score=0.0,  # Will be computed
                rank=1,
                source="hybrid",
            )
        ]

        fused = hybrid_retriever._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0
        assert fused[0].final_score <= 1.0

    def test_apply_score_fusion_vector_only(self, hybrid_retriever: HybridRetriever) -> None:
        """Test score fusion with only vector score."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=0.9,
                graph_score=None,
                entity_coverage_score=0.0,
                confidence_score=0.5,
                diversity_score=0.0,
                final_score=0.0,
                rank=1,
                source="vector",
            )
        ]

        fused = hybrid_retriever._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0

    def test_apply_score_fusion_graph_only(self, hybrid_retriever: HybridRetriever) -> None:
        """Test score fusion with only graph score."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=None,
                graph_score=0.8,
                entity_coverage_score=0.6,
                confidence_score=0.7,
                diversity_score=0.0,
                final_score=0.0,
                rank=1,
                source="graph",
            )
        ]

        fused = hybrid_retriever._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0


class TestDiversityRanking:
    """Tests for diversity-aware ranking."""

    def test_apply_diversity_ranking(self, hybrid_retriever: HybridRetriever) -> None:
        """Test diversity ranking with similar content."""
        chunks = [
            HybridChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc_001",
                content=(
                    "power system thermal control management"
                    if i < 2
                    else "battery voltage monitoring"
                ),
                level=3,
                vector_score=0.9 - (i * 0.05),
                graph_score=0.8,
                entity_coverage_score=0.7,
                confidence_score=0.8,
                diversity_score=0.0,
                final_score=0.9 - (i * 0.05),
                rank=i + 1,
                source="hybrid",
            )
            for i in range(4)
        ]

        diverse = hybrid_retriever._apply_diversity_ranking(chunks)

        assert len(diverse) == 4
        # Check that diversity scores were computed
        assert all(chunk.diversity_score >= 0.0 for chunk in diverse)

    def test_content_similarity(self, hybrid_retriever: HybridRetriever) -> None:
        """Test content similarity calculation."""
        text1 = "power system thermal control"
        text2 = "power system management control"
        text3 = "battery voltage monitoring"

        sim_12 = hybrid_retriever._content_similarity(text1, text2)
        sim_13 = hybrid_retriever._content_similarity(text1, text3)

        # text1 and text2 should be more similar than text1 and text3
        assert sim_12 > sim_13
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0


class TestRetrieveMethod:
    """Tests for the main retrieve method."""

    def test_retrieve_with_auto_strategy(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        sample_semantic_query: ParsedQuery,
        sample_vector_result: RetrievalResult,
    ) -> None:
        """Test retrieve with automatic strategy selection."""
        mock_vector_retriever.retrieve.return_value = sample_vector_result

        result = hybrid_retriever.retrieve(sample_semantic_query, top_k=5)

        assert result is not None
        assert result.query_id == sample_semantic_query.query_id
        assert result.strategy_used in RetrievalStrategy
        assert isinstance(result.chunks, list)
        assert result.retrieval_time_ms > 0

    def test_retrieve_with_forced_strategy(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        sample_semantic_query: ParsedQuery,
        sample_vector_result: RetrievalResult,
    ) -> None:
        """Test retrieve with forced strategy."""
        mock_vector_retriever.retrieve.return_value = sample_vector_result

        result = hybrid_retriever.retrieve(
            sample_semantic_query,
            strategy=RetrievalStrategy.VECTOR_ONLY,
            top_k=5,
        )

        assert result.strategy_used == RetrievalStrategy.VECTOR_ONLY


class TestStatistics:
    """Tests for retrieval statistics."""

    def test_get_statistics(
        self,
        hybrid_retriever: HybridRetriever,
        mock_vector_retriever: Mock,
        mock_graph_retriever: Mock,
    ) -> None:
        """Test getting hybrid retrieval statistics."""
        mock_vector_retriever.get_statistics.return_value = {"total_chunks": 1000}
        mock_graph_retriever.get_statistics.return_value = {"total_entities": 500}

        stats = hybrid_retriever.get_statistics()

        assert "hybrid_config" in stats
        assert "reranking_config" in stats
        assert "vector_retriever" in stats
        assert "graph_retriever" in stats
        assert stats["hybrid_config"]["enabled"] is True
        assert stats["reranking_config"]["enabled"] is True


class TestHybridChunkModel:
    """Tests for HybridChunk model."""

    def test_hybrid_chunk_creation(self) -> None:
        """Test creating HybridChunk instance."""
        chunk = HybridChunk(
            chunk_id="chunk_test_001",
            document_id="doc_001",
            content="Test content",
            level=3,
            vector_score=0.9,
            graph_score=0.8,
            final_score=0.85,
            rank=1,
            source="hybrid",
        )

        assert chunk.chunk_id == "chunk_test_001"
        assert chunk.vector_score == 0.9
        assert chunk.graph_score == 0.8
        assert chunk.final_score == 0.85
        assert chunk.source == "hybrid"

    def test_hybrid_chunk_to_dict(self) -> None:
        """Test HybridChunk to_dict method."""
        chunk = HybridChunk(
            chunk_id="chunk_test_001",
            document_id="doc_001",
            content="Test content",
            level=3,
            final_score=0.85,
            rank=1,
            source="vector",
        )

        chunk_dict = chunk.to_dict()

        assert isinstance(chunk_dict, dict)
        assert chunk_dict["chunk_id"] == "chunk_test_001"
        assert chunk_dict["final_score"] == 0.85


class TestHybridRetrievalResultModel:
    """Tests for HybridRetrievalResult model."""

    def test_hybrid_result_creation(self) -> None:
        """Test creating HybridRetrievalResult instance."""
        result = HybridRetrievalResult(
            query_id="test_query_123",
            query_text="test query",
            strategy_used=RetrievalStrategy.HYBRID_PARALLEL,
            chunks=[],
            graph_paths=[],
            total_results=10,
            vector_results=5,
            graph_results=5,
            merged_results=8,
            retrieval_time_ms=100.0,
            vector_success=True,
            graph_success=True,
        )

        assert result.query_id == "test_query_123"
        assert result.strategy_used == RetrievalStrategy.HYBRID_PARALLEL
        assert result.total_results == 10
        assert result.vector_success is True
        assert result.graph_success is True

    def test_hybrid_result_get_entity_ids(self) -> None:
        """Test getting entity IDs from result."""
        chunks = [
            HybridChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc_001",
                content="test",
                level=3,
                final_score=0.9,
                rank=i + 1,
                source="vector",
                entity_ids=[f"entity_{i}", f"entity_{i+1}"],
            )
            for i in range(3)
        ]

        result = HybridRetrievalResult(
            query_id="test",
            query_text="test",
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            chunks=chunks,
            graph_paths=[],
            total_results=3,
            retrieval_time_ms=50.0,
        )

        entity_ids = result.get_entity_ids()

        assert isinstance(entity_ids, set)
        assert len(entity_ids) > 0

    def test_hybrid_result_get_document_ids(self) -> None:
        """Test getting document IDs from result."""
        chunks = [
            HybridChunk(
                chunk_id=f"chunk_{i}",
                document_id=f"doc_00{i}",
                content="test",
                level=3,
                final_score=0.9,
                rank=i + 1,
                source="vector",
            )
            for i in range(3)
        ]

        result = HybridRetrievalResult(
            query_id="test",
            query_text="test",
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            chunks=chunks,
            graph_paths=[],
            total_results=3,
            retrieval_time_ms=50.0,
        )

        doc_ids = result.get_document_ids()

        assert isinstance(doc_ids, set)
        assert len(doc_ids) == 3
