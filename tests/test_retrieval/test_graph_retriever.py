"""Tests for graph retriever (Task 4.3)."""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.retrieval.graph_retriever import (
    GraphPath,
    GraphRetriever,
    GraphRetrievalResult,
    ResolvedEntity,
    TraversalStrategy,
)
from src.retrieval.query_parser import EntityMention, ParsedQuery, QueryIntent
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
    neo4j.get_entity_by_canonical_name.return_value = None
    neo4j.search_entities.return_value = []
    neo4j.get_entity.return_value = None
    neo4j.find_path.return_value = []
    neo4j.traverse_relationships.return_value = []
    neo4j.get_relationships.return_value = []
    neo4j.execute_cypher.return_value = []
    neo4j.get_statistics.return_value = {}
    return neo4j


@pytest.fixture
def graph_retriever(config: Config, mock_neo4j: Mock) -> GraphRetriever:
    """Create graph retriever with mock Neo4j."""
    return GraphRetriever(config=config, neo4j_manager=mock_neo4j)


@pytest.fixture
def sample_parsed_query() -> ParsedQuery:
    """Create a sample parsed query for testing."""
    return ParsedQuery(
        query_id="test_query_123",
        original_text="What are the components of the Electrical Power System?",
        normalized_text="what are the components of the electrical power system",
        intent=QueryIntent.STRUCTURAL,
        intent_confidence=0.9,
        entity_mentions=[
            EntityMention(
                text="Electrical Power System",
                normalized="electrical_power_system",
                entity_type=EntityType.SYSTEM,
                start_char=30,
                end_char=53,
                confidence=0.9,
            )
        ],
        relationship_types=[RelationshipType.CONTAINS],
        constraints=[],
        expanded_terms={},
        keywords=["components", "electrical", "power", "system"],
        requires_graph_traversal=True,
        max_depth=3,
        timestamp=datetime.now(),
        metadata={},
    )


@pytest.fixture
def sample_entity() -> Dict[str, Any]:
    """Create a sample entity for testing."""
    return {
        "id": "entity_eps_001",
        "canonical_name": "electrical_power_system",
        "entity_type": EntityType.SYSTEM.value,
        "description": "Main power distribution system",
        "confidence_score": 0.9,
        "source_documents": ["doc_001"],
    }


@pytest.fixture
def sample_related_entity() -> Dict[str, Any]:
    """Create a sample related entity for testing."""
    return {
        "id": "entity_battery_001",
        "canonical_name": "battery_subsystem",
        "entity_type": EntityType.SUBSYSTEM.value,
        "description": "Battery power storage",
        "confidence_score": 0.85,
        "depth": 1,
    }


class TestGraphRetrieverInitialization:
    """Tests for GraphRetriever initialization."""

    def test_init_with_config(self, config: Config, mock_neo4j: Mock) -> None:
        """Test initialization with configuration."""
        retriever = GraphRetriever(config=config, neo4j_manager=mock_neo4j)
        assert retriever.config is not None
        assert retriever.neo4j is mock_neo4j
        assert retriever.graph_config is not None

    def test_init_with_defaults(self, mock_neo4j: Mock) -> None:
        """Test initialization with default configuration."""
        retriever = GraphRetriever(neo4j_manager=mock_neo4j)
        assert retriever.config is not None
        assert retriever.graph_config.max_depth > 0


class TestEntityResolution:
    """Tests for entity resolution from query mentions."""

    def test_resolve_entity_exact_match(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock, sample_entity: Dict[str, Any]
    ) -> None:
        """Test entity resolution with exact canonical name match."""
        mock_neo4j.get_entity_by_canonical_name.return_value = sample_entity

        mention = EntityMention(
            text="Electrical Power System",
            normalized="electrical_power_system",
            entity_type=EntityType.SYSTEM,
            start_char=0,
            end_char=23,
            confidence=0.9,
        )

        resolved = graph_retriever._resolve_entities([mention])

        assert len(resolved) == 1
        assert resolved[0].entity_id == sample_entity["id"]
        assert resolved[0].canonical_name == sample_entity["canonical_name"]
        assert resolved[0].confidence >= 0.9
        assert resolved[0].match_method == "exact"

    def test_resolve_entity_search_match(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock, sample_entity: Dict[str, Any]
    ) -> None:
        """Test entity resolution with full-text search."""
        mock_neo4j.get_entity_by_canonical_name.return_value = None
        sample_entity["search_score"] = 0.85
        mock_neo4j.search_entities.return_value = [sample_entity]

        mention = EntityMention(
            text="Power System",
            normalized="power_system",
            entity_type=EntityType.SYSTEM,
            start_char=0,
            end_char=12,
            confidence=0.8,
        )

        resolved = graph_retriever._resolve_entities([mention])

        assert len(resolved) == 1
        assert resolved[0].entity_id == sample_entity["id"]
        assert resolved[0].match_method == "search"
        assert resolved[0].confidence <= 0.9

    def test_resolve_entity_no_match(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock
    ) -> None:
        """Test entity resolution with no matches found."""
        mock_neo4j.get_entity_by_canonical_name.return_value = None
        mock_neo4j.search_entities.return_value = []

        mention = EntityMention(
            text="Unknown System",
            normalized="unknown_system",
            entity_type=EntityType.SYSTEM,
            start_char=0,
            end_char=14,
            confidence=0.7,
        )

        resolved = graph_retriever._resolve_entities([mention])

        assert len(resolved) == 0

    def test_resolve_multiple_entities(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock, sample_entity: Dict[str, Any]
    ) -> None:
        """Test resolving multiple entity mentions."""
        entity2 = {
            "id": "entity_battery_001",
            "canonical_name": "battery_subsystem",
            "entity_type": EntityType.SUBSYSTEM.value,
            "search_score": 0.8,
        }

        def get_entity_side_effect(canonical_name, entity_type):
            if canonical_name == "electrical_power_system":
                return sample_entity
            return None

        mock_neo4j.get_entity_by_canonical_name.side_effect = get_entity_side_effect
        mock_neo4j.search_entities.return_value = [entity2]

        mentions = [
            EntityMention(
                text="Electrical Power System",
                normalized="electrical_power_system",
                entity_type=EntityType.SYSTEM,
                start_char=0,
                end_char=23,
                confidence=0.9,
            ),
            EntityMention(
                text="Battery",
                normalized="battery",
                entity_type=EntityType.SUBSYSTEM,
                start_char=30,
                end_char=37,
                confidence=0.8,
            ),
        ]

        resolved = graph_retriever._resolve_entities(mentions)

        assert len(resolved) == 2
        assert resolved[0].entity_id == sample_entity["id"]
        assert resolved[1].entity_id == entity2["id"]

    def test_resolve_entity_deduplication(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock, sample_entity: Dict[str, Any]
    ) -> None:
        """Test that duplicate entity resolutions are deduplicated."""
        mock_neo4j.get_entity_by_canonical_name.return_value = sample_entity

        mentions = [
            EntityMention(
                text="Power System",
                normalized="electrical_power_system",
                entity_type=EntityType.SYSTEM,
                start_char=0,
                end_char=12,
                confidence=0.9,
            ),
            EntityMention(
                text="EPS",
                normalized="electrical_power_system",
                entity_type=EntityType.SYSTEM,
                start_char=20,
                end_char=23,
                confidence=0.8,
            ),
        ]

        resolved = graph_retriever._resolve_entities(mentions)

        # Should only have one resolved entity despite two mentions
        assert len(resolved) == 1
        assert resolved[0].entity_id == sample_entity["id"]


class TestStrategySelection:
    """Tests for traversal strategy selection."""

    def test_select_sequential_strategy(self, graph_retriever: GraphRetriever) -> None:
        """Test selection of sequential strategy for PRECEDES relationships."""
        query = ParsedQuery(
            query_id="test",
            original_text="test",
            normalized_text="test",
            intent=QueryIntent.PROCEDURAL,
            relationship_types=[RelationshipType.PRECEDES],
            entity_mentions=[],
            timestamp=datetime.now(),
        )

        strategy = graph_retriever._select_strategy(
            query, [RelationshipType.PRECEDES, RelationshipType.REFERENCES]
        )

        assert strategy == TraversalStrategy.SEQUENTIAL

    def test_select_procedural_strategy(self, graph_retriever: GraphRetriever) -> None:
        """Test selection of procedural strategy for REFERENCES relationships."""
        query = ParsedQuery(
            query_id="test",
            original_text="test",
            normalized_text="test",
            intent=QueryIntent.PROCEDURAL,
            relationship_types=[RelationshipType.REFERENCES],
            entity_mentions=[],
            timestamp=datetime.now(),
        )

        strategy = graph_retriever._select_strategy(query, [RelationshipType.REFERENCES])

        assert strategy == TraversalStrategy.PROCEDURAL

    def test_select_hierarchical_strategy(self, graph_retriever: GraphRetriever) -> None:
        """Test selection of hierarchical strategy for CONTAINS relationships."""
        query = ParsedQuery(
            query_id="test",
            original_text="test",
            normalized_text="test",
            intent=QueryIntent.STRUCTURAL,
            relationship_types=[RelationshipType.CONTAINS],
            entity_mentions=[],
            timestamp=datetime.now(),
        )

        strategy = graph_retriever._select_strategy(query, [RelationshipType.CONTAINS])

        assert strategy == TraversalStrategy.HIERARCHICAL

    def test_select_shortest_path_strategy(
        self, graph_retriever: GraphRetriever, config: Config
    ) -> None:
        """Test selection of shortest path strategy for multi-entity queries."""
        query = ParsedQuery(
            query_id="test",
            original_text="test",
            normalized_text="test",
            intent=QueryIntent.HYBRID,
            relationship_types=[],
            entity_mentions=[
                EntityMention(
                    text="Entity1",
                    normalized="entity1",
                    start_char=0,
                    end_char=7,
                    confidence=0.9,
                ),
                EntityMention(
                    text="Entity2",
                    normalized="entity2",
                    start_char=10,
                    end_char=17,
                    confidence=0.9,
                ),
            ],
            timestamp=datetime.now(),
        )

        # Enable shortest path in config
        config.retrieval.graph_search.enable_shortest_path = True
        retriever = GraphRetriever(config=config, neo4j_manager=Mock())

        strategy = retriever._select_strategy(query, [RelationshipType.DEPENDS_ON])

        assert strategy == TraversalStrategy.SHORTEST_PATH


class TestTraversalExecution:
    """Tests for graph traversal execution."""

    def test_hierarchical_traversal(
        self,
        graph_retriever: GraphRetriever,
        mock_neo4j: Mock,
        sample_entity: Dict[str, Any],
        sample_related_entity: Dict[str, Any],
    ) -> None:
        """Test hierarchical traversal (CONTAINS, PART_OF)."""
        resolved = [
            ResolvedEntity(
                entity_id=sample_entity["id"],
                canonical_name=sample_entity["canonical_name"],
                entity_type=EntityType.SYSTEM,
                mention_text="EPS",
                confidence=0.9,
                match_method="exact",
            )
        ]

        # Mock Neo4j responses
        mock_neo4j.traverse_relationships.return_value = [sample_related_entity]
        mock_neo4j.get_entity.side_effect = lambda eid: (
            sample_entity if eid == sample_entity["id"] else sample_related_entity
        )

        paths = graph_retriever._traverse_hierarchical(resolved, max_depth=2)

        assert len(paths) >= 1
        assert mock_neo4j.traverse_relationships.called

    def test_sequential_traversal(
        self,
        graph_retriever: GraphRetriever,
        mock_neo4j: Mock,
        sample_entity: Dict[str, Any],
        sample_related_entity: Dict[str, Any],
    ) -> None:
        """Test sequential traversal (PRECEDES)."""
        resolved = [
            ResolvedEntity(
                entity_id=sample_entity["id"],
                canonical_name=sample_entity["canonical_name"],
                entity_type=EntityType.PROCEDURE,
                mention_text="Startup Procedure",
                confidence=0.9,
                match_method="exact",
            )
        ]

        mock_neo4j.traverse_relationships.return_value = [sample_related_entity]
        mock_neo4j.get_entity.side_effect = lambda eid: (
            sample_entity if eid == sample_entity["id"] else sample_related_entity
        )

        paths = graph_retriever._traverse_sequential(resolved, max_depth=3)

        assert len(paths) >= 1
        assert mock_neo4j.traverse_relationships.call_count >= 2  # Forward and backward

    def test_shortest_path_traversal(
        self,
        graph_retriever: GraphRetriever,
        mock_neo4j: Mock,
        sample_entity: Dict[str, Any],
    ) -> None:
        """Test shortest path finding between entities."""
        entity2 = {
            "id": "entity_002",
            "canonical_name": "entity2",
            "entity_type": EntityType.COMPONENT.value,
        }

        resolved = [
            ResolvedEntity(
                entity_id=sample_entity["id"],
                canonical_name=sample_entity["canonical_name"],
                entity_type=EntityType.SYSTEM,
                mention_text="System",
                confidence=0.9,
                match_method="exact",
            ),
            ResolvedEntity(
                entity_id=entity2["id"],
                canonical_name=entity2["canonical_name"],
                entity_type=EntityType.COMPONENT,
                mention_text="Component",
                confidence=0.85,
                match_method="exact",
            ),
        ]

        # Mock path result
        mock_path = {
            "nodes": [sample_entity, entity2],
            "relationships": [{"type": RelationshipType.CONTAINS.value, "id": "rel_001"}],
            "length": 1,
        }
        mock_neo4j.find_path.return_value = [mock_path]

        paths = graph_retriever._find_shortest_paths(
            resolved, max_depth=3, relationship_types=[RelationshipType.CONTAINS]
        )

        assert len(paths) >= 1
        assert mock_neo4j.find_path.called


class TestPathScoring:
    """Tests for path scoring algorithms."""

    def test_calculate_path_score_short_path(self, graph_retriever: GraphRetriever) -> None:
        """Test path score calculation for short paths."""
        score = graph_retriever._calculate_path_score(
            length=1, source_confidence=0.9, target_confidence=0.85
        )

        # Short paths should have good scores (length=1: distance_score=0.5, confidence=0.875 -> 0.65)
        assert 0.6 <= score <= 1.0

    def test_calculate_path_score_long_path(self, graph_retriever: GraphRetriever) -> None:
        """Test path score calculation for long paths."""
        score = graph_retriever._calculate_path_score(
            length=5, source_confidence=0.9, target_confidence=0.85
        )

        # Long paths should have lower scores
        assert 0.0 <= score <= 0.6

    def test_calculate_path_score_low_confidence(self, graph_retriever: GraphRetriever) -> None:
        """Test path score calculation with low entity confidence."""
        score = graph_retriever._calculate_path_score(
            length=1, source_confidence=0.5, target_confidence=0.5
        )

        # Low confidence should reduce score
        assert 0.0 <= score <= 0.7


class TestGraphRetrieval:
    """Tests for end-to-end graph retrieval."""

    def test_retrieve_with_valid_query(
        self,
        graph_retriever: GraphRetriever,
        mock_neo4j: Mock,
        sample_parsed_query: ParsedQuery,
        sample_entity: Dict[str, Any],
        sample_related_entity: Dict[str, Any],
    ) -> None:
        """Test retrieval with valid parsed query."""
        # Mock entity resolution
        mock_neo4j.get_entity_by_canonical_name.return_value = sample_entity

        # Mock traversal
        mock_neo4j.traverse_relationships.return_value = [sample_related_entity]
        mock_neo4j.get_entity.side_effect = lambda eid: (
            sample_entity if eid == sample_entity["id"] else sample_related_entity
        )

        result = graph_retriever.retrieve(sample_parsed_query)

        assert isinstance(result, GraphRetrievalResult)
        assert result.query_id == sample_parsed_query.query_id
        assert result.query_text == sample_parsed_query.original_text
        assert len(result.resolved_entities) >= 1
        assert result.retrieval_time_ms >= 0

    def test_retrieve_with_no_entities_resolved(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock, sample_parsed_query: ParsedQuery
    ) -> None:
        """Test retrieval when no entities can be resolved."""
        mock_neo4j.get_entity_by_canonical_name.return_value = None
        mock_neo4j.search_entities.return_value = []

        result = graph_retriever.retrieve(sample_parsed_query)

        assert len(result.resolved_entities) == 0
        assert len(result.paths) == 0
        assert len(result.chunk_ids) == 0

    def test_retrieve_with_string_query_raises_error(self, graph_retriever: GraphRetriever) -> None:
        """Test that passing a string query raises ValueError."""
        with pytest.raises(ValueError, match="requires a ParsedQuery object"):
            graph_retriever.retrieve("What is the power system?")

    def test_retrieve_with_custom_depth(
        self,
        graph_retriever: GraphRetriever,
        mock_neo4j: Mock,
        sample_parsed_query: ParsedQuery,
        sample_entity: Dict[str, Any],
    ) -> None:
        """Test retrieval with custom max depth."""
        mock_neo4j.get_entity_by_canonical_name.return_value = sample_entity
        mock_neo4j.traverse_relationships.return_value = []

        result = graph_retriever.retrieve(sample_parsed_query, max_depth=5)

        assert result.max_depth == 5


class TestChunkExtraction:
    """Tests for chunk extraction from graph results."""

    def test_get_chunks_for_entities(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock
    ) -> None:
        """Test extracting chunks for given entity IDs."""
        entity_ids = {"entity_001", "entity_002"}
        mock_chunks = [
            {
                "id": "chunk_001",
                "content": "Test content",
                "entity_ids": ["entity_001"],
                "level": 2,
            },
            {
                "id": "chunk_002",
                "content": "More content",
                "entity_ids": ["entity_002"],
                "level": 3,
            },
        ]

        mock_neo4j.execute_cypher.return_value = [{"c": chunk} for chunk in mock_chunks]

        chunks = graph_retriever.get_chunks_for_entities(entity_ids)

        assert len(chunks) == 2
        assert mock_neo4j.execute_cypher.called

    def test_get_chunks_for_empty_entities(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock
    ) -> None:
        """Test extracting chunks with empty entity set."""
        chunks = graph_retriever.get_chunks_for_entities(set())

        assert len(chunks) == 0
        assert not mock_neo4j.execute_cypher.called


class TestStatistics:
    """Tests for retrieval statistics."""

    def test_get_statistics(self, graph_retriever: GraphRetriever, mock_neo4j: Mock) -> None:
        """Test getting retrieval statistics."""
        mock_neo4j.get_statistics.return_value = {
            "total_entities": 100,
            "total_relationships": 250,
        }

        stats = graph_retriever.get_statistics()

        assert "total_entities" in stats
        assert "total_relationships" in stats
        assert "graph_config" in stats
        assert stats["graph_config"]["max_depth"] > 0

    def test_get_statistics_failure(
        self, graph_retriever: GraphRetriever, mock_neo4j: Mock
    ) -> None:
        """Test statistics retrieval with database error."""
        mock_neo4j.get_statistics.side_effect = Exception("Database error")

        stats = graph_retriever.get_statistics()

        # Should return empty dict on failure
        assert stats == {}
