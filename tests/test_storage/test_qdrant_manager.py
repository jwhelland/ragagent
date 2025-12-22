"""Tests for QdrantManager class.

This test suite verifies the functionality of the Qdrant vector database manager
including collection creation, vector operations, search, and batch operations.
"""

import uuid
from datetime import UTC, datetime
from typing import List

import pytest

from src.storage.qdrant_manager import QdrantManager
from src.utils.config import DatabaseConfig


@pytest.fixture
def db_config() -> DatabaseConfig:
    """Create test database configuration."""
    # Use in-memory/local Qdrant so tests don't require a running Qdrant server.
    return DatabaseConfig(
        qdrant_location=":memory:",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_api_key="",
        qdrant_https=False,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=768,
    )


@pytest.fixture
def qdrant_manager(db_config: DatabaseConfig) -> QdrantManager:
    """Create QdrantManager instance for testing."""
    manager = QdrantManager(
        config=db_config,
        chunk_collection="test_document_chunks",
        entity_collection="test_entities",
    )
    yield manager
    # Cleanup after tests
    manager.close()


@pytest.fixture
def sample_chunks() -> List[dict]:
    """Generate sample document chunks for testing."""
    doc_id = str(uuid.uuid4())
    chunks = []

    for i in range(5):
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": doc_id,
            "level": 3,  # subsection level
            "content": f"This is test chunk {i} about technical systems and procedures.",
            "metadata": {
                "document_title": "Test Technical Manual",
                "section_title": f"Section {i}",
                "page_numbers": [i + 1, i + 2],
                "hierarchy_path": f"1.{i}.1",
                "entity_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
                "has_tables": False,
                "has_figures": i % 2 == 0,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
        chunks.append(chunk)

    return chunks


@pytest.fixture
def sample_vectors(sample_chunks: List[dict]) -> List[List[float]]:
    """Generate sample embedding vectors (768-dimensional for BGE)."""
    import random

    random.seed(42)
    vectors = []
    for _ in sample_chunks:
        # Create random normalized vector
        vector = [random.random() for _ in range(768)]
        # Normalize
        magnitude = sum(x**2 for x in vector) ** 0.5
        vector = [x / magnitude for x in vector]
        vectors.append(vector)

    return vectors


@pytest.fixture
def sample_entities() -> List[dict]:
    """Generate sample entities for testing."""
    entities = [
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "power_subsystem",
            "entity_type": "SYSTEM",
            "description": "The electrical power subsystem manages power distribution",
            "aliases": ["EPS", "Electrical Power System", "Power System"],
            "related_entity_ids": [str(uuid.uuid4())],
        },
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "solar_array",
            "entity_type": "SUBSYSTEM",
            "description": "Solar arrays convert sunlight to electrical power",
            "aliases": ["Solar Panel", "SA"],
            "related_entity_ids": [],
        },
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "battery_pack",
            "entity_type": "COMPONENT",
            "description": "Battery pack stores electrical energy for eclipse operations",
            "aliases": ["Battery", "Energy Storage"],
            "related_entity_ids": [str(uuid.uuid4())],
        },
    ]
    return entities


class TestQdrantManagerInit:
    """Tests for QdrantManager initialization."""

    def test_init_with_valid_config(self, db_config: DatabaseConfig) -> None:
        """Test initialization with valid configuration."""
        manager = QdrantManager(config=db_config)
        assert manager.config == db_config
        assert manager.chunk_collection == "document_chunks"
        assert manager.entity_collection == "entities"
        manager.close()

    def test_init_with_custom_collections(self, db_config: DatabaseConfig) -> None:
        """Test initialization with custom collection names."""
        manager = QdrantManager(
            config=db_config,
            chunk_collection="custom_chunks",
            entity_collection="custom_entities",
        )
        assert manager.chunk_collection == "custom_chunks"
        assert manager.entity_collection == "custom_entities"
        manager.close()

    def test_init_connection_error(self) -> None:
        """Test initialization with invalid connection details."""
        bad_config = DatabaseConfig(
            qdrant_host="invalid_host",
            qdrant_port=9999,
            embedding_dimension=768,
        )
        with pytest.raises(ConnectionError):
            QdrantManager(config=bad_config)


class TestCollectionManagement:
    """Tests for collection creation and management."""

    def test_create_collections(self, qdrant_manager: QdrantManager) -> None:
        """Test creating both collections."""
        qdrant_manager.create_collections(recreate=True)

        assert qdrant_manager.collection_exists(qdrant_manager.chunk_collection)
        assert qdrant_manager.collection_exists(qdrant_manager.entity_collection)

    def test_recreate_collections(self, qdrant_manager: QdrantManager) -> None:
        """Test recreating collections (should delete and recreate)."""
        # Create once
        qdrant_manager.create_collections(recreate=True)

        # Recreate
        qdrant_manager.create_collections(recreate=True)

        # Should still exist
        assert qdrant_manager.collection_exists(qdrant_manager.chunk_collection)

    def test_collection_exists(self, qdrant_manager: QdrantManager) -> None:
        """Test checking if collections exist."""
        qdrant_manager.create_collections(recreate=True)

        assert qdrant_manager.collection_exists(qdrant_manager.chunk_collection) is True
        assert qdrant_manager.collection_exists("nonexistent_collection") is False

    def test_get_collection_info(self, qdrant_manager: QdrantManager) -> None:
        """Test retrieving collection information."""
        qdrant_manager.create_collections(recreate=True)

        info = qdrant_manager.get_collection_info(qdrant_manager.chunk_collection)

        assert "name" in info
        assert "vectors_count" in info
        assert "points_count" in info
        assert "config" in info
        assert info["config"]["distance"] == "Cosine"
        assert info["config"]["size"] == 768


class TestChunkOperations:
    """Tests for chunk upsert and search operations."""

    def test_upsert_chunks(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test upserting chunks with embeddings."""
        qdrant_manager.create_collections(recreate=True)

        count = qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        assert count == len(sample_chunks)

    def test_upsert_chunks_mismatch_length(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test upserting with mismatched lengths raises error."""
        qdrant_manager.create_collections(recreate=True)

        with pytest.raises(ValueError, match="must match"):
            qdrant_manager.upsert_chunks(sample_chunks, sample_vectors[:2])

    def test_upsert_empty_chunks(self, qdrant_manager: QdrantManager) -> None:
        """Test upserting empty lists."""
        qdrant_manager.create_collections(recreate=True)

        count = qdrant_manager.upsert_chunks([], [])
        assert count == 0

    def test_search_chunks(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching for similar chunks."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        # Search using the first vector
        results = qdrant_manager.search_chunks(
            query_vector=sample_vectors[0], top_k=3, score_threshold=0.5
        )

        assert len(results) > 0
        assert results[0]["score"] >= 0.5
        assert "payload" in results[0]
        assert "chunk_id" in results[0]

    def test_search_chunks_with_filters(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching with filters."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        # Filter by document_id
        doc_id = sample_chunks[0]["document_id"]
        results = qdrant_manager.search_chunks(
            query_vector=sample_vectors[0],
            top_k=10,
            filters={"document_id": doc_id},
        )

        # All results should be from the same document
        for result in results:
            assert result["payload"]["document_id"] == doc_id

    def test_get_chunk_by_id(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test retrieving a chunk by ID."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        chunk_id = sample_chunks[0]["chunk_id"]
        chunk = qdrant_manager.get_chunk_by_id(chunk_id)

        assert chunk is not None
        assert chunk["chunk_id"] == chunk_id
        assert "payload" in chunk

    def test_delete_chunks(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test deleting chunks by ID."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        chunk_ids = [sample_chunks[0]["chunk_id"], sample_chunks[1]["chunk_id"]]
        success = qdrant_manager.delete_chunks(chunk_ids)

        assert success is True

        # Verify deletion
        chunk = qdrant_manager.get_chunk_by_id(chunk_ids[0])
        assert chunk is None

    def test_delete_chunks_by_document(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test deleting all chunks from a document."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        doc_id = sample_chunks[0]["document_id"]
        success = qdrant_manager.delete_chunks_by_document(doc_id)

        assert success is True

        # Verify all chunks deleted
        for chunk in sample_chunks:
            result = qdrant_manager.get_chunk_by_id(chunk["chunk_id"])
            assert result is None


class TestEntityOperations:
    """Tests for entity upsert and search operations."""

    def test_upsert_entities(
        self,
        qdrant_manager: QdrantManager,
        sample_entities: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test upserting entities with embeddings."""
        qdrant_manager.create_collections(recreate=True)

        # Use only 3 vectors for 3 entities
        count = qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        assert count == len(sample_entities)

    def test_search_entities(
        self,
        qdrant_manager: QdrantManager,
        sample_entities: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching for similar entities."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        # Search using the first vector
        results = qdrant_manager.search_entities(
            query_vector=sample_vectors[0], top_k=2, score_threshold=0.5
        )

        assert len(results) > 0
        assert results[0]["score"] >= 0.5
        assert "payload" in results[0]
        assert "entity_id" in results[0]

    def test_search_entities_with_type_filter(
        self,
        qdrant_manager: QdrantManager,
        sample_entities: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching entities with type filtering."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        # Search for SYSTEM type only
        results = qdrant_manager.search_entities(
            query_vector=sample_vectors[0],
            top_k=10,
            entity_types=["SYSTEM"],
        )

        # All results should be SYSTEM type
        for result in results:
            assert result["payload"]["entity_type"] == "SYSTEM"

    def test_get_entity_by_id(
        self,
        qdrant_manager: QdrantManager,
        sample_entities: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test retrieving an entity by ID."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        entity_id = sample_entities[0]["entity_id"]
        entity = qdrant_manager.get_entity_by_id(entity_id)

        assert entity is not None
        assert entity["entity_id"] == entity_id
        assert "payload" in entity

    def test_delete_entities(
        self,
        qdrant_manager: QdrantManager,
        sample_entities: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test deleting entities by ID."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        entity_ids = [sample_entities[0]["entity_id"]]
        success = qdrant_manager.delete_entities(entity_ids)

        assert success is True

        # Verify deletion
        entity = qdrant_manager.get_entity_by_id(entity_ids[0])
        assert entity is None


class TestBatchOperations:
    """Tests for batch search operations."""

    def test_batch_search_chunks(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
    ) -> None:
        """Test batch searching for multiple queries."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)

        # Batch search with 3 queries
        query_vectors = sample_vectors[:3]
        results = qdrant_manager.batch_search_chunks(
            query_vectors=query_vectors, top_k=2, score_threshold=0.5
        )

        assert len(results) == len(query_vectors)
        for result_set in results:
            assert len(result_set) > 0


class TestHealthAndStats:
    """Tests for health checks and statistics."""

    def test_health_check_no_collections(self, qdrant_manager: QdrantManager) -> None:
        """Test health check when collections don't exist."""
        # Ensure collections don't exist
        qdrant_manager._delete_collection_if_exists(qdrant_manager.chunk_collection)
        qdrant_manager._delete_collection_if_exists(qdrant_manager.entity_collection)

        is_healthy, message = qdrant_manager.health_check()

        assert is_healthy is False
        assert "missing collections" in message.lower()

    def test_health_check_with_collections(self, qdrant_manager: QdrantManager) -> None:
        """Test health check when collections exist."""
        qdrant_manager.create_collections(recreate=True)

        is_healthy, message = qdrant_manager.health_check()

        assert is_healthy is True
        assert "healthy" in message.lower()

    def test_get_stats(
        self,
        qdrant_manager: QdrantManager,
        sample_chunks: List[dict],
        sample_vectors: List[List[float]],
        sample_entities: List[dict],
    ) -> None:
        """Test getting statistics."""
        qdrant_manager.create_collections(recreate=True)
        qdrant_manager.upsert_chunks(sample_chunks, sample_vectors)
        qdrant_manager.upsert_entities(sample_entities, sample_vectors[:3])

        stats = qdrant_manager.get_stats()

        assert "chunks" in stats
        assert "entities" in stats
        assert stats["chunks"]["count"] == len(sample_chunks)
        assert stats["entities"]["count"] == len(sample_entities)


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager(self, db_config: DatabaseConfig) -> None:
        """Test using QdrantManager as a context manager."""
        with QdrantManager(
            config=db_config,
            chunk_collection="test_chunks",
            entity_collection="test_entities",
        ) as manager:
            assert manager.client is not None

        # Client should be closed after context exit
        # (We can't easily verify this without accessing internals)
