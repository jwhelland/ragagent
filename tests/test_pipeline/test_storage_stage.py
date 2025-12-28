"""Unit tests for StorageStage entity candidate pre-resolution (Strategy 1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.curation.batch_operations import ApprovedEntityLookup
from src.pipeline.stages.storage import StorageStage
from src.utils.config import NormalizationConfig


class MockChunk:
    """Mock chunk for testing."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        merged_entities: list | None = None,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.metadata = {"merged_entities": merged_entities or []}


@pytest.fixture
def mock_config():
    """Create a mock config with proper normalization settings."""
    config = MagicMock()
    config.normalization = NormalizationConfig()
    return config


def test_store_entity_candidates_without_resolve_existing_stores_all(mock_config) -> None:
    """Test that all candidates are stored when resolve_existing=False."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=False)

    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            merged_entities=[
                {
                    "candidate_key": "system:battery_management_system",
                    "canonical_name": "Battery Management System",
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": ["BMS"],
                },
                {
                    "candidate_key": "system:thermal_control_system",
                    "canonical_name": "Thermal Control System",
                    "type": "SYSTEM",
                    "confidence": 0.8,
                    "aliases": ["TCS"],
                },
            ],
        ),
    ]

    stored, auto_resolved = stage._store_entity_candidates(chunks)

    assert stored == 2
    assert auto_resolved == 0
    assert neo4j_manager.upsert_entity_candidate_aggregate.call_count == 2


def test_store_entity_candidates_skips_matched_entities(mock_config) -> None:
    """Test that candidates matching approved entities are skipped."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=True)

    # Pre-populate the approved lookup
    stage._approved_lookup = ApprovedEntityLookup()
    stage._approved_lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            merged_entities=[
                {
                    "candidate_key": "system:battery_management_system",
                    "canonical_name": "Battery Management System",
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": ["BMS"],
                },
                {
                    "candidate_key": "system:thermal_control_system",
                    "canonical_name": "Thermal Control System",
                    "type": "SYSTEM",
                    "confidence": 0.8,
                    "aliases": ["TCS"],
                },
            ],
        ),
    ]

    stored, auto_resolved = stage._store_entity_candidates(chunks)

    # BMS should be auto-resolved, TCS should be stored
    assert stored == 1
    assert auto_resolved == 1
    assert neo4j_manager.upsert_entity_candidate_aggregate.call_count == 1


def test_store_entity_candidates_creates_mentioned_in_for_matched(mock_config) -> None:
    """Test that MENTIONED_IN relationships are created for auto-resolved candidates."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=True)

    # Pre-populate the approved lookup
    stage._approved_lookup = ApprovedEntityLookup()
    stage._approved_lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-new",
            merged_entities=[
                {
                    "candidate_key": "system:battery_management_system",
                    "canonical_name": "Battery Management System",
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": ["BMS"],
                },
            ],
        ),
    ]

    stored, auto_resolved = stage._store_entity_candidates(chunks)

    assert auto_resolved == 1
    # Should create MENTIONED_IN relationship
    neo4j_manager.create_mentioned_in_relationships.assert_called_once_with(
        "entity-123", ["doc-new"]
    )


def test_approved_lookup_loaded_lazily_and_cached(mock_config) -> None:
    """Test that approved entity lookup is loaded lazily and cached."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=True)

    # Initially no lookup
    assert stage._approved_lookup is None

    # Create a mock for build_approved_entity_lookup
    mock_lookup = ApprovedEntityLookup()
    mock_lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Test Entity",
        entity_type="SYSTEM",
        aliases=[],
    )

    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            merged_entities=[
                {
                    "candidate_key": "system:other_entity",
                    "canonical_name": "Other Entity",
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": [],
                },
            ],
        ),
    ]

    with patch(
        "src.curation.batch_operations.build_approved_entity_lookup",
        return_value=mock_lookup,
    ) as mock_build:
        # First call should load the lookup
        stage._store_entity_candidates(chunks)
        assert mock_build.call_count == 1

        # Second call should use cached lookup
        stage._store_entity_candidates(chunks)
        assert mock_build.call_count == 1  # Still 1, not 2


def test_store_entity_candidates_matches_by_alias(mock_config) -> None:
    """Test that candidates can be matched by alias."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=True)

    # Approved entity has alias "BMS"
    stage._approved_lookup = ApprovedEntityLookup()
    stage._approved_lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    # Candidate has canonical name that matches the alias
    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            merged_entities=[
                {
                    "candidate_key": "system:bms",
                    "canonical_name": "BMS",  # Matches the alias
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": [],
                },
            ],
        ),
    ]

    stored, auto_resolved = stage._store_entity_candidates(chunks)

    assert stored == 0
    assert auto_resolved == 1


def test_store_entity_candidates_type_must_match(mock_config) -> None:
    """Test that entity type must match for auto-resolution."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager, resolve_existing=True)

    # Approved entity is SYSTEM type
    stage._approved_lookup = ApprovedEntityLookup()
    stage._approved_lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    # Candidate has same name but different type
    chunks = [
        MockChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            merged_entities=[
                {
                    "candidate_key": "component:battery_management_system",
                    "canonical_name": "Battery Management System",
                    "type": "COMPONENT",  # Different type
                    "confidence": 0.9,
                    "aliases": ["BMS"],
                },
            ],
        ),
    ]

    stored, auto_resolved = stage._store_entity_candidates(chunks)

    # Should NOT auto-resolve because type doesn't match
    assert stored == 1
    assert auto_resolved == 0


def test_resolve_existing_default_false(mock_config) -> None:
    """Test that resolve_existing defaults to False."""
    neo4j_manager = MagicMock()
    qdrant_manager = MagicMock()

    stage = StorageStage(mock_config, neo4j_manager, qdrant_manager)

    assert stage.resolve_existing is False
    assert stage._approved_lookup is None
