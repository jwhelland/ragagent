"""Tests for EntityDeduplicator (Phase 3 Task 3.4)."""

from __future__ import annotations

import numpy as np

from src.normalization.entity_deduplicator import EntityDeduplicator, EntityRecord


class _StubEmbedder:
    def __init__(self, vectors: list[list[float]]):
        self.vectors = [np.array(vec, dtype=np.float32) for vec in vectors]

    def generate(self, texts: list[str]):
        return self.vectors[: len(texts)]


def test_clusters_similar_entities_and_ranks_merge_candidates() -> None:
    entities = [
        EntityRecord(
            entity_id="acs-1",
            name="Attitude Control System",
            description="Controls system orientation",
            entity_type="SYSTEM",
            mention_count=5,
        ),
        EntityRecord(
            entity_id="acs-2",
            name="Attitude Control Subsystem",
            description="Attitude control and stabilization",
            entity_type="SYSTEM",
            mention_count=2,
        ),
        EntityRecord(
            entity_id="thermal-1",
            name="Thermal Control",
            description="Thermal regulation loop",
            entity_type="SYSTEM",
        ),
    ]
    embedder = _StubEmbedder([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    deduplicator = EntityDeduplicator(embedder=embedder)

    result = deduplicator.deduplicate(entities)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert set(cluster.entity_ids) == {"acs-1", "acs-2"}

    suggestion = result.merge_suggestions[0]
    assert suggestion.source_id == "acs-1"  # representative chosen by mention count
    assert suggestion.target_id == "acs-2"
    assert suggestion.auto_merge is True
    assert suggestion.confidence > 0.9


def test_type_mismatch_blocks_auto_merge() -> None:
    entities = [
        EntityRecord(
            entity_id="comms-1", name="Comms Module", entity_type="SYSTEM", mention_count=3
        ),
        EntityRecord(
            entity_id="comms-2",
            name="Communications Payload",
            entity_type="COMPONENT",
            mention_count=2,
        ),
    ]
    embedder = _StubEmbedder([[1.0, 0.0], [1.0, 0.01]])
    deduplicator = EntityDeduplicator(embedder=embedder)

    result = deduplicator.deduplicate(entities)

    assert result.clusters == []
    assert result.merge_suggestions == []


def test_low_similarity_entities_are_not_clustered() -> None:
    entities = [
        EntityRecord(entity_id="power-1", name="Power System", description="Electrical power"),
        EntityRecord(entity_id="thermal-1", name="Thermal Control", description="Heat management"),
    ]
    embedder = _StubEmbedder([[1.0, 0.0], [0.0, 1.0]])
    deduplicator = EntityDeduplicator(embedder=embedder)

    result = deduplicator.deduplicate(entities)

    assert result.clusters == []
    assert result.merge_suggestions == []
