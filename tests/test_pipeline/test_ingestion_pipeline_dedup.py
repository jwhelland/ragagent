"""Integration-style checks for deduplication wiring in IngestionPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.normalization.entity_deduplicator import (
    DeduplicationResult,
    EntityCluster,
    MergeSuggestion,
)
from src.pipeline.stages.extraction import ExtractionStage
from src.utils.config import Config


class _StubDeduplicator:
    def __init__(self, suggestion_count: int = 1) -> None:
        self.suggestion_count = suggestion_count

    def deduplicate(self, entities):
        if len(entities) < 2 or self.suggestion_count == 0:
            return DeduplicationResult(clusters=[], merge_suggestions=[])

        left, right = entities[0], entities[1]
        return DeduplicationResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entity_ids=[left.entity_id, right.entity_id],
                    representative_id=left.entity_id,
                    average_similarity=0.9,
                )
            ],
            merge_suggestions=[
                MergeSuggestion(
                    cluster_id=0,
                    source_id=left.entity_id,
                    target_id=right.entity_id,
                    entity_type=left.entity_type,
                    similarity=0.93,
                    confidence=0.91,
                    auto_merge=False,
                    reason="stub",
                )
            ],
        )


class _Chunk:
    def __init__(self, chunk_id: str, merged_entities: list[dict]) -> None:
        self.chunk_id = chunk_id
        self.metadata = {"merged_entities": merged_entities}


def test_dedup_suggestions_are_attached_to_merged_entities() -> None:
    cfg = Config()
    cfg.normalization.enable_semantic_matching = True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.pipeline.stages.extraction.EmbeddingGenerator", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.SpacyExtractor", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.LLMExtractor", MagicMock())

        stage = ExtractionStage(cfg)
        stage.entity_deduplicator = _StubDeduplicator()

        chunks = [
            _Chunk(
                "chunk-1",
                [
                    {
                        "canonical_name": "Attitude Control System",
                        "canonical_normalized": "attitude control system",
                        "type": "SYSTEM",
                        "aliases": ["ACS"],
                        "mention_count": 3,
                    },
                    {
                        "canonical_name": "Attitude Control Subsystem",
                        "canonical_normalized": "attitude control subsystem",
                        "type": "SYSTEM",
                        "aliases": ["ACS Subsystem"],
                        "mention_count": 1,
                    },
                ],
            )
        ]

        suggestions = stage._deduplicate_merged_entities(chunks)

        assert suggestions == 1
        merged_entities = chunks[0].metadata["merged_entities"]
        keys = [entity.get("candidate_key") for entity in merged_entities]
        assert all(keys)
        # Check that one of them has suggestions
        has_suggestion = False
        for entity in merged_entities:
            if entity.get("dedup_suggestions"):
                has_suggestion = True
                break
        assert has_suggestion, "expected dedup suggestions on at least one entity"
