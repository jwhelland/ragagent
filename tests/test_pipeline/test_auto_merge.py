from unittest.mock import MagicMock

import pytest

from src.ingestion.chunker import Chunk
from src.normalization.entity_deduplicator import DeduplicationResult, MergeSuggestion
from src.pipeline.stages.extraction import ExtractionStage
from src.utils.config import Config


@pytest.fixture
def mock_config():
    config = Config()
    config.normalization.enable_semantic_matching = True
    config.normalization.auto_merge_threshold = 0.95
    # Mock database config to avoid connection attempts during init
    config.database.neo4j_uri = "bolt://localhost:7687"
    config.database.qdrant_host = "localhost"
    return config


def test_auto_merge(mock_config):
    """Test that auto-merge logic correctly merges candidates in chunks."""

    # 1. Setup Candidates
    # Candidate A: "NASA"
    cand_a = {
        "candidate_key": "org:nasa",
        "canonical_name": "NASA",
        "canonical_normalized": "nasa",
        "type": "ORG",
        "confidence": 0.9,
        "aliases": ["National Aeronautics"],
        "description": "Space agency",
        "mention_count": 10,
        "provenance": [{"source": "text", "confidence": 0.9}],
    }

    # Candidate B: "N.A.S.A." (Duplicate)
    cand_b = {
        "candidate_key": "org:n_a_s_a_",
        "canonical_name": "N.A.S.A.",
        "canonical_normalized": "n.a.s.a.",
        "type": "ORG",
        "confidence": 0.8,
        "aliases": ["Space Admin"],
        "description": "US Space Agency",
        "mention_count": 5,
        "provenance": [{"source": "text", "confidence": 0.8}],
    }

    # Candidate C: "SpaceX" (Unrelated)
    cand_c = {
        "candidate_key": "org:spacex",
        "canonical_name": "SpaceX",
        "canonical_normalized": "spacex",
        "type": "ORG",
        "confidence": 0.95,
        "mention_count": 8,
        "aliases": [],
        "description": "",
        "provenance": [],
    }

    # 2. Setup Chunk
    chunk = Chunk(
        chunk_id="chunk1",
        document_id="doc1",
        level=1,
        content="NASA and N.A.S.A. are the same.",
        metadata={"merged_entities": [cand_a, cand_b, cand_c]},
    )

    # 3. Setup Stage and Mock Deduplicator
    # Mock dependencies
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.pipeline.stages.extraction.EmbeddingGenerator", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.SpacyExtractor", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.LLMExtractor", MagicMock())

        stage = ExtractionStage(mock_config)
        stage.entity_deduplicator = MagicMock()

        # Define what deduplicator returns
        # It should return a suggestion to merge A and B
        suggestion = MergeSuggestion(
            cluster_id=1,
            source_id="org:nasa",  # Survivor (has higher mentions)
            target_id="org:n_a_s_a_",
            entity_type="ORG",
            similarity=0.99,
            confidence=0.99,
            auto_merge=True,
            reason="Test",
        )

        # First call returns suggestion
        result1 = DeduplicationResult(clusters=[], merge_suggestions=[suggestion])

        # Second call (recursive) returns no suggestions
        result2 = DeduplicationResult(clusters=[], merge_suggestions=[])

        stage.entity_deduplicator.deduplicate.side_effect = [result1, result2]

        # 4. Run Logic
        chunks = [chunk]
        stage._deduplicate_merged_entities(chunks)

        # 5. Assertions
        merged_entities = chunk.metadata["merged_entities"]

        # Should have 2 entities now (NASA and SpaceX)
        assert len(merged_entities) == 2

        # Check NASA
        nasa = next(e for e in merged_entities if e["candidate_key"] == "org:nasa")
        assert nasa["mention_count"] == 15  # 10 + 5
        assert "Space Admin" in nasa["aliases"]  # From B
        assert "National Aeronautics" in nasa["aliases"]  # From A
        assert "N.A.S.A." in nasa["aliases"]  # From B's name
        assert len(nasa["description"]) >= len("US Space Agency")

        # Check SpaceX is untouched
        spacex = next(e for e in merged_entities if e["candidate_key"] == "org:spacex")
        assert spacex["mention_count"] == 8
