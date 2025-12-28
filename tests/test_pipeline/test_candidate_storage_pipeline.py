from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.pipeline.stages.storage import StorageStage
from src.storage.schemas import EntityCandidate, RelationshipCandidate
from src.utils.config import Config


class _Chunk:
    def __init__(self, chunk_id: str, document_id: str, metadata: Dict[str, Any]) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.metadata = metadata


class _FakeNeo4j:
    def __init__(self) -> None:
        self.entity_candidates: List[EntityCandidate] = []
        self.relationship_candidates: List[RelationshipCandidate] = []

    def upsert_entity_candidate_aggregate(self, candidate: EntityCandidate) -> str:
        self.entity_candidates.append(candidate)
        return candidate.candidate_key

    def upsert_relationship_candidate_aggregate(self, candidate: RelationshipCandidate) -> str:
        self.relationship_candidates.append(candidate)
        return candidate.candidate_key


def test_store_entity_candidates_from_merged_entities() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    # Mock Qdrant
    fake_qdrant = MagicMock()
    fake_neo4j = _FakeNeo4j()

    stage = StorageStage(cfg, fake_neo4j, fake_qdrant)

    chunk = _Chunk(
        chunk_id="c1",
        document_id="doc-1",
        metadata={
            "merged_entities": [
                {
                    "canonical_name": "Solar Array",
                    "type": "COMPONENT",
                    "confidence": 0.91,
                    "aliases": ["arrays"],
                    "description": "Generates power",
                    "mention_count": 2,
                    "conflicting_types": ["SYSTEM"],
                    "provenance": [{"extractor": "llm"}, {"extractor": "spacy"}],
                    "candidate_key": "COMPONENT:solar_array",  # ExtractionStage adds this
                }
            ]
        },
    )

    stored = stage._store_entity_candidates([chunk])

    assert stored == 1
    assert len(fake_neo4j.entity_candidates) == 1
    cand = fake_neo4j.entity_candidates[0]
    assert cand.canonical_name == "Solar Array"
    assert cand.candidate_type.value == "COMPONENT"
    assert cand.mention_count == 2
    assert cand.source_documents == ["doc-1"]
    assert cand.chunk_ids == ["c1"]
    assert cand.conflicting_types == ["SYSTEM"]
    assert cand.candidate_key == "COMPONENT:solar_array"


def test_store_relationship_candidates_from_llm_relationships() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    fake_qdrant = MagicMock()
    fake_neo4j = _FakeNeo4j()

    stage = StorageStage(cfg, fake_neo4j, fake_qdrant)

    chunk = _Chunk(
        chunk_id="c2",
        document_id="doc-2",
        metadata={
            "llm_relationships": [
                {
                    "source": "battery",
                    "target": "solar_array",
                    "type": "DEPENDS_ON",
                    "description": "Battery charging depends on arrays",
                    "confidence": 0.88,
                    "source_extractor": "llm",
                }
            ]
        },
    )

    stored = stage._store_relationship_candidates([chunk])

    assert stored == 1
    assert len(fake_neo4j.relationship_candidates) == 1
    rel = fake_neo4j.relationship_candidates[0]
    assert rel.source == "battery"
    assert rel.target == "solar_array"
    assert rel.type == "DEPENDS_ON"
    assert rel.source_documents == ["doc-2"]
    assert rel.chunk_ids == ["c2"]
