"""Unit tests for relationship candidate promotion behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationTable,
)
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType
from src.utils.config import Config


class _FakeManager:
    def __init__(self) -> None:
        self.status_updates: List[Tuple[str, CandidateStatus]] = []
        self.relationship_status_updates: List[Tuple[str, CandidateStatus]] = []
        self.entity_upserts: List[Dict[str, Any]] = []
        self.relationship_upserts: List[Dict[str, Any]] = []
        self.relationship_candidate_rows: List[Dict[str, Any]] = []

    def upsert_entity(self, entity) -> str:  # type: ignore[override]
        self.entity_upserts.append(entity.model_dump())
        return entity.id

    def get_entity(self, entity_id: str) -> Dict[str, Any] | None:
        for ent in self.entity_upserts:
            if ent["id"] == entity_id:
                return ent
        return None

    def update_entity_candidate_status(self, identifier: str, status: CandidateStatus) -> bool:
        self.status_updates.append((identifier, status))
        return True

    def get_relationship_candidates_involving_keys(
        self, keys: List[str], *, statuses: List[str] | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        return list(self.relationship_candidate_rows)

    def update_relationship_candidate_status(
        self, identifier: str, status: CandidateStatus
    ) -> bool:
        self.relationship_status_updates.append((identifier, status))
        return True

    def upsert_relationship(self, relationship) -> str:  # type: ignore[override]
        self.relationship_upserts.append(relationship.model_dump())
        return relationship.id


def test_promotes_approved_related_to_candidates(tmp_path: Path) -> None:
    """An already-approved RelationshipCandidate should still be promotable later."""
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "u.json",
    )

    # Target endpoint already approved in normalization table.
    table.upsert(
        NormalizationEntry(
            raw_text="Jet Propulsion Laboratory",
            canonical_id="ent-jpl",
            canonical_name="Jet Propulsion Laboratory",
            entity_type="ORGANIZATION",
            method=NormalizationMethod.MANUAL,
            confidence=1.0,
            status="approved",
        )
    )
    # Simulate the target entity existing in Neo4j.
    manager.entity_upserts.append({"id": "ent-jpl"})

    # Relationship candidate already approved (e.g., user approved it in TUI before endpoints resolved).
    manager.relationship_candidate_rows = [
        {
            "id": "relcand-1",
            "candidate_key": "nasa:RELATED_TO:jet_propulsion_laboratory",
            "source": "NASA",
            "target": "Jet Propulsion Laboratory",
            "type": "RELATED_TO",
            "description": "",
            "confidence_score": 0.5,
            "status": "approved",
            "mention_count": 1,
            "source_documents": [],
            "chunk_ids": [],
            "provenance_events": [],
        }
    ]

    candidate = EntityCandidate(
        id=None,
        candidate_key="ORGANIZATION:nasa",
        canonical_name="NASA",
        candidate_type=EntityType.ORGANIZATION,
        confidence_score=0.9,
        aliases=[],
        status=CandidateStatus.PENDING,
        mention_count=1,
        source_documents=[],
        chunk_ids=[],
        conflicting_types=[],
        provenance_events=[],
    )
    entity_id = service.approve_candidate(candidate)

    assert manager.relationship_upserts, "expected a promoted Relationship upsert"
    assert manager.relationship_status_updates == [("relcand-1", CandidateStatus.APPROVED)]
    assert manager.relationship_upserts[0]["source_entity_id"] == entity_id
    assert manager.relationship_upserts[0]["target_entity_id"] == "ent-jpl"
