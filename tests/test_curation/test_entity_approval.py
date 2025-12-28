"""Unit tests for entity curation operations (Tasks 3.7/3.8 helpers)."""

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
        self.deleted_entities: List[str] = []
        self.candidate_updates: List[Dict[str, Any]] = []
        self.relationship_upserts: List[Dict[str, Any]] = []
        self.relationship_candidate_rows: List[Dict[str, Any]] = []
        self.mentioned_in_relationships: List[Tuple[str, List[str]]] = []

    def upsert_entity(self, entity) -> str:  # type: ignore[override]
        self.entity_upserts.append(entity.model_dump())
        return entity.id

    def delete_entity(self, entity_id: str) -> bool:
        self.deleted_entities.append(entity_id)
        return True

    def update_entity_candidate_status(self, identifier: str, status: CandidateStatus) -> bool:
        self.status_updates.append((identifier, status))
        return True

    def update_entity_candidate(self, identifier: str, properties: Dict[str, Any]) -> bool:
        self.candidate_updates.append({"identifier": identifier, "properties": dict(properties)})
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

    def get_entity(self, entity_id: str) -> Dict[str, Any] | None:
        for ent in self.entity_upserts:
            if ent["id"] == entity_id:
                return ent
        # Dummy fallback for test setup simplicity if not in upserts
        if entity_id == "existing-entity-1":
            return {
                "id": "existing-entity-1",
                "canonical_name": "Existing Entity",
                "entity_type": "SYSTEM",
                "aliases": ["Old Alias"],
                "mention_count": 10,
                "source_documents": ["old-doc"],
                "chunk_ids": ["old-chunk"],
                "merged_candidate_keys": [],
            }
        return None

    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        # Find and update the dummy entity representation
        for ent in self.entity_upserts:
            if ent["id"] == entity_id:
                ent.update(properties)
                return True
        self.entity_upserts.append({"id": entity_id, **properties})  # simplified tracking
        return True

    def create_mentioned_in_relationships(
        self, entity_id: str, document_ids: List[str]
    ) -> int:
        self.mentioned_in_relationships.append((entity_id, list(document_ids)))
        return len(document_ids)


def _candidate(**kwargs: Any) -> EntityCandidate:
    base = dict(
        id=None,
        candidate_key="cand-1",
        canonical_name="Power System",
        candidate_type=EntityType.SYSTEM,
        confidence_score=0.9,
    )
    base.update(kwargs)
    return EntityCandidate(**base)


def test_merge_candidate_into_entity_and_undo(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    candidate = _candidate(
        candidate_key="cand-new",
        canonical_name="New Alias",
        aliases=["Another Alias"],
        mention_count=5,
    )
    target_entity_id = "existing-entity-1"

    success = service.merge_candidate_into_entity(target_entity_id, candidate)
    assert success is True

    # Check updates
    assert manager.status_updates[-1] == ("cand-new", CandidateStatus.MERGED_INTO_ENTITY)

    # Check entity update (find the update in our fake manager)
    updated_entity = None
    for ent in manager.entity_upserts:
        if ent["id"] == target_entity_id:
            updated_entity = ent
            break
    assert updated_entity is not None
    assert "New Alias" in updated_entity["aliases"]
    assert updated_entity["mention_count"] == 15  # 10 + 5

    # Undo
    assert service.undo_last_operation() is True

    # Verify rollback (status reset, entity properties reverted)
    assert manager.status_updates[-1] == ("cand-new", CandidateStatus.PENDING)

    # Verify entity state reverted
    # Since _FakeManager updates in-place, we check the current state of the single record
    reverted_entity = None
    for ent in manager.entity_upserts:
        if ent["id"] == target_entity_id:
            reverted_entity = ent
            break

    assert reverted_entity is not None
    # Depending on how the fallback get_entity worked, checking strict equality might be tricky
    # if previous_values didn't capture the fallback state perfectly.
    # But we know mention_count should be back to 10
    assert reverted_entity["mention_count"] == 10
    assert "New Alias" not in reverted_entity["aliases"]


def test_approve_candidate_promotes_and_updates_normalization(tmp_path: Path) -> None:
    manager = _FakeManager()
    table_path = tmp_path / "normalization.json"
    table = NormalizationTable(table_path=table_path)
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    candidate = _candidate(aliases=["EPS"], source_documents=["doc-1"])
    entity_id = service.approve_candidate(candidate)

    assert manager.entity_upserts[0]["id"] == entity_id
    assert manager.status_updates[-1] == (candidate.candidate_key, CandidateStatus.APPROVED)
    assert manager.entity_upserts[0]["properties"]["display_name"] == "Power System"

    record = table.lookup(candidate.canonical_name)
    assert record is not None
    assert record.canonical_id == entity_id
    assert record.status == "approved"


def test_reject_and_undo_restore_status(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    candidate = _candidate(candidate_key="cand-undo")
    service.reject_candidate(candidate, reason="noise")
    assert manager.status_updates[-1] == (candidate.candidate_key, CandidateStatus.REJECTED)

    assert service.undo_last_operation() is True
    assert manager.status_updates[-1] == (candidate.candidate_key, CandidateStatus.PENDING)


def test_merge_candidates_and_undo(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    primary = _candidate(
        candidate_key="cand-primary", canonical_name="Power System", mention_count=3
    )
    duplicate = _candidate(candidate_key="cand-dup", canonical_name="Power Sys", mention_count=2)

    entity_id = service.merge_candidates(primary, [duplicate])
    assert manager.entity_upserts[0]["id"] == entity_id
    assert manager.entity_upserts[0]["properties"]["display_name"] == "Power System"
    assert manager.entity_upserts[0]["properties"]["candidate_key"] == "cand-primary"
    assert manager.status_updates == [
        ("cand-primary", CandidateStatus.APPROVED),
        ("cand-dup", CandidateStatus.REJECTED),
    ]

    record = table.lookup(primary.canonical_name)
    assert record is not None
    assert record.canonical_id == entity_id

    assert service.undo_last_operation() is True
    assert manager.deleted_entities[-1] == entity_id


def test_approve_promotes_resolved_relationship_candidates(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    table.upsert(
        NormalizationEntry(
            raw_text="Attitude Control System",
            canonical_id="ent-b",
            canonical_name="Attitude Control System",
            entity_type="SYSTEM",
            method=NormalizationMethod.MANUAL,
            confidence=1.0,
            status="approved",
        )
    )

    manager.relationship_candidate_rows = [
        {
            "id": "relcand-1",
            "candidate_key": "power_system:DEPENDS_ON:attitude_control_system",
            "source": "Power System",
            "target": "Attitude Control System",
            "type": "DEPENDS_ON",
            "description": "",
            "confidence_score": 0.8,
            "status": "pending",
            "mention_count": 1,
            "source_documents": [],
            "chunk_ids": [],
            "provenance_events": [],
        }
    ]

    candidate = _candidate(candidate_key="power-system-key", canonical_name="Power System")
    entity_id = service.approve_candidate(candidate)

    assert manager.relationship_upserts, "expected a promoted Relationship upsert"
    assert manager.relationship_status_updates == [("relcand-1", CandidateStatus.APPROVED)]
    assert manager.relationship_upserts[0]["source_entity_id"] == entity_id
    assert manager.relationship_upserts[0]["target_entity_id"] == "ent-b"


def test_merge_candidate_into_entity_creates_mentioned_in_relationships(tmp_path: Path) -> None:
    """Verify that merging a candidate creates MENTIONED_IN edges for new documents.

    This tests the fix for the critical issue where merge_candidate_into_entity()
    was not creating MENTIONED_IN relationships for cross-document mentions.
    """
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )

    # Create a candidate from a NEW document (new-doc-1, new-doc-2)
    # The existing entity already has "old-doc" (see _FakeManager.get_entity)
    candidate = _candidate(
        candidate_key="cand-cross-doc",
        canonical_name="Cross Doc Entity",
        aliases=["CDE"],
        mention_count=3,
        source_documents=["new-doc-1", "new-doc-2", "old-doc"],  # includes overlap
        chunk_ids=["new-chunk-1"],
    )
    target_entity_id = "existing-entity-1"

    success = service.merge_candidate_into_entity(target_entity_id, candidate)
    assert success is True

    # Verify MENTIONED_IN relationships were created for NEW documents only
    assert len(manager.mentioned_in_relationships) == 1
    entity_id, doc_ids = manager.mentioned_in_relationships[0]
    assert entity_id == target_entity_id
    # Should contain new-doc-1 and new-doc-2, but NOT old-doc (already linked)
    assert set(doc_ids) == {"new-doc-1", "new-doc-2"}
