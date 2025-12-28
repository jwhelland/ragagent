"""Batch curation operation tests (Task 3.8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.curation.batch_operations import (
    ApprovedEntityLookup,
    BatchCurationService,
    build_approved_entity_lookup,
    normalize_name,
)
from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import NormalizationTable
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityStatus, EntityType
from src.utils.config import Config, CurationConfig


class _FakeManager:
    def __init__(self) -> None:
        self.status_updates: List[Tuple[str, CandidateStatus]] = []
        self.entity_upserts: List[Dict[str, Any]] = []
        self.deleted_entities: List[str] = []
        self.relationship_candidate_rows: List[Dict[str, Any]] = []
        # Storage for entities (for get_entity and list_entities)
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.mentioned_in_created: List[Tuple[str, List[str]]] = []

    def upsert_entity(self, entity) -> str:  # type: ignore[override]
        entity_dict = entity.model_dump()
        self.entity_upserts.append(entity_dict)
        self.entities[entity.id] = entity_dict
        return entity.id

    def delete_entity(self, entity_id: str) -> bool:
        self.deleted_entities.append(entity_id)
        if entity_id in self.entities:
            del self.entities[entity_id]
        return True

    def update_entity_candidate_status(self, identifier: str, status: CandidateStatus) -> bool:
        self.status_updates.append((identifier, status))
        return True

    def update_entity_candidate(self, identifier: str, properties: Dict[str, Any]) -> bool:
        return True

    def get_relationship_candidates_involving_keys(
        self, keys: List[str], *, statuses: List[str] | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        return list(self.relationship_candidate_rows)

    def update_relationship_candidate_status(
        self, identifier: str, status: CandidateStatus
    ) -> bool:
        return True

    def upsert_relationship(self, relationship) -> str:  # type: ignore[override]
        return relationship.id

    def get_entity(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Dict[str, Any]]:
        """Get entity by ID (for merge_candidate_into_entity)."""
        return self.entities.get(entity_id)

    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update entity properties (for merge_candidate_into_entity)."""
        if entity_id in self.entities:
            self.entities[entity_id].update(properties)
            return True
        return False

    def create_mentioned_in_relationships(self, entity_id: str, document_ids: List[str]) -> int:
        """Track MENTIONED_IN relationships created."""
        self.mentioned_in_created.append((entity_id, document_ids))
        return len(document_ids)

    def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        status: Optional[EntityStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List entities with optional filtering."""
        results = list(self.entities.values())
        if entity_type:
            results = [e for e in results if e.get("entity_type") == entity_type.value]
        if status:
            results = [e for e in results if e.get("status") == status.value]
        return results[offset : offset + limit]

    def add_entity(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: EntityType,
        aliases: List[str] | None = None,
        status: EntityStatus = EntityStatus.APPROVED,
        source_documents: List[str] | None = None,
    ) -> None:
        """Helper to add entities for testing."""
        self.entities[entity_id] = {
            "id": entity_id,
            "canonical_name": canonical_name,
            "entity_type": entity_type.value,
            "aliases": aliases or [],
            "status": status.value,
            "source_documents": source_documents or [],
            "chunk_ids": [],
            "mention_count": 1,
        }


def _candidate(key: str, confidence: float) -> EntityCandidate:
    return EntityCandidate(
        id=None,
        candidate_key=key,
        canonical_name=key,
        candidate_type=EntityType.SYSTEM,
        confidence_score=confidence,
    )


def test_batch_approve_respects_threshold(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )
    batch_service = BatchCurationService(service, CurationConfig(auto_approve_threshold=0.8))

    candidates = [
        _candidate("cand-high", 0.9),
        _candidate("cand-low", 0.5),
    ]

    preview = batch_service.preview_batch_approve(candidates)
    assert preview.to_approve == ["cand-high"]
    assert "cand-low" in preview.skipped

    result = batch_service.batch_approve(candidates, dry_run=False)
    assert len(result.approved_entities) == 1
    assert manager.status_updates[-1] == ("cand-high", CandidateStatus.APPROVED)


def test_batch_merge_clusters_handles_multiple_groups(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )
    batch_service = BatchCurationService(service, CurationConfig())

    clusters = [
        [_candidate("primary-1", 0.9), _candidate("dup-1", 0.7)],
        [_candidate("primary-2", 0.95), _candidate("dup-2", 0.6)],
    ]

    result = batch_service.batch_merge_clusters(clusters)
    assert len(result.merged_entities) == 2
    assert manager.status_updates.count(("primary-1", CandidateStatus.APPROVED)) == 1
    assert manager.status_updates.count(("dup-1", CandidateStatus.REJECTED)) == 1
    assert manager.status_updates.count(("primary-2", CandidateStatus.APPROVED)) == 1


# ============================================================================
# Tests for Auto-Approve Existing functionality (Strategy 2)
# ============================================================================


class TestNormalizeName:
    """Tests for the normalize_name helper function."""

    def test_lowercases_name(self) -> None:
        assert normalize_name("Battery Management System") == "battery management system"

    def test_strips_whitespace(self) -> None:
        assert normalize_name("  BMS  ") == "bms"

    def test_handles_mixed_case_and_whitespace(self) -> None:
        assert normalize_name("  Power SUPPLY  ") == "power supply"


class TestApprovedEntityLookup:
    """Tests for the ApprovedEntityLookup class."""

    def test_add_and_find_by_canonical_name(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity(
            entity_id="ent-123",
            canonical_name="Battery Management System",
            entity_type="SYSTEM",
        )

        # Exact match (case-insensitive)
        result = lookup.find_match("battery management system", "SYSTEM")
        assert result == "ent-123"

        # Different case
        result = lookup.find_match("Battery Management System", "SYSTEM")
        assert result == "ent-123"

    def test_find_by_alias(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity(
            entity_id="ent-123",
            canonical_name="Battery Management System",
            entity_type="SYSTEM",
            aliases=["BMS", "Battery Controller"],
        )

        # Find by alias
        assert lookup.find_match("BMS", "SYSTEM") == "ent-123"
        assert lookup.find_match("battery controller", "SYSTEM") == "ent-123"

    def test_candidate_alias_matches_entity_name(self) -> None:
        """Test that a candidate's alias can match an entity's canonical name."""
        lookup = ApprovedEntityLookup()
        lookup.add_entity(
            entity_id="ent-123",
            canonical_name="BMS",
            entity_type="SYSTEM",
        )

        # Candidate has "BMS" as alias
        result = lookup.find_match(
            "Battery Management System",
            "SYSTEM",
            aliases=["BMS", "Battery Controller"],
        )
        assert result == "ent-123"

    def test_no_match_for_different_type(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity(
            entity_id="ent-123",
            canonical_name="BMS",
            entity_type="SYSTEM",
        )

        # Same name but different type - should not match
        result = lookup.find_match("BMS", "COMPONENT")
        assert result is None

    def test_no_match_for_unknown_name(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity(
            entity_id="ent-123",
            canonical_name="BMS",
            entity_type="SYSTEM",
        )

        result = lookup.find_match("Unknown System", "SYSTEM")
        assert result is None

    def test_len_returns_unique_entity_count(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity("ent-1", "System A", "SYSTEM", aliases=["SysA"])
        lookup.add_entity("ent-2", "System B", "SYSTEM", aliases=["SysB"])

        assert len(lookup) == 2

    def test_contains_checks_entity_ids(self) -> None:
        lookup = ApprovedEntityLookup()
        lookup.add_entity("ent-123", "System A", "SYSTEM")

        assert "ent-123" in lookup
        assert "ent-999" not in lookup


class TestBuildApprovedEntityLookup:
    """Tests for the build_approved_entity_lookup function."""

    def test_builds_lookup_from_approved_entities(self) -> None:
        manager = _FakeManager()
        manager.add_entity(
            entity_id="ent-1",
            canonical_name="Battery Management System",
            entity_type=EntityType.SYSTEM,
            aliases=["BMS"],
            status=EntityStatus.APPROVED,
        )
        manager.add_entity(
            entity_id="ent-2",
            canonical_name="Power Supply Unit",
            entity_type=EntityType.COMPONENT,
            aliases=["PSU"],
            status=EntityStatus.APPROVED,
        )
        # This one should be excluded (not approved)
        manager.add_entity(
            entity_id="ent-3",
            canonical_name="Draft Entity",
            entity_type=EntityType.SYSTEM,
            status=EntityStatus.DRAFT,
        )

        lookup = build_approved_entity_lookup(manager)

        # Should include the two approved entities
        assert len(lookup) == 2
        assert "ent-1" in lookup
        assert "ent-2" in lookup
        assert "ent-3" not in lookup

        # Should be able to find by name and alias
        assert lookup.find_match("Battery Management System", "SYSTEM") == "ent-1"
        assert lookup.find_match("BMS", "SYSTEM") == "ent-1"
        assert lookup.find_match("PSU", "COMPONENT") == "ent-2"


def _candidate_with_details(
    key: str,
    name: str,
    entity_type: EntityType,
    confidence: float = 0.8,
    aliases: List[str] | None = None,
    conflicting_types: List[str] | None = None,
    source_documents: List[str] | None = None,
    status: CandidateStatus = CandidateStatus.PENDING,
) -> EntityCandidate:
    """Helper to create detailed candidates for testing."""
    return EntityCandidate(
        id=f"id-{key}",
        candidate_key=key,
        canonical_name=name,
        candidate_type=entity_type,
        confidence_score=confidence,
        aliases=aliases or [],
        conflicting_types=conflicting_types or [],
        source_documents=source_documents or [],
        status=status,
    )


class TestPreviewAutoApproveExisting:
    """Tests for the preview_auto_approve_existing method."""

    def test_identifies_matching_candidates(self, tmp_path: Path) -> None:
        manager = _FakeManager()
        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        # Build lookup with one approved entity
        lookup = ApprovedEntityLookup()
        lookup.add_entity("ent-1", "Battery Management System", "SYSTEM", aliases=["BMS"])

        candidates = [
            _candidate_with_details("SYSTEM:bms", "BMS", EntityType.SYSTEM),
            _candidate_with_details("SYSTEM:other", "Other System", EntityType.SYSTEM),
        ]

        preview = batch_service.preview_auto_approve_existing(candidates, lookup)

        assert len(preview.to_merge) == 1
        assert preview.to_merge[0] == ("SYSTEM:bms", "ent-1")
        assert "SYSTEM:other" in preview.skipped_no_match

    def test_skips_candidates_with_conflicting_types(self, tmp_path: Path) -> None:
        manager = _FakeManager()
        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        lookup = ApprovedEntityLookup()
        lookup.add_entity("ent-1", "BMS", "SYSTEM")

        # This candidate has conflicting types - should be skipped
        candidates = [
            _candidate_with_details(
                "SYSTEM:bms",
                "BMS",
                EntityType.SYSTEM,
                conflicting_types=["COMPONENT"],
            ),
        ]

        preview = batch_service.preview_auto_approve_existing(candidates, lookup)

        assert len(preview.to_merge) == 0
        assert "SYSTEM:bms" in preview.skipped_conflicting

    def test_skips_non_pending_candidates(self, tmp_path: Path) -> None:
        manager = _FakeManager()
        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        lookup = ApprovedEntityLookup()
        lookup.add_entity("ent-1", "BMS", "SYSTEM")

        candidates = [
            _candidate_with_details(
                "SYSTEM:bms", "BMS", EntityType.SYSTEM, status=CandidateStatus.APPROVED
            ),
        ]

        preview = batch_service.preview_auto_approve_existing(candidates, lookup)

        # Should not include already-approved candidates
        assert len(preview.to_merge) == 0


class TestAutoApproveExistingMatches:
    """Tests for the auto_approve_existing_matches method."""

    def test_dry_run_returns_preview_only(self, tmp_path: Path) -> None:
        manager = _FakeManager()
        manager.add_entity(
            entity_id="ent-1",
            canonical_name="Battery Management System",
            entity_type=EntityType.SYSTEM,
            aliases=["BMS"],
            status=EntityStatus.APPROVED,
            source_documents=["doc-1"],
        )

        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        lookup = build_approved_entity_lookup(manager)
        candidates = [
            _candidate_with_details(
                "SYSTEM:bms",
                "BMS",
                EntityType.SYSTEM,
                source_documents=["doc-2"],
            ),
        ]

        result = batch_service.auto_approve_existing_matches(candidates, lookup, dry_run=True)

        assert result.preview_only is True
        assert len(result.merged_entities) == 0
        # No status updates should have been made
        assert (
            len([s for s in manager.status_updates if s[1] == CandidateStatus.MERGED_INTO_ENTITY])
            == 0
        )

    def test_merges_matching_candidates(self, tmp_path: Path) -> None:
        manager = _FakeManager()
        manager.add_entity(
            entity_id="ent-1",
            canonical_name="Battery Management System",
            entity_type=EntityType.SYSTEM,
            aliases=["BMS"],
            status=EntityStatus.APPROVED,
            source_documents=["doc-1"],
        )

        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        lookup = build_approved_entity_lookup(manager)
        candidates = [
            _candidate_with_details(
                "SYSTEM:bms",
                "BMS",
                EntityType.SYSTEM,
                source_documents=["doc-2"],
            ),
        ]

        result = batch_service.auto_approve_existing_matches(candidates, lookup, dry_run=False)

        assert result.preview_only is False
        assert "ent-1" in result.merged_entities
        # Candidate should be marked as MERGED_INTO_ENTITY (uses candidate.id, not candidate_key)
        assert ("id-SYSTEM:bms", CandidateStatus.MERGED_INTO_ENTITY) in manager.status_updates

    def test_returns_unique_entity_ids_when_multiple_merge(self, tmp_path: Path) -> None:
        """Test that when multiple candidates merge into the same entity, it's listed once."""
        manager = _FakeManager()
        manager.add_entity(
            entity_id="ent-1",
            canonical_name="Battery Management System",
            entity_type=EntityType.SYSTEM,
            aliases=["BMS", "Battery Controller"],
            status=EntityStatus.APPROVED,
        )

        table = NormalizationTable(table_path=tmp_path / "norm.json")
        service = EntityCurationService(
            manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
        )
        batch_service = BatchCurationService(service, CurationConfig())

        lookup = build_approved_entity_lookup(manager)
        # Two candidates that both match the same entity
        candidates = [
            _candidate_with_details("SYSTEM:bms", "BMS", EntityType.SYSTEM),
            _candidate_with_details(
                "SYSTEM:battery_controller",
                "Battery Controller",
                EntityType.SYSTEM,
            ),
        ]

        result = batch_service.auto_approve_existing_matches(candidates, lookup, dry_run=False)

        # Entity should only appear once in merged_entities
        assert result.merged_entities.count("ent-1") == 1
