"""Entity curation operations (Tasks 3.7).

Provides approve/merge/reject/edit operations plus undo + audit trail.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationRecord,
    NormalizationTable,
)
from src.normalization.string_normalizer import StringNormalizer
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.schemas import (
    CandidateStatus,
    Entity,
    EntityCandidate,
    EntityStatus,
    EntityType,
    ExtractionMethod,
    Relationship,
    RelationshipCandidate,
    RelationshipProvenance,
    RelationshipType,
)
from src.utils.candidate_keys import normalize_candidate_key_fragment
from src.utils.config import Config
from src.utils.embeddings import EmbeddingGenerator


class NormalizationCheckpoint(BaseModel):
    """Track normalization changes for undo."""

    raw_text: str
    previous_record: NormalizationRecord | None = None


class StatusCheckpoint(BaseModel):
    identifier: str
    previous_status: CandidateStatus
    candidate_key: str | None = None
    created_relationship_id: str | None = None


class UndoAction(BaseModel):
    """Encapsulate undo information for a curation operation."""

    operation: str
    entity_id: str | None = None
    candidate_statuses: List[StatusCheckpoint] = Field(default_factory=list)
    relationship_candidate_statuses: List[StatusCheckpoint] = Field(default_factory=list)
    normalization_changes: List[NormalizationCheckpoint] = Field(default_factory=list)
    identifier: str | None = None
    previous_status: CandidateStatus | None = None
    previous_values: Dict[str, Any] = Field(default_factory=dict)

    def compact(self) -> Dict[str, Any]:
        """Return a compact, JSON-friendly representation for audit/logging."""
        return {
            "operation": self.operation,
            "entity_id": self.entity_id,
            "candidate_statuses": [
                (c.identifier, c.previous_status.value) for c in self.candidate_statuses
            ],
            "relationship_candidate_statuses": [
                (c.identifier, c.previous_status.value)
                for c in self.relationship_candidate_statuses
            ],
            "normalization_changes": [c.raw_text for c in self.normalization_changes],
            "identifier": self.identifier,
        }


class UndoStackState(BaseModel):
    actions: List[UndoAction] = Field(default_factory=list)


class CurationAuditTrail:
    """Simple JSONL audit trail for curation actions."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, payload: Dict[str, object]) -> None:
        """Append an audit entry to disk."""
        if not self.enabled:
            return

        entry = {
            "event": event,
            "payload": payload,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


class EntityCurationService:
    """Orchestrate curation actions on EntityCandidates."""

    def __init__(
        self,
        manager: Neo4jManager,
        normalization_table: NormalizationTable,
        config: Config | None = None,
        audit_path: Path | None = None,
        undo_stack_path: Path | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        qdrant_manager: QdrantManager | None = None,
    ) -> None:
        self.manager = manager
        self.normalization_table = normalization_table
        self.config = config or Config()
        self._undo_stack_path = undo_stack_path or Path("data/curation/undo_stack.json")
        self._undo_stack = self._load_undo_stack(self._undo_stack_path)
        self._audit = CurationAuditTrail(
            audit_path or Path("logs/curation_audit.jsonl"),
            enabled=self.config.curation.enable_audit_trail,
        )
        self._string_normalizer = StringNormalizer(self.config.normalization)
        self.embedding_generator = embedding_generator
        self.qdrant_manager = qdrant_manager

    def _embed_and_upsert_entity(self, entity: Entity) -> None:
        """Generate embedding for entity and upsert to Qdrant."""
        if not self.embedding_generator or not self.qdrant_manager:
            return

        try:
            # Construct text representation: Canonical Name + Description + Aliases
            text_parts = [entity.canonical_name]
            if entity.description:
                text_parts.append(entity.description)
            if entity.aliases:
                text_parts.extend(entity.aliases)

            text_to_embed = ". ".join(text_parts)

            # Generate embedding
            embedding = self.embedding_generator.generate_single(text_to_embed)

            # Upsert to Qdrant
            self.qdrant_manager.upsert_entities(
                entities=[entity.model_dump()],
                vectors=[embedding.tolist()],
            )
            logger.debug(f"Upserted embedding for entity {entity.canonical_name}")
        except Exception as e:
            logger.error(f"Failed to embed/upsert entity {entity.canonical_name}: {e}")

    # Public API -----------------------------------------------------
    def approve_candidate(
        self, candidate: EntityCandidate, *, update_normalization: bool = True
    ) -> str:
        """Approve a candidate and promote it to a production Entity."""
        entity = self._entity_from_candidate(candidate, EntityStatus.APPROVED)
        entity_id = self.manager.upsert_entity(entity)
        # Ensure entity_id is set on the object for embedding/upsert
        entity.id = entity_id
        self._embed_and_upsert_entity(entity)

        identifier = self._candidate_identifier(candidate)
        previous_status = candidate.status
        self.manager.update_entity_candidate_status(identifier, CandidateStatus.APPROVED)

        normalization_changes: List[NormalizationCheckpoint] = []
        if update_normalization:
            normalization_changes = self._upsert_normalization_entries(
                entity_id=entity_id,
                canonical_name=candidate.canonical_name,
                entity_type=candidate.candidate_type,
                raw_texts=[candidate.canonical_name, *candidate.aliases],
                status="approved",
            )

        relationship_candidate_statuses = self._promote_related_relationship_candidates(
            raw_mentions=[candidate.canonical_name, *candidate.aliases]
        )

        self._push_undo(
            UndoAction(
                operation="approve_candidate",
                entity_id=entity_id,
                candidate_statuses=[
                    StatusCheckpoint(
                        identifier=identifier,
                        previous_status=previous_status,
                        candidate_key=candidate.candidate_key,
                    )
                ],
                relationship_candidate_statuses=relationship_candidate_statuses,
                normalization_changes=normalization_changes,
            )
        )

        self._record_audit(
            "approve_candidate",
            {
                "candidate_key": candidate.candidate_key,
                "entity_id": entity_id,
                "canonical_name": candidate.canonical_name,
            },
        )
        return entity_id

    def reject_candidate(self, candidate: EntityCandidate, reason: str = "") -> None:
        """Reject a candidate without deleting it."""
        identifier = self._candidate_identifier(candidate)
        previous_status = candidate.status
        self.manager.update_entity_candidate_status(identifier, CandidateStatus.REJECTED)

        self._push_undo(
            UndoAction(
                operation="reject_candidate",
                identifier=identifier,
                previous_status=previous_status,
            )
        )
        self._record_audit(
            "reject_candidate",
            {"candidate_key": candidate.candidate_key, "reason": reason},
        )

    def edit_candidate(
        self, candidate: EntityCandidate, updates: Dict[str, object]
    ) -> EntityCandidate:
        """Edit candidate properties."""
        if not updates:
            return candidate

        allowed_fields = {
            "canonical_name",
            "aliases",
            "description",
            "candidate_type",
            "confidence_score",
        }
        sanitized = {k: v for k, v in updates.items() if k in allowed_fields}
        if not sanitized:
            return candidate

        identifier = self._candidate_identifier(candidate)
        previous_snapshot = {key: getattr(candidate, key) for key in sanitized}
        self.manager.update_entity_candidate(identifier, sanitized)

        updated_candidate = candidate.model_copy(update=sanitized)

        self._push_undo(
            UndoAction(
                operation="edit_candidate",
                identifier=identifier,
                previous_values=previous_snapshot,
            )
        )
        self._record_audit(
            "edit_candidate",
            {"candidate_key": candidate.candidate_key, "changes": sanitized},
        )
        return updated_candidate

    def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        *,
        aliases: Sequence[str] | None = None,
        description: str = "",
        confidence_score: float = 1.0,
        source_documents: Sequence[str] | None = None,
        chunk_ids: Sequence[str] | None = None,
    ) -> str:
        """Create a new approved entity directly (no EntityCandidate required)."""
        aliases = list(aliases or [])
        source_documents = list(source_documents or [])
        chunk_ids = list(chunk_ids or [])

        entity = Entity(
            canonical_name=name,
            entity_type=entity_type,
            aliases=sorted(set(aliases)),
            description=description,
            confidence_score=confidence_score,
            extraction_method=ExtractionMethod.MANUAL,
            status=EntityStatus.APPROVED,
            mention_count=1,
            source_documents=sorted(set(source_documents)),
            properties={"chunk_ids": sorted(set(chunk_ids)), "display_name": name},
        )
        entity_id = self.manager.upsert_entity(entity)
        entity.id = entity_id
        self._embed_and_upsert_entity(entity)

        normalization_changes = self._upsert_normalization_entries(
            entity_id=entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            raw_texts=[name, *aliases],
            status="approved",
        )
        relationship_candidate_statuses = self._promote_related_relationship_candidates(
            raw_mentions=[name, *aliases]
        )

        self._push_undo(
            UndoAction(
                operation="create_entity",
                entity_id=entity_id,
                relationship_candidate_statuses=relationship_candidate_statuses,
                normalization_changes=normalization_changes,
            )
        )
        self._record_audit(
            "create_entity",
            {
                "entity_id": entity_id,
                "canonical_name": entity.canonical_name,
                "entity_type": entity_type.value,
            },
        )
        return entity_id

    def merge_candidates(
        self, primary: EntityCandidate, duplicates: Sequence[EntityCandidate]
    ) -> str:
        """Merge multiple candidates into a single approved entity."""
        if not duplicates:
            return self.approve_candidate(primary)

        all_candidates = [primary, *duplicates]
        aggregate_aliases = self._collect_aliases(all_candidates)
        source_documents = self._collect_unique(all_candidates, "source_documents")
        chunk_ids = self._collect_unique(all_candidates, "chunk_ids")
        mention_count = sum(candidate.mention_count for candidate in all_candidates)
        confidence_score = max(candidate.confidence_score for candidate in all_candidates)
        description = primary.description or next(
            (c.description for c in duplicates if c.description), ""
        )

        entity = Entity(
            canonical_name=primary.canonical_name,
            entity_type=primary.candidate_type,
            aliases=sorted(aggregate_aliases),
            description=description,
            confidence_score=confidence_score,
            extraction_method=ExtractionMethod.MANUAL,
            status=EntityStatus.APPROVED,
            mention_count=mention_count,
            source_documents=source_documents,
            properties={
                "chunk_ids": chunk_ids,
                "merged_candidate_keys": [c.candidate_key for c in all_candidates],
                "candidate_key": primary.candidate_key,
                "display_name": primary.canonical_name,
            },
        )
        entity_id = self.manager.upsert_entity(entity)
        entity.id = entity_id
        self._embed_and_upsert_entity(entity)

        previous_statuses = []
        for candidate in all_candidates:
            identifier = self._candidate_identifier(candidate)
            previous_statuses.append(
                StatusCheckpoint(
                    identifier=identifier,
                    previous_status=candidate.status,
                    candidate_key=candidate.candidate_key,
                )
            )
            new_status = (
                CandidateStatus.APPROVED if candidate is primary else CandidateStatus.REJECTED
            )
            self.manager.update_entity_candidate_status(identifier, new_status)

        normalization_changes = self._upsert_normalization_entries(
            entity_id=entity_id,
            canonical_name=primary.canonical_name,
            entity_type=primary.candidate_type,
            raw_texts=list(aggregate_aliases) + [primary.canonical_name],
            status="approved",
        )

        relationship_candidate_statuses = self._promote_related_relationship_candidates(
            raw_mentions=list(aggregate_aliases) + [primary.canonical_name]
        )

        self._push_undo(
            UndoAction(
                operation="merge_candidates",
                entity_id=entity_id,
                candidate_statuses=previous_statuses,
                relationship_candidate_statuses=relationship_candidate_statuses,
                normalization_changes=normalization_changes,
            )
        )

        self._record_audit(
            "merge_candidates",
            {
                "entity_id": entity_id,
                "primary": primary.candidate_key,
                "merged": [c.candidate_key for c in duplicates],
                "aliases_added": list(aggregate_aliases),
            },
        )
        return entity_id

    def merge_candidate_into_entity(self, entity_id: str, candidate: EntityCandidate) -> bool:
        """Merge a candidate into an existing approved entity."""
        existing_entity_data = self.manager.get_entity(entity_id)
        if not existing_entity_data:
            logger.error(f"Target entity {entity_id} not found for merge")
            return False

        # 1. Update existing entity
        # We need to manually update fields like aliases, mention_count, etc.
        current_aliases = set(existing_entity_data.get("aliases", []))
        new_aliases = set(candidate.aliases) | {candidate.canonical_name}
        merged_aliases = sorted(current_aliases | new_aliases)

        current_docs = set(existing_entity_data.get("source_documents", []))
        new_docs = set(candidate.source_documents)
        merged_docs = sorted(current_docs | new_docs)

        # Note: Neo4j get_entity returns all properties, we need to filter or just update what we want

        merged_props = {
            "aliases": merged_aliases,
            "source_documents": merged_docs,
            "mention_count": existing_entity_data.get("mention_count", 0) + candidate.mention_count,
            "last_updated": datetime.now().isoformat(),
        }

        # Handle chunk_ids in properties if they exist
        existing_chunks = set(existing_entity_data.get("chunk_ids", []))
        new_chunks = set(candidate.chunk_ids)
        if existing_chunks or new_chunks:
            merged_props["chunk_ids"] = sorted(existing_chunks | new_chunks)

        # Handle merged_candidate_keys
        existing_keys = set(existing_entity_data.get("merged_candidate_keys", []))
        if candidate.candidate_key:
            merged_props["merged_candidate_keys"] = sorted(
                existing_keys | {candidate.candidate_key}
            )

        self.manager.update_entity(entity_id, merged_props)

        # Update embeddings for the modified entity
        try:
            # We need to reconstruct the entity object or fetch it to re-embed
            # For efficiency, we can construct a partial object if we have enough data,
            # or just fetch the updated one.
            updated_entity_data = self.manager.get_entity(entity_id)
            if updated_entity_data:
                # Need to convert dict to Entity model to use the helper
                # Note: get_entity returns a dict
                updated_entity = Entity(**updated_entity_data)
                self._embed_and_upsert_entity(updated_entity)
        except Exception as e:
            logger.warning(f"Failed to update embedding for merged entity {entity_id}: {e}")

        # 2. Update candidate status
        identifier = self._candidate_identifier(candidate)
        previous_status = candidate.status
        self.manager.update_entity_candidate_status(identifier, CandidateStatus.MERGED_INTO_ENTITY)

        # 3. Update normalization table
        normalization_changes = self._upsert_normalization_entries(
            entity_id=entity_id,
            canonical_name=existing_entity_data.get("canonical_name", "entity"),
            entity_type=EntityType(existing_entity_data.get("entity_type", "CONCEPT")),
            raw_texts=[candidate.canonical_name, *candidate.aliases],
            status="approved",
        )

        # 4. Promote related relationship candidates
        relationship_candidate_statuses = self._promote_related_relationship_candidates(
            raw_mentions=[candidate.canonical_name, *candidate.aliases]
        )

        # 5. Push to undo stack
        # For merging into existing entity, we need to store the previous state of the entity to revert properly
        self._push_undo(
            UndoAction(
                operation="merge_candidate_into_entity",
                entity_id=entity_id,
                candidate_statuses=[
                    StatusCheckpoint(
                        identifier=identifier,
                        previous_status=previous_status,
                        candidate_key=candidate.candidate_key,
                    )
                ],
                relationship_candidate_statuses=relationship_candidate_statuses,
                normalization_changes=normalization_changes,
                previous_values=existing_entity_data,  # Store full snapshot for undo
            )
        )

        self._record_audit(
            "merge_candidate_into_entity",
            {
                "entity_id": entity_id,
                "candidate_key": candidate.candidate_key,
                "aliases_added": list(new_aliases),
            },
        )
        return True

    def approve_relationship_candidate(self, candidate: RelationshipCandidate) -> str | None:
        """Approve a relationship candidate.

        If endpoints are resolved/approved, creates the Relationship.
        Otherwise, just marks status as APPROVED.
        """
        identifier = candidate.id or candidate.candidate_key
        previous_status = candidate.status
        self.manager.update_relationship_candidate_status(identifier, CandidateStatus.APPROVED)

        # Try to promote immediately
        promotion_statuses = self._promote_related_relationship_candidates(
            raw_mentions=[candidate.source, candidate.target]
        )

        primary_checkpoint = StatusCheckpoint(
            identifier=identifier,
            previous_status=previous_status,
            candidate_key=candidate.candidate_key,
        )
        promoted_self = next(
            (
                s
                for s in promotion_statuses
                if s.candidate_key == candidate.candidate_key or s.identifier == identifier
            ),
            None,
        )
        if promoted_self:
            primary_checkpoint.created_relationship_id = promoted_self.created_relationship_id

        # Note: promotion may return a checkpoint for the relationship candidate itself, but it
        # will reflect "previous_status=APPROVED" since we already approved it above. Keep the
        # original previous_status captured in primary_checkpoint instead.
        self._push_undo(
            UndoAction(
                operation="approve_relationship_candidate",
                relationship_candidate_statuses=[primary_checkpoint]
                + [
                    s
                    for s in promotion_statuses
                    if s.candidate_key != candidate.candidate_key and s.identifier != identifier
                ],
            )
        )

        self._record_audit(
            "approve_relationship_candidate",
            {
                "candidate_key": candidate.candidate_key,
                "source": candidate.source,
                "target": candidate.target,
                "promoted": bool(primary_checkpoint.created_relationship_id),
            },
        )
        return identifier

    def reject_relationship_candidate(
        self, candidate: RelationshipCandidate, reason: str = ""
    ) -> None:
        """Reject a relationship candidate."""
        identifier = candidate.id or candidate.candidate_key
        previous_status = candidate.status
        self.manager.update_relationship_candidate_status(identifier, CandidateStatus.REJECTED)

        self._push_undo(
            UndoAction(
                operation="reject_relationship_candidate",
                relationship_candidate_statuses=[
                    StatusCheckpoint(
                        identifier=identifier,
                        previous_status=previous_status,
                        candidate_key=candidate.candidate_key,
                    )
                ],
            )
        )
        self._record_audit(
            "reject_relationship_candidate",
            {"candidate_key": candidate.candidate_key, "reason": reason},
        )

    def undo_last_operation(self) -> bool:
        """Undo the most recent curation operation."""
        if not self._undo_stack:
            return False

        action = self._undo_stack.pop()
        self._persist_undo_stack()
        logger.info("Undoing curation operation {}", action.operation)

        def delete_created_relationships(checkpoints: List[StatusCheckpoint]) -> None:
            if not hasattr(self.manager, "delete_relationship"):
                return
            for checkpoint in checkpoints:
                if not checkpoint.created_relationship_id:
                    continue
                try:
                    self.manager.delete_relationship(checkpoint.created_relationship_id)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    continue

        if action.operation in {"approve_candidate", "merge_candidates"}:
            entity_id = action.entity_id
            if isinstance(entity_id, str):
                self.manager.delete_entity(entity_id)
                if self.qdrant_manager:
                    try:
                        self.qdrant_manager.delete_entities([entity_id])
                    except Exception as e:
                        logger.warning(f"Failed to delete entity {entity_id} from Qdrant: {e}")

            for checkpoint in action.candidate_statuses:
                self.manager.update_entity_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )

            for checkpoint in action.normalization_changes:
                if checkpoint.previous_record:
                    self.normalization_table.restore_record(checkpoint.previous_record)
                else:
                    self.normalization_table.remove(checkpoint.raw_text)

            for checkpoint in action.relationship_candidate_statuses:
                self.manager.update_relationship_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )
            delete_created_relationships(action.relationship_candidate_statuses)

        elif action.operation == "merge_candidate_into_entity":
            entity_id = action.entity_id
            previous_values = action.previous_values
            if isinstance(entity_id, str) and previous_values:
                self.manager.update_entity(entity_id, previous_values)
                # Re-embed the entity with previous values to revert Qdrant state
                try:
                    # We might need to fetch the full entity again to be safe,
                    # but if previous_values contains the critical text fields, we might be okay.
                    # Best to fetch the restored entity from Neo4j to be sure.
                    restored_entity_data = self.manager.get_entity(entity_id)
                    if restored_entity_data and self.embedding_generator and self.qdrant_manager:
                        restored_entity = Entity(**restored_entity_data)
                        self._embed_and_upsert_entity(restored_entity)
                except Exception as e:
                    logger.warning(f"Failed to revert embedding for entity {entity_id}: {e}")

            for checkpoint in action.candidate_statuses:
                self.manager.update_entity_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )

            for checkpoint in action.normalization_changes:
                if checkpoint.previous_record:
                    self.normalization_table.restore_record(checkpoint.previous_record)
                else:
                    self.normalization_table.remove(checkpoint.raw_text)

            for checkpoint in action.relationship_candidate_statuses:
                self.manager.update_relationship_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )
            delete_created_relationships(action.relationship_candidate_statuses)

        elif action.operation == "reject_candidate":
            identifier = action.identifier
            previous_status = action.previous_status
            if isinstance(previous_status, CandidateStatus):
                self.manager.update_entity_candidate_status(identifier, previous_status)

        elif action.operation == "edit_candidate":
            identifier = action.identifier
            previous_values = action.previous_values or {}
            if isinstance(previous_values, dict):
                self.manager.update_entity_candidate(identifier, previous_values)

        elif action.operation == "approve_relationship_candidate":
            for checkpoint in action.relationship_candidate_statuses:
                self.manager.update_relationship_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )
            delete_created_relationships(action.relationship_candidate_statuses)

        elif action.operation == "reject_relationship_candidate":
            for checkpoint in action.relationship_candidate_statuses:
                self.manager.update_relationship_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )

        elif action.operation == "create_entity":
            entity_id = action.entity_id
            if isinstance(entity_id, str):
                self.manager.delete_entity(entity_id)
                if self.qdrant_manager:
                    try:
                        self.qdrant_manager.delete_entities([entity_id])
                    except Exception as e:
                        logger.warning(f"Failed to delete entity {entity_id} from Qdrant: {e}")

            for checkpoint in action.normalization_changes:
                if checkpoint.previous_record:
                    self.normalization_table.restore_record(checkpoint.previous_record)
                else:
                    self.normalization_table.remove(checkpoint.raw_text)

            for checkpoint in action.relationship_candidate_statuses:
                self.manager.update_relationship_candidate_status(
                    checkpoint.identifier, checkpoint.previous_status
                )
            delete_created_relationships(action.relationship_candidate_statuses)

        self._record_audit("undo", {"operation": action.operation})
        return True

    def undo_checkpoint(self) -> int:
        """Return a checkpoint marker for the current undo stack."""
        return len(self._undo_stack)

    def rollback_to_checkpoint(self, checkpoint: int) -> None:
        """Undo operations until reaching the given checkpoint."""
        while len(self._undo_stack) > checkpoint:
            self.undo_last_operation()

    # Helpers --------------------------------------------------------
    def _entity_from_candidate(self, candidate: EntityCandidate, status: EntityStatus) -> Entity:
        return Entity(
            canonical_name=candidate.canonical_name,
            entity_type=candidate.candidate_type,
            aliases=sorted(set(candidate.aliases)),
            description=candidate.description,
            confidence_score=candidate.confidence_score,
            extraction_method=ExtractionMethod.MANUAL,
            status=status,
            mention_count=candidate.mention_count,
            source_documents=list(candidate.source_documents),
            properties={
                "candidate_key": candidate.candidate_key,
                "chunk_ids": candidate.chunk_ids,
                "display_name": candidate.canonical_name,
            },
        )

    def _candidate_identifier(self, candidate: EntityCandidate) -> str:
        return candidate.id or candidate.candidate_key

    def _collect_aliases(self, candidates: Sequence[EntityCandidate]) -> List[str]:
        aliases = set()
        for candidate in candidates:
            aliases.add(candidate.canonical_name)
            aliases.update(candidate.aliases)
        return sorted(aliases)

    def _collect_unique(self, candidates: Sequence[EntityCandidate], field: str) -> List[str]:
        values = []
        for candidate in candidates:
            values.extend(getattr(candidate, field, []) or [])
        return sorted(set(values))

    def _upsert_normalization_entries(
        self,
        *,
        entity_id: str,
        canonical_name: str,
        entity_type: EntityType,
        raw_texts: Iterable[str],
        status: str,
    ) -> List[NormalizationCheckpoint]:
        changes: List[NormalizationCheckpoint] = []

        variants_by_key: Dict[str, List[str]] = {}
        representative_raw_text: Dict[str, str] = {}
        existing_by_key: Dict[str, NormalizationRecord | None] = {}

        for raw_text in raw_texts:
            cleaned = (raw_text or "").strip()
            if not cleaned:
                continue
            try:
                key = self.normalization_table.normalize_key(cleaned)
            except Exception:  # noqa: BLE001
                continue

            variants_by_key.setdefault(key, []).append(cleaned)
            representative_raw_text.setdefault(key, cleaned)
            if key not in existing_by_key:
                existing_by_key[key] = self.normalization_table.lookup(cleaned)

        if not variants_by_key:
            return []

        entries: List[NormalizationEntry] = []
        for key, variants in variants_by_key.items():
            for variant in sorted(set(variants)):
                entries.append(
                    NormalizationEntry(
                        raw_text=variant,
                        canonical_id=entity_id,
                        canonical_name=canonical_name,
                        entity_type=entity_type.value,
                        method=NormalizationMethod.MANUAL,
                        confidence=1.0,
                        status=status,
                    )
                )
            changes.append(
                NormalizationCheckpoint(
                    raw_text=representative_raw_text[key],
                    previous_record=existing_by_key.get(key),
                )
            )

        self.normalization_table.bulk_upsert(entries)
        return changes

    def _push_undo(self, action: UndoAction) -> None:
        self._undo_stack.append(action)
        self._persist_undo_stack()
        self._record_audit("operation_logged", {"undo": action.compact()})

    def _record_audit(self, event: str, payload: Dict[str, object]) -> None:
        self._audit.record(event, payload)

    def _load_undo_stack(self, path: Path) -> List[UndoAction]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = UndoStackState.model_validate(payload)
            return list(state.actions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load undo stack from {}: {}", path, exc)
            return []

    def _persist_undo_stack(self) -> None:
        self._undo_stack_path.parent.mkdir(parents=True, exist_ok=True)
        state = UndoStackState(actions=self._undo_stack)
        self._undo_stack_path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _normalize_candidate_key(self, value: str) -> str:
        return normalize_candidate_key_fragment(value, normalizer=self._string_normalizer)

    def _promote_related_relationship_candidates(
        self, *, raw_mentions: Sequence[str]
    ) -> List[StatusCheckpoint]:
        """Promote RelationshipCandidates to production relationships when both endpoints resolve."""
        keys = sorted({self._normalize_candidate_key(m) for m in raw_mentions if (m or "").strip()})
        keys = [k for k in keys if k]
        if not keys:
            return []

        rows = self.manager.get_relationship_candidates_involving_keys(
            keys,
            statuses=[CandidateStatus.PENDING.value, CandidateStatus.APPROVED.value],
        )
        if not rows:
            return []

        status_checkpoints: List[StatusCheckpoint] = []
        promoted = 0
        for row in rows:
            try:
                rc = RelationshipCandidate(**row)
            except Exception:  # noqa: BLE001
                continue
            if rc.status not in {CandidateStatus.PENDING, CandidateStatus.APPROVED}:
                continue

            if not (rc.source or "").strip() or not (rc.target or "").strip():
                continue

            try:
                source_record = self.normalization_table.lookup(rc.source)
                target_record = self.normalization_table.lookup(rc.target)
            except Exception:  # noqa: BLE001
                # Bad/empty normalization key or unexpected normalization errors should not
                # break relationship promotion; just skip the candidate.
                continue
            if not (source_record and target_record):
                continue
            if (
                source_record.status.lower() != "approved"
                or target_record.status.lower() != "approved"
            ):
                continue
            if not (source_record.canonical_id and target_record.canonical_id):
                continue

            try:
                rel_type = RelationshipType(str(rc.type))
            except Exception:  # noqa: BLE001
                continue

            relationship_id = self._deterministic_uuid(
                f"relationship:{source_record.canonical_id}:{rel_type.value}:{target_record.canonical_id}"
            )

            provenance = self._relationship_provenance_from_candidate(rc)
            relationship = Relationship(
                id=relationship_id,
                type=rel_type,
                source_entity_id=source_record.canonical_id,
                target_entity_id=target_record.canonical_id,
                description=rc.description or "",
                confidence_score=rc.confidence_score,
                extraction_method=ExtractionMethod.MANUAL,
                status=EntityStatus.APPROVED,
                provenance=provenance,
                properties={"relationship_candidate_key": rc.candidate_key},
            )

            try:
                existed = False
                if hasattr(self.manager, "get_relationship"):
                    existed = bool(self.manager.get_relationship(relationship_id))  # type: ignore[attr-defined]
                self.manager.upsert_relationship(relationship)
            except Exception:  # noqa: BLE001
                continue
            prev_status = rc.status
            self.manager.update_relationship_candidate_status(
                rc.id or rc.candidate_key, CandidateStatus.APPROVED
            )
            status_checkpoints.append(
                StatusCheckpoint(
                    identifier=rc.id or rc.candidate_key,
                    previous_status=prev_status,
                    candidate_key=rc.candidate_key,
                    created_relationship_id=None if existed else relationship_id,
                )
            )
            promoted += 1

        if promoted:
            self._record_audit("promote_relationship_candidates", {"count": promoted})
        return status_checkpoints

    def _relationship_provenance_from_candidate(
        self, candidate: RelationshipCandidate
    ) -> List[RelationshipProvenance]:
        provenances: List[RelationshipProvenance] = []
        for raw in candidate.provenance_events:
            try:
                payload = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            doc_id = str(payload.get("document_id") or "").strip()
            if not doc_id:
                continue
            chunk_id = payload.get("chunk_id")
            confidence = float(payload.get("confidence") or candidate.confidence_score or 0.0)
            provenances.append(
                RelationshipProvenance(
                    document_id=doc_id,
                    chunk_id=str(chunk_id) if chunk_id else None,
                    confidence_score=confidence,
                    extracted_text=str(payload.get("extracted_text") or "") or None,
                    source_extractor=str(payload.get("source_extractor") or "") or None,
                )
            )
        if provenances:
            return provenances
        return [
            RelationshipProvenance(document_id=doc_id)
            for doc_id in (candidate.source_documents or [])
            if doc_id
        ]

    def _deterministic_uuid(self, key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class NeighborhoodIssue(BaseModel):
    """Represents a pending relationship blocked by a peer entity's status."""

    relationship_candidate: RelationshipCandidate
    peer_name: str
    issue_type: (
        str  # "promotable" (peer approved), "resolvable" (peer pending), "missing" (peer unknown)
    )
    peer_candidate_key: str | None = None
    peer_entity_id: str | None = None

    def __str__(self) -> str:
        return f"Relation '{self.relationship_candidate.type}' to '{self.peer_name}' is {self.issue_type}"


def get_neighborhood_issues(
    service: EntityCurationService, entity_name: str, aliases: List[str]
) -> List[NeighborhoodIssue]:
    """Identify pending relationships involving the given entity and classify the blocker."""
    raw_identifiers = [text for text in [entity_name, *aliases] if (text or "").strip()]
    keys = sorted(
        {
            service._normalize_candidate_key(text)  # noqa: SLF001
            for text in raw_identifiers
            if (text or "").strip()
        }
    )
    raw_issues = service.manager.get_pending_relationships_with_peer_status(keys)

    issues: List[NeighborhoodIssue] = []
    for row in raw_issues:
        rc = RelationshipCandidate(**row["relationship"])
        peer_name = row["peer_name"]
        peer_cand = row["peer_candidate"]
        peer_ent = row["peer_entity"]

        issue_type = "missing"
        peer_key = None
        peer_id = None

        if peer_ent and peer_ent.get("status") == "approved":
            issue_type = "promotable"
            peer_id = peer_ent.get("id")
        elif peer_cand and peer_cand.get("status") == "pending":
            issue_type = "resolvable"
            peer_key = peer_cand.get("candidate_key")
        elif peer_cand and peer_cand.get("status") == "rejected":
            # If peer was explicitly rejected, we usually don't prompt again,
            # but the relationship is still pending.
            # For now, treat as "missing" (needs a new entity created) or "resolvable" (un-reject?)
            # Let's treat as "missing" so user can force-create if they really want.
            issue_type = "missing"

        issues.append(
            NeighborhoodIssue(
                relationship_candidate=rc,
                peer_name=peer_name,
                issue_type=issue_type,
                peer_candidate_key=peer_key,
                peer_entity_id=peer_id,
            )
        )
    return issues
