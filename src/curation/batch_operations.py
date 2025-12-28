"""Batch curation operations (Task 3.8)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from src.curation.entity_approval import EntityCurationService
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityStatus, EntityType
from src.utils.config import CurationConfig


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, strip whitespace)."""
    return name.lower().strip()


class ApprovedEntityLookup:
    """Alias-aware lookup for approved entities.

    Maps (normalized_name, entity_type) -> entity_id for canonical names and all aliases.
    """

    def __init__(self) -> None:
        # Maps (normalized_name, type_value) -> entity_id
        self._lookup: Dict[Tuple[str, str], str] = {}
        # Set of all entity IDs in the lookup
        self._entity_ids: Set[str] = set()

    def add_entity(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str,
        aliases: List[str] | None = None,
    ) -> None:
        """Add an entity and its aliases to the lookup."""
        self._entity_ids.add(entity_id)
        # Add canonical name
        key = (normalize_name(canonical_name), entity_type)
        self._lookup[key] = entity_id
        # Add all aliases
        for alias in aliases or []:
            alias_key = (normalize_name(alias), entity_type)
            # Don't overwrite if already exists (first entity wins)
            if alias_key not in self._lookup:
                self._lookup[alias_key] = entity_id

    def find_match(
        self, name: str, entity_type: str, aliases: List[str] | None = None
    ) -> Optional[str]:
        """Find matching entity_id for a candidate.

        Checks canonical name and all aliases against the lookup.

        Args:
            name: Candidate's canonical name
            entity_type: Candidate's entity type
            aliases: Candidate's aliases

        Returns:
            entity_id if exact match found, None otherwise
        """
        # Check canonical name
        key = (normalize_name(name), entity_type)
        if key in self._lookup:
            return self._lookup[key]

        # Check all aliases
        for alias in aliases or []:
            alias_key = (normalize_name(alias), entity_type)
            if alias_key in self._lookup:
                return self._lookup[alias_key]

        return None

    def __len__(self) -> int:
        return len(self._entity_ids)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entity_ids


def build_approved_entity_lookup(manager: Neo4jManager) -> ApprovedEntityLookup:
    """Build an alias-aware lookup of all approved entities from Neo4j.

    Args:
        manager: Connected Neo4jManager instance

    Returns:
        ApprovedEntityLookup with all approved entities and their aliases
    """
    lookup = ApprovedEntityLookup()

    # Query all approved entities with their aliases
    for entity_type in EntityType:
        entities = manager.list_entities(
            entity_type=entity_type,
            status=EntityStatus.APPROVED,
            limit=10000,  # Reasonable upper bound
            offset=0,
        )
        for entity in entities:
            entity_id = entity.get("id")
            canonical_name = entity.get("canonical_name")
            aliases = entity.get("aliases", [])

            if entity_id and canonical_name:
                lookup.add_entity(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    entity_type=entity_type.value,
                    aliases=aliases,
                )

    logger.info(f"Built approved entity lookup with {len(lookup)} entities")
    return lookup


class BatchOperationPreview(BaseModel):
    """Preview of a batch operation."""

    to_approve: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)
    threshold: float
    total_candidates: int


class AutoApproveMatchPreview(BaseModel):
    """Preview of auto-approve existing matches operation."""

    # List of (candidate_key, matched_entity_id) tuples
    to_merge: List[Tuple[str, str]] = Field(default_factory=list)
    # Skipped due to conflicting types
    skipped_conflicting: List[str] = Field(default_factory=list)
    # Skipped because no match found
    skipped_no_match: List[str] = Field(default_factory=list)
    # Skipped due to ambiguous match (multiple entities could match)
    skipped_ambiguous: List[str] = Field(default_factory=list)
    total_candidates: int
    total_approved_entities: int


class BatchOperationResult(BaseModel):
    """Result of a batch operation."""

    approved_entities: List[str] = Field(default_factory=list)
    merged_entities: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)
    preview_only: bool = False
    rolled_back: bool = False


class BatchCurationService:
    """Execute batch curation actions with preview/rollback support."""

    def __init__(
        self,
        curation_service: EntityCurationService,
        config: CurationConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.curation_service = curation_service
        self.config = config or CurationConfig()
        self.progress_callback = progress_callback

    def preview_batch_approve(
        self, candidates: Sequence[EntityCandidate], *, threshold: float | None = None
    ) -> BatchOperationPreview:
        threshold = threshold or self.config.auto_approve_threshold
        to_approve: List[str] = []
        skipped: List[str] = []

        for candidate in candidates:
            if candidate.status != CandidateStatus.PENDING:
                skipped.append(candidate.candidate_key)
                continue
            if candidate.confidence_score >= threshold:
                to_approve.append(candidate.candidate_key)
            else:
                skipped.append(candidate.candidate_key)

        return BatchOperationPreview(
            to_approve=to_approve,
            skipped=skipped,
            threshold=threshold,
            total_candidates=len(candidates),
        )

    def batch_approve(
        self,
        candidates: Sequence[EntityCandidate],
        *,
        threshold: float | None = None,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        preview = self.preview_batch_approve(candidates, threshold=threshold)
        if dry_run:
            logger.info("Dry-run: would approve {} candidates", len(preview.to_approve))
            return BatchOperationResult(
                approved_entities=[],
                skipped=preview.skipped,
                preview_only=True,
            )

        checkpoint = self.curation_service.undo_checkpoint()
        approved_entities: List[str] = []
        try:
            for idx, candidate in enumerate(candidates):
                self._tick(f"Approving {idx + 1}/{len(candidates)}")
                if candidate.candidate_key not in preview.to_approve:
                    continue
                entity_id = self.curation_service.approve_candidate(candidate)
                approved_entities.append(entity_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Batch approve failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            return BatchOperationResult(
                approved_entities=[],
                skipped=preview.skipped,
                failed=[str(exc)],
                rolled_back=True,
            )

        return BatchOperationResult(
            approved_entities=approved_entities,
            skipped=preview.skipped,
        )

    def batch_merge_clusters(
        self,
        clusters: Sequence[Sequence[EntityCandidate]],
        *,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        if dry_run:
            merged = [cluster[0].candidate_key for cluster in clusters if cluster]
            logger.info("Dry-run: would merge {} clusters", len(merged))
            return BatchOperationResult(merged_entities=[], preview_only=True)

        checkpoint = self.curation_service.undo_checkpoint()
        merged_entities: List[str] = []
        failed: List[str] = []

        try:
            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                self._tick(f"Merging cluster {idx + 1}/{len(clusters)}")
                primary, *duplicates = cluster
                entity_id = self.curation_service.merge_candidates(primary, duplicates)
                merged_entities.append(entity_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Batch merge failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            failed.append(str(exc))
            return BatchOperationResult(
                merged_entities=[],
                failed=failed,
                rolled_back=True,
            )

        return BatchOperationResult(merged_entities=merged_entities, failed=failed)

    def batch_merge_into_entity(
        self,
        entity_id: str,
        candidates: Sequence[EntityCandidate],
        *,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        """Merge multiple candidates into an existing entity."""
        if dry_run:
            logger.info(
                "Dry-run: would merge {} candidates into entity {}", len(candidates), entity_id
            )
            return BatchOperationResult(merged_entities=[], preview_only=True)

        checkpoint = self.curation_service.undo_checkpoint()
        try:
            for idx, candidate in enumerate(candidates):
                self._tick(f"Merging candidate {idx + 1}/{len(candidates)} into entity {entity_id}")
                success = self.curation_service.merge_candidate_into_entity(entity_id, candidate)
                if not success:
                    raise Exception(f"Failed to merge candidate {candidate.candidate_key}")
        except Exception as exc:
            logger.exception("Batch merge into entity failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            return BatchOperationResult(
                merged_entities=[],
                failed=[str(exc)],
                rolled_back=True,
            )

        return BatchOperationResult(merged_entities=[entity_id])

    def _tick(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            logger.debug(message)

    def preview_auto_approve_existing(
        self,
        candidates: Sequence[EntityCandidate],
        lookup: ApprovedEntityLookup,
    ) -> AutoApproveMatchPreview:
        """Preview which candidates would be merged into existing approved entities.

        Args:
            candidates: Pending entity candidates to check
            lookup: Pre-built lookup of approved entities

        Returns:
            Preview showing which candidates would be merged vs skipped
        """
        to_merge: List[Tuple[str, str]] = []
        skipped_conflicting: List[str] = []
        skipped_no_match: List[str] = []

        for candidate in candidates:
            # Skip non-pending candidates
            if candidate.status != CandidateStatus.PENDING:
                continue

            # Skip candidates with conflicting types (ambiguous, requires human review)
            if candidate.conflicting_types:
                skipped_conflicting.append(candidate.candidate_key)
                continue

            # Try to find a matching approved entity
            entity_id = lookup.find_match(
                name=candidate.canonical_name,
                entity_type=candidate.candidate_type.value,
                aliases=candidate.aliases,
            )

            if entity_id:
                to_merge.append((candidate.candidate_key, entity_id))
            else:
                skipped_no_match.append(candidate.candidate_key)

        return AutoApproveMatchPreview(
            to_merge=to_merge,
            skipped_conflicting=skipped_conflicting,
            skipped_no_match=skipped_no_match,
            skipped_ambiguous=[],  # Not currently tracked, could be added
            total_candidates=len(candidates),
            total_approved_entities=len(lookup),
        )

    def auto_approve_existing_matches(
        self,
        candidates: Sequence[EntityCandidate],
        lookup: ApprovedEntityLookup,
        *,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        """Merge candidates that exactly match existing approved entities.

        This implements Strategy 2 from the entity deduplication plan:
        - Scans pending candidates
        - Finds exact matches against approved entities (by name or alias, same type)
        - Merges matching candidates into the existing entity
        - Skips candidates with conflicting_types (requires human review)

        WARNING: This command must NOT run concurrently with ragagent-review-interactive,
        ragagent-ingest, or other batch-approve instances. Concurrent execution may cause
        data corruption.

        Args:
            candidates: Pending entity candidates to check
            lookup: Pre-built lookup of approved entities
            dry_run: If True, only preview what would happen

        Returns:
            BatchOperationResult with merged entities and skipped candidates
        """
        preview = self.preview_auto_approve_existing(candidates, lookup)

        if dry_run:
            logger.info(
                "Dry-run: would merge {} candidates into existing entities",
                len(preview.to_merge),
            )
            return BatchOperationResult(
                merged_entities=[],
                skipped=preview.skipped_conflicting + preview.skipped_no_match,
                preview_only=True,
            )

        # Execute the merges
        checkpoint = self.curation_service.undo_checkpoint()
        merged_entities: List[str] = []
        failed: List[str] = []

        # Build a lookup from candidate_key to candidate for efficient access
        candidate_lookup: Dict[str, EntityCandidate] = {c.candidate_key: c for c in candidates}

        try:
            for idx, (candidate_key, entity_id) in enumerate(preview.to_merge):
                self._tick(
                    f"Merging candidate {idx + 1}/{len(preview.to_merge)} into entity {entity_id}"
                )

                candidate = candidate_lookup.get(candidate_key)
                if not candidate:
                    failed.append(f"Candidate not found: {candidate_key}")
                    continue

                try:
                    success = self.curation_service.merge_candidate_into_entity(
                        entity_id, candidate
                    )
                    if success:
                        merged_entities.append(entity_id)
                    else:
                        failed.append(f"Failed to merge {candidate_key}")
                except Exception as e:
                    logger.warning(f"Error merging {candidate_key}: {e}")
                    failed.append(f"{candidate_key}: {str(e)}")

        except Exception as exc:
            logger.exception("Auto-approve batch failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            return BatchOperationResult(
                merged_entities=[],
                skipped=preview.skipped_conflicting + preview.skipped_no_match,
                failed=[str(exc)],
                rolled_back=True,
            )

        # Deduplicate merged_entities (same entity may be merged into multiple times)
        unique_merged = list(dict.fromkeys(merged_entities))

        return BatchOperationResult(
            merged_entities=unique_merged,
            skipped=preview.skipped_conflicting + preview.skipped_no_match,
            failed=failed,
        )
