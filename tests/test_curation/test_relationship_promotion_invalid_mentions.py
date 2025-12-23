"""Unit tests for robust relationship promotion on malformed candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import NormalizationTable
from src.storage.schemas import CandidateStatus
from src.utils.config import Config


class _FakeManager:
    def __init__(self) -> None:
        self.relationship_candidate_rows: List[Dict[str, Any]] = []

    def get_relationship_candidates_involving_keys(
        self, keys: List[str], *, statuses: List[str] | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        return list(self.relationship_candidate_rows)

    def get_entity(self, entity_id: str) -> Dict[str, Any] | None:
        return None


def test_promotion_skips_candidates_with_unusable_target(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "u.json",
    )

    manager.relationship_candidate_rows = [
        {
            "id": "relcand-bad",
            "candidate_key": "foo:RELATED_TO:bar",
            "source": "Foo",
            "target": "!!!",  # normalizes to empty -> used to raise
            "type": "RELATED_TO",
            "description": "",
            "confidence_score": 0.5,
            "status": CandidateStatus.PENDING.value,
            "mention_count": 1,
            "source_documents": [],
            "chunk_ids": [],
            "provenance_events": [],
        }
    ]

    # Should not raise.
    promoted = service._promote_related_relationship_candidates(
        raw_mentions=["foo", "bar"]
    )  # noqa: SLF001
    assert promoted == []
