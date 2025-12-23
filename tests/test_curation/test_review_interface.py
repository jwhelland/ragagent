"""CLI tests for review interface (Phase 3 Task 3.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from src.curation import review_interface
from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationTable,
)
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType
from src.utils.config import reset_config

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure config is reset and env vars don't interfere."""
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    reset_config()
    yield
    reset_config()


def _seed_normalization_table(path: Path) -> None:
    table = NormalizationTable(table_path=path)
    table.bulk_upsert(
        [
            NormalizationEntry(
                raw_text="Electric Power Subsystem",
                canonical_id="SYS-1",
                canonical_name="Electric Power Subsystem",
                method=NormalizationMethod.EXACT,
                confidence=0.93,
                status="pending",
            ),
            NormalizationEntry(
                raw_text="EPS",
                canonical_id="SYS-1",
                canonical_name="Electric Power Subsystem",
                method=NormalizationMethod.ACRONYM,
                confidence=0.9,
                status="approved",
            ),
            NormalizationEntry(
                raw_text="Attitude Control System",
                canonical_id="SYS-2",
                canonical_name="Attitude Control System",
                method=NormalizationMethod.FUZZY,
                confidence=0.82,
                status="pending",
            ),
        ]
    )


class _FakeCandidateStore:
    def __init__(self, candidates: List[EntityCandidate]) -> None:
        self._candidates = candidates

    def list_candidates(
        self,
        *,
        status: str | None,
        candidate_types: List[EntityType] | None,
        min_confidence: float | None,
        limit: int,
        offset: int,
    ) -> List[EntityCandidate]:
        items = list(self._candidates)
        if status:
            items = [c for c in items if c.status.value == status]
        if candidate_types:
            allowed = {t.value for t in candidate_types}
            items = [c for c in items if c.candidate_type.value in allowed]
        if min_confidence is not None:
            items = [c for c in items if c.confidence_score >= min_confidence]
        return items[offset : offset + limit]

    def get_candidate(self, query: str) -> EntityCandidate | None:
        for c in self._candidates:
            if c.id == query or c.candidate_key == query or c.canonical_name == query:
                return c
        return None

    def search(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        q = query.lower()
        matched = [
            c
            for c in self._candidates
            if q in c.canonical_name.lower() or q in (c.description or "").lower()
        ]
        return [c.model_dump() for c in matched[:limit]]

    def stats(self) -> Dict[str, Any]:
        totals = {"total": len(self._candidates)}
        return {"totals": totals, "by_type": []}

    def relationship_stats(self) -> Dict[str, Any]:
        return {
            "totals": {"total": 2, "pending": 2, "approved": 0, "rejected": 0},
            "by_type": [{"candidate_type": "DEPENDS_ON", "count": 2}],
        }

    def close(self) -> None:
        return None


def test_candidate_queue_works_with_fake_store(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        EntityCandidate(
            id="cand-1",
            candidate_key="k1",
            canonical_name="Electric Power Subsystem",
            candidate_type=EntityType.SYSTEM,
            confidence_score=0.93,
            status=CandidateStatus.PENDING,
            mention_count=5,
            source_documents=["doc-1"],
            chunk_ids=["chunk-1", "chunk-2"],
        ),
        EntityCandidate(
            id="cand-2",
            candidate_key="k2",
            canonical_name="Attitude Control System",
            candidate_type=EntityType.SYSTEM,
            confidence_score=0.82,
            status=CandidateStatus.PENDING,
            mention_count=2,
            source_documents=["doc-2"],
            chunk_ids=["chunk-3"],
        ),
    ]

    def _factory(_cfg):
        return _FakeCandidateStore(candidates)

    monkeypatch.setattr(review_interface, "create_candidate_store", _factory)

    result = runner.invoke(
        review_interface.app,
        ["queue", "--config", "config/config.yaml", "--limit", "5"],
    )

    assert result.exit_code == 0
    assert "Electric Power Subsystem" in result.stdout
    assert "Attitude Control System" in result.stdout


def test_candidate_search_and_show_work_with_fake_store(monkeypatch) -> None:
    candidates = [
        EntityCandidate(
            id="cand-1",
            candidate_key="k1",
            canonical_name="Electric Power Subsystem",
            candidate_type=EntityType.SYSTEM,
            confidence_score=0.93,
            status=CandidateStatus.PENDING,
            mention_count=5,
            source_documents=["doc-1"],
            chunk_ids=["chunk-1", "chunk-2"],
        ),
        EntityCandidate(
            id="cand-2",
            candidate_key="k2",
            canonical_name="Attitude Control System",
            candidate_type=EntityType.SYSTEM,
            confidence_score=0.82,
            status=CandidateStatus.PENDING,
            mention_count=2,
            source_documents=["doc-2"],
            chunk_ids=["chunk-3"],
            description="Controls attitude via reaction wheels.",
        ),
    ]

    def _factory(_cfg):
        return _FakeCandidateStore(candidates)

    monkeypatch.setattr(review_interface, "create_candidate_store", _factory)

    search_result = runner.invoke(
        review_interface.app,
        ["search", "reaction", "--config", "config/config.yaml"],
    )
    assert search_result.exit_code == 0
    assert "Attitude Control System" in search_result.stdout

    show_result = runner.invoke(
        review_interface.app,
        ["show", "cand-2", "--config", "config/config.yaml"],
    )
    assert show_result.exit_code == 0
    assert "Attitude Control System" in show_result.stdout


def test_normalization_subcommands_still_work(tmp_path: Path) -> None:
    table_path = tmp_path / "table.json"
    _seed_normalization_table(table_path)

    result = runner.invoke(
        review_interface.app,
        [
            "normalization",
            "queue",
            "--table-path",
            str(table_path),
            "--config",
            "config/config.yaml",
            "--status",
            "pending",
            "--limit",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "Electric Power Subsystem" in result.stdout
    assert "Attitude Control System" in result.stdout


def test_stats_command_works_with_fake_store(monkeypatch) -> None:
    candidates = [
        EntityCandidate(
            id="cand-1",
            candidate_key="k1",
            canonical_name="Electric Power Subsystem",
            candidate_type=EntityType.SYSTEM,
            confidence_score=0.93,
            status=CandidateStatus.PENDING,
            mention_count=5,
            source_documents=["doc-1"],
            chunk_ids=["chunk-1", "chunk-2"],
        )
    ]

    def _factory(_cfg):
        return _FakeCandidateStore(candidates)

    monkeypatch.setattr(review_interface, "create_candidate_store", _factory)

    result = runner.invoke(
        review_interface.app,
        ["stats", "--config", "config/config.yaml"],
    )

    assert result.exit_code == 0
    assert "Entity Candidate Stats" in result.stdout
    assert "Relationship Candidate Stats" in result.stdout
