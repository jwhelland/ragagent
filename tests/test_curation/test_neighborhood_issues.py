"""Unit tests for neighborhood issue detection helpers."""

from __future__ import annotations

from pathlib import Path

from src.curation.entity_approval import EntityCurationService, get_neighborhood_issues
from src.normalization.normalization_table import NormalizationTable
from src.utils.config import Config


class _CapturingManager:
    def __init__(self) -> None:
        self.last_identifiers: list[str] | None = None

    def get_pending_relationships_with_peer_status(self, identifiers: list[str], limit: int = 50):
        self.last_identifiers = list(identifiers)
        return []


def test_get_neighborhood_issues_includes_normalized_identifiers(tmp_path: Path) -> None:
    manager = _CapturingManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(manager=manager, normalization_table=table, config=Config())

    get_neighborhood_issues(service, "AC Motor", ["AC_Motor", "  AC motor  "])

    assert manager.last_identifiers is not None
    assert "AC Motor" in manager.last_identifiers
    assert "AC_Motor" in manager.last_identifiers
    assert "  AC motor  " in manager.last_identifiers
    assert "ac_motor" in manager.last_identifiers
