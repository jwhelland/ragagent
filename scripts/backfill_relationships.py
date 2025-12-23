#!/usr/bin/env python3
"""Backfill promoted graph relationships from RelationshipCandidate nodes.

This script is useful after:
- approving entities/candidates (so normalization table has canonical IDs)
- approving relationship candidates in the TUI (which previously could block promotion)
- adding new RelationshipType enum values

It scans RelationshipCandidate nodes, collects endpoint keys from their candidate_key,
and runs the promotion logic to create actual Neo4j relationships when both endpoints
resolve to approved entities.

Usage:
  uv run python scripts/backfill_relationships.py
  uv run python scripts/backfill_relationships.py --limit 50000
  uv run python scripts/backfill_relationships.py --statuses pending approved
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import NormalizationTable
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import CandidateStatus
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill graph relationships by promoting RelationshipCandidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--normalization-table",
        default=None,
        help="Override normalization table JSON path (defaults to config).",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=[CandidateStatus.PENDING.value, CandidateStatus.APPROVED.value],
        help="RelationshipCandidate statuses to consider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200_000,
        help="Max RelationshipCandidate rows to scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only (no writes).",
    )
    return parser.parse_args()


def _endpoint_keys_from_candidate_key(candidate_key: str) -> tuple[str, str] | None:
    # Expected format: "{source_key}:{type}:{target_key}"
    parts = [p for p in (candidate_key or "").split(":") if p]
    if len(parts) < 3:
        return None
    return parts[0], parts[-1]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    table_path = (
        Path(args.normalization_table)
        if args.normalization_table
        else Path(config.normalization.normalization_table_path)
    )
    norm_table = NormalizationTable(table_path=table_path, config=config.normalization)

    manager = Neo4jManager(config.database)
    manager.connect()
    service = EntityCurationService(manager=manager, normalization_table=norm_table, config=config)

    try:
        with manager.session() as session:
            rows = session.run(
                """
                MATCH (c:RelationshipCandidate)
                WHERE c.status IN $statuses
                RETURN c.candidate_key as candidate_key
                LIMIT $limit
                """,
                statuses=list(args.statuses),
                limit=int(args.limit),
            ).data()

        endpoint_keys: set[str] = set()
        for row in rows:
            candidate_key = str(row.get("candidate_key") or "")
            endpoints = _endpoint_keys_from_candidate_key(candidate_key)
            if not endpoints:
                continue
            source_key, target_key = endpoints
            endpoint_keys.add(source_key)
            endpoint_keys.add(target_key)

        logger.info(
            "Scanned RelationshipCandidates",
            candidate_rows=len(rows),
            endpoint_keys=len(endpoint_keys),
            statuses=list(args.statuses),
        )

        if args.dry_run:
            return 0

        # Promotion logic uses normalization table lookups on rc.source/rc.target, but uses endpoint
        # keys to efficiently load relevant RelationshipCandidate nodes.
        promoted = service._promote_related_relationship_candidates(  # noqa: SLF001
            raw_mentions=sorted(endpoint_keys)
        )
        logger.success("Promoted {} relationship candidates into graph edges", len(promoted))

        with manager.session() as session:
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            by_type = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as n
                ORDER BY n DESC
                LIMIT 20
                """
            ).data()
        logger.info("Graph relationship count: {}", rel_count)
        logger.info("Top relationship types: {}", by_type)
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
