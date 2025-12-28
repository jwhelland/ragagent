#!/usr/bin/env python3
"""Entity discovery CLI script.

This script analyzes EntityCandidate nodes stored in Neo4j and generates a discovery report:
- frequency stats by type/status
- chunk-level co-occurrence edges and clusters
- merge suggestions (fuzzy by default; optional semantic via embeddings)
- suggested new entity type labels from unknown conflicting types

Usage:
    python scripts/run_discovery.py
    python scripts/run_discovery.py --min-confidence 0.7 --max-candidates 500
    python scripts/run_discovery.py --output-dir data/discovery/custom --no-viz
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline.discovery_pipeline import DiscoveryParameters, DiscoveryPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run corpus-wide entity discovery analysis and generate a report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/discovery/<timestamp>)",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence filter")
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Include status (repeatable; default: pending, approved, rejected)",
    )
    parser.add_argument(
        "--candidate-type",
        action="append",
        default=[],
        help="Include candidate type (repeatable; default: all)",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=2000, help="Max candidates to analyze"
    )
    parser.add_argument(
        "--max-entities-per-chunk",
        type=int,
        default=50,
        help="Safety cap for co-occurrence counting per chunk",
    )
    parser.add_argument(
        "--min-cooccurrence",
        type=int,
        default=2,
        help="Min co-mention count for edges/clusters",
    )
    parser.add_argument(
        "--max-edges", type=int, default=500, help="Max co-occurrence edges to keep"
    )
    parser.add_argument(
        "--max-clusters", type=int, default=50, help="Max co-occurrence clusters to keep"
    )
    parser.add_argument(
        "--enable-semantic-merge",
        action="store_true",
        help="Attempt embedding-based semantic merge suggestions (may require model availability)",
    )
    parser.add_argument(
        "--max-merge-suggestions",
        type=int,
        default=100,
        help="Max merge suggestions (per method)",
    )
    parser.add_argument(
        "--resolve-existing",
        action="store_true",
        default=False,
        help="Auto-resolve candidates matching approved entities (Strategy 1). "
        "Matching candidates are filtered from the report and tracked separately.",
    )
    parser.add_argument("--no-viz", action="store_true", help="Do not write GraphViz DOT output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    config = load_config(args.config)
    pipeline = DiscoveryPipeline(config)

    statuses = tuple(s.lower() for s in (args.status or [])) or ("pending", "approved", "rejected")
    candidate_types = tuple(t.upper() for t in (args.candidate_type or []))
    params = DiscoveryParameters(
        min_confidence=float(args.min_confidence),
        statuses=statuses,
        candidate_types=candidate_types,
        max_candidates=int(args.max_candidates),
        max_entities_per_chunk=int(args.max_entities_per_chunk),
        min_cooccurrence=int(args.min_cooccurrence),
        max_edges=int(args.max_edges),
        max_clusters=int(args.max_clusters),
        enable_semantic_merge=bool(args.enable_semantic_merge),
        max_merge_suggestions=int(args.max_merge_suggestions),
        resolve_existing=bool(args.resolve_existing),
    )

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/discovery") / timestamp

    report = pipeline.run(
        output_dir=output_dir,
        parameters=params,
        create_visualization=not args.no_viz,
    )
    logger.info(
        "Done. Reports: MD={}, HTML={}",
        report.artifacts.get("report_markdown"),
        report.artifacts.get("report_html"),
    )
    if report.auto_resolved:
        logger.info(
            "Auto-resolved {} candidates to existing entities (see report for details)",
            len(report.auto_resolved),
        )


if __name__ == "__main__":
    main()
