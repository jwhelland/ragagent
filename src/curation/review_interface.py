"""CLI review interface (Task 3.6 core).

Primary focus: reviewing EntityCandidate nodes (filters, sorting, search, details).
Also includes a subcommand group for reviewing the normalization table (Task 3.5).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import typer
from rich.console import Console
from rich.table import Table

from src.curation.batch_operations import BatchCurationService
from src.curation.entity_approval import EntityCurationService, get_neighborhood_issues
from src.normalization.normalization_table import (
    NormalizationMethod,
    NormalizationRecord,
    NormalizationTable,
)
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType
from src.utils.config import Config, load_config

app = typer.Typer(help="Review and search entity candidates.")
normalization_app = typer.Typer(help="Review and search normalization mappings.")
app.add_typer(normalization_app, name="normalization")

console = Console(color_system=None, force_terminal=False, width=120)


class CandidateStore(Protocol):
    def list_candidates(
        self,
        *,
        status: str | None,
        candidate_types: Sequence[EntityType] | None,
        min_confidence: float | None,
        limit: int,
        offset: int,
    ) -> List[EntityCandidate]: ...

    def get_candidate(self, query: str) -> EntityCandidate | None: ...

    def search(self, query: str, *, limit: int) -> List[Dict[str, Any]]: ...

    def stats(self) -> Dict[str, Any]: ...

    def relationship_stats(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


class Neo4jCandidateStore:
    def __init__(self, config: Config) -> None:
        self._manager = Neo4jManager(config.database)
        self._manager.connect()

    def list_candidates(
        self,
        *,
        status: str | None,
        candidate_types: Sequence[EntityType] | None,
        min_confidence: float | None,
        limit: int,
        offset: int,
    ) -> List[EntityCandidate]:
        rows = self._manager.get_entity_candidates(
            status=status,
            candidate_types=list(candidate_types) if candidate_types else None,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
        return [EntityCandidate(**row) for row in rows]

    def get_candidate(self, query: str) -> EntityCandidate | None:
        row = self._manager.get_entity_candidate(query)
        return EntityCandidate(**row) if row else None

    def search(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        return self._manager.search_entity_candidates(query, limit=limit)

    def stats(self) -> Dict[str, Any]:
        return self._manager.get_entity_candidate_statistics()

    def relationship_stats(self) -> Dict[str, Any]:
        with self._manager.session() as session:
            totals = session.run(
                """
                MATCH (c:RelationshipCandidate)
                RETURN
                    count(c) as total,
                    count(CASE WHEN c.status = 'pending' THEN 1 END) as pending,
                    count(CASE WHEN c.status = 'approved' THEN 1 END) as approved,
                    count(CASE WHEN c.status = 'rejected' THEN 1 END) as rejected
                """
            ).single()

            by_type = session.run(
                """
                MATCH (c:RelationshipCandidate)
                RETURN c.type as candidate_type, count(c) as count
                ORDER BY count DESC
                """
            ).data()

            return {"totals": dict(totals) if totals else {}, "by_type": by_type}

    def close(self) -> None:
        self._manager.close()


def create_candidate_store(config: Config) -> CandidateStore:
    return Neo4jCandidateStore(config)


def _create_curation_service(
    cfg: Config, table_path: Path | None
) -> tuple[EntityCurationService, NormalizationTable, Neo4jManager]:
    table = _load_normalization_table(cfg, table_path)
    manager = Neo4jManager(cfg.database)
    manager.connect()
    service = EntityCurationService(manager, table, cfg)
    return service, table, manager


def _render_candidate_table(
    candidates: Sequence[EntityCandidate], *, title: str, total_hint: str | None = None
) -> None:
    table_title = title if total_hint is None else f"{title} ({total_hint})"
    table = Table(title=table_title)
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Conf")
    table.add_column("Status")
    table.add_column("Mentions", justify="right")
    table.add_column("Docs", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("ID", style="magenta")

    for candidate in candidates:
        table.add_row(
            candidate.canonical_name,
            candidate.candidate_type.value,
            f"{candidate.confidence_score:.2f}",
            candidate.status.value,
            str(candidate.mention_count),
            str(len(candidate.source_documents)),
            str(len(candidate.chunk_ids)),
            candidate.id or "-",
        )

    console.print(table)


def _load(config_path: Path) -> Config:
    return load_config(config_path)


def _parse_types(types: Sequence[str]) -> List[EntityType]:
    parsed: List[EntityType] = []
    for t in types:
        parsed.append(EntityType(t.upper()))
    return parsed


def _parse_single_type(type_value: str | None) -> EntityType | None:
    if not type_value:
        return None
    return EntityType(type_value.upper())


def _ensure_candidate(store: CandidateStore, query: str) -> EntityCandidate:
    candidate = store.get_candidate(query)
    if not candidate:
        console.print("[red]No matching candidate found.[/red]")
        raise typer.Exit(code=1)
    return candidate


def _handle_neighborhood_issues(
    service: EntityCurationService, store: CandidateStore, entity_name: str, aliases: List[str]
) -> None:
    """Interactively resolve blocked relationships."""
    issues = get_neighborhood_issues(service, entity_name, aliases)
    if not issues:
        return

    console.print(
        f"\n[bold cyan]Found {len(issues)} pending relationship(s) blocked by peers:[/bold cyan]"
    )

    for issue in issues:
        console.print(
            f"\nRelationship: [blue]{issue.relationship_candidate.source}[/blue] --[{issue.relationship_candidate.type}]--> [blue]{issue.relationship_candidate.target}[/blue]"
        )
        console.print(f"Peer: [yellow]{issue.peer_name}[/yellow] ({issue.issue_type})")

        if issue.issue_type == "promotable":
            # Just approve the relationship, the peer is already approved
            if typer.confirm(f"Promote relationship to approved entity '{issue.peer_name}'?"):
                service.approve_relationship_candidate(issue.relationship_candidate)
                console.print("[green]Relationship promoted.[/green]")

        elif issue.issue_type == "resolvable" and issue.peer_candidate_key:
            if typer.confirm(f"Approve pending candidate '{issue.peer_name}'?"):
                try:
                    peer_candidate = _ensure_candidate(store, issue.peer_candidate_key)
                    service.approve_candidate(peer_candidate)
                    console.print(
                        f"[green]Approved peer '{peer_candidate.canonical_name}'.[/green]"
                    )
                    # Recursively check new neighborhood
                    _handle_neighborhood_issues(
                        service, store, peer_candidate.canonical_name, peer_candidate.aliases
                    )
                except Exception as e:
                    console.print(f"[red]Failed to approve peer: {e}[/red]")

        elif issue.issue_type == "missing":
            if typer.confirm(f"Create new entity for '{issue.peer_name}'?"):
                # Simple interactive creation
                new_type_str = typer.prompt("Entity Type", default="CONCEPT")
                try:
                    new_type = EntityType(new_type_str.upper())

                    # Create synthetic candidate
                    synthetic = EntityCandidate(
                        candidate_key=issue.peer_name.lower().replace(" ", "_"),
                        canonical_name=issue.peer_name,
                        candidate_type=new_type,
                        status=CandidateStatus.PENDING,
                        confidence_score=1.0,
                        description="Created during neighborhood approval",
                        source_documents=issue.relationship_candidate.source_documents,
                        chunk_ids=issue.relationship_candidate.chunk_ids,
                    )

                    # Approve it
                    service.approve_candidate(synthetic)
                    console.print(f"[green]Created and approved entity '{issue.peer_name}'.[/green]")
                    # Recursively check new neighborhood
                    _handle_neighborhood_issues(
                        service, store, synthetic.canonical_name, synthetic.aliases
                    )
                except ValueError:
                    console.print(f"[red]Invalid entity type: {new_type_str}[/red]")
                except Exception as e:
                    console.print(f"[red]Failed to create entity: {e}[/red]")


@app.command("queue")
def queue(
    min_confidence: float = typer.Option(0.0, help="Minimum confidence filter."),
    entity_type: List[str] = typer.Option([], help="Filter by entity type (repeatable)."),
    status: str = typer.Option("pending", help="Filter by status (pending/approved/rejected)."),
    sort_by: str = typer.Option(
        "confidence",
        help="Sort by confidence|mentions.",
        show_default=True,
    ),
    limit: int = typer.Option(20, help="Max rows to display.", min=1),
    offset: int = typer.Option(0, help="Offset for paging.", min=0),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Display entity candidate review queue with filters and sorting."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    try:
        candidate_types = _parse_types(entity_type) if entity_type else None
        candidates = store.list_candidates(
            status=status or None,
            candidate_types=candidate_types,
            min_confidence=min_confidence if min_confidence > 0 else None,
            limit=limit,
            offset=offset,
        )

        if sort_by == "mentions":
            candidates = sorted(
                candidates,
                key=lambda c: (-(c.mention_count or 0), -(c.confidence_score or 0.0)),
            )

        if not candidates:
            console.print("[yellow]No entity candidates found for the given filters.[/yellow]")
            return

        _render_candidate_table(
            candidates,
            title="Entity Candidate Queue",
            total_hint=f"offset={offset}, limit={limit}",
        )
    finally:
        store.close()


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query for entity candidates."),
    limit: int = typer.Option(20, help="Maximum results.", min=1),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Search EntityCandidates (full-text) by name/description/aliases."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    try:
        results = store.search(query, limit=limit)
        if not results:
            console.print("[yellow]No matches found.[/yellow]")
            return

        candidates = [EntityCandidate(**item) for item in results]
        _render_candidate_table(candidates, title="Entity Candidate Search Results")
    finally:
        store.close()


@app.command("show")
def show(
    query: str = typer.Argument(..., help="Candidate id, candidate_key, or canonical name."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Show detailed information for a single EntityCandidate."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    try:
        candidate = store.get_candidate(query)
        if not candidate:
            console.print("[red]No matching candidate found.[/red]")
            raise typer.Exit(code=1)

        console.print(f"[bold]{candidate.canonical_name}[/bold] ({candidate.id or '-'})")
        console.print(
            f"Type: {candidate.candidate_type.value} | Status: {candidate.status.value} "
            f"| Confidence: {candidate.confidence_score:.2f}"
        )
        console.print(f"Mentions: {candidate.mention_count}")
        if candidate.aliases:
            console.print(f"Aliases: {', '.join(candidate.aliases)}")
        if candidate.description:
            console.print(f"Description: {candidate.description}")
        if candidate.source_documents:
            preview = ", ".join(candidate.source_documents[:10])
            console.print(f"Source documents ({len(candidate.source_documents)}): {preview}")
        if candidate.chunk_ids:
            preview = ", ".join(candidate.chunk_ids[:10])
            console.print(f"Chunk IDs ({len(candidate.chunk_ids)}): {preview}")
    finally:
        store.close()


@app.command("stats")
def stats(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Show candidate counts by status/type."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    try:
        raw = store.stats()
        totals = raw.get("totals") or {}
        by_type = raw.get("by_type") or []

        console.print("[bold]Entity Candidate Stats[/bold]")
        console.print(
            "Totals: "
            + ", ".join(
                f"{key}={value}"
                for key, value in [
                    ("total", totals.get("total")),
                    ("pending", totals.get("pending")),
                    ("approved", totals.get("approved")),
                    ("rejected", totals.get("rejected")),
                ]
                if value is not None
            )
        )
        if by_type:
            console.print("By type:")
            for row in by_type:
                console.print(f"  {row.get('candidate_type')}: {row.get('count')}")

        # Add Relationship Stats
        rel_totals: Dict[str, Any] = {}
        rel_by_type: List[Dict[str, Any]] = []
        relationship_stats = getattr(store, "relationship_stats", None)
        if callable(relationship_stats):
            rel_raw = relationship_stats() or {}
            rel_totals = rel_raw.get("totals") or {}
            rel_by_type = rel_raw.get("by_type") or []

        if rel_totals:
            console.print("\n[bold]Relationship Candidate Stats[/bold]")
            console.print(
                "Totals: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in [
                        ("total", rel_totals.get("total")),
                        ("pending", rel_totals.get("pending")),
                        ("approved", rel_totals.get("approved")),
                        ("rejected", rel_totals.get("rejected")),
                    ]
                    if value is not None
                )
            )
            if rel_by_type:
                console.print("By type:")
                for row in rel_by_type:
                    console.print(f"  {row.get('candidate_type')}: {row.get('count')}")

    finally:
        store.close()


@app.command("approve")
def approve(
    query: str = typer.Argument(..., help="Candidate id/key/name to approve."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
    recursive: bool = typer.Option(True, help="Check and prompt for blocked relationships."),
) -> None:
    """Approve a candidate and promote to production entity."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    try:
        candidate = _ensure_candidate(store, query)
        entity_id = service.approve_candidate(candidate)
        console.print(f"[green]Approved {candidate.canonical_name} -> entity {entity_id}[/green]")

        if recursive:
            _handle_neighborhood_issues(service, store, candidate.canonical_name, candidate.aliases)

    finally:
        store.close()
        manager.close()


@app.command("reject")
def reject(
    query: str = typer.Argument(..., help="Candidate id/key/name to reject."),
    reason: str = typer.Option("", help="Optional rejection reason."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Reject a candidate (kept for audit)."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    try:
        candidate = _ensure_candidate(store, query)
        service.reject_candidate(candidate, reason=reason)
        console.print(f"[yellow]Rejected {candidate.canonical_name}[/yellow]")
    finally:
        store.close()
        manager.close()


@app.command("edit")
def edit(
    query: str = typer.Argument(..., help="Candidate id/key/name to edit."),
    canonical_name: str = typer.Option("", help="New canonical name."),
    candidate_type: str = typer.Option("", help="New candidate type."),
    description: str = typer.Option("", help="New description."),
    confidence: Optional[float] = typer.Option(None, help="New confidence score."),
    aliases: List[str] = typer.Option([], help="Replace aliases (repeatable)."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Edit candidate fields."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    try:
        candidate = _ensure_candidate(store, query)
        updates: Dict[str, Any] = {}
        if canonical_name:
            updates["canonical_name"] = canonical_name
        parsed_type = _parse_single_type(candidate_type)
        if parsed_type:
            updates["candidate_type"] = parsed_type
        if description:
            updates["description"] = description
        if confidence is not None:
            updates["confidence_score"] = confidence
        if aliases:
            updates["aliases"] = aliases

        updated = service.edit_candidate(candidate, updates)
        console.print(f"[green]Updated {updated.candidate_key}[/green]")
    finally:
        store.close()
        manager.close()


@app.command("merge")
def merge(
    primary: str = typer.Argument(..., help="Primary candidate id/key/name."),
    duplicate: List[str] = typer.Argument(..., help="Duplicate candidate ids/keys/names."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Merge multiple candidates into a single approved entity."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    try:
        primary_candidate = _ensure_candidate(store, primary)
        duplicates = [_ensure_candidate(store, q) for q in duplicate]
        entity_id = service.merge_candidates(primary_candidate, duplicates)
        console.print(
            f"[green]Merged {len(duplicates)} candidates into entity {entity_id} "
            f"(primary={primary_candidate.candidate_key})[/green]"
        )
    finally:
        store.close()
        manager.close()


@app.command("undo")
def undo(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
) -> None:
    """Undo the most recent curation operation."""
    cfg = _load(config)
    service, _table, manager = _create_curation_service(cfg, table_path)
    try:
        if service.undo_last_operation():
            console.print("[green]Reverted last curation operation.[/green]")
        else:
            console.print("[yellow]Nothing to undo.[/yellow]")
    finally:
        manager.close()


@app.command("batch-approve")
def batch_approve(
    min_confidence: Optional[float] = typer.Option(
        None, help="Confidence threshold for auto-approve."
    ),
    limit: int = typer.Option(50, help="Number of candidates to consider.", min=1),
    offset: int = typer.Option(0, help="Offset for paging.", min=0),
    dry_run: bool = typer.Option(False, help="Preview without executing."),
    preview_limit: int = typer.Option(20, help="Max items to show in preview.", min=1),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Approve candidates above threshold in batch."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    batch_service = BatchCurationService(service, cfg.curation)
    try:
        candidates = store.list_candidates(
            status=CandidateStatus.PENDING.value,
            candidate_types=None,
            min_confidence=None,
            limit=limit,
            offset=offset,
        )
        preview = batch_service.preview_batch_approve(candidates, threshold=min_confidence)
        if dry_run:
            console.print(
                f"[cyan]Preview: threshold={preview.threshold:.2f}, "
                f"{len(preview.to_approve)}/{preview.total_candidates} candidates would be approved[/cyan]"
            )
            for key in preview.to_approve[:preview_limit]:
                console.print(f"  {key}")
            return

        result = batch_service.batch_approve(candidates, threshold=min_confidence, dry_run=False)
        if result.preview_only:
            console.print("[cyan]Preview-only (no changes applied).[/cyan]")
        else:
            console.print(
                f"[green]Approved {len(result.approved_entities)} candidates "
                f"(skipped {len(result.skipped)})[/green]"
            )
        if result.failed:
            console.print(f"[red]Failures: {result.failed}[/red]")
        if result.rolled_back:
            console.print("[red]Rolled back batch due to errors.[/red]")
    finally:
        store.close()
        manager.close()


def _parse_cluster_arg(cluster_arg: str) -> tuple[str, List[str]]:
    """Parse cluster arg format primary:dup1,dup2."""
    if ":" not in cluster_arg:
        return cluster_arg, []
    primary, dup_str = cluster_arg.split(":", maxsplit=1)
    duplicates = [d for d in dup_str.split(",") if d]
    return primary, duplicates


@app.command("batch-merge")
def batch_merge(
    cluster: List[str] = typer.Argument(..., help="Cluster spec primary:dup1,dup2 (repeatable)."),
    dry_run: bool = typer.Option(False, help="Preview without executing."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Merge clusters of similar candidates in batch."""
    cfg = _load(config)
    store = create_candidate_store(cfg)
    service, _table, manager = _create_curation_service(cfg, table_path)
    batch_service = BatchCurationService(service, cfg.curation)
    try:
        clusters: List[List[EntityCandidate]] = []
        for raw in cluster:
            primary_key, duplicate_keys = _parse_cluster_arg(raw)
            primary_candidate = _ensure_candidate(store, primary_key)
            duplicate_candidates = [_ensure_candidate(store, dup) for dup in duplicate_keys]
            clusters.append([primary_candidate, *duplicate_candidates])

        result = batch_service.batch_merge_clusters(clusters, dry_run=dry_run)
        if result.preview_only:
            console.print(f"[cyan]Preview: {len(clusters)} clusters would be merged.[/cyan]")
        else:
            console.print(f"[green]Merged {len(result.merged_entities)} clusters.[/green]")
        if result.failed:
            console.print(f"[red]Failures: {result.failed}[/red]")
        if result.rolled_back:
            console.print("[red]Rolled back batch due to errors.[/red]")
    finally:
        store.close()
        manager.close()


def _render_normalization_table(
    records: List[NormalizationRecord], *, limit: int, sort_by: str
) -> None:
    sort_key = {
        "confidence": lambda r: (-r.confidence, r.updated_at),
        "updated": lambda r: r.updated_at,
        "name": lambda r: r.canonical_name.lower(),
    }.get(sort_by, lambda r: (-r.confidence, r.updated_at))

    rows = sorted(records, key=sort_key)[:limit]

    table = Table(title=f"Normalization Queue (showing {len(rows)}/{len(records)})")
    table.add_column("Canonical", style="cyan")
    table.add_column("ID", style="magenta")
    table.add_column("Type")
    table.add_column("Method")
    table.add_column("Conf")
    table.add_column("Status")
    table.add_column("Updated")
    table.add_column("Variants")

    for record in rows:
        table.add_row(
            record.canonical_name,
            record.canonical_id,
            record.entity_type or "-",
            record.method.value,
            f"{record.confidence:.2f}",
            record.status,
            record.updated_at.strftime("%Y-%m-%d"),
            ", ".join(record.raw_variants[:3]) + ("..." if len(record.raw_variants) > 3 else ""),
        )

    console.print(table)


def _load_normalization_table(cfg: Config, table_path: Path | None) -> NormalizationTable:
    path = table_path or Path(cfg.normalization.normalization_table_path)
    return NormalizationTable(table_path=path, config=cfg.normalization)


@normalization_app.command("queue")
def normalization_queue(
    min_confidence: float = typer.Option(0.0, help="Minimum confidence filter."),
    entity_type: str = typer.Option("", help="Filter by entity type."),
    status: str = typer.Option("", help="Filter by status (pending/approved/rejected)."),
    method: NormalizationMethod | None = typer.Option(None, help="Filter by normalization method."),
    sort_by: str = typer.Option("confidence", help="Sort by confidence|updated|name."),
    limit: int = typer.Option(20, help="Max rows to display.", min=1),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Display normalization mappings with filters and sorting."""
    cfg = _load(config)
    table = _load_normalization_table(cfg, table_path)
    records = table.all_records()
    if min_confidence > 0:
        records = [r for r in records if r.confidence >= min_confidence]
    if entity_type:
        records = [r for r in records if (r.entity_type or "").lower() == entity_type.lower()]
    if status:
        records = [r for r in records if r.status.lower() == status.lower()]
    if method:
        records = [r for r in records if r.method == method]

    if not records:
        console.print("[yellow]No normalization records found for the given filters.[/yellow]")
        return

    _render_normalization_table(records, limit=limit, sort_by=sort_by)


@normalization_app.command("show")
def normalization_show(
    query: str = typer.Argument(..., help="Raw mention, canonical name, or canonical ID."),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Show details for a specific normalization entry."""
    cfg = _load(config)
    table = _load_normalization_table(cfg, table_path)

    record = table.lookup(query)
    if not record:
        matches = [
            r
            for r in table.all_records()
            if query.lower() in r.canonical_name.lower() or query.lower() in r.canonical_id.lower()
        ]
        record = matches[0] if matches else None

    if not record:
        console.print("[red]No matching normalization record found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{record.canonical_name}[/bold] ({record.canonical_id})")
    console.print(f"Type: {record.entity_type or '-'} | Method: {record.method.value}")
    console.print(f"Confidence: {record.confidence:.2f} | Status: {record.status}")
    console.print(
        f"Updated: {record.updated_at.isoformat()} | Created: {record.created_at.isoformat()}"
    )
    console.print(f"Variants: {', '.join(record.raw_variants)}")
    if record.notes:
        console.print(f"Notes: {record.notes}")
    if record.source:
        console.print(f"Source: {record.source}")


@normalization_app.command("search")
def normalization_search(
    term: str = typer.Argument(..., help="Search term for canonical name/id/variants."),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence filter."),
    limit: int = typer.Option(20, help="Maximum results.", min=1),
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Search normalization mappings by substring."""
    cfg = _load(config)
    table = _load_normalization_table(cfg, table_path)
    records = table.all_records()
    if min_confidence > 0:
        records = [r for r in records if r.confidence >= min_confidence]

    term_lower = term.lower()
    matches = [
        r
        for r in records
        if term_lower in r.canonical_name.lower()
        or term_lower in r.canonical_id.lower()
        or any(term_lower in variant.lower() for variant in r.raw_variants)
    ][:limit]

    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return

    _render_normalization_table(matches, limit=limit, sort_by="confidence")


@normalization_app.command("stats")
def normalization_stats(
    table_path: Path | None = typer.Option(None, help="Override normalization table path."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config file."),
) -> None:
    """Show basic statistics about the normalization table."""
    cfg = _load(config)
    table = _load_normalization_table(cfg, table_path)
    records = table.all_records()

    by_status = Counter(record.status for record in records)
    by_type = Counter((record.entity_type or "UNKNOWN").upper() for record in records)

    console.print("[bold]Normalization Table Stats[/bold]")
    console.print(f"Total records: {len(records)}")
    console.print(
        "Status: "
        + ", ".join(
            f"{status}={count}" for status, count in sorted(by_status.items(), reverse=True)
        )
    )
    console.print(
        "Types: "
        + ", ".join(
            f"{entity_type}={count}" for entity_type, count in sorted(by_type.items(), reverse=True)
        )
    )


def run() -> None:
    """Entrypoint for Typer."""
    app()


if __name__ == "__main__":
    run()
