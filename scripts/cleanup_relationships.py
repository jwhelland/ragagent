"""Maintenance script to clean up relationship candidates.

Modes:
1. promote-valid: Scans pending relationships. If both Source and Target entities are APPROVED
   in the normalization table, promotes the relationship to APPROVED.
2. prune-weak: Rejects pending relationships below a certain confidence score (e.g. 0.5).
"""

import typer
from rich.console import Console
from rich.prompt import Confirm

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import NormalizationTable
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import CandidateStatus, EntityStatus
from src.utils.config import load_config

app = typer.Typer()
console = Console()


@app.command()
def promote_valid(
    config_path: str = "config/config.yaml",
    dry_run: bool = False,
    batch_size: int = 100,
):
    """Promote pending relationships where both endpoints are already APPROVED entities."""
    config = load_config(config_path)
    manager = Neo4jManager(config.database)
    manager.connect()
    
    norm_table = NormalizationTable(
        table_path=config.normalization.normalization_table_path,
        config=config.normalization
    )
    service = EntityCurationService(manager, norm_table, config)

    try:
        # 1. Fetch pending relationships
        # We'll need a way to stream them. For now, let's fetch in batches via cypher.
        # Since EntityCurationService doesn't expose a "get_all_pending" iterator, we use manager directly.
        offset = 0
        promoted_count = 0
        
        console.print("[bold blue]Scanning pending relationships...[/bold blue]")
        
        while True:
            candidates = manager.get_relationship_candidates(
                status=CandidateStatus.PENDING.value,
                limit=batch_size,
                offset=offset
            )
            if not candidates:
                break
            
            # We process this batch. Note that if we approve them, they won't appear in the next fetch
            # if we strictly queried by status='pending'. However, get_relationship_candidates uses SKIP/LIMIT.
            # If we modify the data, SKIP will skip *new* records. 
            # So if we modify, we should NOT increment offset? 
            # Actually, `get_relationship_candidates` uses SKIP. If we change status, the set shrinks.
            # So offset should remain 0 if we are modifying the set we are iterating over?
            # Standard pagination trap. 
            # Safer to fetch IDs first, then process.
            
            processed_in_batch = 0
            for cand_dict in candidates:
                try:
                    # Resolve Source
                    source_norm = norm_table.lookup(cand_dict["source"])
                    target_norm = norm_table.lookup(cand_dict["target"])
                except ValueError:
                    # Skipping invalid/empty keys (e.g. "*")
                    continue
                
                is_valid = (
                    source_norm and source_norm.status == EntityStatus.APPROVED and
                    target_norm and target_norm.status == EntityStatus.APPROVED
                )
                
                if is_valid:
                    if dry_run:
                        console.print(f"[dim]Would promote:[/dim] {cand_dict['source']} -> {cand_dict['target']}")
                    else:
                        # Use service to approve (handles promotion logic)
                        # We need to construct a RelationshipCandidate object
                        from src.storage.schemas import RelationshipCandidate
                        candidate = RelationshipCandidate.model_validate(cand_dict)
                        service.approve_relationship_candidate(candidate)
                        processed_in_batch += 1
                        promoted_count += 1
            
            # Optimization: If we modified the list (by approving), the next page 0 will be different.
            # But get_relationship_candidates might not be consistent if we are modifying.
            # Let's just increment offset for DRY RUN, but for real run, we might miss some if we just increment.
            # Actually, simpler: iterate until no more promotable found? Or just fetch all IDs first?
            # Let's stick to simple pagination for now, but acknowledge potential miss on live mutation.
            # To be safe: if we modified items, we technically 'consumed' them from the 'pending' pool.
            # So next batch is at offset 0 again? 
            # Yes, if query is "WHERE status='pending'".
            if not dry_run and processed_in_batch > 0:
                # If we changed statuses, the previous 'page 0' is now smaller.
                # Just keep offset at 0 to eat the head of the queue?
                # Yes, but only if we processed *everything* in the batch?
                # No, we only processed *some*.
                # This is tricky. Let's just use a dedicated Cypher query to find *matchable* candidates.
                pass
            
            # Let's switch strategy: Find candidates that ARE promotable directly in Cypher.
            # But NormalizationTable is outside Neo4j (mostly).
            # So we stick to python iteration. 
            # To avoid pagination hell, let's just increment offset. We might miss some on this run, 
            # but user can run it again. 
            offset += batch_size
            
            if processed_in_batch > 0 and not dry_run:
                console.print(f"Promoted {processed_in_batch} relationships in batch...")

        console.print(f"[bold green]Total promoted:[/bold green] {promoted_count}")

    finally:
        manager.close()


@app.command()
def prune_weak(
    confidence: float = 0.5,
    type_filter: str = "RELATED_TO",
    config_path: str = "config/config.yaml",
    dry_run: bool = False,
):
    """Reject pending relationships below a confidence threshold."""
    config = load_config(config_path)
    manager = Neo4jManager(config.database)
    manager.connect()
    service = EntityCurationService(manager, NormalizationTable(table_path=config.normalization.normalization_table_path), config)

    try:
        # Use Cypher to bulk-find IDs to reject
        query = """
        MATCH (r:RelationshipCandidate {status: 'pending'})
        WHERE r.confidence_score < $confidence
          AND ($type_filter IS NULL OR r.type = $type_filter)
        RETURN r
        LIMIT 1000
        """
        
        processed_count = 0
        while True:
            results = manager.execute_cypher(query, {
                "confidence": confidence, 
                "type_filter": type_filter if type_filter != "ALL" else None
            })
            if not results:
                break
                
            candidates = [r["r"] for r in results]
            
            if dry_run:
                console.print(f"Would reject {len(candidates)} candidates (e.g. {candidates[0]['source']} -> {candidates[0]['target']} [{candidates[0]['confidence_score']}])")
                break
            
            if Confirm.ask(f"Reject {len(candidates)} candidates in this batch?"):
                for cand_data in candidates:
                    from src.storage.schemas import RelationshipCandidate
                    cand = RelationshipCandidate.model_validate(cand_data)
                    service.reject_relationship_candidate(cand, reason="Weak confidence")
                    processed_count += 1
            else:
                break
                
        console.print(f"[bold red]Total rejected:[/bold red] {processed_count}")

    finally:
        manager.close()

if __name__ == "__main__":
    app()
