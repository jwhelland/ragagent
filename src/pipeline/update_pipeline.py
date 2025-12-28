"""Pipeline for detecting document changes and updating the graph."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.schemas import EntityType
from src.utils.config import Config


class FileState(BaseModel):
    """Represents the state of a file in the update process."""

    path: str = Field(..., description="Absolute path to the file")
    checksum: str = Field(..., description="SHA-256 checksum of the file")
    status: str = Field(
        ..., description="Status: 'new', 'modified', 'deleted', 'unchanged', 'renamed'"
    )
    document_id: Optional[str] = Field(None, description="Existing document ID if known")
    last_ingested_at: Optional[str] = Field(None, description="Last ingestion timestamp")
    old_path: Optional[str] = Field(None, description="Previous path if renamed")

    @property
    def filename(self) -> str:
        return Path(self.path).name


class ChangeReport(BaseModel):
    """Report of detected changes."""

    new_files: List[FileState] = []
    modified_files: List[FileState] = []
    deleted_files: List[FileState] = []
    unchanged_files: List[FileState] = []
    renamed_files: List[FileState] = []

    @property
    def has_changes(self) -> bool:
        return bool(
            self.new_files or self.modified_files or self.deleted_files or self.renamed_files
        )

    def summary(self) -> str:
        """Return a human-readable summary of changes."""
        return (
            f"Change Report: {len(self.new_files)} new, "
            f"{len(self.modified_files)} modified, "
            f"{len(self.deleted_files)} deleted, "
            f"{len(self.renamed_files)} renamed, "
            f"{len(self.unchanged_files)} unchanged."
        )


class UpdatePipeline:
    """Pipeline for incremental document updates."""

    def __init__(self, config: Config, neo4j_manager: Optional[Neo4jManager] = None):
        """Initialize the update pipeline."""
        self.config = config
        self.neo4j_manager = neo4j_manager or Neo4jManager(config.database)
        self.qdrant_manager = QdrantManager(config.database)
        self.ingestion_pipeline = IngestionPipeline(config)

    def process_report(self, report: ChangeReport, dry_run: bool = False) -> Dict[str, int]:
        """Process the changes detected in the report.

        Args:
            report: The change report to process.
            dry_run: If True, only log what would be done.

        Returns:
            Dictionary with counts of processed files.
        """
        stats = {
            "new": 0,
            "modified": 0,
            "deleted": 0,
            "renamed": 0,
            "failed": 0,
            "unchanged": len(report.unchanged_files),
        }

        if dry_run:
            logger.info("DRY RUN: No changes will be persisted to the database.")

        # 1. Handle Deleted Files
        for file_state in report.deleted_files:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete document: {file_state.path}")
                stats["deleted"] += 1
                continue

            logger.info(f"Deleting document: {file_state.path}")
            if self._delete_document(file_state.document_id):
                stats["deleted"] += 1
            else:
                stats["failed"] += 1

        # 2. Handle Renamed Files
        for file_state in report.renamed_files:
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would rename document {file_state.document_id} from {file_state.old_path} to {file_state.path}"
                )
                stats["renamed"] += 1
                continue

            logger.info(
                f"Renaming document {file_state.document_id} from {file_state.old_path} to {file_state.path}"
            )
            if self._rename_document(file_state.document_id, file_state.path):
                stats["renamed"] += 1
            else:
                stats["failed"] += 1

        # 3. Handle Modified Files
        for file_state in report.modified_files:
            if dry_run:
                logger.info(f"[DRY RUN] Would update modified document: {file_state.path}")
                stats["modified"] += 1
                continue

            logger.info(f"Updating modified document: {file_state.path}")
            # Re-ingest
            result = self.ingestion_pipeline.process_document(file_state.path, force_reingest=True)
            if result.success:
                stats["modified"] += 1
            else:
                stats["failed"] += 1

        # 4. Handle New Files
        for file_state in report.new_files:
            if dry_run:
                logger.info(f"[DRY RUN] Would ingest new document: {file_state.path}")
                stats["new"] += 1
                continue

            logger.info(f"Ingesting new document: {file_state.path}")
            result = self.ingestion_pipeline.process_document(file_state.path)
            if result.success:
                stats["new"] += 1
            else:
                stats["failed"] += 1

        return stats

    def _rename_document(self, document_id: Optional[str], new_path: str) -> bool:
        """Update the file_path of a document in Neo4j."""
        if not document_id:
            return False

        try:
            self.neo4j_manager.connect()
            # We need to expose a method to update document properties or run a cypher query
            # For now, I'll assume we can use execute_query or similar
            query = "MATCH (d:Document {id: $id}) SET d.file_path = $new_path, d.filename = $filename RETURN d"
            filename = Path(new_path).name
            self.neo4j_manager.query(
                query, {"id": document_id, "new_path": new_path, "filename": filename}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to rename document {document_id}: {e}")
            return False
        finally:
            self.neo4j_manager.close()

    def _delete_document(self, document_id: Optional[str]) -> bool:
        """Delete document and its chunks from all databases."""
        if not document_id:
            return False

        success = True

        # Delete from Qdrant
        try:
            self.qdrant_manager.delete_chunks_by_document(document_id)
        except Exception as e:
            logger.error(f"Failed to delete chunks from Qdrant for {document_id}: {e}")
            success = False

        # Delete from Neo4j
        try:
            self.neo4j_manager.connect()
            self.neo4j_manager.delete_document(document_id)
        except Exception as e:
            logger.error(f"Failed to delete document from Neo4j {document_id}: {e}")
            success = False
        finally:
            self.neo4j_manager.close()

        return success

    def detect_changes(
        self,
        search_paths: List[Path | str] | Path | str,
        extensions: Optional[List[str]] = None,
    ) -> ChangeReport:
        """Detect changes in documents across specified paths.

        Args:
            search_paths: Single path or list of paths to scan (directories or files).
            extensions: List of file extensions to include (default: .pdf, .txt, .md).

        Returns:
            ChangeReport containing classified files.
        """
        if not extensions:
            extensions = [".pdf", ".txt", ".md", ".markdown"]

        # Normalize search paths
        if isinstance(search_paths, (str, Path)):
            search_paths = [search_paths]

        search_paths = [Path(p).resolve() for p in search_paths]

        # 1. Scan local files
        local_files: Dict[str, str] = {}  # path -> checksum

        for path in search_paths:
            if path.is_file():
                if path.suffix.lower() in extensions:
                    local_files[str(path)] = self._compute_checksum(path)
            elif path.is_dir():
                for ext in extensions:
                    for file_path in path.rglob(f"*{ext}"):
                        local_files[str(file_path)] = self._compute_checksum(file_path)

        logger.info(f"Scanned {len(local_files)} local files")

        # 2. Fetch existing documents from DB
        try:
            self.neo4j_manager.connect()
            db_docs = self.neo4j_manager.list_entities(
                entity_type=EntityType.DOCUMENT, limit=100000
            )
        finally:
            self.neo4j_manager.close()

        # Map DB docs
        db_by_path: Dict[str, Dict] = {}
        db_by_checksum: Dict[str, List[Dict]] = (
            {}
        )  # Map checksum to list of docs (duplicates possible)

        for doc in db_docs:
            fpath = doc.get("file_path")
            if fpath:
                db_by_path[fpath] = doc

            checksum = doc.get("checksum")
            if checksum:
                if checksum not in db_by_checksum:
                    db_by_checksum[checksum] = []
                db_by_checksum[checksum].append(doc)

        # 3. Classify files
        report = ChangeReport()
        claimed_doc_ids: Set[str] = set()

        # Check Local Files against DB
        for path, checksum in local_files.items():
            # Match by Exact Path first
            if path in db_by_path:
                doc = db_by_path[path]
                db_checksum = doc.get("checksum")

                doc_id = doc.get("id")
                if doc_id:
                    claimed_doc_ids.add(doc_id)

                state = FileState(
                    path=path,
                    checksum=checksum,
                    status="unchanged",
                    document_id=doc_id,
                    last_ingested_at=doc.get("last_ingested_at"),
                )

                if db_checksum == checksum:
                    report.unchanged_files.append(state)
                else:
                    state.status = "modified"
                    report.modified_files.append(state)
                continue

            # Match by Checksum (Rename Detection)
            potential_matches = db_by_checksum.get(checksum, [])

            # Find a match that hasn't been claimed yet
            found_match = None
            for doc in potential_matches:
                if doc.get("id") not in claimed_doc_ids:
                    found_match = doc
                    break

            if found_match:
                # Same content, different path -> RENAME
                doc_id = found_match.get("id")
                claimed_doc_ids.add(doc_id)

                state = FileState(
                    path=path,
                    checksum=checksum,
                    status="renamed",
                    document_id=doc_id,
                    last_ingested_at=found_match.get("last_ingested_at"),
                    old_path=found_match.get("file_path"),
                )
                report.renamed_files.append(state)
            else:
                # No path match, no checksum match (or all checksum matches taken) -> NEW
                report.new_files.append(FileState(path=path, checksum=checksum, status="new"))

        # Check for Deleted Files
        # A file is deleted if it's in DB but not in the scanned local_files
        # AND it belongs to one of the search_paths
        # AND it hasn't been claimed (renamed)

        for doc in db_docs:
            doc_id = doc.get("id")
            if doc_id in claimed_doc_ids:
                continue

            db_path = doc.get("file_path")
            if not db_path:
                continue

            # Check if this DB doc falls within our search scope
            in_scope = False
            for search_path in search_paths:
                try:
                    # Check if db_path is relative to search_path
                    Path(db_path).relative_to(search_path)
                    in_scope = True
                    break
                except ValueError:
                    continue

            if in_scope:
                # If we are here, it means the file at db_path is gone
                # AND we didn't find its content elsewhere (rename).
                report.deleted_files.append(
                    FileState(
                        path=db_path,
                        checksum=doc.get("checksum", ""),
                        status="deleted",
                        document_id=doc.get("id"),
                        last_ingested_at=doc.get("last_ingested_at"),
                    )
                )

        return report

    def _compute_checksum(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
