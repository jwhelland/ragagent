import re
from datetime import datetime

from loguru import logger

from src.pipeline.base import PipelineContext, PipelineStage
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.schemas import Document, EntityType


class CheckpointStage(PipelineStage):
    """Stage for handling resume/rollback and status tracking."""

    def __init__(self, config, neo4j_manager: Neo4jManager, qdrant_manager: QdrantManager):
        super().__init__("Checkpoint")
        self.config = config
        self.neo4j_manager = neo4j_manager
        self.qdrant_manager = qdrant_manager

    def run(self, context: PipelineContext) -> PipelineContext:
        parsed_doc = context.parsed_document
        if not parsed_doc:
            raise ValueError("Parsed document required for checkpointing")

        if self.config.pipeline.enable_checkpointing:
            existing = self.neo4j_manager.get_entity(parsed_doc.document_id, EntityType.DOCUMENT)
            existing_checksum = (existing or {}).get("checksum")
            existing_status = (existing or {}).get("ingestion_status")

            if (
                existing
                and existing_checksum
                and existing_checksum == parsed_doc.metadata.get("checksum")
                and existing_status == "completed"
                and not context.force_reingest
            ):
                # Already ingested successfully; skip.
                try:
                    existing_chunks = self.neo4j_manager.get_chunks_by_document(
                        parsed_doc.document_id
                    )
                    chunks_created = len(existing_chunks)
                except Exception:
                    chunks_created = 0

                context.update_stats("chunks_created", chunks_created, "set")
                context.skipped = True
                context.skip_reason = (
                    f"Document {parsed_doc.document_id} already ingested (checksum match)"
                )
                return context

            # If doc exists but isn't completed or checksum changed, clean up.
            if existing:
                logger.info(
                    "Re-ingesting document {} (status={!r}, checksum_changed={}, force={})",
                    parsed_doc.document_id,
                    existing_status,
                    existing_checksum != parsed_doc.metadata.get("checksum"),
                    context.force_reingest,
                )
                self._cleanup_document_chunks(parsed_doc.document_id)

            # Check for outdated versions of the same file (same path, different ID)
            # This handles file updates where content change = new ID
            file_path = parsed_doc.metadata.get("file_path")
            if file_path:
                try:
                    # Find other documents with this file path
                    query = """
                    MATCH (d:DOCUMENT)
                    WHERE d.file_path = $file_path AND d.id <> $current_id
                    RETURN d.id as id
                    """
                    results = self.neo4j_manager.execute_cypher(
                        query, {"file_path": file_path, "current_id": parsed_doc.document_id}
                    )

                    for record in results:
                        old_doc_id = record["id"]
                        logger.info(
                            f"Detected outdated document version {old_doc_id} for path {file_path}. Cleaning up."
                        )
                        # Delete the old document entirely
                        self.neo4j_manager.delete_document(old_doc_id)
                        # Also clean up Qdrant chunks for the old ID
                        try:
                            self.qdrant_manager.delete_chunks_by_document(old_doc_id)
                        except Exception as e:
                            logger.warning(f"Qdrant cleanup failed for old document {old_doc_id}: {e}")

                except Exception as e:
                    logger.warning(f"Failed to check for outdated document versions: {e}")

        # Mark document as ingesting
        self._upsert_document_status(parsed_doc, status="ingesting")
        return context

    def _cleanup_document_chunks(self, document_id: str) -> None:
        try:
            self.qdrant_manager.delete_chunks_by_document(document_id)
        except Exception as e:
            logger.warning(f"Qdrant cleanup failed for document {document_id}: {e}")

        try:
            self.neo4j_manager.delete_chunks_by_document(document_id)
        except Exception as e:
            logger.warning(f"Neo4j chunk cleanup failed for document {document_id}: {e}")

    def _upsert_document_status(self, parsed_doc, status: str, error: str | None = None) -> None:
        filename = parsed_doc.metadata.get("filename", "")
        title = parsed_doc.metadata.get("title") or filename or "document"
        canonical_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip()).strip("_").lower() or "document"
        )

        props = {
            "ingestion_status": status,
            "last_ingested_at": datetime.now().isoformat(),
            "file_path": parsed_doc.metadata.get("file_path"),
        }
        if error:
            props["ingestion_error"] = error

        document = Document(
            id=parsed_doc.document_id,
            canonical_name=canonical_name,
            filename=filename,
            title=parsed_doc.metadata.get("title"),
            version=parsed_doc.metadata.get("version"),
            date=parsed_doc.metadata.get("date"),
            author=parsed_doc.metadata.get("author"),
            page_count=parsed_doc.page_count,
            checksum=parsed_doc.metadata.get("checksum"),
            properties=props,
        )

        if hasattr(self.neo4j_manager, "upsert_entity"):
            self.neo4j_manager.upsert_entity(document)
        else:
            self.neo4j_manager.create_entity(document)
