"""End-to-end document ingestion pipeline.

This module orchestrates the complete document processing workflow:
1. Document parsing (PDF via Docling; text/markdown via lightweight parser)
2. Text cleaning and preprocessing
3. Hierarchical chunking
4. Embedding generation
5. Storage in Neo4j (graph) and Qdrant (vectors)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.pipeline.base import Pipeline, PipelineContext
from src.pipeline.stages.checkpoint import CheckpointStage
from src.pipeline.stages.chunking import ChunkingStage
from src.pipeline.stages.cleaning import CleaningStage, RewritingStage
from src.pipeline.stages.embedding import EmbeddingStage
from src.pipeline.stages.extraction import ExtractionStage
from src.pipeline.stages.parsing import ParsingStage
from src.pipeline.stages.storage import StorageStage
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import Config


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    success: bool
    chunks_created: int = 0
    entities_created: int = 0
    relationships_filtered: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Orchestrates the complete document processing workflow from PDF to
    stored chunks and embeddings in both graph and vector databases.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the ingestion pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self._debug_logging = str(getattr(config.logging, "level", "INFO")).upper() == "DEBUG"

        # Managers
        self.neo4j_manager: Neo4jManager | None = None
        self.qdrant_manager: QdrantManager | None = None

        # The Pipeline
        self.pipeline: Pipeline | None = None

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_created": 0,
            "llm_entities_extracted": 0,
            "llm_relationships_extracted": 0,
            "rule_based_relationships_extracted": 0,
            "merged_entities_created": 0,
            "entity_candidates_stored": 0,
            "relationship_candidates_stored": 0,
            "acronym_definitions_added": 0,
            "dedup_merge_suggestions": 0,
            "total_processing_time": 0.0,
            "relationships_filtered": 0,
        }

        logger.info("IngestionPipeline initialized")

    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        if self.pipeline is not None:
            return

        self._silence_external_http_logs()

        # Init managers
        if self.neo4j_manager is None:
            self.neo4j_manager = Neo4jManager(self.config.database)
            self.neo4j_manager.connect()

        if self.qdrant_manager is None:
            self.qdrant_manager = QdrantManager(self.config.database)

        # Build pipeline stages
        stages = [
            ParsingStage(self.config),
            CheckpointStage(self.config, self.neo4j_manager, self.qdrant_manager),
            CleaningStage(self.config),
            RewritingStage(self.config),
            ChunkingStage(self.config),
            ExtractionStage(self.config),
            EmbeddingStage(self.config),
            StorageStage(self.config, self.neo4j_manager, self.qdrant_manager),
        ]

        self.pipeline = Pipeline(stages)
        logger.debug("Pipeline components initialized")

    def process_document(
        self,
        pdf_path: Path | str,
        *,
        force_reingest: bool = False,
        topics: List[str] | None = None,
    ) -> IngestionResult:
        """Process a single document end-to-end.

        Args:
            pdf_path: Path to the document file
            force_reingest: If True, ignore checkpoint skip and reprocess
            topics: Optional list of topics

        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        logger.info(f"Processing document: {pdf_path.name}")

        try:
            self.initialize_components()
            assert self.pipeline is not None

            context = PipelineContext(
                file_path=pdf_path, config=self.config, force_reingest=force_reingest, topics=topics
            )

            # Run pipeline
            context = self.pipeline.run(context)

            # Update global stats
            for key, value in context.stats.items():
                if key in self.stats:
                    self.stats[key] += value
                else:
                    self.stats[key] = value  # Or init it

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["documents_processed"] += 1

            if context.skipped:
                return IngestionResult(
                    document_id=(
                        context.parsed_document.document_id
                        if context.parsed_document
                        else "unknown"
                    ),
                    success=True,
                    chunks_created=context.stats.get("chunks_created", 0),
                    processing_time=processing_time,
                )

            logger.success(
                f"Document processed successfully: {len(context.chunks)} chunks, {processing_time:.2f}s"
            )

            return IngestionResult(
                document_id=(
                    context.parsed_document.document_id if context.parsed_document else "unknown"
                ),
                success=True,
                chunks_created=len(context.chunks),
                entities_created=context.stats.get("entities_created", 0),
                relationships_filtered=context.stats.get("relationships_filtered", 0),
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed: {e}")

            # Error handling logic - Rollback is partially handled in Checkpoint/Storage stages via cleanup on start
            # But immediate rollback on error was in old pipeline.
            # "Best-effort rollback of partial chunk writes"

            doc_id = str(pdf_path)
            # Try to get doc_id from context if available
            if "context" in locals() and context.parsed_document:
                doc_id = context.parsed_document.document_id
                # Attempt cleanup
                try:
                    # reusing CheckpointStage logic or manually calling managers
                    if self.qdrant_manager:
                        self.qdrant_manager.delete_chunks_by_document(doc_id)
                    if self.neo4j_manager:
                        self.neo4j_manager.delete_chunks_by_document(doc_id)
                        self._upsert_failed_status(context.parsed_document, str(e))
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup failed: {cleanup_err}")

            return IngestionResult(
                document_id=doc_id,
                success=False,
                processing_time=processing_time,
                error=str(e),
            )

    def _upsert_failed_status(self, parsed_doc, error_msg):
        # Helper to mark failed status
        import re
        from datetime import datetime

        from src.storage.schemas import Document

        filename = parsed_doc.metadata.get("filename", "")
        title = parsed_doc.metadata.get("title") or filename or "document"
        canonical_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip()).strip("_").lower() or "document"
        )
        doc = Document(
            id=parsed_doc.document_id,
            canonical_name=canonical_name,
            filename=filename,
            title=parsed_doc.metadata.get("title"),
            version=parsed_doc.metadata.get("version"),
            date=parsed_doc.metadata.get("date"),
            author=parsed_doc.metadata.get("author"),
            page_count=parsed_doc.page_count,
            checksum=parsed_doc.metadata.get("checksum"),
            properties={
                "ingestion_status": "failed",
                "ingestion_error": error_msg,
                "last_ingested_at": datetime.now().isoformat(),
                "file_path": parsed_doc.metadata.get("file_path"),
            },
        )
        if self.neo4j_manager:
            if hasattr(self.neo4j_manager, "upsert_entity"):
                self.neo4j_manager.upsert_entity(doc)
            else:
                self.neo4j_manager.create_entity(doc)

    def process_batch(
        self,
        pdf_paths: List[Path | str],
        *,
        force_reingest: bool = False,
        topics: List[str] | None = None,
    ) -> List[IngestionResult]:
        """Process multiple PDF documents."""
        logger.info(f"Processing batch of {len(pdf_paths)} documents")

        results = []
        for pdf_path in pdf_paths:
            result = self.process_document(pdf_path, force_reingest=force_reingest, topics=topics)
            results.append(result)

            successful = sum(1 for r in results if r.success)
            logger.info(
                f"Progress: {len(results)}/{len(pdf_paths)} processed, {successful} successful"
            )

        total_chunks = sum(r.chunks_created for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        successful_count = sum(1 for r in results if r.success)

        logger.info(
            f"Batch processing complete: {successful_count}/{len(pdf_paths)} successful, "
            f"{total_chunks} chunks created, {total_time:.2f}s total"
        )
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return self.stats.copy()

    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health = {}
        try:
            if not self.neo4j_manager:
                self.neo4j_manager = Neo4jManager(self.config.database)
            health["neo4j"] = self.neo4j_manager.health_check()
        except Exception:
            health["neo4j"] = False
        return health

    def close(self) -> None:
        """Close database connections."""
        if self.neo4j_manager:
            self.neo4j_manager.close()
        # QdrantManager usually doesn't strictly require close, but good practice if method exists
        # or if we need to release http client resources.
        # Checking QdrantManager... usually it's just neo4j that needs explicit close.

    def _silence_external_http_logs(self) -> None:
        if self._debug_logging:
            return
        noisy_loggers = (
            "httpx",
            "httpcore",
            "openai",
            "openai._base_client",
            "openai._http_client",
            "docling",
            "docling.document_converter",
        )
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.ERROR)