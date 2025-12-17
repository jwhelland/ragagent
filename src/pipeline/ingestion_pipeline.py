"""End-to-end document ingestion pipeline.

This module orchestrates the complete document processing workflow:
1. PDF parsing with structure extraction
2. Text cleaning and preprocessing
3. Hierarchical chunking
4. Embedding generation
5. Storage in Neo4j (graph) and Qdrant (vectors)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

import re
from datetime import datetime

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.text_rewriter import TextRewriter
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.schemas import Chunk as GraphChunk
from src.storage.schemas import Document, EntityType
from src.utils.config import Config
from src.utils.embeddings import EmbeddingGenerator


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    success: bool
    chunks_created: int = 0
    entities_created: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Orchestrates the complete document processing workflow from PDF to
    stored chunks and embeddings in both graph and vector databases.

    Example:
        >>> pipeline = IngestionPipeline(config)
        >>> result = pipeline.process_document("document.pdf")
        >>> print(f"Processed {result.chunks_created} chunks")
    """

    def __init__(self, config: Config) -> None:
        """Initialize the ingestion pipeline.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize components
        self.pdf_parser = None
        self.text_cleaner = None
        self.text_rewriter = None
        self.chunker = None
        self.embeddings = None
        self.neo4j_manager = None
        self.qdrant_manager = None

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_created": 0,
            "total_processing_time": 0.0,
        }

        logger.info("IngestionPipeline initialized")

    def initialize_components(self) -> None:
        """Initialize all pipeline components.

        This is called lazily when first needed to avoid startup overhead.
        """
        if self.pdf_parser is None:
            from src.ingestion.pdf_parser import PDFParser

            self.pdf_parser = PDFParser(self.config.ingestion.pdf_parser)

        if self.text_cleaner is None:
            self.text_cleaner = TextCleaner(self.config.ingestion.text_cleaning)

        if self.text_rewriter is None and self.config.ingestion.text_rewriting.enabled:
            self.text_rewriter = TextRewriter(self.config.ingestion.text_rewriting)

        if self.chunker is None:
            self.chunker = HierarchicalChunker(self.config.ingestion.chunking)

        if self.embeddings is None:
            self.embeddings = EmbeddingGenerator(self.config.database)

        if self.neo4j_manager is None:
            self.neo4j_manager = Neo4jManager(self.config.database)
            self.neo4j_manager.connect()

        if self.qdrant_manager is None:
            self.qdrant_manager = QdrantManager(self.config.database)

        logger.debug("All pipeline components initialized")

    def process_document(self, pdf_path: Path | str) -> IngestionResult:
        """Process a single PDF document end-to-end.

        Implements basic resume/rollback semantics:
        - Resume/skip if the document exists in Neo4j with matching checksum and status=completed.
        - If a prior run was interrupted (status != completed) or checksum differs, we clean up
          existing chunks in both DBs and re-ingest.
        - On failures during storage, we roll back partial chunk writes and mark the document failed.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        logger.info(f"Processing document: {pdf_path.name}")

        parsed_doc: ParsedDocument | None = None

        try:
            # Initialize components if needed
            self.initialize_components()

            # Step 1: Parse PDF
            logger.debug("Step 1: Parsing PDF")
            parsed_doc = self.pdf_parser.parse_pdf(pdf_path)

            if parsed_doc.error:
                raise Exception(f"PDF parsing failed: {parsed_doc.error}")

            # Resume / cleanup logic (based on deterministic document_id + checksum)
            if self.config.pipeline.enable_checkpointing:
                existing = self.neo4j_manager.get_entity(
                    parsed_doc.document_id, EntityType.DOCUMENT
                )
                existing_checksum = (existing or {}).get("checksum")
                existing_status = (existing or {}).get("ingestion_status")

                if (
                    existing
                    and existing_checksum
                    and existing_checksum == parsed_doc.metadata.get("checksum")
                    and existing_status == "completed"
                ):
                    # Already ingested successfully; skip.
                    try:
                        existing_chunks = self.neo4j_manager.get_chunks_by_document(
                            parsed_doc.document_id
                        )
                        chunks_created = len(existing_chunks)
                    except Exception:
                        chunks_created = 0

                    processing_time = time.time() - start_time
                    logger.info(
                        f"Skipping already-ingested document {parsed_doc.document_id} (checksum match, status=completed)"
                    )
                    return IngestionResult(
                        document_id=parsed_doc.document_id,
                        success=True,
                        chunks_created=chunks_created,
                        entities_created=0,
                        processing_time=processing_time,
                    )

                # If doc exists but isn't completed or checksum changed, clean up partial/old chunks.
                if existing:
                    logger.info(
                        f"Re-ingesting document {parsed_doc.document_id} (status={existing_status!r}, checksum_changed={existing_checksum != parsed_doc.metadata.get('checksum')})"
                    )
                    self._cleanup_document_chunks(parsed_doc.document_id)

            # Mark document as ingesting (status tracking)
            self._upsert_document_status(parsed_doc, status="ingesting")

            # Step 2: Clean text
            logger.debug("Step 2: Cleaning text")
            if self.config.ingestion.text_cleaning.enabled:
                parsed_doc.raw_text = self.text_cleaner.clean(parsed_doc.raw_text)

            # Step 2.5: Optional rewriting (disabled by default)
            if self.config.ingestion.text_rewriting.enabled:
                logger.debug("Step 2.5: Rewriting text (optional)")
                self._rewrite_parsed_document(parsed_doc)

            # Step 3: Create chunks
            logger.debug("Step 3: Creating chunks")
            chunks = self.chunker.chunk_document(parsed_doc)

            if not chunks:
                raise Exception("No chunks created from document")

            # Step 4: Generate embeddings
            logger.debug("Step 4: Generating embeddings")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.generate(chunk_texts)

            if len(embeddings) != len(chunks):
                raise Exception(
                    f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
                )

            # Step 5: Store in databases
            logger.debug("Step 5: Storing in databases")
            self._store_document_and_chunks(parsed_doc, chunks, embeddings)

            # Mark document as completed
            self._upsert_document_status(parsed_doc, status="completed")

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["total_processing_time"] += processing_time

            logger.success(
                f"Document processed successfully: {len(chunks)} chunks, {processing_time:.2f}s"
            )

            return IngestionResult(
                document_id=parsed_doc.document_id,
                success=True,
                chunks_created=len(chunks),
                entities_created=0,  # Will be updated when entity extraction is added
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed: {e}")

            # Best-effort rollback of partial chunk writes
            if parsed_doc is not None:
                try:
                    self._cleanup_document_chunks(parsed_doc.document_id)
                except Exception as rollback_err:
                    logger.warning(f"Rollback cleanup failed: {rollback_err}")

                try:
                    self._upsert_document_status(parsed_doc, status="failed", error=str(e))
                except Exception as status_err:
                    logger.warning(f"Failed to update document status to failed: {status_err}")

                doc_id_for_result = parsed_doc.document_id
            else:
                doc_id_for_result = str(pdf_path)

            return IngestionResult(
                document_id=doc_id_for_result,
                success=False,
                chunks_created=0,
                entities_created=0,
                processing_time=processing_time,
                error=str(e),
            )

    def process_batch(self, pdf_paths: List[Path | str]) -> List[IngestionResult]:
        """Process multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of IngestionResult objects
        """
        logger.info(f"Processing batch of {len(pdf_paths)} documents")

        results = []
        for pdf_path in pdf_paths:
            result = self.process_document(pdf_path)
            results.append(result)

            # Log progress
            successful = sum(1 for r in results if r.success)
            logger.info(
                f"Progress: {len(results)}/{len(pdf_paths)} processed, {successful} successful"
            )

        # Summary
        total_chunks = sum(r.chunks_created for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        successful_count = sum(1 for r in results if r.success)

        logger.info(
            f"Batch processing complete: {successful_count}/{len(pdf_paths)} successful, "
            f"{total_chunks} chunks created, {total_time:.2f}s total"
        )

        return results

    def _rewrite_parsed_document(self, parsed_doc: ParsedDocument) -> None:
        """Rewrite parsed document content based on config chunk_level.

        - section: rewrite each section.content (and raw_text)
        - subsection: rewrite each subsection.content (and raw_text)
        """
        if not self.text_rewriter:
            # Enabled in config, but not initialized (shouldn't happen), fail safe.
            self.text_rewriter = TextRewriter(self.config.ingestion.text_rewriting)

        rewriting_cfg = self.config.ingestion.text_rewriting
        chunk_level = rewriting_cfg.chunk_level

        # Preserve original for audit if requested
        if rewriting_cfg.preserve_original:
            parsed_doc.metadata.setdefault("rewriting", {})
            parsed_doc.metadata["rewriting"]["original_raw_text"] = parsed_doc.raw_text

        rewritten_count = 0

        if chunk_level == "section":
            original_sections: Dict[str, str] = {}
            for section in parsed_doc.structure.get("sections", []):
                key = getattr(section, "hierarchy_path", "") or getattr(section, "title", "")
                if rewriting_cfg.preserve_original:
                    original_sections[key] = section.content

                result = self.text_rewriter.rewrite(
                    section.content,
                    metadata={
                        "section_title": getattr(section, "title", ""),
                        "hierarchy_path": getattr(section, "hierarchy_path", ""),
                    },
                )
                if result.used_rewrite:
                    section.content = result.rewritten
                    rewritten_count += 1

                for subsection in getattr(section, "subsections", []) or []:
                    # If chunk_level is section, still keep subsections as-is (they are downstream).
                    pass

            if rewriting_cfg.preserve_original:
                parsed_doc.metadata.setdefault("rewriting", {})
                parsed_doc.metadata["rewriting"]["original_sections"] = original_sections

        elif chunk_level == "subsection":
            original_subsections: Dict[str, str] = {}
            for section in parsed_doc.structure.get("sections", []):
                for subsection in getattr(section, "subsections", []) or []:
                    key = getattr(subsection, "hierarchy_path", "") or getattr(
                        subsection, "title", ""
                    )
                    if rewriting_cfg.preserve_original:
                        original_subsections[key] = subsection.content

                    result = self.text_rewriter.rewrite(
                        subsection.content,
                        metadata={
                            "subsection_title": getattr(subsection, "title", ""),
                            "hierarchy_path": getattr(subsection, "hierarchy_path", ""),
                        },
                    )
                    if result.used_rewrite:
                        subsection.content = result.rewritten
                        rewritten_count += 1
        else:
            logger.warning(f"Unknown rewriting chunk_level: {chunk_level}. Skipping rewriting.")
            return

        # Always rewrite doc-level raw_text too (so L1 chunk matches improved text),
        # but preserve original above when requested.
        doc_result = self.text_rewriter.rewrite(
            parsed_doc.raw_text,
            metadata={
                "document_title": parsed_doc.metadata.get("title", ""),
                "filename": parsed_doc.metadata.get("filename", ""),
            },
        )
        if doc_result.used_rewrite:
            parsed_doc.raw_text = doc_result.rewritten
            rewritten_count += 1

        parsed_doc.metadata.setdefault("rewriting", {})
        parsed_doc.metadata["rewriting"].update(
            {
                "enabled": True,
                "chunk_level": chunk_level,
                "preserve_original": rewriting_cfg.preserve_original,
                "rewritten_units": rewritten_count,
            }
        )

    def _store_document_and_chunks(
        self,
        parsed_doc: ParsedDocument,
        chunks: List[Any],  # HierarchicalChunker.Chunk
        embeddings: List[Any],  # numpy arrays
    ) -> None:
        """Store document metadata and chunks in databases.

        Args:
            parsed_doc: Parsed document
            chunks: List of chunk objects
            embeddings: List of embedding vectors
        """
        # Upsert document entity (must include canonical_name for base Entity model).
        filename = parsed_doc.metadata.get("filename", "")
        title = parsed_doc.metadata.get("title") or filename or "document"
        canonical_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip()).strip("_").lower() or "document"
        )

        document = Document(
            id=parsed_doc.document_id,  # deterministic ID (checksum-based)
            canonical_name=canonical_name,
            filename=filename,
            title=parsed_doc.metadata.get("title"),
            version=parsed_doc.metadata.get("version"),
            date=parsed_doc.metadata.get("date"),
            author=parsed_doc.metadata.get("author"),
            page_count=parsed_doc.page_count,
            checksum=parsed_doc.metadata.get("checksum"),
            properties={
                "ingestion_status": "ingesting",
                "last_ingested_at": datetime.now().isoformat(),
            },
        )

        # Idempotent document upsert (supports resume/retries)
        if hasattr(self.neo4j_manager, "upsert_entity"):
            doc_id = self.neo4j_manager.upsert_entity(document)
        else:
            # Back-compat for tests/fakes that only implement create_entity()
            doc_id = self.neo4j_manager.create_entity(document)

        # Prepare chunk data for Qdrant
        chunk_payloads = []
        chunk_vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            # Ensure metadata has entity_ids where QdrantManager expects it
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("entity_ids", metadata.get("entity_ids", []))

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "level": chunk.level,
                "content": chunk.content,
                "metadata": metadata,
                "timestamp": metadata.get("created_at", ""),
            }

            chunk_payloads.append(payload)
            chunk_vectors.append(embedding.tolist())  # Convert numpy array to list

        # Store chunks in Qdrant
        self.qdrant_manager.upsert_chunks(chunk_payloads, chunk_vectors)

        # Store chunks in Neo4j (idempotent per chunk_id; safe for retries)
        for chunk in chunks:
            graph_chunk = GraphChunk(
                id=chunk.chunk_id,
                document_id=chunk.document_id,
                level=chunk.level,
                parent_chunk_id=chunk.parent_chunk_id,
                child_chunk_ids=chunk.child_chunk_ids,
                content=chunk.content,
                section_title=chunk.metadata.get("section_title")
                or chunk.metadata.get("subsection_title"),
                page_numbers=chunk.metadata.get("page_numbers", []),
                hierarchy_path=chunk.metadata.get("hierarchy_path"),
                token_count=chunk.token_count,
                entity_ids=chunk.metadata.get("entity_ids", []),
                has_tables=chunk.metadata.get("has_tables", False),
                has_figures=chunk.metadata.get("has_figures", False),
                created_at=datetime.now(),
            )
            if hasattr(self.neo4j_manager, "upsert_chunk"):
                self.neo4j_manager.upsert_chunk(graph_chunk)
            else:
                # Back-compat for tests/fakes that only implement create_chunk()
                self.neo4j_manager.create_chunk(graph_chunk)

        logger.debug(f"Stored document {doc_id} with {len(chunks)} chunks")

    def _cleanup_document_chunks(self, document_id: str) -> None:
        """Best-effort cleanup of all chunks for a document in both databases.

        Used for:
        - resuming after interrupted ingestion
        - rollback after failures
        """
        # Qdrant cleanup
        try:
            if self.qdrant_manager:
                self.qdrant_manager.delete_chunks_by_document(document_id)
        except Exception as e:
            logger.warning(f"Qdrant cleanup failed for document {document_id}: {e}")

        # Neo4j cleanup
        try:
            if self.neo4j_manager:
                self.neo4j_manager.delete_chunks_by_document(document_id)
        except Exception as e:
            logger.warning(f"Neo4j chunk cleanup failed for document {document_id}: {e}")

    def _upsert_document_status(
        self, parsed_doc: ParsedDocument, status: str, error: str | None = None
    ) -> None:
        """Upsert the Document node with ingestion status fields."""
        filename = parsed_doc.metadata.get("filename", "")
        title = parsed_doc.metadata.get("title") or filename or "document"
        canonical_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip()).strip("_").lower() or "document"
        )

        props = {
            "ingestion_status": status,
            "last_ingested_at": datetime.now().isoformat(),
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()

    def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components.

        Returns:
            Dictionary with component health status
        """
        health = {}

        try:
            health["neo4j"] = self.neo4j_manager.health_check() if self.neo4j_manager else False
        except:
            health["neo4j"] = False

        try:
            qdrant_health, _ = (
                self.qdrant_manager.health_check() if self.qdrant_manager else (False, "")
            )
            health["qdrant"] = qdrant_health
        except:
            health["qdrant"] = False

        # Other components don't have health checks
        health["pdf_parser"] = self.pdf_parser is not None
        health["text_cleaner"] = self.text_cleaner is not None
        health["chunker"] = self.chunker is not None
        health["embeddings"] = self.embeddings is not None

        return health

    def close(self) -> None:
        """Clean up pipeline resources."""
        if self.neo4j_manager:
            self.neo4j_manager.close()
        if self.qdrant_manager:
            self.qdrant_manager.close()
        if self.embeddings:
            self.embeddings.clear_cache()

        logger.info("IngestionPipeline closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
