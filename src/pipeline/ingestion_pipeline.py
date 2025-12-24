"""End-to-end document ingestion pipeline.

This module orchestrates the complete document processing workflow:
1. Document parsing (PDF via Docling; text/markdown via lightweight parser)
2. Text cleaning and preprocessing
3. Hierarchical chunking
4. Embedding generation
5. Storage in Neo4j (graph) and Qdrant (vectors)
"""

import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.extraction import EntityMerger, LLMExtractor, SpacyExtractor
from src.extraction.cooccurrence_extractor import CooccurrenceRelationshipExtractor
from src.extraction.dependency_extractor import DependencyRelationshipExtractor
from src.extraction.pattern_extractor import PatternRelationshipExtractor
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.text_rewriter import TextRewriter
from src.normalization import EntityDeduplicator, EntityRecord, MergeSuggestion, StringNormalizer
from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.schemas import (
    Chunk as GraphChunk,
)
from src.storage.schemas import (
    Document,
    EntityCandidate,
    EntityType,
    RelationshipCandidate,
)
from src.utils.config import Config
from src.utils.embeddings import EmbeddingGenerator

if TYPE_CHECKING:
    from src.ingestion.pdf_parser import PDFParser
    from src.normalization.acronym_resolver import AcronymResolver


class IngestionResult(BaseModel):
    """Result of document ingestion."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    success: bool
    chunks_created: int = 0
    entities_created: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


class ExtractionProgress:
    """Lightweight extraction progress tracker with heartbeat logging."""

    def __init__(self, total_chunks: int, heartbeat_seconds: float = 15.0) -> None:
        self.total_chunks = max(0, total_chunks)
        self.heartbeat_seconds = heartbeat_seconds
        self.stages: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def enable_stage(self, name: str, label: str) -> None:
        self.stages[name] = {
            "label": label,
            "total": self.total_chunks,
            "done": 0,
            "start": time.time(),
            "last_log": 0.0,
        }

    def update(self, name: str, *, increment: int = 1) -> None:
        stage = self.stages.get(name)
        if not stage:
            return
        with self._lock:
            stage["done"] = min(stage["done"] + increment, stage["total"])
            now = time.time()
            elapsed = max(now - stage["start"], 1e-6)
            rate = stage["done"] / elapsed
            remaining = max(stage["total"] - stage["done"], 0)
            eta_seconds = remaining / rate if rate > 0 else float("inf")

            should_log = (
                stage["done"] == stage["total"]
                or stage["last_log"] == 0
                or (now - stage["last_log"]) >= self.heartbeat_seconds
            )
            if should_log:
                percent = (stage["done"] / max(stage["total"], 1)) * 100
                logger.info(
                    "Extraction {}: {}/{} ({:.0f}%), {:.2f} cps, ETA {}",
                    stage["label"],
                    stage["done"],
                    stage["total"],
                    percent,
                    rate,
                    self._format_eta(eta_seconds),
                )
                stage["last_log"] = now

    def _format_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "unknown"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"


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
        self._debug_logging = str(getattr(config.logging, "level", "INFO")).upper() == "DEBUG"
        self._http_logs_silenced = False

        # Initialize components
        self.pdf_parser: PDFParser | None = None
        self.text_cleaner: TextCleaner | None = None
        self.text_rewriter: TextRewriter | None = None
        self.chunker: HierarchicalChunker | None = None
        self.embeddings: EmbeddingGenerator | None = None
        self.neo4j_manager: Neo4jManager | None = None
        self.qdrant_manager: QdrantManager | None = None
        self.spacy_extractor: SpacyExtractor | None = None
        self.pattern_extractor: PatternRelationshipExtractor | None = None
        self.dependency_extractor: DependencyRelationshipExtractor | None = None
        self.cooccurrence_extractor: CooccurrenceRelationshipExtractor | None = None
        self.llm_extractor: LLMExtractor | None = None
        self.entity_merger: EntityMerger | None = None
        self.string_normalizer: StringNormalizer | None = None
        self.acronym_resolver: AcronymResolver | None = None
        self.entity_deduplicator: EntityDeduplicator | None = None

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
        }

        # Per-document extraction caches
        self._spacy_entities_by_chunk: Dict[str | None, List[Any]] = {}
        self._llm_entities_by_chunk: Dict[str | None, List[Any]] = {}

        logger.info("IngestionPipeline initialized")

    def initialize_components(self) -> None:
        """Initialize all pipeline components.

        This is called lazily when first needed to avoid startup overhead.
        """
        self._silence_external_http_logs()

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
        embeddings_generator = self.embeddings

        if self.string_normalizer is None:
            self.string_normalizer = StringNormalizer(self.config.normalization)

        if self.entity_deduplicator is None and self.config.normalization.enable_semantic_matching:
            self.entity_deduplicator = EntityDeduplicator(
                config=self.config.normalization,
                embedder=embeddings_generator,
                database_config=self.config.database,
            )

        if self.acronym_resolver is None and self.config.normalization.enable_acronym_resolution:
            from src.normalization.acronym_resolver import AcronymResolver

            string_normalizer = self.string_normalizer
            assert string_normalizer is not None
            self.acronym_resolver = AcronymResolver(
                config=self.config.normalization,
                normalizer=string_normalizer,
            )

        if self.spacy_extractor is None:
            try:
                self.spacy_extractor = SpacyExtractor(self.config.extraction.spacy)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "spaCy extractor initialization failed; continuing without spaCy extraction",
                    error=str(exc),
                )

        if self.pattern_extractor is None:
            try:
                self.pattern_extractor = PatternRelationshipExtractor()
            except Exception as exc:
                logger.warning(
                    "Pattern extractor initialization failed",
                    error=str(exc)
                )

        if self.dependency_extractor is None:
            try:
                # Reuse spaCy NLP if available to save memory
                nlp = self.spacy_extractor.nlp if self.spacy_extractor else None
                self.dependency_extractor = DependencyRelationshipExtractor(nlp=nlp)
            except Exception as exc:
                logger.warning(
                    "Dependency extractor initialization failed",
                    error=str(exc)
                )

        if self.cooccurrence_extractor is None:
            self.cooccurrence_extractor = CooccurrenceRelationshipExtractor()

        if self.llm_extractor is None and self.config.extraction.enable_llm:
            try:
                self.llm_extractor = LLMExtractor(
                    self.config.extraction.llm,
                    prompts_path=self.config.extraction.llm_prompt_template,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM extractor initialization failed; disabling LLM extraction",
                    error=str(exc),
                )
                self.config.extraction.enable_llm = False

        if self.entity_merger is None:
            self.entity_merger = EntityMerger(
                allowed_types=self.config.extraction.entity_types,
                normalizer=self.string_normalizer,
            )

        if self.neo4j_manager is None:
            self.neo4j_manager = Neo4jManager(self.config.database)
        neo4j_manager = self.neo4j_manager
        assert neo4j_manager is not None
        neo4j_manager.connect()

        if self.qdrant_manager is None:
            self.qdrant_manager = QdrantManager(self.config.database)

        logger.debug("All pipeline components initialized")

    def process_document(
        self, pdf_path: Path | str, *, force_reingest: bool = False
    ) -> IngestionResult:
        """Process a single document end-to-end.

        Implements basic resume/rollback semantics:
        - Resume/skip if the document exists in Neo4j with matching checksum and status=completed.
        - If a prior run was interrupted (status != completed) or checksum differs, we clean up
          existing chunks in both DBs and re-ingest.
        - On failures during storage, we roll back partial chunk writes and mark the document failed.

        Args:
            pdf_path: Path to the document file (.pdf, .txt, .md)
            force_reingest: If True, ignore checkpoint skip and reprocess the document (after cleanup)

        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        suffix = pdf_path.suffix.lower()

        logger.info(f"Processing document: {pdf_path.name}")

        parsed_doc: ParsedDocument | None = None

        try:
            # Reset per-document caches
            self._spacy_entities_by_chunk = {}
            self._llm_entities_by_chunk = {}

            # Initialize components if needed
            self.initialize_components()

            pdf_parser = self.pdf_parser
            text_cleaner = self.text_cleaner
            chunker = self.chunker
            embeddings_generator = self.embeddings
            neo4j_manager = self.neo4j_manager
            assert text_cleaner is not None
            assert chunker is not None
            assert embeddings_generator is not None
            assert neo4j_manager is not None

            # Step 1: Parse document
            logger.debug("Step 1: Parsing document")
            if suffix == ".pdf":
                assert pdf_parser is not None
                parsed_doc = pdf_parser.parse_pdf(pdf_path)
            elif suffix in {".txt", ".md", ".markdown"}:
                from src.ingestion.text_file_parser import TextFileParser

                parsed_doc = TextFileParser().parse_file(pdf_path)
            else:
                raise ValueError(
                    f"Unsupported document type: {suffix} (supported: .pdf, .txt, .md, .markdown)"
                )

            if parsed_doc.error:
                raise Exception(f"Document parsing failed: {parsed_doc.error}")

                # Resume / cleanup logic (based on deterministic document_id + checksum)
            if self.config.pipeline.enable_checkpointing:
                existing = neo4j_manager.get_entity(parsed_doc.document_id, EntityType.DOCUMENT)
                existing_checksum = (existing or {}).get("checksum")
                existing_status = (existing or {}).get("ingestion_status")

                if (
                    existing
                    and existing_checksum
                    and existing_checksum == parsed_doc.metadata.get("checksum")
                    and existing_status == "completed"
                    and not force_reingest
                ):
                    # Already ingested successfully; skip.
                    try:
                        existing_chunks = neo4j_manager.get_chunks_by_document(
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
                        "Re-ingesting document {} (status={!r}, checksum_changed={}, force={})",
                        parsed_doc.document_id,
                        existing_status,
                        existing_checksum != parsed_doc.metadata.get("checksum"),
                        force_reingest,
                    )
                    self._cleanup_document_chunks(parsed_doc.document_id)

            # Mark document as ingesting (status tracking)
            self._upsert_document_status(parsed_doc, status="ingesting")

            # Step 2: Clean text
            logger.debug("Step 2: Cleaning text")
            if self.config.ingestion.text_cleaning.enabled:
                parsed_doc.raw_text = text_cleaner.clean(parsed_doc.raw_text)

            # Step 2.5: Optional rewriting (disabled by default)
            if self.config.ingestion.text_rewriting.enabled:
                logger.debug("Step 2.5: Rewriting text (optional)")
                self._rewrite_parsed_document(parsed_doc)

            # Step 3: Create chunks
            logger.debug("Step 3: Creating chunks")
            chunks = chunker.chunk_document(parsed_doc)

            if not chunks:
                raise Exception("No chunks created from document")

            # Step 3b: Build/update acronym dictionary from chunks (normalization aid)
            self.stats["acronym_definitions_added"] += self._update_acronym_dictionary(chunks)

            progress = ExtractionProgress(len(chunks))
            if self.spacy_extractor:
                progress.enable_stage("spacy", "spaCy entities")
            if self.config.extraction.enable_llm and self.llm_extractor:
                progress.enable_stage("llm_entities", "LLM entities")
                progress.enable_stage("llm_relationships", "LLM relationships")

            # Step 4: Extract entities with spaCy + LLM (in parallel when possible)
            logger.debug("Step 4: Extracting entities (spaCy + LLM)")
            spacy_entities_created = 0
            llm_entities_created = 0
            llm_relationships_created = 0

            # Filter chunks for LLM extraction (Sections/Subsections vs Paragraphs)
            llm_extraction_chunks = self._get_extraction_chunks(chunks)
            llm_progress = ExtractionProgress(len(llm_extraction_chunks))

            can_parallelize = (
                self.spacy_extractor is not None
                and self.llm_extractor is not None
                and self.config.extraction.enable_llm
            )
            if can_parallelize:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    spacy_future = executor.submit(self._extract_spacy_entities, chunks, progress)
                    llm_future = executor.submit(self._extract_llm_entities, llm_extraction_chunks, llm_progress)
                    spacy_entities_created = spacy_future.result()
                    llm_entities_created = llm_future.result()
            else:
                logger.debug("Step 4a: Extracting entities with spaCy")
                spacy_entities_created = self._extract_spacy_entities(chunks, progress)
                if self.config.extraction.enable_llm:
                    if not self.llm_extractor:
                        logger.warning(
                            "LLM extraction enabled but extractor not initialized; skipping"
                        )
                    else:
                        logger.debug("Step 4b: Extracting entities with LLM")
                        llm_entities_created = self._extract_llm_entities(llm_extraction_chunks, llm_progress)

            if self.config.extraction.enable_llm and self.llm_extractor:
                logger.debug("Step 4c: Extracting relationships with LLM")
                llm_relationships_created = self._extract_llm_relationships(llm_extraction_chunks, llm_progress)

                # Step 4c.1: Propagate entities to child chunks
                # This ensures that L4 paragraphs inherit entities found in their L2/L3 parent sections
                self._propagate_llm_entities(chunks)

            # Step 4c.2: Extract rule-based relationships (Pattern + Dependency)
            logger.debug("Step 4c.2: Extracting rule-based relationships")
            rule_based_relationships_created = self._extract_rule_based_relationships(chunks, progress)

            # Step 4d: Merge entities across extractors
            logger.debug("Step 4d: Merging extracted entities")
            merged_entities_created = self._merge_entities(chunks)
            self._enrich_merged_entities_with_acronyms(chunks)

            entities_created = (
                merged_entities_created
                if self.entity_merger
                else spacy_entities_created + llm_entities_created
            )

            # Step 5: Generate embeddings
            logger.debug("Step 5: Deduplicating merged entities with embeddings")
            dedup_suggestions = self._deduplicate_merged_entities(chunks)

            logger.debug("Step 6: Generating embeddings")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embeddings_generator.generate(chunk_texts)

            if len(embeddings) != len(chunks):
                raise Exception(
                    f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
                )

            # Step 6: Store in databases
            logger.debug("Step 7: Storing in databases")
            self._store_document_and_chunks(parsed_doc, chunks, embeddings)

            # Step 7b: Store entity/relationship candidates for curation
            logger.debug("Step 7b: Storing extraction candidates")
            entity_candidates_stored = self._store_entity_candidates(chunks)
            relationship_candidates_stored = self._store_relationship_candidates(chunks)

            # Mark document as completed
            self._upsert_document_status(parsed_doc, status="completed")

            # Persist updated acronym mappings (best-effort; don't fail ingestion)
            if self.acronym_resolver and self.config.normalization.enable_acronym_resolution:
                try:
                    self.acronym_resolver.store_mappings()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to store acronym mappings", error=str(exc))

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["total_processing_time"] += processing_time
            self.stats["entities_created"] = (
                self.stats.get("entities_created", 0) + entities_created
            )
            self.stats["llm_entities_extracted"] = (
                self.stats.get("llm_entities_extracted", 0) + llm_entities_created
            )
            self.stats["llm_relationships_extracted"] = (
                self.stats.get("llm_relationships_extracted", 0) + llm_relationships_created
            )
            self.stats["rule_based_relationships_extracted"] = (
                self.stats.get("rule_based_relationships_extracted", 0) + rule_based_relationships_created
            )
            self.stats["merged_entities_created"] = (
                self.stats.get("merged_entities_created", 0) + merged_entities_created
            )
            self.stats["dedup_merge_suggestions"] = (
                self.stats.get("dedup_merge_suggestions", 0) + dedup_suggestions
            )
            self.stats["entity_candidates_stored"] = (
                self.stats.get("entity_candidates_stored", 0) + entity_candidates_stored
            )
            self.stats["relationship_candidates_stored"] = (
                self.stats.get("relationship_candidates_stored", 0) + relationship_candidates_stored
            )

            logger.success(
                f"Document processed successfully: {len(chunks)} chunks, {processing_time:.2f}s"
            )

            return IngestionResult(
                document_id=parsed_doc.document_id,
                success=True,
                chunks_created=len(chunks),
                entities_created=entities_created,
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

    def _update_acronym_dictionary(self, chunks: List[Any]) -> int:
        if not (self.acronym_resolver and self.config.normalization.enable_acronym_resolution):
            return 0
        try:
            return int(self.acronym_resolver.update_dictionary_from_chunks(chunks))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Acronym dictionary update failed", error=str(exc))
            return 0

    def _enrich_merged_entities_with_acronyms(self, chunks: List[Any]) -> None:
        """Ensure both acronym and expansion appear in candidate aliases."""
        if not (self.acronym_resolver and self.config.normalization.enable_acronym_resolution):
            return

        acronym_re = re.compile(r"\b[A-Z][A-Z0-9&/\-]{1,10}\b")

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue

            context = getattr(chunk, "content", "") or ""

            for candidate in merged:
                canonical_name = str(candidate.get("canonical_name") or "").strip()
                alias_list = list(candidate.get("aliases") or [])

                seen: set[str] = set()
                for value in [canonical_name, *alias_list]:
                    if value:
                        seen.add(str(value))

                acronyms: set[str] = set()
                for value in [canonical_name, *alias_list]:
                    for match in acronym_re.finditer(str(value or "")):
                        token = match.group(0)
                        if len(token) > 1:
                            acronyms.add(token)

                if not acronyms:
                    continue

                for acronym in sorted(acronyms):
                    resolution = self.acronym_resolver.resolve(acronym, context=context)
                    if not resolution:
                        continue

                    for alias in resolution.aliases:
                        if alias and alias not in seen:
                            alias_list.append(alias)
                            seen.add(alias)

                    for mention in [canonical_name, *list(candidate.get("aliases") or [])]:
                        if not mention or acronym not in str(mention):
                            continue
                        expanded = str(mention).replace(acronym, resolution.expansion)
                        if expanded and expanded not in seen:
                            alias_list.append(expanded)
                            seen.add(expanded)

                candidate["aliases"] = alias_list

            metadata["merged_entities"] = merged
            chunk.metadata = metadata

    def _deduplicate_merged_entities(self, chunks: List[Any]) -> int:
        """Run embedding-based deduplication across merged entity candidates."""
        if not (self.entity_deduplicator and self.config.normalization.enable_semantic_matching):
            return 0

        aggregated: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue

            for candidate in merged:
                canonical_name = str(candidate.get("canonical_name") or "").strip()
                if not canonical_name:
                    continue

                type_label = str(candidate.get("type") or "UNKNOWN").upper()
                canonical_normalized = str(
                    candidate.get("canonical_normalized") or canonical_name
                ).strip()
                candidate_key = str(
                    candidate.get("candidate_key")
                    or self._candidate_key(type_label, canonical_normalized, canonical_name)
                )
                candidate["candidate_key"] = candidate_key

                mention_count = int(candidate.get("mention_count") or 1)
                aliases = [alias for alias in candidate.get("aliases") or [] if alias]
                description = str(candidate.get("description") or "").strip()

                record = aggregated.get(candidate_key)
                if record:
                    record["mention_count"] += mention_count
                    for alias in aliases:
                        if alias not in record["aliases"]:
                            record["aliases"].append(alias)
                    if not record["description"] and description:
                        # Keep longest description
                        if len(description) > len(record["description"]):
                            record["description"] = description
                else:
                    aggregated[candidate_key] = {
                        "entity_id": candidate_key,
                        "name": canonical_name,
                        "entity_type": type_label,
                        "description": description,
                        "aliases": aliases,
                        "mention_count": max(1, mention_count),
                    }

        if not aggregated:
            return 0

        try:
            records = [EntityRecord(**payload) for payload in aggregated.values()]
            result = self.entity_deduplicator.deduplicate(records)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deduplication failed", error=str(exc))
            return 0

        # Step 1: Handle Auto-Merges
        auto_merge_suggestions = [s for s in result.merge_suggestions if s.auto_merge]
        if auto_merge_suggestions:
            merges_performed = self._perform_auto_merges(chunks, auto_merge_suggestions)
            if merges_performed > 0:
                logger.info(f"Automatically merged {merges_performed} duplicate entities")
                # Recursive call to re-deduplicate the now-cleaner set of entities.
                # This ensures remaining suggestions are valid for the new entities.
                return self._deduplicate_merged_entities(chunks)

        # Step 2: Store suggestions for manual review
        suggestions_by_key: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for suggestion in result.merge_suggestions:
            payload = suggestion.model_dump()
            suggestions_by_key[suggestion.source_id].append(payload)
            suggestions_by_key[suggestion.target_id].append(payload)

        suggestion_count = len(result.merge_suggestions)
        if suggestion_count == 0:
            return 0

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue

            changed = False
            for candidate in merged:
                candidate_key = candidate.get("candidate_key")
                if candidate_key and candidate_key in suggestions_by_key:
                    candidate["dedup_suggestions"] = suggestions_by_key[candidate_key]
                    changed = True

            if changed:
                metadata["merged_entities"] = merged
                chunk.metadata = metadata

        logger.info(
            "Deduplication produced {} merge suggestions across {} clusters",
            suggestion_count,
            len(result.clusters),
        )

        return suggestion_count

    def _perform_auto_merges(
        self, chunks: List[Any], suggestions: List[MergeSuggestion]
    ) -> int:
        """Execute automatic merges on the chunks in-place."""
        if not suggestions:
            return 0

        # 1. Identify connected components (merge groups)
        parent: Dict[str, str] = {}

        def find(i: str) -> str:
            if i not in parent:
                parent[i] = i
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: str, j: str) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        involved_keys = set()
        for s in suggestions:
            union(s.source_id, s.target_id)
            involved_keys.add(s.source_id)
            involved_keys.add(s.target_id)

        # 2. Gather stats to pick survivors
        candidate_stats: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            for cand in merged:
                key = cand.get("candidate_key")
                if key and key in involved_keys:
                    if key not in candidate_stats:
                        candidate_stats[key] = {
                            "canonical_name": cand.get("canonical_name"),
                            "mention_count": 0,
                            "type": cand.get("type"),
                            "key": key,
                        }
                    candidate_stats[key]["mention_count"] += int(cand.get("mention_count", 1))

        # 3. Determine survivor for each group
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for key in involved_keys:
            # Only process keys present in this batch (sanity check)
            if key in candidate_stats:
                root = find(key)
                groups[root].append(key)

        replacement_map: Dict[str, str] = {}  # old_key -> survivor_key

        for members in groups.values():
            if len(members) < 2:
                continue

            # Sort by mention count desc, then name length desc, then lexical
            survivor_key = sorted(
                members,
                key=lambda k: (
                    candidate_stats[k]["mention_count"],
                    len(candidate_stats[k]["canonical_name"] or ""),
                    k,
                ),
                reverse=True,
            )[0]

            for m in members:
                if m != survivor_key:
                    replacement_map[m] = survivor_key

        if not replacement_map:
            return 0

        # 4. Aggregate data into survivor records
        merged_data_cache: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            for cand in merged:
                key = cand.get("candidate_key")
                if not key:
                    continue

                # Check if this candidate is part of a merge group (either survivor or victim)
                is_survivor = key in merged_data_cache or (key in involved_keys and key not in replacement_map)
                is_victim = key in replacement_map

                if not (is_survivor or is_victim):
                    continue

                survivor_key = replacement_map.get(key, key)

                # Initialize survivor record if needed
                if survivor_key not in merged_data_cache:
                    base_stats = candidate_stats.get(survivor_key)
                    if not base_stats:
                        continue # Should not happen

                    merged_data_cache[survivor_key] = {
                        "canonical_name": base_stats["canonical_name"],
                        "canonical_normalized": "", # Will re-normalize if needed
                        "type": base_stats["type"],
                        "candidate_key": survivor_key,
                        "confidence": 0.0,
                        "aliases": set(),
                        "description": "",
                        "mention_count": 0,
                        "conflicting_types": set(),
                        "provenance": [],
                    }

                target = merged_data_cache[survivor_key]

                # Merge logic
                target["confidence"] = max(float(target["confidence"]), float(cand.get("confidence", 0.0)))
                target["mention_count"] += int(cand.get("mention_count", 1))

                # Aliases
                for alias in cand.get("aliases") or []:
                    if alias:
                        target["aliases"].add(alias)
                # Add own canonical name as alias if different from survivor
                if cand.get("canonical_name") and cand.get("canonical_name") != target["canonical_name"]:
                     target["aliases"].add(cand.get("canonical_name"))

                # Description (keep longest)
                desc = str(cand.get("description", ""))
                if len(desc) > len(str(target["description"])):
                     target["description"] = desc

                # Conflicting types
                for ct in cand.get("conflicting_types") or []:
                    target["conflicting_types"].add(ct)
                # If merging different types, add original type to conflicts
                if cand.get("type") and cand.get("type") != target["type"]:
                    target["conflicting_types"].add(cand.get("type"))

                # Provenance
                target["provenance"].extend(cand.get("provenance") or [])

        # 5. Write back to chunks
        merges_count = len(replacement_map)

        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            new_merged = []
            seen_keys_in_chunk = set()

            for cand in merged:
                key = cand.get("candidate_key")

                # If this candidate is not involved in any merge, keep it
                if not key or (key not in replacement_map and key not in merged_data_cache):
                    new_merged.append(cand)
                    continue

                # It is involved
                survivor_key = replacement_map.get(key, key)

                if survivor_key not in seen_keys_in_chunk:
                    # Retrieve the fully merged data
                    if survivor_key in merged_data_cache:
                        data = merged_data_cache[survivor_key]

                        # Convert sets to lists
                        final_cand = data.copy()
                        final_cand["aliases"] = list(data["aliases"])
                        final_cand["conflicting_types"] = list(data["conflicting_types"])

                        new_merged.append(final_cand)
                        seen_keys_in_chunk.add(survivor_key)

            chunk.metadata["merged_entities"] = new_merged

        return merges_count

    def _get_extraction_chunks(self, chunks: List[Any]) -> List[Any]:
        """Determine the optimal set of chunks for LLM extraction.

        Prefers Level 2 (Sections) or Level 3 (Subsections) to provide better
        context. Falls back to Level 4 (Paragraphs) if no structural chunks exist
        (e.g., for unstructured text files).

        Args:
            chunks: List of all hierarchical chunks

        Returns:
            List of chunks to be used for LLM extraction
        """
        # 1. Try Level 3 (Subsections) - Good balance of context and size
        l3_chunks = [c for c in chunks if getattr(c, "level", 0) == 3]
        if l3_chunks:
            logger.debug(f"Using {len(l3_chunks)} Level 3 chunks for LLM extraction")
            return l3_chunks

        # 2. Try Level 2 (Sections) - If no subsections exist
        l2_chunks = [c for c in chunks if getattr(c, "level", 0) == 2]
        if l2_chunks:
            logger.debug(f"Using {len(l2_chunks)} Level 2 chunks for LLM extraction")
            return l2_chunks

        # 3. Fallback to Level 4 (Paragraphs) - For unstructured docs
        l4_chunks = [c for c in chunks if getattr(c, "level", 0) == 4]
        if l4_chunks:
            logger.debug(f"Using {len(l4_chunks)} Level 4 chunks for LLM extraction (fallback)")
            return l4_chunks

        # 4. Final fallback to Level 1 (Document) if it's a very small file
        l1_chunks = [c for c in chunks if getattr(c, "level", 0) == 1]
        logger.debug(f"Using {len(l1_chunks)} Level 1 chunks for LLM extraction (final fallback)")
        return l1_chunks

    def process_batch(
        self, pdf_paths: List[Path | str], *, force_reingest: bool = False
    ) -> List[IngestionResult]:
        """Process multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files
            force_reingest: If True, ignore checkpoint skip and reprocess documents (after cleanup)

        Returns:
            List of IngestionResult objects
        """
        logger.info(f"Processing batch of {len(pdf_paths)} documents")

        results = []
        for pdf_path in pdf_paths:
            result = self.process_document(pdf_path, force_reingest=force_reingest)
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

    def _silence_external_http_logs(self) -> None:
        """Reduce noisy external logs unless debug logging is enabled."""
        if self._debug_logging or self._http_logs_silenced:
            return

        noisy_loggers = (
            "httpx",
            "httpcore",
            "openai",
            "openai._base_client",
            "openai._http_client",
            # Docling emits deprecation warnings like "strict_text"; suppress unless debugging.
            "docling",
            "docling.document_converter",
        )
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.ERROR)
        self._http_logs_silenced = True

    def _rewrite_parsed_document(self, parsed_doc: ParsedDocument) -> None:
        """Rewrite parsed document content based on config chunk_level.

        - section: rewrite each section.content (and raw_text)
        - subsection: rewrite each subsection.content (and raw_text)
        """
        if not self.text_rewriter:
            # Enabled in config, but not initialized (shouldn't happen), fail safe.
            self.text_rewriter = TextRewriter(self.config.ingestion.text_rewriting)
        text_rewriter = self.text_rewriter
        assert text_rewriter is not None

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

                result = text_rewriter.rewrite(
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

                    result = text_rewriter.rewrite(
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
        doc_result = text_rewriter.rewrite(
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

    def _extract_spacy_entities(
        self, chunks: List[Any], progress: ExtractionProgress | None = None
    ) -> int:
        """Extract entities from chunks using the spaCy extractor."""
        if not self.spacy_extractor:
            return 0

        try:
            by_chunk = self.spacy_extractor.extract_from_chunks(chunks)
            self._spacy_entities_by_chunk = by_chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy extraction failed; proceeding without entities", error=str(exc))
            return 0

        total = 0
        for chunk in chunks:
            entities = by_chunk.get(getattr(chunk, "chunk_id", None), [])
            if not entities:
                if progress:
                    progress.update("spacy")
                continue

            total += len(entities)
            chunk.metadata.setdefault("spacy_entities", [])

            for ent in entities:
                chunk.metadata["spacy_entities"].append(
                    {
                        "name": ent.name,
                        "type": ent.type,
                        "confidence": ent.confidence,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "sentence": ent.sentence,
                        "context": ent.context,
                        "source": (ent.metadata or {}).get("source", ent.source),
                    }
                )

            if progress:
                progress.update("spacy")

        # Ensure final heartbeat when there were zero entities but chunks processed
        if progress and not chunks:
            progress.update("spacy")

        return total

    def _extract_llm_entities(
        self, chunks: List[Any], progress: ExtractionProgress | None = None
    ) -> int:
        """Extract entities from chunks using the configured LLM extractor."""
        if not self.llm_extractor:
            return 0

        total = 0
        llm_entities_by_chunk: DefaultDict[str | None, List[Any]] = defaultdict(list)
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            try:
                entities = self.llm_extractor.extract_entities(
                    chunk,
                    document_context={
                        "document_title": metadata.get("document_title"),
                        "section_title": metadata.get("section_title")
                        or metadata.get("hierarchy_path"),
                        "page_numbers": metadata.get("page_numbers"),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM entity extraction failed for chunk",
                    chunk_id=getattr(chunk, "chunk_id", None),
                    error=str(exc),
                )
                entities = []

            if entities:
                total += len(entities)
                metadata.setdefault("llm_entities", [])
                llm_entities_by_chunk[getattr(chunk, "chunk_id", None)].extend(entities)

                for ent in entities:
                    metadata["llm_entities"].append(
                        {
                            "name": ent.name,
                            "type": ent.type,
                            "description": ent.description,
                            "aliases": ent.aliases,
                            "confidence": ent.confidence,
                            "source": ent.source,
                            "chunk_id": ent.chunk_id or getattr(chunk, "chunk_id", None),
                            "document_id": ent.document_id or getattr(chunk, "document_id", None),
                        }
                    )

                if hasattr(chunk, "metadata"):
                    chunk.metadata = metadata
            if progress:
                progress.update("llm_entities")

        self._llm_entities_by_chunk = dict(llm_entities_by_chunk)
        if progress and not chunks:
            progress.update("llm_entities")
        return total

    def _normalize_candidate_key(self, value: str) -> str:
        from src.utils.candidate_keys import normalize_candidate_key_fragment

        return normalize_candidate_key_fragment(value, normalizer=self.string_normalizer)

    def _candidate_key(self, type_label: str, canonical_normalized: str, fallback: str) -> str:
        base = canonical_normalized or fallback
        normalized = self._normalize_candidate_key(base)
        return f"{type_label}:{normalized}" if normalized else f"{type_label}:{fallback}"

    def _store_entity_candidates(self, chunks: List[Any]) -> int:
        """Store merged entity candidates in Neo4j for later curation."""
        neo4j_manager = self.neo4j_manager
        if not neo4j_manager:
            return 0
        if not hasattr(neo4j_manager, "upsert_entity_candidate_aggregate"):
            return 0

        stored = 0
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue

            chunk_id = getattr(chunk, "chunk_id", None)
            document_id = getattr(chunk, "document_id", None)

            for cand in merged:
                try:
                    cand_type = EntityType(str(cand.get("type")))
                except Exception:  # noqa: BLE001
                    continue

                canonical_name = str(cand.get("canonical_name") or "").strip()
                canonical_normalized = str(
                    cand.get("canonical_normalized") or canonical_name
                ).strip()
                if not canonical_name:
                    continue

                key = str(
                    cand.get("candidate_key")
                    or f"{cand_type.value}:{self._normalize_candidate_key(canonical_normalized)}"
                )
                event = EntityCandidate.provenance_event(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "observed_at": datetime.now().isoformat(),
                        "provenance": cand.get("provenance") or [],
                        "confidence": cand.get("confidence"),
                        "source": "pipeline",
                    }
                )

                candidate = EntityCandidate(
                    id=None,
                    candidate_key=key,
                    canonical_name=canonical_name,
                    candidate_type=cand_type,
                    aliases=list(cand.get("aliases") or []),
                    description=str(cand.get("description") or ""),
                    confidence_score=float(cand.get("confidence") or 0.0),
                    mention_count=int(cand.get("mention_count") or 1),
                    source_documents=[document_id] if document_id else [],
                    chunk_ids=[chunk_id] if chunk_id else [],
                    conflicting_types=list(cand.get("conflicting_types") or []),
                    provenance_events=[event],
                )
                try:
                    neo4j_manager.upsert_entity_candidate_aggregate(candidate)
                    stored += 1
                    metadata.setdefault("entity_candidate_keys", []).append(key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to store entity candidate",
                        chunk_id=chunk_id,
                        error=str(exc),
                    )

        return stored

    def _store_relationship_candidates(self, chunks: List[Any]) -> int:
        """Store relationship candidates in Neo4j for later curation."""
        neo4j_manager = self.neo4j_manager
        if not neo4j_manager:
            return 0
        if not hasattr(neo4j_manager, "upsert_relationship_candidate_aggregate"):
            return 0

        stored = 0
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}

            # Combine LLM and rule-based relationships
            rels = []
            rels.extend(metadata.get("llm_relationships") or [])
            rels.extend(metadata.get("rule_based_relationships") or [])

            if not rels:
                continue

            chunk_id = getattr(chunk, "chunk_id", None)
            document_id = getattr(chunk, "document_id", None)

            for rel in rels:
                source = str(rel.get("source") or "").strip()
                target = str(rel.get("target") or "").strip()
                rel_type = str(rel.get("type") or "").strip()
                if not (source and target and rel_type):
                    continue

                key = (
                    f"{self._normalize_candidate_key(source)}:"
                    f"{rel_type}:"
                    f"{self._normalize_candidate_key(target)}"
                )
                event = RelationshipCandidate.provenance_event(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "observed_at": datetime.now().isoformat(),
                        "source_extractor": rel.get("source_extractor") or "llm",
                        "confidence": rel.get("confidence"),
                        "source": "pipeline",
                    }
                )

                candidate = RelationshipCandidate(
                    id=None,
                    candidate_key=key,
                    source=source,
                    target=target,
                    type=rel_type,
                    description=str(rel.get("description") or ""),
                    confidence_score=float(rel.get("confidence") or 0.0),
                    mention_count=1,
                    source_documents=[document_id] if document_id else [],
                    chunk_ids=[chunk_id] if chunk_id else [],
                    provenance_events=[event],
                )
                try:
                    neo4j_manager.upsert_relationship_candidate_aggregate(candidate)
                    stored += 1
                    metadata.setdefault("relationship_candidate_keys", []).append(key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to store relationship candidate",
                        chunk_id=chunk_id,
                        error=str(exc),
                    )

        return stored

    def _extract_rule_based_relationships(
        self, chunks: List[Any], progress: ExtractionProgress | None = None
    ) -> int:
        """Extract relationships using pattern and dependency extractors."""
        total = 0
        if self.pattern_extractor:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", {}) or {}
                rels = self.pattern_extractor.extract_relationships(chunk)
                if rels:
                    metadata.setdefault("rule_based_relationships", []).extend([
                        r.model_dump() for r in rels
                    ])
                    total += len(rels)
                chunk.metadata = metadata

        if self.dependency_extractor:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", {}) or {}
                rels = self.dependency_extractor.extract_relationships(chunk)
                if rels:
                    metadata.setdefault("rule_based_relationships", []).extend([
                        r.model_dump() for r in rels
                    ])
                    total += len(rels)
                chunk.metadata = metadata

        if self.cooccurrence_extractor:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", {}) or {}
                known_entities = []
                known_entities.extend(metadata.get("llm_entities", []))
                known_entities.extend(metadata.get("spacy_entities", []))

                rels = self.cooccurrence_extractor.extract_relationships(
                    chunk, known_entities=known_entities
                )
                if rels:
                    metadata.setdefault("rule_based_relationships", []).extend([
                        r.model_dump() for r in rels
                    ])
                    total += len(rels)
                chunk.metadata = metadata

        return total

    def _extract_llm_relationships(
        self, chunks: List[Any], progress: ExtractionProgress | None = None
    ) -> int:
        """Extract relationships from chunks using the configured LLM extractor."""
        if not self.llm_extractor:
            return 0

        total = 0
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            known_entities = []
            known_entities.extend(metadata.get("llm_entities", []))
            known_entities.extend(metadata.get("spacy_entities", []))

            try:
                relationships = self.llm_extractor.extract_relationships(
                    chunk,
                    known_entities=known_entities,
                    document_context={
                        "document_title": metadata.get("document_title"),
                        "section_title": metadata.get("section_title")
                        or metadata.get("hierarchy_path"),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM relationship extraction failed for chunk",
                    chunk_id=getattr(chunk, "chunk_id", None),
                    error=str(exc),
                )
                relationships = []

            if relationships:
                total += len(relationships)
                metadata.setdefault("llm_relationships", [])

                for rel in relationships:
                    metadata["llm_relationships"].append(
                        {
                            "source": rel.source,
                            "type": rel.type,
                            "target": rel.target,
                            "description": rel.description,
                            "confidence": rel.confidence,
                            "bidirectional": rel.bidirectional,
                            "chunk_id": rel.chunk_id or getattr(chunk, "chunk_id", None),
                            "document_id": rel.document_id or getattr(chunk, "document_id", None),
                            "source_extractor": rel.source_extractor,
                        }
                    )

                if hasattr(chunk, "metadata"):
                    chunk.metadata = metadata
            if progress:
                progress.update("llm_relationships")

        if progress and not chunks:
            progress.update("llm_relationships")
        return total

    def _propagate_llm_entities(self, chunks: List[Any]) -> int:
        """Propagate extracted LLM entities from parent chunks to their children.

        When entities are extracted from Level 2/3 chunks, they should also be
        associated with the specific Level 4 (Paragraph) chunks where they are
        mentioned. This ensures vector retrieval on L4 chunks can still link
        back to the entities.

        Args:
            chunks: List of all chunks

        Returns:
            Number of propagated entity mentions
        """
        propagated_count = 0
        chunk_map = {getattr(c, "chunk_id", None): c for c in chunks}

        # Identify parent chunks (L2/L3) that have extracted entities
        for parent in chunks:
            parent_id = getattr(parent, "chunk_id", None)
            child_ids = getattr(parent, "child_chunk_ids", []) or []
            if not child_ids:
                continue

            # Get source entities from this parent
            source_entities = self._llm_entities_by_chunk.get(parent_id, [])
            if not source_entities:
                continue

            # Check each child for mentions
            for child_id in child_ids:
                child = chunk_map.get(child_id)
                if not child:
                    continue

                content = getattr(child, "content", "").lower()
                child_metadata = getattr(child, "metadata", {}) or {}

                # Ensure child has a list for new entities
                if "llm_entities" not in child_metadata:
                    child_metadata["llm_entities"] = []
                    # If this child had no LLM extraction done (typical for L4 now),
                    # we need to initialize its entry in the cache too.
                    if child_id not in self._llm_entities_by_chunk:
                        self._llm_entities_by_chunk[child_id] = []

                # Find entities mentioned in this child
                for ent in source_entities:
                    # Check name
                    name_match = str(ent.name).lower() in content

                    # Check aliases
                    alias_match = False
                    for alias in ent.aliases:
                        if str(alias).lower() in content:
                            alias_match = True
                            break

                    if name_match or alias_match:
                        # Propagate this entity to the child
                        propagated_entity = ent.model_copy()
                        propagated_entity.chunk_id = child_id
                        # Note: We keep the original confidence score.

                        # Add to pipeline cache
                        self._llm_entities_by_chunk[child_id].append(propagated_entity)

                        # Add to chunk metadata
                        child_metadata["llm_entities"].append({
                            "name": ent.name,
                            "type": ent.type,
                            "description": ent.description,
                            "aliases": ent.aliases,
                            "confidence": ent.confidence,
                            "source": ent.source,
                            "chunk_id": child_id,
                            "document_id": ent.document_id,
                            "propagated_from": parent_id
                        })
                        propagated_count += 1

                child.metadata = child_metadata

        if propagated_count > 0:
            logger.debug(f"Propagated {propagated_count} entity mentions to child chunks")

        return propagated_count

    def _merge_entities(self, chunks: List[Any]) -> int:
        """Merge spaCy and LLM entities into unified candidates."""
        if not self.entity_merger:
            return 0

        total = 0
        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            metadata = getattr(chunk, "metadata", None) or {}

            spacy_entities = self._spacy_entities_by_chunk.get(chunk_id, [])
            llm_entities = self._llm_entities_by_chunk.get(chunk_id, [])

            merged = self.entity_merger.merge(spacy_entities, llm_entities)
            if not merged:
                continue

            metadata.setdefault("merged_entities", [])
            for candidate in merged:
                candidate_key = self._candidate_key(
                    candidate.resolved_type,
                    candidate.canonical_normalized,
                    candidate.canonical_name,
                )
                metadata["merged_entities"].append(
                    {
                        "canonical_name": candidate.canonical_name,
                        "canonical_normalized": candidate.canonical_normalized,
                        "type": candidate.resolved_type,
                        "candidate_key": candidate_key,
                        "confidence": candidate.combined_confidence,
                        "aliases": candidate.aliases,
                        "description": candidate.description,
                        "mention_count": candidate.mention_count,
                        "conflicting_types": candidate.conflicting_types,
                        "provenance": [prov.model_dump() for prov in candidate.provenance],
                    }
                )
            if hasattr(chunk, "metadata"):
                chunk.metadata = metadata
            else:
                chunk.metadata = metadata

            total += len(merged)

        return total

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
        neo4j_manager = self.neo4j_manager
        qdrant_manager = self.qdrant_manager
        assert neo4j_manager is not None
        assert qdrant_manager is not None

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
                "file_path": parsed_doc.metadata.get("file_path"),
            },
        )

        # Idempotent document upsert (supports resume/retries)
        if hasattr(neo4j_manager, "upsert_entity"):
            doc_id = neo4j_manager.upsert_entity(document)
        else:
            # Back-compat for tests/fakes that only implement create_entity()
            doc_id = neo4j_manager.create_entity(document)

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
        qdrant_manager.upsert_chunks(chunk_payloads, chunk_vectors)

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
            if hasattr(neo4j_manager, "upsert_chunk"):
                neo4j_manager.upsert_chunk(graph_chunk)
            else:
                # Back-compat for tests/fakes that only implement create_chunk()
                neo4j_manager.create_chunk(graph_chunk)

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
        neo4j_manager = self.neo4j_manager
        assert neo4j_manager is not None
        if hasattr(neo4j_manager, "upsert_entity"):
            neo4j_manager.upsert_entity(document)
        else:
            neo4j_manager.create_entity(document)

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
        except Exception:  # noqa: BLE001
            health["neo4j"] = False

        try:
            qdrant_health, _ = (
                self.qdrant_manager.health_check() if self.qdrant_manager else (False, "")
            )
            health["qdrant"] = qdrant_health
        except Exception:  # noqa: BLE001
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

    def __enter__(self) -> "IngestionPipeline":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
