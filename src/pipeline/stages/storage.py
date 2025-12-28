import re
from datetime import datetime
from typing import Any, List

from loguru import logger

from src.normalization.string_normalizer import StringNormalizer
from src.pipeline.base import PipelineContext, PipelineStage
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
from src.utils.candidate_keys import normalize_candidate_key_fragment


class StorageStage(PipelineStage):
    """Stage for storing results in databases."""

    def __init__(self, config, neo4j_manager: Neo4jManager, qdrant_manager: QdrantManager):
        super().__init__("Storage")
        self.config = config
        self.neo4j_manager = neo4j_manager
        self.qdrant_manager = qdrant_manager
        self.string_normalizer = StringNormalizer(config.normalization)

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.parsed_document or not context.chunks:
            return context

        logger.debug("Storing in databases")
        self._store_document_and_chunks(context.parsed_document, context.chunks, context.embeddings)

        if context.topics:
            logger.debug(f"Linking to topics: {context.topics}")
            self.neo4j_manager.save_document_topics(
                context.parsed_document.document_id, context.topics
            )

        logger.debug("Storing extraction candidates")
        entity_candidates_stored = self._store_entity_candidates(context.chunks)
        relationship_candidates_stored = self._store_relationship_candidates(context.chunks)

        context.update_stats("entity_candidates_stored", entity_candidates_stored)
        context.update_stats("relationship_candidates_stored", relationship_candidates_stored)

        # Mark document as completed
        self._upsert_document_status(context.parsed_document, status="completed")

        return context

    def _store_document_and_chunks(self, parsed_doc, chunks, embeddings):
        filename = parsed_doc.metadata.get("filename", "")
        title = parsed_doc.metadata.get("title") or filename or "document"
        canonical_name = (
            re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip()).strip("_").lower() or "document"
        )

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
            properties={
                "ingestion_status": "ingesting",
                "last_ingested_at": datetime.now().isoformat(),
                "file_path": parsed_doc.metadata.get("file_path"),
            },
        )

        if hasattr(self.neo4j_manager, "upsert_entity"):
            self.neo4j_manager.upsert_entity(document)
        else:
            self.neo4j_manager.create_entity(document)

        chunk_payloads = []
        chunk_vectors = []

        for chunk, embedding in zip(chunks, embeddings):
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
            chunk_vectors.append(embedding.tolist())

        self.qdrant_manager.upsert_chunks(chunk_payloads, chunk_vectors)

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
                self.neo4j_manager.create_chunk(graph_chunk)

    def _store_entity_candidates(self, chunks: List[Any]) -> int:
        if not hasattr(self.neo4j_manager, "upsert_entity_candidate_aggregate"):
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
                except Exception:
                    continue

                canonical_name = str(cand.get("canonical_name") or "").strip()
                str(cand.get("canonical_normalized") or canonical_name).strip()
                if not canonical_name:
                    continue

                # We need normalize_candidate_key_fragment. It's not available here directly.
                # I imported it above.

                # But I need StringNormalizer logic.
                # I can create a temporary normalizer or reuse the one in ExtractionStage...
                # Actually, `candidate_key` SHOULD already be in `cand` dict from ExtractionStage!
                # Yes, ExtractionStage sets it.

                key = str(cand.get("candidate_key") or "")
                # If key is missing (shouldn't be), we might need to regenerate it.
                # But ExtractionStage guarantees it.

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
                    self.neo4j_manager.upsert_entity_candidate_aggregate(candidate)
                    stored += 1
                except Exception as exc:
                    logger.warning(f"Failed to store entity candidate: {exc}")
        return stored

    def _store_relationship_candidates(self, chunks: List[Any]) -> int:
        if not hasattr(self.neo4j_manager, "upsert_relationship_candidate_aggregate"):
            return 0
        stored = 0
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
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

                source_norm = normalize_candidate_key_fragment(
                    source, normalizer=self.string_normalizer
                )
                target_norm = normalize_candidate_key_fragment(
                    target, normalizer=self.string_normalizer
                )

                key = f"{source_norm}:{rel_type}:{target_norm}"

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
                    self.neo4j_manager.upsert_relationship_candidate_aggregate(candidate)
                    stored += 1
                except Exception as exc:
                    logger.warning(f"Failed to store relationship candidate: {exc}")

        return stored

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
