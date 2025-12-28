"""Unit tests for ingestion pipeline mapping logic (Phase 1 Task 1.10).

These tests avoid requiring Neo4j/Qdrant by monkeypatching managers and focusing on:
- Document entity creation fields
- Chunk mapping to graph schema model
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

from src.pipeline.stages.storage import StorageStage
from src.storage.schemas import Chunk as GraphChunk
from src.utils.config import Config


class _IngestChunk(BaseModel):
    chunk_id: str
    document_id: str
    level: int
    parent_chunk_id: str | None = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


class _FakeNeo4j:
    def __init__(self) -> None:
        self.created_entities: List[Any] = []
        self.created_chunks: List[GraphChunk] = []

    def create_entity(self, entity: Any) -> str:
        self.created_entities.append(entity)
        return getattr(entity, "id", "doc-id")

    def create_chunk(self, chunk: GraphChunk) -> str:
        self.created_chunks.append(chunk)
        return chunk.id


class _FakeQdrant:
    def __init__(self) -> None:
        self.upserts: List[Any] = []

    def upsert_chunks(self, payloads: List[dict], vectors: List[list]) -> int:
        self.upserts.append((payloads, vectors))
        return len(payloads)


def test_store_document_and_chunks_maps_models() -> None:
    cfg = Config.from_yaml("config/config.yaml")

    # Inject fakes (no network/db)
    neo4j = _FakeNeo4j()
    qdrant = _FakeQdrant()

    stage = StorageStage(cfg, neo4j, qdrant)

    class _Parsed:
        document_id = "doc-123"
        page_count = 1
        metadata = {"filename": "file.pdf", "title": "My Title"}

    parsed_doc = _Parsed()

    chunks = [
        _IngestChunk(
            chunk_id="c1",
            document_id="doc-123",
            level=1,
            parent_chunk_id=None,
            child_chunk_ids=[],
            content="hello",
            metadata={"page_numbers": [1], "hierarchy_path": "1"},
            token_count=10,
        )
    ]
    embeddings = [np.zeros(768, dtype=np.float32)]

    stage._store_document_and_chunks(parsed_doc, chunks, embeddings)

    # Document entity was created with canonical_name and id
    created_doc = neo4j.created_entities[0]
    assert created_doc.id == "doc-123"
    assert created_doc.canonical_name  # non-empty
    assert created_doc.filename == "file.pdf"

    # Chunk mapping used GraphChunk model
    assert len(neo4j.created_chunks) == 1
    created_chunk = neo4j.created_chunks[0]
    assert isinstance(created_chunk, GraphChunk)
    assert created_chunk.id == "c1"
    assert created_chunk.document_id == "doc-123"
