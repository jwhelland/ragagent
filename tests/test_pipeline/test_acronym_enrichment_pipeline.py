"""Unit tests for ingestion pipeline acronym enrichment (Phase 3 Task 3.3)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from src.pipeline.stages.chunking import ChunkingStage
from src.pipeline.stages.extraction import ExtractionStage
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


def test_pipeline_enriches_candidate_aliases_with_expansion() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.normalization.enable_acronym_resolution = True

    # Setup ChunkingStage for dictionary update
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.pipeline.stages.chunking.HierarchicalChunker", MagicMock())
        chunking_stage = ChunkingStage(cfg)

        # Setup ExtractionStage for enrichment
        mp.setattr("src.pipeline.stages.extraction.EmbeddingGenerator", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.SpacyExtractor", MagicMock())
        mp.setattr("src.pipeline.stages.extraction.LLMExtractor", MagicMock())
        extraction_stage = ExtractionStage(cfg)

        chunk = _IngestChunk(
            chunk_id="c1",
            document_id="doc1",
            level=2,
            content="Telemetry and Command (T&C) subsystem handles uplink commands.",
            metadata={
                "merged_entities": [
                    {
                        "canonical_name": "T&C",
                        "canonical_normalized": "t&c",
                        "type": "SYSTEM",
                        "confidence": 0.9,
                        "aliases": ["T&C subsystem"],
                        "description": "",
                        "mention_count": 1,
                        "conflicting_types": [],
                        "provenance": [],
                    }
                ]
            },
        )

        # Manually share the acronym resolver between stages (in real pipeline, they are separate instances but share the underlying dictionary file/logic?
        # Actually in IngestionPipeline, it shared the instance `self.acronym_resolver`.
        # In modular pipeline, `ChunkingStage` has one, `ExtractionStage` has another.
        # But `AcronymResolver` loads/saves to file.
        # For the test, we need them to share state or use the same instance.
        # I'll inject the resolver from chunking to extraction for this test.
        extraction_stage.acronym_resolver = chunking_stage.acronym_resolver

        chunking_stage._update_acronym_dictionary([chunk])
        extraction_stage._enrich_merged_entities_with_acronyms([chunk])

        merged = chunk.metadata["merged_entities"][0]
        assert "Telemetry and Command" in merged["aliases"]
        assert "Telemetry and Command subsystem" in merged["aliases"]
