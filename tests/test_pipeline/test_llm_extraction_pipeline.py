from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.extraction.models import ExtractedEntity, ExtractedRelationship
from src.pipeline.base import PipelineContext
from src.pipeline.stages.extraction import ExtractionStage
from src.utils.config import Config


class _Chunk:
    def __init__(
        self, chunk_id: str, document_id: str, content: str, metadata: Dict[str, Any]
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.level = 3
        self.content = content
        self.metadata = metadata


class _FakeLLMExtractor:
    def __init__(self) -> None:
        self.entity_calls: List[Dict[str, Any]] = []
        self.relationship_known_entities: List[Any] = []

    def extract_entities(self, chunk: Any, *, document_context: Dict[str, Any] | None = None):
        self.entity_calls.append(
            {"chunk_id": getattr(chunk, "chunk_id", None), "context": document_context}
        )
        return [
            ExtractedEntity(
                name="solar array",
                type="COMPONENT",
                description="Generates power",
                aliases=["arrays"],
                confidence=0.9,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source="llm",
            )
        ]

    def extract_relationships(
        self,
        chunk: Any,
        *,
        known_entities: List[Any] | None = None,
        document_context: Dict[str, Any] | None = None,
    ):
        self.relationship_known_entities = list(known_entities or [])
        return [
            ExtractedRelationship(
                source="battery",
                target="solar_array",
                type="DEPENDS_ON",
                description="Battery charging depends on arrays",
                confidence=0.88,
                bidirectional=False,
                chunk_id=getattr(chunk, "chunk_id", None),
                document_id=getattr(chunk, "document_id", None),
                source_extractor="llm",
            )
        ]


def test_llm_entities_added_to_chunk_metadata() -> None:
    cfg = Config.from_yaml("config/config.yaml")

    with MagicMock() as mock_embedder:
        stage = ExtractionStage(cfg)
        stage.llm_extractor = _FakeLLMExtractor()
        stage.embeddings = mock_embedder  # Mock embedder
        stage.spacy_extractor = MagicMock()  # Mock spacy to avoid load

        chunk = _Chunk(
            chunk_id="c1",
            document_id="doc-1",
            content="Solar arrays provide power",
            metadata={"document_title": "Power", "section_title": "EPS"},
        )

        context = PipelineContext(file_path=Path("dummy"), config=cfg)

        count = stage._extract_llm_entities([chunk], context)

        assert count == 1
        llm_entities = chunk.metadata.get("llm_entities", [])
        assert len(llm_entities) == 1
        assert llm_entities[0]["name"] == "solar array"
        assert llm_entities[0]["type"] == "COMPONENT"
        assert llm_entities[0]["confidence"] == 0.9

        # Check cache
        assert "c1" in context.llm_entities_by_chunk
        assert len(context.llm_entities_by_chunk["c1"]) == 1


def test_llm_relationships_use_known_entities_and_add_metadata() -> None:
    cfg = Config.from_yaml("config/config.yaml")

    with MagicMock() as mock_embedder:
        stage = ExtractionStage(cfg)
        fake_extractor = _FakeLLMExtractor()
        stage.llm_extractor = fake_extractor
        stage.embeddings = mock_embedder
        stage.spacy_extractor = MagicMock()

        chunk = _Chunk(
            chunk_id="c2",
            document_id="doc-1",
            content="The battery depends on the solar array for charging.",
            metadata={
                "document_title": "Power",
                "section_title": "EPS",
                "llm_entities": [{"name": "battery", "type": "COMPONENT"}],
                "spacy_entities": [{"text": "solar array", "label": "COMPONENT"}],
            },
        )

        context = PipelineContext(file_path=Path("dummy"), config=cfg)

        count = stage._extract_llm_relationships([chunk], context)

        assert count == 1
        llm_relationships = chunk.metadata.get("llm_relationships", [])
        assert len(llm_relationships) == 1
        rel = llm_relationships[0]
        assert rel["source"] == "battery"
        assert rel["target"] == "solar_array"
        assert rel["type"] == "DEPENDS_ON"
        assert len(fake_extractor.relationship_known_entities) == 2
