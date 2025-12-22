from __future__ import annotations

from typing import Dict

import pytest

from src.extraction.llm_extractor import (
    LLMExtractedEntity,
    LLMExtractedRelationship,
    LLMExtractor,
)
from src.utils.config import LLMConfig


def _noop_sleep(_: float) -> None:
    return None


def test_llm_extractor_parses_entity_response(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMConfig(provider="openai", retry_attempts=1, model="test-model")
    extractor = LLMExtractor(
        config=config, prompts_path="config/extraction_prompts.yaml", sleep_fn=_noop_sleep
    )

    calls: Dict[str, str] = {}

    def fake_openai(*, system: str, user: str) -> str:
        calls["system"] = system
        calls["user"] = user
        return """
        {
          "entities": [
            {
              "name": "solar array",
              "type": "COMPONENT",
              "description": "Generates power from sunlight",
              "aliases": ["solar panels"],
              "confidence": 0.88
            }
          ]
        }
        """

    monkeypatch.setattr(extractor, "_call_openai", fake_openai)

    chunk = {
        "content": "The system uses a solar array for primary power generation.",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "metadata": {"document_title": "Power System", "section_title": "EPS"},
    }

    entities = extractor.extract_entities(chunk)

    assert len(entities) == 1
    entity = entities[0]
    assert entity.name == "solar array"
    assert entity.type == "COMPONENT"
    assert entity.confidence == pytest.approx(0.88)
    assert entity.chunk_id == "chunk-1"
    assert "solar array" in calls.get("user", "")


def test_llm_extractor_parses_relationships(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMConfig(provider="anthropic", retry_attempts=1, model="test-model")
    extractor = LLMExtractor(
        config=config, prompts_path="config/extraction_prompts.yaml", sleep_fn=_noop_sleep
    )

    calls: Dict[str, str] = {}

    def fake_anthropic(*, system: str, user: str) -> str:
        calls["system"] = system
        calls["user"] = user
        return """
        Preamble text before JSON.
        {
          "relationships": [
            {
              "source": "battery",
              "type": "DEPENDS_ON",
              "target": "solar_array",
              "description": "Battery relies on solar array charging",
              "confidence": 0.92,
              "bidirectional": false
            }
          ]
        }
        """

    monkeypatch.setattr(extractor, "_call_anthropic", fake_anthropic)

    chunk = {
        "content": "The battery depends on the solar array for charging during daylight.",
        "chunk_id": "chunk-2",
        "document_id": "doc-1",
        "metadata": {"document_title": "Power System", "section_title": "EPS"},
    }

    known_entities = [
        LLMExtractedEntity(
            name="battery", type="COMPONENT", description="", aliases=[], confidence=0.9
        ),
        {"name": "solar_array", "type": "COMPONENT"},
    ]

    relationships = extractor.extract_relationships(chunk, known_entities=known_entities)

    assert len(relationships) == 1
    rel: LLMExtractedRelationship = relationships[0]
    assert rel.source == "battery"
    assert rel.target == "solar_array"
    assert rel.type == "DEPENDS_ON"
    assert rel.confidence == pytest.approx(0.92)
    assert "battery" in calls.get("user", "")
    assert "solar_array" in calls.get("user", "")


def test_llm_extractor_retries_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMConfig(provider="openai", retry_attempts=2, model="test-model")
    extractor = LLMExtractor(
        config=config, prompts_path="config/extraction_prompts.yaml", sleep_fn=_noop_sleep
    )

    attempts = {"count": 0}

    def flaky_openai(*, system: str, user: str) -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("Simulated timeout")
        return '{"entities": []}'

    monkeypatch.setattr(extractor, "_call_openai", flaky_openai)

    entities = extractor.extract_entities({"content": "short text"})

    assert attempts["count"] == 2
    assert entities == []
