import json
from pathlib import Path

import spacy

from src.extraction.spacy_extractor import SpacyExtractor
from src.utils.config import SpacyConfig


def _write_patterns(tmp_path: Path) -> Path:
    patterns = [
        {"label": "SYSTEM", "pattern": [{"LOWER": "system"}, {"LOWER": "bus"}]},
        {"label": "COMPONENT", "pattern": [{"LOWER": "motor"}, {"LOWER": "controller"}]},
    ]
    path = tmp_path / "patterns.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in patterns))
    return path


def _build_nlp() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def test_spacy_extractor_with_patterns_and_domain_terms(tmp_path: Path) -> None:
    pattern_file = _write_patterns(tmp_path)
    config = SpacyConfig(
        model="en_dummy",
        custom_patterns=str(pattern_file),
        confidence_threshold=0.0,
    )
    extractor = SpacyExtractor(config=config, nlp=_build_nlp())

    chunk = {
        "content": "The system bus uses a motor controller in the power subsystem.",
        "chunk_id": "c1",
        "document_id": "doc1",
    }

    results = extractor.extract_from_chunks([chunk])
    entities = results.get("c1", [])

    assert {e.label for e in entities} == {"SYSTEM", "SUBSYSTEM", "COMPONENT"}
    assert {e.text for e in entities} == {"system bus", "power subsystem", "motor controller"}
    assert all(0.0 <= e.confidence <= 1.0 for e in entities)
    assert all(e.sentence for e in entities)
    sources = {e.metadata.get("source") if e.metadata else None for e in entities}
    assert "domain_matcher" in sources


def test_spacy_extractor_respects_confidence_threshold(tmp_path: Path) -> None:
    pattern_file = _write_patterns(tmp_path)
    config = SpacyConfig(
        model="en_dummy",
        custom_patterns=str(pattern_file),
        confidence_threshold=0.95,
    )
    extractor = SpacyExtractor(config=config, nlp=_build_nlp())

    chunk = {
        "content": "The system bus uses a motor controller in the power subsystem.",
        "chunk_id": "c1",
        "document_id": "doc1",
    }

    results = extractor.extract_from_chunks([chunk])
    assert results.get("c1") == []
