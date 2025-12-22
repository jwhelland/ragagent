"""Tests for AcronymResolver (Phase 3 Task 3.3)."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingestion.chunker import Chunk
from src.normalization.acronym_resolver import AcronymResolver


def test_extracts_definitions_and_builds_dictionary() -> None:
    resolver = AcronymResolver()

    text = (
        "The Telemetry and Command (T&C) subsystem processes commands. "
        "Attitude Control System (ACS) maintains pointing."
    )

    definitions = resolver.extract_definitions(text)

    assert ("T&C", "Telemetry and Command") in {
        (item.acronym, item.expansion) for item in definitions
    }

    dictionary = resolver.build_dictionary_from_texts([text])
    assert "T&C" in dictionary
    assert dictionary["T&C"].expansions["Telemetry and Command"] == 1


def test_context_disambiguation_prefers_matching_context() -> None:
    resolver = AcronymResolver()
    resolver.build_dictionary_from_texts(
        [
            "The Power Distribution Unit (PDU) conditions system power buses.",
            "Protocol Data Unit (PDU) defines a network layer packet structure.",
        ]
    )

    network_context = "Each protocol data unit travels through the network stack."
    power_context = "The system replaced the old PDU in the power bay."

    network_resolution = resolver.resolve("PDU", context=network_context)
    power_resolution = resolver.resolve("PDU", context=power_context)

    assert network_resolution
    assert network_resolution.expansion == "Protocol Data Unit"
    assert power_resolution
    assert power_resolution.expansion == "Power Distribution Unit"


def test_overrides_take_priority(tmp_path: Path) -> None:
    override_file = tmp_path / "overrides.yaml"
    override_file.write_text("PDU: Power Distribution Unit\n", encoding="utf-8")

    resolver = AcronymResolver(overrides_path=override_file)
    resolver.build_dictionary_from_texts(["Protocol Data Unit (PDU) is defined in the spec."])

    resolution = resolver.resolve("PDU", context="power bus monitoring")

    assert resolution
    assert resolution.method == "override"
    assert resolution.expansion == "Power Distribution Unit"
    assert resolution.confidence >= 0.95


def test_resolve_in_text_uses_chunk_context() -> None:
    resolver = AcronymResolver()
    chunk = Chunk(
        chunk_id="c1",
        document_id="doc1",
        level=3,
        parent_chunk_id=None,
        content="Natural Language Processing (NLP) improves language understanding.",
        metadata={},
        token_count=0,
    )

    resolver.build_dictionary_from_chunks([chunk])
    resolutions = resolver.resolve_in_text(chunk.content)

    assert any(
        res.acronym == "NLP" and res.expansion == "Natural Language Processing"
        for res in resolutions
    )


def test_store_mappings_writes_yaml(tmp_path: Path) -> None:
    storage_path = tmp_path / "acronyms.yaml"
    resolver = AcronymResolver(storage_path=storage_path)
    resolver.build_dictionary_from_texts(["Telemetry and Command (T&C) is critical."])

    output_path = resolver.store_mappings()
    assert output_path.exists()

    saved = yaml.safe_load(output_path.read_text())
    assert saved["dictionary"]["T&C"]["expansions"]["Telemetry and Command"] == 1


def test_supports_two_letter_acronyms() -> None:
    resolver = AcronymResolver()
    resolver.build_dictionary_from_texts(["Artificial Intelligence (AI) is advancing rapidly."])

    resolution = resolver.resolve("AI", context="intelligence systems")

    assert resolution
    assert resolution.expansion == "Artificial Intelligence"


def test_load_mappings_round_trip(tmp_path: Path) -> None:
    storage_path = tmp_path / "acronyms.yaml"
    writer = AcronymResolver(storage_path=storage_path)
    writer.build_dictionary_from_texts(["Telemetry and Command (T&C) is critical."])
    writer.store_mappings()

    reader = AcronymResolver(storage_path=storage_path)
    reader.load_mappings()

    resolution = reader.resolve("T&C", context="command handling")
    assert resolution
    assert resolution.expansion == "Telemetry and Command"
