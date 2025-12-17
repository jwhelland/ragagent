"""Tests for TextCleaner (Phase 1 Task 1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ingestion.text_cleaner import TextCleaner
from src.utils.config import TextCleaningConfig


def _write_patterns(path: Path) -> None:
    patterns = {
        "headers": {
            "enabled": True,
            "patterns": [r"(?i)^CONFIDENTIAL.*$"],
        },
        "footers": {
            "enabled": True,
            "patterns": [r"(?i)^copyright.*$"],
        },
        "page_numbers": {
            "enabled": True,
            "patterns": [r"^\s*\d+\s*$"],
        },
        "watermarks": {
            "enabled": True,
            "patterns": [r"(?i)DRAFT"],
        },
        "noise": {
            "enabled": True,
            "patterns": [r"^\s*[\|_-]{5,}\s*$"],
        },
        "preserve": {
            "enabled": True,
            "patterns": [
                r"(?s)```.*?```",  # code blocks
                r"(?s)`[^`]+`",  # inline code
                r"\$.*?\$",  # equation-ish
                r"\\\(.*?\\\)",  # latex-ish
                r"\b[A-Z]{2,}\b",  # acronyms
            ],
        },
        "ocr_corrections": {
            "enabled": True,
            "patterns": [
                {"pattern": r"\bO\b", "replacement": "0", "context": "numeric"},
            ],
        },
    }
    path.write_text(yaml.safe_dump(patterns), encoding="utf-8")


def test_cleaner_removes_headers_footers_page_numbers(tmp_path: Path) -> None:
    patterns_file = tmp_path / "patterns.yaml"
    _write_patterns(patterns_file)

    cfg = TextCleaningConfig(
        enabled=True,
        patterns_file=str(patterns_file),
        remove_headers=True,
        remove_footers=True,
        remove_page_numbers=True,
        normalize_whitespace=True,
        preserve_code_blocks=True,
        preserve_equations=True,
        preserve_technical_terms=True,
    )
    cleaner = TextCleaner(cfg)

    raw = "\n".join(
        [
            "CONFIDENTIAL - INTERNAL USE ONLY",
            "Some important content here.",
            "-----",
            "12",
            "copyright 2025 ACME",
        ]
    )

    cleaned = cleaner.clean(raw)

    assert "CONFIDENTIAL" not in cleaned
    assert "copyright" not in cleaned.lower()
    assert "\n12\n" not in f"\n{cleaned}\n"
    assert "Some important content here." in cleaned


def test_cleaner_preserves_code_blocks(tmp_path: Path) -> None:
    patterns_file = tmp_path / "patterns.yaml"
    _write_patterns(patterns_file)

    cfg = TextCleaningConfig(
        enabled=True,
        patterns_file=str(patterns_file),
        remove_headers=True,
        remove_footers=True,
        remove_page_numbers=True,
        normalize_whitespace=True,
        preserve_code_blocks=True,
        preserve_equations=True,
        preserve_technical_terms=True,
    )
    cleaner = TextCleaner(cfg)

    raw = "\n".join(
        [
            "CONFIDENTIAL",
            "```",
            "CMD_SET_MODE 0x1A2B",
            "```",
            "copyright 2025",
        ]
    )

    cleaned = cleaner.clean(raw)

    assert "CONFIDENTIAL" not in cleaned
    assert "copyright" not in cleaned.lower()
    assert "CMD_SET_MODE 0x1A2B" in cleaned
    assert "```" in cleaned


def test_cleaner_applies_ocr_numeric_correction(tmp_path: Path) -> None:
    patterns_file = tmp_path / "patterns.yaml"
    _write_patterns(patterns_file)

    cfg = TextCleaningConfig(
        enabled=True,
        patterns_file=str(patterns_file),
        remove_headers=False,
        remove_footers=False,
        remove_page_numbers=False,
        normalize_whitespace=True,
        preserve_code_blocks=True,
        preserve_equations=True,
        preserve_technical_terms=True,
    )
    cleaner = TextCleaner(cfg)

    raw = "Voltage is 2 O 5 V"  # O should become 0 within numeric context
    cleaned = cleaner.clean(raw)

    assert "2 0 5" in cleaned
