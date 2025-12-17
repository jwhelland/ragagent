"""Tests for PDF parsing/metadata that avoid heavy Docling dependencies.

Phase 1 Task 1.5 requires:
- Graceful error handling for missing/corrupt PDFs
- Metadata extraction (title/author/page count)

These tests focus on:
- [`PDFParser.parse_pdf()`](src/ingestion/pdf_parser.py:105) raising FileNotFoundError
  without importing/initializing Docling.
- [`MetadataExtractor.extract_metadata()`](src/ingestion/metadata_extractor.py:41) extracting
  basic metadata using pypdf.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.pdf_parser import PDFParser


def test_pdf_parser_missing_file_raises(tmp_path: Path) -> None:
    parser = PDFParser()
    missing = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        parser.parse_pdf(missing)


def test_metadata_extractor_reads_pdf_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata(
        {
            "/Title": "Test Title",
            "/Author": "Test Author",
            "/Subject": "Test Subject",
            "/Creator": "UnitTest",
        }
    )

    with pdf_path.open("wb") as f:
        writer.write(f)

    extractor = MetadataExtractor()
    md = extractor.extract_metadata(pdf_path)

    assert md["filename"] == "test.pdf"
    assert md["page_count"] == 1
    assert md["title"] == "Test Title"
    assert md["author"] == "Test Author"
