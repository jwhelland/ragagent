"""Unit tests for ingestion pipeline routing (Phase 1 Task 1.2).

Ensures .txt/.md files use lightweight parser instead of Docling.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.base import PipelineContext
from src.pipeline.stages.parsing import ParsingStage
from src.utils.config import Config


class _FailingPDFParser:
    def parse_pdf(self, *args, **kwargs):
        raise RuntimeError("PDF parser should not be called for text files")


def test_process_document_txt_bypasses_pdf_parser(tmp_path: Path) -> None:
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("Hello from text.\n\nSecond paragraph.\n", encoding="utf-8")

    cfg = Config.from_yaml("config/config.yaml")

    stage = ParsingStage(cfg)
    # Inject failing parser
    stage.pdf_parser = _FailingPDFParser()

    context = PipelineContext(file_path=doc_path, config=cfg)

    # Should not raise RuntimeError
    stage.run(context)

    assert context.parsed_document is not None
    assert context.parsed_document.raw_text.startswith("Hello from text")
