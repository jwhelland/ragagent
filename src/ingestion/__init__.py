"""Ingestion module for document processing and chunking.

NOTE: Keep this module lightweight.

Importing Docling/Transformers can trigger heavy optional dependencies during test collection
(e.g., Docling -> Transformers -> TorchVision -> stdlib `lzma`), which may not be available
in all Python builds/environments.

We therefore expose public symbols via lazy imports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "HierarchicalChunker",
    "Chunk",
    "MetadataExtractor",
    "PDFParser",
    "ParsedDocument",
    "TextCleaner",
    "TextRewriter",
    "RewriteResult",
]


_LAZY_EXPORTS = {
    # chunker
    "HierarchicalChunker": ("src.ingestion.chunker", "HierarchicalChunker"),
    "Chunk": ("src.ingestion.chunker", "Chunk"),
    # metadata
    "MetadataExtractor": ("src.ingestion.metadata_extractor", "MetadataExtractor"),
    # pdf parser
    "PDFParser": ("src.ingestion.pdf_parser", "PDFParser"),
    "ParsedDocument": ("src.ingestion.pdf_parser", "ParsedDocument"),
    # cleaning
    "TextCleaner": ("src.ingestion.text_cleaner", "TextCleaner"),
    # rewriting
    "TextRewriter": ("src.ingestion.text_rewriter", "TextRewriter"),
    "RewriteResult": ("src.ingestion.text_rewriter", "RewriteResult"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk, HierarchicalChunker
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.pdf_parser import PDFParser, ParsedDocument
    from src.ingestion.text_cleaner import TextCleaner
    from src.ingestion.text_rewriter import RewriteResult, TextRewriter
