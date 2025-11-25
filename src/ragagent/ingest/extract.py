from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image  # type: ignore
import io

from ..config import settings

try:
    # Docling optional import; used when available for layout-aware parsing
    from docling.document_converter import DocumentConverter  # type: ignore
except Exception:  # noqa: BLE001
    DocumentConverter = None  # type: ignore

try:
    import easyocr  # type: ignore
except Exception:  # noqa: BLE001
    easyocr = None  # type: ignore


def ocr_page(image_bytes: bytes, languages: str) -> str:
    if easyocr is None:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return ""
    reader = easyocr.Reader(languages.split(","), gpu=False)
    result = reader.readtext(np.array(img), detail=0, paragraph=True)
    return "\n".join(result)


def _extract_with_docling(pdf_path: Path) -> Dict:
    if DocumentConverter is None:
        raise RuntimeError("Docling not available")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    # Best-effort conversion to a page-oriented structure with tables
    pages: List[Dict] = []
    all_empty = True
    for p in result.document.pages:  # type: ignore[attr-defined]
        text = getattr(p, "text", "") or ""
        tables = []
        for ti, t in enumerate(getattr(p, "tables", []) or []):
            # Attempt to render table to Markdown; fallback to cell join
            table_md = None
            try:
                table_md = t.to_markdown()  # type: ignore[attr-defined]
            except Exception:
                try:
                    rows = []
                    for row in t.cells:  # type: ignore[attr-defined]
                        rows.append(" | ".join(c.text for c in row))
                    table_md = "\n".join(rows)
                except Exception:
                    table_md = ""
            tables.append({
                "table_id": f"{pdf_path.stem}:p{p.number}:t{ti}",  # type: ignore[attr-defined]
                "markdown": table_md,
            })
        if text.strip() or tables:
            all_empty = False
        pages.append({
            "page_number": getattr(p, "number", None) or len(pages) + 1,
            "text": text,
            "tables": tables,
        })
    if all_empty:
        # If Docling yields no text/tables at all (e.g., pure image PDF
        # where we are not tapping its OCR output), fall back to PyMuPDF+OCR.
        raise RuntimeError("Docling produced empty pages; falling back to PyMuPDF")
    return {"file": str(pdf_path), "pages": pages}


def _extract_with_pymupdf(pdf_path: Path) -> Dict:
    doc = fitz.open(pdf_path)
    pages: List[Dict] = []
    for i, page in enumerate(doc):
        number = i + 1
        text = page.get_text("text") or ""
        if not text.strip():
            pix = page.get_pixmap(dpi=200)
            text = ocr_page(pix.tobytes("png"), settings.ocr_languages)
        pages.append({
            "page_number": number,
            "text": text,
            "tables": [],
        })
    return {"file": str(pdf_path), "pages": pages}


def extract_pdf(pdf_path: Path) -> Dict:
    try:
        return _extract_with_docling(pdf_path)
    except Exception:
        return _extract_with_pymupdf(pdf_path)
