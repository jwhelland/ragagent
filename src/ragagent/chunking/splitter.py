from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    chunks: List[str] = []
    text = text.strip()
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

