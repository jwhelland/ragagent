from __future__ import annotations

from typing import Iterable, List, Sequence

from ..logging_setup import get_logger
from .models import ContextBundle, ContextChunk, RetrievedChunk


logger = get_logger(__name__)


class ContextAssembler:
    """
    Assemble a compact, citation-friendly context window from retrieved chunks.

    The assembler takes vector and graph retrieval results, merges and
    de-duplicates them, applies length limits, and produces:
    - A list of ``ContextChunk`` objects (tags + snippets + raw chunks).
    - A single formatted string with section headers like ``[S1] ...`` that
      is fed directly to the LLM.
    - A citation map from section tags (``S1``, ``S2``, ...) back to the
      underlying chunk identifiers and provenance.

    This is the final step before calling the agent/LLM.
    """
    def __init__(
        self,
        *,
        max_chunks: int = 8,
        max_chars_per_chunk: int = 900,
    ) -> None:
        """
        Initialize a ContextAssembler.

        Args:
            max_chunks: Maximum number of chunks to include in the final
                assembled context, across all retrievers.
            max_chars_per_chunk: Maximum number of characters to keep from
                each chunk's text when building snippets for the LLM.
        """
        self._max_chunks = max_chunks
        self._max_chars = max_chars_per_chunk

    def build(
        self,
        vector_chunks: Sequence[RetrievedChunk],
        graph_chunks: Sequence[RetrievedChunk],
    ) -> ContextBundle:
        """
        Merge vector and graph chunks into a formatted LLM context bundle.

        Steps:
        1. Merge and de-duplicate chunks, preserving vector order and then
           appending graph results.
        2. Trim to at most ``max_chunks`` items.
        3. For each chunk, create a short snippet and an ``[S#]`` descriptor.
        4. Build a citation map so model outputs can be grounded in specific
           documents, pages, and tables.

        Args:
            vector_chunks: Primary chunks returned from the vector retriever.
            graph_chunks: Secondary chunks from the graph retriever.

        Returns:
            A ``ContextBundle`` containing formatted context text, per-section
            metadata, and a citation lookup table.

        Raises:
            ValueError: If no chunks are available after merging.
        """
        ordered = self._merge(vector_chunks, graph_chunks)
        if not ordered:
            raise ValueError("no_context_available")

        selected = ordered[: self._max_chunks]
        context_chunks: List[ContextChunk] = []
        sections: List[str] = []
        citation_map = {}

        for idx, chunk in enumerate(selected, start=1):
            tag = f"S{idx}"
            snippet = self._compress_text(chunk.text)
            descriptor = self._format_descriptor(chunk)
            sections.append(f"[{tag}] {descriptor}\n{snippet}")
            citation_map[tag] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "table_id": chunk.table_id,
                "origin": chunk.origin,
            }
            context_chunks.append(ContextChunk(tag=tag, chunk=chunk, snippet=snippet))

        formatted = "\n\n".join(sections)
        return ContextBundle(chunks=context_chunks, formatted=formatted, citation_map=citation_map)

    def _merge(
        self,
        vector_chunks: Sequence[RetrievedChunk],
        graph_chunks: Sequence[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Merge and de-duplicate vector and graph chunks into a single list.

        Vector chunks are added first (sorted by score), followed by graph
        chunks (also sorted by score). Chunks with duplicate ``chunk_id``
        are only kept once, preserving the first occurrence.

        Args:
            vector_chunks: Chunks produced by the vector retriever.
            graph_chunks: Chunks produced by the graph retriever.

        Returns:
            An ordered list of unique ``RetrievedChunk`` instances.
        """
        ordered: List[RetrievedChunk] = []
        seen: set[str] = set()

        def extend(chunks: Iterable[RetrievedChunk], *, sort_desc: bool = True):
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=sort_desc)
            for chunk in sorted_chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                ordered.append(chunk)

        extend(vector_chunks)
        extend(graph_chunks)
        return ordered

    def _compress_text(self, text: str) -> str:
        """
        Truncate a chunk's text to the configured character limit.

        The method tries to avoid cutting words in half by backing up to
        the last whitespace boundary before appending an ellipsis when
        truncation is required.
        """
        text = (text or "").strip()
        if len(text) <= self._max_chars:
            return text
        snippet = text[: self._max_chars].rsplit(" ", 1)[0]
        return snippet + " …"

    def _format_descriptor(self, chunk: RetrievedChunk) -> str:
        """
        Build a human-readable descriptor for a chunk's provenance.

        The descriptor is used in the formatted context header line
        (for example, ``[S1] Doc <id> (p.3, table T1)``) and is intended
        to give enough information for reliable citation references.
        """
        page_part = f"p.{chunk.page}" if chunk.page is not None else "page?"
        table_part = f", table {chunk.table_id}" if chunk.table_id else ""
        return f"Doc {chunk.doc_id} ({page_part}{table_part})"
