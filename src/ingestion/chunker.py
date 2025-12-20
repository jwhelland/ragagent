"""Hierarchical document chunking module.

This module implements a 4-level hierarchical chunking strategy:
- Level 1: Document (entire document)
- Level 2: Section (major sections)
- Level 3: Subsection (detailed subsections)
- Level 4: Paragraph (individual paragraphs)

Chunks maintain parent-child relationships and preserve page numbers.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.pdf_parser import ParsedDocument, Section
from src.utils.config import ChunkingConfig


class Chunk(BaseModel):
    """Represents a document chunk at any hierarchy level."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    document_id: str
    level: int
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


class HierarchicalChunker:
    """Create hierarchical chunks from parsed documents.

    This class implements a 4-level chunking strategy that preserves document
    structure and maintains parent-child relationships. Token counting ensures
    chunks don't exceed size limits.

    Example:
        >>> chunker = HierarchicalChunker(config)
        >>> chunks = chunker.chunk_document(parsed_doc)
        >>> print(f"Created {len(chunks)} chunks")
        >>> level_2_chunks = [c for c in chunks if c.level == 2]
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """Initialize the hierarchical chunker.

        Args:
            config: Chunking configuration. If None, uses default settings.
        """
        self.config = config or ChunkingConfig()

        logger.info(
            f"Initialized HierarchicalChunker: strategy={self.config.strategy}, "
            f"max_tokens={self.config.max_tokens}, overlap={self.config.overlap_tokens}"
        )

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """Create hierarchical chunks from a parsed document.

        Args:
            parsed_doc: Parsed document with structure and content

        Returns:
            List of chunks at all hierarchy levels
        """
        if parsed_doc.error:
            logger.warning(f"Cannot chunk document with error: {parsed_doc.error}")
            return []

        logger.info(f"Chunking document: {parsed_doc.metadata.get('filename', 'unknown')}")

        all_chunks: List[Chunk] = []

        # Level 1: Document chunk (entire document)
        doc_chunk = self._create_document_chunk(parsed_doc)
        all_chunks.append(doc_chunk)

        # Level 2: Section chunks
        section_chunks = self._create_section_chunks(parsed_doc, doc_chunk.chunk_id)
        all_chunks.extend(section_chunks)

        # Update document chunk with child IDs (may be populated below)
        doc_chunk.child_chunk_ids = [c.chunk_id for c in section_chunks]

        # Fallback: if no structural sections were detected, chunk the whole document by paragraphs
        if not section_chunks:
            paragraph_chunks = self._create_paragraph_chunks(
                doc_chunk.content,
                parsed_doc.document_id,
                doc_chunk.chunk_id,
                doc_chunk.metadata,
            )
            all_chunks.extend(paragraph_chunks)
            doc_chunk.child_chunk_ids = [c.chunk_id for c in paragraph_chunks]
            logger.info(
                "No sections detected; chunked document into {} paragraph-level chunks",
                len(paragraph_chunks),
            )
            return all_chunks

        # Level 3: Subsection chunks
        for section_chunk in section_chunks:
            section = self._find_section_by_metadata(
                parsed_doc.structure.get("sections", []), section_chunk.metadata
            )
            if section and section.subsections:
                subsection_chunks = self._create_subsection_chunks(
                    section.subsections,
                    parsed_doc.document_id,
                    section_chunk.chunk_id,
                    section_chunk.metadata.get("hierarchy_path", ""),
                )
                all_chunks.extend(subsection_chunks)
                section_chunk.child_chunk_ids = [c.chunk_id for c in subsection_chunks]

                # Level 4: Paragraph chunks
                for subsection_chunk in subsection_chunks:
                    paragraph_chunks = self._create_paragraph_chunks(
                        subsection_chunk.content,
                        parsed_doc.document_id,
                        subsection_chunk.chunk_id,
                        subsection_chunk.metadata,
                    )
                    all_chunks.extend(paragraph_chunks)
                    subsection_chunk.child_chunk_ids = [c.chunk_id for c in paragraph_chunks]
            else:
                # No subsections, create paragraphs directly from section
                paragraph_chunks = self._create_paragraph_chunks(
                    section_chunk.content,
                    parsed_doc.document_id,
                    section_chunk.chunk_id,
                    section_chunk.metadata,
                )
                all_chunks.extend(paragraph_chunks)
                section_chunk.child_chunk_ids = [c.chunk_id for c in paragraph_chunks]

        logger.success(
            f"Created {len(all_chunks)} chunks: "
            f"L1={sum(1 for c in all_chunks if c.level == 1)}, "
            f"L2={sum(1 for c in all_chunks if c.level == 2)}, "
            f"L3={sum(1 for c in all_chunks if c.level == 3)}, "
            f"L4={sum(1 for c in all_chunks if c.level == 4)}"
        )

        return all_chunks

    def _create_document_chunk(self, parsed_doc: ParsedDocument) -> Chunk:
        """Create level 1 chunk (entire document).

        Args:
            parsed_doc: Parsed document

        Returns:
            Document-level chunk
        """
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=parsed_doc.document_id,
            level=1,
            parent_chunk_id=None,
            content=parsed_doc.raw_text,
            metadata={
                "document_title": parsed_doc.metadata.get("title", ""),
                "filename": parsed_doc.metadata.get("filename", ""),
                "page_count": parsed_doc.page_count,
                "page_numbers": list(range(1, parsed_doc.page_count + 1)),
                "hierarchy_path": "1",
                "has_tables": len(parsed_doc.tables) > 0,
                "has_figures": len(parsed_doc.figures) > 0,
                "table_count": len(parsed_doc.tables),
                "figure_count": len(parsed_doc.figures),
            },
            token_count=self._count_tokens(parsed_doc.raw_text),
        )

        return chunk

    def _create_section_chunks(self, parsed_doc: ParsedDocument, parent_id: str) -> List[Chunk]:
        """Create level 2 chunks (sections).

        Args:
            parsed_doc: Parsed document
            parent_id: ID of parent (document) chunk

        Returns:
            List of section-level chunks
        """
        sections = parsed_doc.structure.get("sections", [])
        chunks: List[Chunk] = []

        for i, section in enumerate(sections, 1):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=parsed_doc.document_id,
                level=2,
                parent_chunk_id=parent_id,
                content=section.content,
                metadata={
                    "document_title": parsed_doc.metadata.get("title", ""),
                    "section_title": section.title,
                    "page_numbers": list(range(section.start_page, section.end_page + 1)),
                    "hierarchy_path": section.hierarchy_path or str(i),
                    "has_tables": len(section.tables) > 0,
                    "has_figures": len(section.figures) > 0,
                },
                token_count=self._count_tokens(section.content),
            )
            chunks.append(chunk)

        return chunks

    def _create_subsection_chunks(
        self, subsections: List[Section], document_id: str, parent_id: str, parent_path: str
    ) -> List[Chunk]:
        """Create level 3 chunks (subsections).

        Args:
            subsections: List of subsection objects
            document_id: Document ID
            parent_id: ID of parent (section) chunk
            parent_path: Hierarchy path of parent

        Returns:
            List of subsection-level chunks
        """
        chunks: List[Chunk] = []

        for i, subsection in enumerate(subsections, 1):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                level=3,
                parent_chunk_id=parent_id,
                content=subsection.content,
                metadata={
                    "subsection_title": subsection.title,
                    "page_numbers": list(range(subsection.start_page, subsection.end_page + 1)),
                    "hierarchy_path": subsection.hierarchy_path or f"{parent_path}.{i}",
                    "has_tables": len(subsection.tables) > 0,
                    "has_figures": len(subsection.figures) > 0,
                },
                token_count=self._count_tokens(subsection.content),
            )
            chunks.append(chunk)

        return chunks

    def _create_paragraph_chunks(
        self, content: str, document_id: str, parent_id: str, parent_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create level 4 chunks (paragraphs).

        Args:
            content: Text content to split into paragraphs
            document_id: Document ID
            parent_id: ID of parent chunk
            parent_metadata: Metadata from parent chunk

        Returns:
            List of paragraph-level chunks
        """
        chunks: List[Chunk] = []

        # Split content into paragraphs (by double newline or more)
        paragraphs = re.split(r"\n\s*\n", content)

        # Filter out empty paragraphs and very short ones
        paragraphs = [
            p.strip()
            for p in paragraphs
            if p.strip() and len(p.strip()) >= self.config.min_chunk_size
        ]

        for i, paragraph in enumerate(paragraphs):
            token_count = self._count_tokens(paragraph)

            # If paragraph is too long, split it further
            if token_count > self.config.max_tokens:
                # Split by sentence boundaries
                sentences = self._split_sentences(paragraph)
                sub_chunks = self._create_sentence_chunks(
                    sentences, document_id, parent_id, parent_metadata, i
                )
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    level=4,
                    parent_chunk_id=parent_id,
                    content=paragraph,
                    metadata={
                        **parent_metadata,
                        "paragraph_index": i,
                        "hierarchy_path": f"{parent_metadata.get('hierarchy_path', '')}.{i + 1}",
                    },
                    token_count=token_count,
                )
                chunks.append(chunk)

        return chunks

    def _create_sentence_chunks(
        self,
        sentences: List[str],
        document_id: str,
        parent_id: str,
        parent_metadata: Dict[str, Any],
        paragraph_index: int,
    ) -> List[Chunk]:
        """Create paragraph chunks from sentences with overlap.

        Args:
            sentences: List of sentences
            document_id: Document ID
            parent_id: ID of parent chunk
            parent_metadata: Metadata from parent chunk
            paragraph_index: Index of the original paragraph

        Returns:
            List of paragraph chunks
        """
        chunks: List[Chunk] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If adding this sentence exceeds max tokens, save current chunk
            if current_tokens + sentence_tokens > self.config.max_tokens and current_chunk:
                chunk_content = " ".join(current_chunk)
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    level=4,
                    parent_chunk_id=parent_id,
                    content=chunk_content,
                    metadata={
                        **parent_metadata,
                        "paragraph_index": paragraph_index,
                        "sub_chunk_index": len(chunks),
                        "hierarchy_path": f"{parent_metadata.get('hierarchy_path', '')}.{paragraph_index + 1}.{len(chunks) + 1}",
                    },
                    token_count=current_tokens,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_size = min(len(current_chunk), 2)  # Keep last 1-2 sentences
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_tokens = sum(self._count_tokens(s) for s in current_chunk)

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk if not empty
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                level=4,
                parent_chunk_id=parent_id,
                content=chunk_content,
                metadata={
                    **parent_metadata,
                    "paragraph_index": paragraph_index,
                    "sub_chunk_index": len(chunks),
                    "hierarchy_path": f"{parent_metadata.get('hierarchy_path', '')}.{paragraph_index + 1}.{len(chunks) + 1}",
                },
                token_count=current_tokens,
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text.

        Uses simple word-based counting. For more accurate counting,
        could use tiktoken or transformers tokenizer.

        Args:
            text: Text to count tokens in

        Returns:
            Approximate token count
        """
        # Simple approximation: ~1.3 tokens per word
        words = len(text.split())
        return int(words * 1.3)

    def _find_section_by_metadata(
        self, sections: List[Section], metadata: Dict[str, Any]
    ) -> Optional[Section]:
        """Find a section by matching metadata.

        Args:
            sections: List of sections to search
            metadata: Metadata to match

        Returns:
            Matching section or None
        """
        section_title = metadata.get("section_title", "")
        hierarchy_path = metadata.get("hierarchy_path", "")

        for section in sections:
            if section.title == section_title or section.hierarchy_path == hierarchy_path:
                return section

        return None

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}

        return {
            "total_chunks": len(chunks),
            "by_level": {
                1: sum(1 for c in chunks if c.level == 1),
                2: sum(1 for c in chunks if c.level == 2),
                3: sum(1 for c in chunks if c.level == 3),
                4: sum(1 for c in chunks if c.level == 4),
            },
            "avg_tokens": sum(c.token_count for c in chunks) / len(chunks),
            "max_tokens": max(c.token_count for c in chunks),
            "min_tokens": min(c.token_count for c in chunks),
            "total_tokens": sum(c.token_count for c in chunks),
        }
