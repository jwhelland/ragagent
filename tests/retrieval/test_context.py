import pytest

from ragagent.retrieval.context import ContextAssembler
from ragagent.retrieval.models import RetrievedChunk


def test_build_creates_context_bundle():
    """build() creates ContextBundle with tagged chunks."""
    assembler = ContextAssembler(max_chunks=5, max_chars_per_chunk=100)
    vector_chunks = [
        RetrievedChunk(
            chunk_id="v1", doc_id="doc1", text="Vector content", score=0.9, page=1
        ),
    ]
    graph_chunks = []

    result = assembler.build(vector_chunks, graph_chunks)

    assert len(result.chunks) == 1
    assert result.chunks[0].tag == "S1"
    assert result.chunks[0].snippet == "Vector content"
    assert "S1" in result.citation_map
    assert result.citation_map["S1"]["chunk_id"] == "v1"
    assert result.citation_map["S1"]["doc_id"] == "doc1"
    assert result.citation_map["S1"]["page"] == 1


def test_build_merges_vector_and_graph_chunks():
    """build() merges vector and graph results, deduplicating by chunk_id."""
    assembler = ContextAssembler(max_chunks=10)
    vector_chunks = [
        RetrievedChunk(chunk_id="v1", doc_id="doc1", text="Vector 1", score=0.9),
        RetrievedChunk(chunk_id="shared", doc_id="doc2", text="Shared", score=0.8),
    ]
    graph_chunks = [
        RetrievedChunk(
            chunk_id="shared", doc_id="doc2", text="Shared", score=0.7, origin="graph"
        ),
        RetrievedChunk(
            chunk_id="g1", doc_id="doc3", text="Graph 1", score=0.6, origin="graph"
        ),
    ]

    result = assembler.build(vector_chunks, graph_chunks)

    # Should have 3 unique chunks (shared appears once)
    assert len(result.chunks) == 3
    chunk_ids = {c.chunk.chunk_id for c in result.chunks}
    assert chunk_ids == {"v1", "shared", "g1"}


def test_build_respects_max_chunks_limit():
    """build() limits output to max_chunks."""
    assembler = ContextAssembler(max_chunks=2)
    vector_chunks = [
        RetrievedChunk(
            chunk_id=f"v{i}", doc_id="doc1", text=f"Text {i}", score=1.0 - i * 0.1
        )
        for i in range(5)
    ]

    result = assembler.build(vector_chunks, [])

    assert len(result.chunks) == 2
    assert result.chunks[0].tag == "S1"
    assert result.chunks[1].tag == "S2"


def test_build_raises_on_empty_context():
    """build() raises ValueError when no chunks provided."""
    assembler = ContextAssembler()

    with pytest.raises(ValueError, match="no_context_available"):
        assembler.build([], [])


def test_build_formatted_output_structure():
    """build() creates formatted string with tags and descriptors."""
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="doc1", text="Content one", score=0.9, page=5
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="doc2",
            text="Content two",
            score=0.8,
            page=10,
            table_id="table_3",
        ),
    ]

    result = assembler.build(chunks, [])

    assert "[S1] Doc doc1 (p.5)\nContent one" in result.formatted
    assert "[S2] Doc doc2 (p.10, table table_3)\nContent two" in result.formatted
    assert "\n\n" in result.formatted  # Double newline separator


def test_build_includes_citation_metadata():
    """build() populates citation_map with all metadata."""
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="doc1",
            text="Test",
            score=0.9,
            page=3,
            table_id="t1",
            origin="vector",
        ),
    ]

    result = assembler.build(chunks, [])

    citation = result.citation_map["S1"]
    assert citation["chunk_id"] == "c1"
    assert citation["doc_id"] == "doc1"
    assert citation["page"] == 3
    assert citation["table_id"] == "t1"
    assert citation["origin"] == "vector"


def test_merge_deduplicates_by_chunk_id():
    """_merge() removes duplicates based on chunk_id."""
    assembler = ContextAssembler()
    vector = [RetrievedChunk(chunk_id="dup", doc_id="doc1", text="A", score=0.9)]
    graph = [
        RetrievedChunk(
            chunk_id="dup", doc_id="doc1", text="A", score=0.8, origin="graph"
        )
    ]

    merged = assembler._merge(vector, graph)

    assert len(merged) == 1
    assert merged[0].chunk_id == "dup"


def test_merge_sorts_by_score_descending():
    """_merge() sorts chunks by score (highest first)."""
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(chunk_id="low", doc_id="doc1", text="Low", score=0.5),
        RetrievedChunk(chunk_id="high", doc_id="doc2", text="High", score=0.9),
        RetrievedChunk(chunk_id="mid", doc_id="doc3", text="Mid", score=0.7),
    ]

    merged = assembler._merge(chunks, [])

    assert merged[0].chunk_id == "high"
    assert merged[1].chunk_id == "mid"
    assert merged[2].chunk_id == "low"


def test_merge_prioritizes_vector_over_graph():
    """_merge() processes vector chunks before graph chunks."""
    assembler = ContextAssembler()
    vector = [RetrievedChunk(chunk_id="v1", doc_id="doc1", text="Vec", score=0.5)]
    graph = [
        RetrievedChunk(
            chunk_id="g1", doc_id="doc2", text="Graph", score=0.9, origin="graph"
        )
    ]

    merged = assembler._merge(vector, graph)

    # Vector chunk appears first despite lower score
    assert merged[0].chunk_id == "v1"
    assert merged[1].chunk_id == "g1"


def test_compress_text_preserves_short_text():
    """_compress_text() returns text unchanged if within max_chars."""
    assembler = ContextAssembler(max_chars_per_chunk=100)
    text = "Short text."

    result = assembler._compress_text(text)

    assert result == "Short text."


def test_compress_text_truncates_long_text():
    """_compress_text() truncates text exceeding max_chars."""
    assembler = ContextAssembler(max_chars_per_chunk=20)
    text = "This is a very long text that exceeds the character limit."

    result = assembler._compress_text(text)

    assert len(result) <= 23  # max + " …"
    assert result.endswith(" …")
    assert not result.endswith("  …")  # Should break at word boundary


def test_compress_text_breaks_at_word_boundary():
    """_compress_text() breaks at last space before limit."""
    assembler = ContextAssembler(max_chars_per_chunk=15)
    text = "word1 word2 word3 word4"

    result = assembler._compress_text(text)

    # Should break after "word1 word2" (11 chars) not mid-word
    assert result == "word1 word2 …"


def test_compress_text_handles_empty_input():
    """_compress_text() handles empty or whitespace text."""
    assembler = ContextAssembler()

    assert assembler._compress_text("") == ""
    assert assembler._compress_text("   ") == ""
    assert assembler._compress_text(None) == ""


def test_format_descriptor_with_page():
    """_format_descriptor() formats chunk with page number."""
    assembler = ContextAssembler()
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="doc123", text="Text", score=0.9, page=42
    )

    result = assembler._format_descriptor(chunk)

    assert result == "Doc doc123 (p.42)"


def test_format_descriptor_with_table():
    """_format_descriptor() includes table_id when present."""
    assembler = ContextAssembler()
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="doc1", text="Text", score=0.9, page=5, table_id="table_7"
    )

    result = assembler._format_descriptor(chunk)

    assert result == "Doc doc1 (p.5, table table_7)"


def test_format_descriptor_missing_page():
    """_format_descriptor() handles missing page number."""
    assembler = ContextAssembler()
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="doc1", text="Text", score=0.9, page=None
    )

    result = assembler._format_descriptor(chunk)

    assert result == "Doc doc1 (page?)"


def test_format_descriptor_with_table_no_page():
    """_format_descriptor() shows table even without page."""
    assembler = ContextAssembler()
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="Text",
        score=0.9,
        page=None,
        table_id="table_2",
    )

    result = assembler._format_descriptor(chunk)

    assert result == "Doc doc1 (page?, table table_2)"


def test_build_handles_unicode_text():
    """build() correctly handles Unicode characters in text."""
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="doc1", text="Hello 世界 🌍", score=0.9, page=1
        ),
    ]

    result = assembler.build(chunks, [])

    assert "Hello 世界 🌍" in result.formatted
    assert result.chunks[0].snippet == "Hello 世界 🌍"


def test_build_context_bundle_as_prompt_section():
    """ContextBundle.as_prompt_section() returns formatted string."""
    assembler = ContextAssembler()
    chunks = [
        RetrievedChunk(chunk_id="c1", doc_id="doc1", text="Test", score=0.9, page=1)
    ]

    result = assembler.build(chunks, [])

    assert result.as_prompt_section() == result.formatted
