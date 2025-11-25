from ragagent.chunking.splitter import chunk_text


def test_chunk_text_empty_string():
    """Empty input returns empty list."""
    result = chunk_text("")
    assert result == []


def test_chunk_text_whitespace_only():
    """Whitespace-only input returns empty list."""
    result = chunk_text("   \n\t  ")
    assert result == []


def test_chunk_text_single_chunk_fits():
    """Text shorter than max_chars returns single chunk."""
    text = "Short text that fits in one chunk."
    result = chunk_text(text, max_chars=100)
    assert len(result) == 1
    assert result[0] == text


def test_chunk_text_exact_size():
    """Text exactly at max_chars returns single chunk."""
    text = "a" * 100
    result = chunk_text(text, max_chars=100)
    assert len(result) == 1
    assert result[0] == text


def test_chunk_text_creates_multiple_chunks():
    """Text longer than max_chars creates multiple chunks."""
    text = "a" * 250
    result = chunk_text(text, max_chars=100, overlap=20)
    # Chunks: 0-100, 80-180, 160-250 = 3 chunks
    assert len(result) == 3
    assert len(result[0]) == 100
    assert len(result[1]) == 100
    assert len(result[2]) == 90  # Last chunk goes to end


def test_chunk_text_overlap_applied():
    """Overlap creates overlapping content between chunks."""
    text = "0123456789" * 30  # 300 chars
    result = chunk_text(text, max_chars=100, overlap=20)

    # Check chunks overlap correctly
    assert len(result) > 1
    # Second chunk should start 80 chars into first chunk (100-20)
    assert result[0][80:100] == result[1][:20]


def test_chunk_text_no_overlap():
    """Setting overlap=0 creates no overlap."""
    text = "a" * 300
    result = chunk_text(text, max_chars=100, overlap=0)
    assert len(result) == 3
    assert result[0] == "a" * 100
    assert result[1] == "a" * 100
    assert result[2] == "a" * 100


def test_chunk_text_strips_input():
    """Leading/trailing whitespace is stripped."""
    text = "  \n  test content  \n  "
    result = chunk_text(text, max_chars=100)
    assert len(result) == 1
    assert result[0] == "test content"


def test_chunk_text_default_params():
    """Default parameters work correctly."""
    text = "a" * 3000
    result = chunk_text(text)
    # With max_chars=1200, overlap=100, should create multiple chunks
    assert len(result) > 1
    assert all(len(chunk) <= 1200 for chunk in result)


def test_chunk_text_last_chunk_no_overlap():
    """Last chunk doesn't have overlap applied."""
    text = "a" * 250
    result = chunk_text(text, max_chars=100, overlap=20)
    # Last chunk should go to end of text
    assert result[-1][-1] == "a"
    assert len("".join(result)) > len(text)  # Due to overlap


def test_chunk_text_unicode_handling():
    """Unicode characters are handled correctly."""
    text = "Hello 世界 " * 200
    result = chunk_text(text, max_chars=100, overlap=10)
    assert len(result) > 1
    # All chunks should be valid strings
    assert all(isinstance(chunk, str) for chunk in result)
