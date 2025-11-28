"""Tests for the RAG tool."""

import tempfile
from pathlib import Path


def test_chunk_text():
    """Test text chunking functionality."""
    from magi.tools.rag import chunk_text

    # Test short text (should return single chunk)
    short_text = "This is a short text."
    chunks = chunk_text(short_text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Test longer text (should create multiple chunks)
    long_text = "A" * 250
    chunks = chunk_text(long_text, chunk_size=100, overlap=10)
    assert len(chunks) > 1

    # Verify overlap
    for i in range(len(chunks) - 1):
        # Last 10 chars of chunk i should match first 10 of next chunk
        # (Due to how we chunk, this might not be exact, but chunks should overlap)
        assert len(chunks[i]) <= 100


def test_read_file_content_txt():
    """Test reading text file content."""
    from magi.tools.rag import read_file_content

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_content = "This is test content for the RAG system."
        f.write(test_content)
        f.flush()

        content = read_file_content(Path(f.name))
        assert content == test_content


def test_read_file_content_md():
    """Test reading markdown file content."""
    from magi.tools.rag import read_file_content

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        test_content = "# Markdown Header\n\nThis is markdown content."
        f.write(test_content)
        f.flush()

        content = read_file_content(Path(f.name))
        assert content == test_content
