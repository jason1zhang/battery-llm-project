"""
Unit tests for data pipeline components.
"""

import os
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.data_pipeline.loader import DocumentLoader, DirectoryLoader
from src.data_pipeline.chunker import (
    create_chunker,
    RecursiveChunker,
    SemanticChunker,
)
from src.data_pipeline.embedder import create_embedder


class TestDocumentLoader:
    """Test document loaders."""

    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")

        loader = DocumentLoader(str(test_file))
        docs = loader.load()

        assert len(docs) > 0
        assert "test" in docs[0].page_content.lower()

    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nThis is a test.")

        loader = DocumentLoader(str(test_file))
        docs = loader.load()

        assert len(docs) > 0

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader = DocumentLoader("/nonexistent/file.txt")
            loader.load()


class TestTextChunker:
    """Test text chunkers."""

    def test_recursive_chunker(self):
        """Test recursive chunker."""
        docs = [
            Document(
                page_content="This is first paragraph.\n\nThis is second paragraph.",
                metadata={"source": "test"},
            )
        ]

        chunker = create_chunker(chunk_size=20, chunk_overlap=5, strategy="recursive")
        chunks = chunker.split_documents(docs)

        assert len(chunks) > 0

    def test_semantic_chunker(self):
        """Test semantic chunker."""
        docs = [
            Document(
                page_content="First sentence. Second sentence. Third sentence.",
                metadata={"source": "test"},
            )
        ]

        chunker = create_chunker(chunk_size=50, chunk_overlap=10, strategy="semantic")
        chunks = chunker.split_documents(docs)

        assert len(chunks) > 0

    def test_chunker_preserves_metadata(self):
        """Test that chunker preserves document metadata."""
        docs = [
            Document(
                page_content="This is a test document with enough content to be chunked.",
                metadata={"source": "test_file", "page": 1},
            )
        ]

        chunker = create_chunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.split_documents(docs)

        assert all("source" in chunk.metadata for chunk in chunks)


class TestEmbedder:
    """Test embedding generation."""

    def test_embedder_creation(self):
        """Test embedder can be created."""
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        assert embedder is not None
        assert embedder.embeddings is not None

    def test_embed_query(self):
        """Test embedding a single query."""
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        embedding = embedder.embed_query("test query")

        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        docs = [
            Document(page_content="First document"),
            Document(page_content="Second document"),
        ]

        embeddings = embedder.embed_documents(docs)

        assert len(embeddings) == 2
        assert all(len(e) > 0 for e in embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
