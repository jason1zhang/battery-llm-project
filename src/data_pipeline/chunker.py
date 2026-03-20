"""
Text chunking strategies for document processing.
"""

from typing import List, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_core.documents import Document


class TextChunker:
    """Base class for text chunking."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        raise NotImplementedError


class RecursiveChunker(TextChunker):
    """Recursive character text splitter with customizable separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n",
            "\n",
            ".",
            "?",
            "!",
            ";",
            ",",
            " ",
            "",
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        return splitter.split_documents(documents)


class MarkdownChunker(TextChunker):
    """Split documents by Markdown headers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents by Markdown structure."""
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)


class SemanticChunker(TextChunker):
    """Semantic chunking based on sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents at semantic boundaries (sentences/paragraphs)."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",  # Paragraph breaks
                "\n\n",    # Line breaks
                ". ",      # Sentence ends
                "? ",
                "! ",
                "; ",
                ", ",
                "",
            ],
            length_function=len,
        )
        return splitter.split_documents(documents)


class HybridChunker(TextChunker):
    """Combine multiple chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "semantic",
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy = strategy

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using selected strategy."""
        if self.strategy == "recursive":
            chunker = RecursiveChunker(
                self.chunk_size,
                self.chunk_overlap,
            )
        elif self.strategy == "markdown":
            chunker = MarkdownChunker(
                self.chunk_size,
                self.chunk_overlap,
            )
        elif self.strategy == "semantic":
            chunker = SemanticChunker(
                self.chunk_size,
                self.chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        return chunker.split_documents(documents)


def create_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str = "semantic",
) -> TextChunker:
    """
    Factory function to create a chunker.

    Args:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy (recursive, markdown, semantic)

    Returns:
        TextChunker instance
    """
    return HybridChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
