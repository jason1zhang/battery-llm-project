"""
Document loaders for various file formats.
Supports PDF, TXT, and Markdown files.
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document


class DocumentLoader:
    """Base class for document loading."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load document and return list of Document objects."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        return self._load_file()

    def _load_file(self) -> List[Document]:
        """Load file based on extension."""
        ext = self.file_path.suffix.lower()

        if ext == ".pdf":
            return self._load_pdf()
        elif ext in [".txt", ".text"]:
            return self._load_text()
        elif ext in [".md", ".markdown"]:
            return self._load_markdown()
        elif ext == ".html":
            return self._load_html()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self) -> List[Document]:
        """Load PDF file."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(self.file_path))
            return loader.load()
        except ImportError:
            raise ImportError("pypdf is required for PDF loading")

    def _load_text(self) -> List[Document]:
        """Load text file."""
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(self.file_path), encoding=self.encoding)
        return loader.load()

    def _load_markdown(self) -> List[Document]:
        """Load Markdown file as plain text."""
        # Read markdown as plain text
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            content = f.read()

        from langchain_core.documents import Document
        return [Document(page_content=content, metadata={"source": str(self.file_path)})]

    def _load_html(self) -> List[Document]:
        """Load HTML file."""
        try:
            from langchain_community.document_loaders import UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(str(self.file_path))
            return loader.load()
        except ImportError:
            # Fallback to plain text
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": str(self.file_path)})]


class DirectoryLoader:
    """Load all documents from a directory."""

    def __init__(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ):
        self.directory = Path(directory)
        self.recursive = recursive
        self.extensions = extensions or [".pdf", ".txt", ".md", ".html"]

    def load(self) -> List[Document]:
        """Load all documents from the directory."""
        documents = []

        pattern = "**/*" if self.recursive else "*"

        for file_path in self.directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                try:
                    loader = DocumentLoader(str(file_path))
                    docs = loader.load()
                    # Add source metadata
                    for doc in docs:
                        doc.metadata["source_file"] = str(file_path.name)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents


def load_documents(
    source: str,
    is_directory: bool = False,
    **kwargs,
) -> List[Document]:
    """
    Convenience function to load documents.

    Args:
        source: File path or directory path
        is_directory: If True, treat source as directory
        **kwargs: Additional arguments for loaders

    Returns:
        List of Document objects
    """
    if is_directory:
        loader = DirectoryLoader(source, **kwargs)
    else:
        loader = DocumentLoader(source, **kwargs)

    return loader.load()
