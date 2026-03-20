"""
Unit tests for RAG components.
"""

import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from unittest.mock import Mock, patch

from src.rag.retriever import (
    VectorRetriever,
    HybridRetriever,
    create_retriever,
)
from src.rag.generator import (
    ResponseGenerator,
    GroundedGenerator,
)
from src.rag.pipeline import RAGPipeline


class TestVectorRetriever:
    """Test vector retriever."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store."""
        mock = Mock(spec=Chroma)
        mock.as_retriever.return_value = Mock()
        return mock

    def test_vector_retriever_creation(self, mock_vectorstore):
        """Test vector retriever can be created."""
        retriever = VectorRetriever(
            vectorstore=mock_vectorstore,
            k=4,
        )

        assert retriever is not None
        assert retriever.k == 4


class TestHybridRetriever:
    """Test hybrid retriever."""

    def test_hybrid_retriever_creation(self):
        """Test hybrid retriever creation."""
        mock_vectorstore = Mock(spec=Chroma)
        mock_vectorstore.as_retriever.return_value = Mock()

        docs = [Document(page_content="Test content", metadata={})]

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            documents=docs,
            k=4,
            keyword_weight=0.3,
            semantic_weight=0.7,
        )

        assert retriever is not None
        assert retriever.k == 4


class TestResponseGenerator:
    """Test response generator."""

    def test_generator_creation(self):
        """Test generator can be created."""
        # This tests the generator creation without actually loading a model
        generator = ResponseGenerator(temperature=0.7, max_tokens=100)

        assert generator is not None
        assert generator.temperature == 0.7
        assert generator.max_tokens == 100


class TestRAGPipeline:
    """Test RAG pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        pipeline = RAGPipeline()

        assert pipeline is not None

    def test_pipeline_with_config(self):
        """Test pipeline with custom config."""
        config = {
            "model": {
                "generator_model": {"name": "test"},
                "embedding_model": {"name": "test"},
            },
            "rag": {
                "retrieval": {"k": 4},
                "generation": {"temperature": 0.5},
            },
        }

        pipeline = RAGPipeline(config=config)

        assert pipeline is not None
        assert pipeline.config == config


class TestGroundedGenerator:
    """Test grounded generator."""

    def test_grounded_generator_creation(self):
        """Test grounded generator can be created."""
        mock_generator = Mock(spec=ResponseGenerator)
        mock_generator.generate.return_value = "Test answer"

        grounded = GroundedGenerator(
            generator=mock_generator,
            citation_enabled=True,
            hallucination_check=True,
        )

        assert grounded is not None

    def test_citation_extraction(self):
        """Test citation extraction."""
        mock_generator = Mock(spec=ResponseGenerator)
        mock_generator.generate.return_value = "Test answer"

        grounded = GroundedGenerator(generator=mock_generator)

        docs = [
            Document(
                page_content="Test content",
                metadata={"source_file": "test.md"},
            )
        ]

        citations = grounded._extract_citations(docs)

        assert len(citations) == 1
        assert citations[0]["source"] == "test.md"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
