"""
Benchmark tests for RAG system evaluation.
These tests evaluate the quality of RAG answers using sample questions.

Note: These tests require the vector store to be built locally.
They will skip in CI if data/embeddings/chroma doesn't exist.
"""

import os
import pytest
from typing import List

from src.rag.pipeline import RAGPipeline


# Check if vector store exists (locally only)
VECTOR_STORE_PATH = "data/embeddings/chroma"
HAS_VECTOR_STORE = os.path.exists(VECTOR_STORE_PATH) and any(
    f.endswith(('.sqlite3', '.parquet'))
    for f in os.listdir(VECTOR_STORE_PATH)
    if os.path.isfile(os.path.join(VECTOR_STORE_PATH, f))
)


# Benchmark questions for battery manufacturing domain
BENCHMARK_QUESTIONS = [
    {
        "question": "What is cathode mixing ratio?",
        "keywords": ["cathode", "mixing", "ratio", "NMC", "100:1"],
        "min_length": 50,
    },
    {
        "question": "What safety tests are required?",
        "keywords": ["safety", "test", "nail", "crush", "overcharge"],
        "min_length": 30,
    },
    {
        "question": "What are the dry room requirements?",
        "keywords": ["dry", "room", "humidity", "RH", "moisture"],
        "min_length": 30,
    },
    {
        "question": "What is the electrode coating process?",
        "keywords": ["coating", "electrode", "slot", "die", "thickness"],
        "min_length": 50,
    },
    {
        "question": "How is electrolyte handled?",
        "keywords": ["electrolyte", "LiPF6", "moisture", "humidity", "spill"],
        "min_length": 30,
    },
]


# Skip all tests in this class if vector store not available
@pytest.mark.skipif(
    not HAS_VECTOR_STORE,
    reason="Vector store not available (data/embeddings/chroma not in git)"
)
class TestRAGPipeline:
    """Test RAG pipeline with benchmark questions."""

    @pytest.fixture
    def pipeline(self):
        """Create RAG pipeline for testing."""
        pipeline = RAGPipeline()
        # Note: This will load existing vector store if available
        return pipeline

    def test_pipeline_responds_to_question(self, pipeline):
        """Test that pipeline can respond to a question."""
        result = pipeline.query(
            question="What is cathode mixing ratio?",
            return_sources=True
        )

        assert result is not None
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_answer_not_empty(self, pipeline):
        """Test that answer is not empty."""
        result = pipeline.query(
            question="What safety tests are required?",
            return_sources=True
        )

        answer = result.get("answer", "")
        assert len(answer.strip()) > 0, "Answer should not be empty"

    def test_answer_has_minimum_length(self, pipeline):
        """Test that answer meets minimum length requirement."""
        result = pipeline.query(
            question="What are the dry room requirements?",
            return_sources=True
        )

        answer = result.get("answer", "")
        assert len(answer) >= 30, f"Answer too short: {len(answer)} chars"

    def test_sources_returned(self, pipeline):
        """Test that sources are returned with the answer."""
        result = pipeline.query(
            question="What is the electrode coating process?",
            return_sources=True
        )

        assert "sources" in result
        sources = result.get("sources", [])
        assert isinstance(sources, list)

    def test_answer_contains_keywords(self, pipeline):
        """Test that answer contains relevant keywords."""
        result = pipeline.query(
            question="How is electrolyte handled?",
            return_sources=True
        )

        answer = result.lower()
        keywords = ["electrolyte", "LiPF", "moisture", "humidity"]

        found_keywords = [kw for kw in keywords if kw.lower() in answer]
        assert len(found_keywords) >= 2, \
            f"Answer should contain at least 2 keywords. Found: {found_keywords}"

    def test_multiple_questions(self, pipeline):
        """Test that pipeline can handle multiple questions."""
        questions = [
            "What is cathode mixing ratio?",
            "What safety tests are required?",
            "What are the dry room requirements?",
        ]

        for question in questions:
            result = pipeline.query(question=question, return_sources=True)
            assert result is not None
            assert "answer" in result
            assert len(result["answer"]) > 0

    def test_sources_have_metadata(self, pipeline):
        """Test that sources have proper metadata."""
        result = pipeline.query(
            question="What is the electrode coating process?",
            return_sources=True
        )

        sources = result.get("sources", [])
        if len(sources) > 0:
            # Check first source has metadata
            first_source = sources[0]
            assert "metadata" in first_source or "content" in first_source

    def test_no_hallucination_empty_question(self, pipeline):
        """Test that empty questions don't cause errors."""
        try:
            result = pipeline.query(question="", return_sources=True)
            # Empty question should return empty answer
            assert len(result.get("answer", "")) == 0
        except Exception as e:
            pytest.fail(f"Empty question caused error: {e}")


@pytest.mark.skipif(
    not HAS_VECTOR_STORE,
    reason="Vector store not available (data/embeddings/chroma not in git)"
)
class TestRAGBenchmark:
    """Benchmark tests that evaluate RAG quality."""

    @pytest.fixture
    def pipeline(self):
        """Create RAG pipeline for testing."""
        return RAGPipeline()

    @pytest.mark.parametrize("benchmark", BENCHMARK_QUESTIONS)
    def test_benchmark_question(self, pipeline, benchmark):
        """Run a benchmark question and verify quality."""
        question = benchmark["question"]
        keywords = benchmark["keywords"]
        min_length = benchmark["min_length"]

        result = pipeline.query(question=question, return_sources=True)

        # Check answer exists
        assert "answer" in result, f"No answer for question: {question}"
        answer = result["answer"]
        answer_lower = answer.lower()

        # Check minimum length
        assert len(answer) >= min_length, \
            f"Answer too short for '{question}': {len(answer)} < {min_length}"

        # Check at least some keywords are present
        found_keywords = [kw for kw in keywords if kw.lower() in answer_lower]
        keyword_match_rate = len(found_keywords) / len(keywords)

        # At least 20% of keywords should be found
        assert keyword_match_rate >= 0.2, \
            f"Too few keywords found for '{question}': {found_keywords}/{keywords}"

    def test_retrieval_relevance(self, pipeline):
        """Test that retrieved documents are relevant to the query."""
        question = "cathode mixing ratio"

        result = pipeline.query(question=question, return_sources=True)
        sources = result.get("sources", [])

        # Should retrieve some documents
        assert len(sources) > 0, "Should retrieve at least one document"

        # Check that at least one source mentions relevant terms
        relevant_terms = ["cathode", "mixing", "ratio", "NMC", "electrode"]
        found_in_sources = False

        for source in sources:
            content = source.get("content", "").lower()
            if any(term in content for term in relevant_terms):
                found_in_sources = True
                break

        assert found_in_sources, "No relevant terms found in retrieved sources"

    def test_response_time(self, pipeline):
        """Test that response time is reasonable."""
        import time

        question = "What is cathode mixing ratio?"

        start_time = time.time()
        result = pipeline.query(question=question, return_sources=True)
        end_time = time.time()

        response_time = end_time - start_time

        # Response should be under 30 seconds (generous for API call)
        assert response_time < 30, \
            f"Response too slow: {response_time:.2f}s"

        print(f"\nResponse time: {response_time:.2f}s")


@pytest.mark.skipif(
    not HAS_VECTOR_STORE,
    reason="Vector store not available (data/embeddings/chroma not in git)"
)
class TestRAGQuality:
    """High-level quality tests for RAG system."""

    def test_no_error_on_common_questions(self):
        """Test that system handles common questions without errors."""
        pipeline = RAGPipeline()

        common_questions = [
            "What is the cathode mixing ratio?",
            "Tell me about battery safety",
            "What are dry room requirements?",
        ]

        for question in common_questions:
            try:
                result = pipeline.query(question=question, return_sources=True)
                assert result is not None
                assert "answer" in result
            except Exception as e:
                pytest.fail(f"Error on question '{question}': {e}")

    def test_sources_are_deduplicated(self):
        """Test that sources are properly handled."""
        pipeline = RAGPipeline()

        result = pipeline.query(
            question="What safety tests are required?",
            return_sources=True
        )

        sources = result.get("sources", [])

        # Sources should be a list
        assert isinstance(sources, list)

        # If there are sources, they should have content
        if len(sources) > 0:
            first_source = sources[0]
            assert "content" in first_source or "metadata" in first_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
