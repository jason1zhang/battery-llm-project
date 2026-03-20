"""
Unit tests for evaluation components.
"""

import pytest
from src.fine_tuning.evaluator import (
    ModelEvaluator,
    RAGEvaluator,
    HallucinationDetector,
)
from src.fine_tuning.dataset_prep import (
    BatteryDomainPreparer,
    prepare_battery_dataset,
)


class TestRAGEvaluator:
    """Test RAG evaluator."""

    def test_evaluator_creation(self):
        """Test evaluator can be created."""
        evaluator = RAGEvaluator()

        assert evaluator is not None

    def test_calculate_bleu(self):
        """Test BLEU calculation."""
        evaluator = RAGEvaluator()

        reference = "The battery is lithium ion"
        hypothesis = "The battery is lithium ion"

        bleu = evaluator.calculate_bleu(reference, hypothesis)

        assert bleu > 0

    def test_calculate_rouge(self):
        """Test ROUGE calculation."""
        evaluator = RAGEvaluator()

        reference = "The battery is lithium ion"
        hypothesis = "The battery uses lithium ion technology"

        rouge = evaluator.calculate_rouge(reference, hypothesis)

        assert "rouge_1" in rouge
        assert "rouge_l" in rouge

    def test_evaluate_response_quality(self):
        """Test response quality evaluation."""
        evaluator = RAGEvaluator()

        result = evaluator.evaluate_response_quality(
            question="What is battery chemistry?",
            generated_answer="Lithium ion is the battery chemistry used.",
            reference_answer="Lithium ion is a type of battery chemistry.",
        )

        assert "answer_length" in result
        assert "question_keyword_coverage" in result


class TestHallucinationDetector:
    """Test hallucination detector."""

    def test_detector_creation(self):
        """Test detector can be created."""
        detector = HallucinationDetector()

        assert detector is not None

    def test_keyword_faithfulness(self):
        """Test keyword-based faithfulness."""
        detector = HallucinationDetector()

        answer = "Lithium ion battery with high energy density"
        context = "Lithium ion battery has high energy density and long cycle life"

        score = detector._keyword_faithfulness(answer, context)

        assert 0 <= score <= 1

    def test_self_consistency(self):
        """Test self-consistency check."""
        detector = HallucinationDetector()

        answers = [
            "Lithium ion batteries are used",
            "Lithium ion batteries are used in devices",
            "Lithium ion batteries are used",
        ]

        score = detector.self_consistency_check("test question", answers)

        assert 0 <= score <= 1


class TestDatasetPreparer:
    """Test dataset preparation."""

    def test_battery_preparer_creation(self):
        """Test preparer can be created."""
        preparer = BatteryDomainPreparer()

        assert preparer is not None

    def test_add_example(self):
        """Test adding examples."""
        preparer = BatteryDomainPreparer()

        preparer.add_example(
            instruction="Test question?",
            input_text="Test input",
            output="Test output",
        )

        assert len(preparer.examples) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
