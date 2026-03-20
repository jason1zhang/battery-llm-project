"""
Model evaluation utilities.
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelEvaluator:
    """Evaluate language model outputs."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load model for evaluation."""
        if self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
            )

    def calculate_perplexity(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        Calculate perplexity of generated text.

        Args:
            texts: List of generated texts
            max_length: Maximum sequence length

        Returns:
            Average perplexity
        """
        if self.model is None:
            self.load_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        perplexities = []

        for text in texts:
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = encodings.input_ids
            labels = input_ids

            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)

        return np.mean(perplexities)


class RAGEvaluator:
    """Evaluate RAG system performance."""

    def __init__(self):
        self.results = []

    def calculate_bleu(
        self,
        reference: str,
        hypothesis: str,
    ) -> float:
        """
        Calculate BLEU score.

        Args:
            reference: Reference text
            hypothesis: Generated text

        Returns:
            BLEU score
        """
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()

        # Use smoothing for short sentences
        smoothie = SmoothingFunction().method1

        return sentence_bleu(
            [reference_tokens],
            hypothesis_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )

    def calculate_rouge(
        self,
        reference: str,
        hypothesis: str,
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            reference: Reference text
            hypothesis: Generated text

        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)

        return {
            "rouge_1": scores['rouge1'].fmeasure,
            "rouge_2": scores['rouge2'].fmeasure,
            "rouge_l": scores['rougeL'].fmeasure,
        }

    def calculate_retrieval_precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 4,
    ) -> float:
        """
        Calculate precision@k for retrieval.

        Args:
            retrieved_docs: List of retrieved document contents
            relevant_docs: List of relevant document contents
            k: Number of top results to consider

        Returns:
            Precision@k
        """
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        num_relevant = sum(1 for doc in retrieved_k if doc in relevant_set)
        return num_relevant / k if k > 0 else 0.0

    def calculate_retrieval_recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 4,
    ) -> float:
        """
        Calculate recall@k for retrieval.

        Args:
            retrieved_docs: List of retrieved document contents
            relevant_docs: List of relevant document contents
            k: Number of top results to consider

        Returns:
            Recall@k
        """
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)

        num_relevant = sum(1 for doc in retrieved_k if doc in relevant_set)
        return num_relevant / len(relevant_docs) if len(relevant_docs) > 0 else 0.0

    def evaluate_response_quality(
        self,
        question: str,
        generated_answer: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate response quality.

        Args:
            question: The input question
            generated_answer: Generated answer
            reference_answer: Optional reference answer

        Returns:
            Evaluation metrics
        """
        result = {
            "answer_length": len(generated_answer.split()),
            "answer_word_count": len(generated_answer.split()),
        }

        # Calculate BLEU/ROUGE if reference available
        if reference_answer:
            result["bleu"] = self.calculate_bleu(reference_answer, generated_answer)
            result.update(self.calculate_rouge(reference_answer, generated_answer))

        # Check if answer addresses the question
        question_keywords = set(question.lower().split())
        answer_keywords = set(generated_answer.lower().split())
        keyword_overlap = len(question_keywords & answer_keywords) / len(question_keywords)

        result["question_keyword_coverage"] = keyword_overlap

        return result

    def add_result(self, result: Dict[str, Any]):
        """Add evaluation result."""
        self.results.append(result)

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all evaluation metrics."""
        if not self.results:
            return {}

        metrics = {}
        for key in self.results[0].keys():
            values = [r[key] for r in self.results if key in r]
            if all(isinstance(v, (int, float)) for v in values):
                metrics[f"avg_{key}"] = np.mean(values)

        return metrics


class HallucinationDetector:
    """Detect hallucinations in generated responses."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def calculate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Calculate faithfulness of answer to context.

        Args:
            answer: Generated answer
            context: Source context

        Returns:
            Faithfulness score (0-1)
        """
        if self.embedder is None:
            # Use simple keyword-based approach
            return self._keyword_faithfulness(answer, context)

        # Use embedding similarity
        answer_embedding = self.embedder.embed_query(answer)
        context_embedding = self.embedder.embed_query(context)

        similarity = cosine_similarity(
            [answer_embedding],
            [context_embedding],
        )[0][0]

        return float(similarity)

    def _keyword_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """Calculate keyword-based faithfulness."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words) / len(answer_words)
        return float(overlap)

    def self_consistency_check(
        self,
        question: str,
        answers: List[str],
    ) -> float:
        """
        Check consistency across multiple generations.

        Args:
            question: Input question
            answers: Multiple generated answers

        Returns:
            Consistency score (0-1)
        """
        if len(answers) < 2:
            return 1.0

        # Use keyword overlap
        answer_sets = [set(a.lower().split()) for a in answers]

        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(answer_sets)):
            for j in range(i + 1, len(answer_sets)):
                if answer_sets[i]:
                    overlap = len(answer_sets[i] & answer_sets[j]) / len(
                        answer_sets[i] | answer_sets[j]
                    )
                    overlaps.append(overlap)

        return np.mean(overlaps) if overlaps else 0.0


def evaluate_rag_system(
    pipeline,
    test_questions: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Evaluate a RAG pipeline.

    Args:
        pipeline: RAGPipeline instance
        test_questions: List of test questions with expected answers

    Returns:
        Evaluation results
    """
    evaluator = RAGEvaluator()

    for item in test_questions:
        question = item["question"]
        reference = item.get("reference_answer")

        # Get pipeline response
        result = pipeline.query(question, return_sources=True)

        # Evaluate
        eval_result = evaluator.evaluate_response_quality(
            question=question,
            generated_answer=result["answer"],
            reference_answer=reference,
        )

        # Add retrieval metrics
        if "sources" in result:
            retrieved = [s["content"] for s in result["sources"]]
            eval_result["retrieved_docs"] = len(retrieved)

        evaluator.add_result(eval_result)

    return evaluator.get_average_metrics()
