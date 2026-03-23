"""
Retrieval system for RAG pipeline.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


class VectorRetriever:
    """Vector store based retriever."""

    def __init__(
        self,
        vectorstore: Chroma,
        k: int = 4,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.retriever = self._create_retriever()

    def _create_retriever(self):
        """Create the base retriever."""
        search_kwargs = {"k": self.k}

        if self.score_threshold is not None:
            search_kwargs["score_threshold"] = self.score_threshold

        return self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        return self.retriever.invoke(query)

    def retrieve_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k,
        )
        return docs_and_scores


class KeywordRetriever:
    """BM25 keyword-based retriever."""

    def __init__(
        self,
        documents: List[Document],
        k: int = 4,
        preprocess_func=None,
    ):
        self.k = k
        self.retriever = self._create_retriever(documents, preprocess_func)

    def _create_retriever(self, documents: List[Document], preprocess_func):
        """Create BM25 retriever."""
        # Get text content from documents
        texts = [doc.page_content for doc in documents]
        retriever = BM25Retriever.from_texts(
            texts=texts,
            preprocess_func=preprocess_func,
        )
        retriever.k = self.k
        return retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        return self.retriever.invoke(query)


class HybridRetriever:
    """Combine semantic and keyword retrieval."""

    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        k: int = 4,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        score_threshold: Optional[float] = None,
    ):
        self.k = k
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Create individual retrievers
        self.vector_retriever = VectorRetriever(
            vectorstore,
            k=k * 2,  # Get more for re-ranking
            score_threshold=score_threshold,
        )
        self.keyword_retriever = KeywordRetriever(
            documents,
            k=k * 2,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve using hybrid approach."""
        # Get results from both retrievers
        semantic_docs = self.vector_retriever.get_relevant_documents(query)
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)

        # Combine and re-rank using reciprocal rank fusion
        return self._reciprocal_rank_fusion(semantic_docs, keyword_docs)

    def _reciprocal_rank_fusion(
        self,
        semantic_docs: List[Document],
        keyword_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """Combine results using RRF algorithm."""
        doc_scores = {}

        # Score semantic results
        for rank, doc in enumerate(semantic_docs):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            score = 1.0 / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score * self.semantic_weight)
            else:
                doc_scores[doc_id] = (doc, score * self.semantic_weight)

        # Score keyword results
        for rank, doc in enumerate(keyword_docs):
            doc_id = doc.page_content[:100]
            score = 1.0 / (k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score * self.keyword_weight)
            else:
                doc_scores[doc_id] = (doc, score * self.keyword_weight)

        # Sort by combined score
        ranked = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.k]]


class MultiQueryRetriever:
    """Generate multiple queries for better retrieval."""

    def __init__(
        self,
        base_retriever,
        llm=None,
        k: int = 4,
    ):
        self.base_retriever = base_retriever
        self.llm = llm
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve using multiple query variations."""
        # Generate query variations
        queries = self._generate_queries(query)

        # Retrieve documents for all queries
        all_docs = []
        seen_contents = set()

        for q in queries:
            docs = self.base_retriever.get_relevant_documents(q)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(content_hash)

        # Return top k unique documents
        return all_docs[: self.k]

    def _generate_queries(self, query: str) -> List[str]:
        """Generate query variations."""
        if self.llm is None:
            # Use original query if no LLM provided
            return [query]

        # Use LLM to generate variations (simplified)
        prompt = f"""Generate 3 different variations of this query to improve document retrieval.
Original query: {query}

Variations should:
- Rephrase the question
- Use synonyms
- Break into sub-questions

Return only the variations, one per line:"""

        # This would use the LLM to generate - simplified here
        return [
            query,
            query.lower(),
            query.replace("?", ""),
        ]


def create_retriever(
    vectorstore: Chroma,
    documents: List[Document],
    k: int = 4,
    use_hybrid: bool = True,
    use_reranking: bool = False,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> any:
    """
    Factory function to create a retriever.

    Args:
        vectorstore: Chroma vector store
        documents: List of documents for keyword retrieval
        k: Number of documents to retrieve
        use_hybrid: Use hybrid (semantic + keyword) retrieval
        use_reranking: Use cross-encoder re-ranking (not available)
        keyword_weight: Weight for keyword search
        semantic_weight: Weight for semantic search
        reranker_model: Model for re-ranking

    Returns:
        Configured retriever
    """
    if use_hybrid:
        base_retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=documents,
            k=k,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
        )
    else:
        base_retriever = VectorRetriever(
            vectorstore=vectorstore,
            k=k,
        )

    # Re-ranking is disabled due to langchain compatibility issues
    # if use_reranking:
    #     return ReRankedRetriever(...)

    return base_retriever
