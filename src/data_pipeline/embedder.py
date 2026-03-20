"""
Embedding generation using TFIDF, HuggingFace, or MiniMax.
"""

import os
import requests
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document


class MiniMaxEmbedder:
    """MiniMax API based embedder (works in China)."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "embeddings-bge-1536-para",
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.dimension = 1536  # Default dimension for MiniMax embeddings

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single text/query using MiniMax API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "texts": [text],
            }

            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0].get("embedding", [])
                    return np.array(embedding)
            else:
                print(f"MiniMax API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"MiniMax embedding error: {e}")

        # Return zero vector if failed
        return np.zeros(self.dimension)


class TFIDFEmbedder:
    """TFIDF based embedder (works offline)."""

    def __init__(
        self,
        max_features: int = 384,
        dimension: int = 384,
    ):
        self.max_features = max_features
        self.dimension = dimension
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.fitted = False

    def fit(self, documents: List[Document]):
        """Fit the vectorizer on documents."""
        texts = [doc.page_content for doc in documents]
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Embed a list of documents (or strings)."""
        # Handle both Document objects and strings
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))

        if not self.fitted:
            # Fit on sample texts
            sample_docs = [Document(page_content=t) for t in texts[:10]]
            self.fit(sample_docs)

        embeddings = self.vectorizer.transform(texts).toarray()
        return [emb for emb in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        # For queries, we need to transform using the fitted vectorizer
        # If not fitted, fit on a dummy document first
        if not self.fitted:
            self.fit([Document(page_content=query)])

        embedding = self.vectorizer.transform([query]).toarray()[0]

        # Pad to dimension if needed
        if len(embedding) < self.dimension:
            embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
        elif len(embedding) > self.dimension:
            embedding = embedding[:self.dimension]

        return embedding

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.dimension


class Embedder:
    """Wrapper for embedding generation - tries HuggingFace first, falls back to TFIDF."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
        use_offline: bool = False,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {"device": "cpu"}
        self.encode_kwargs = encode_kwargs or {
            "normalize_embeddings": True,
            "batch_size": 32,
        }
        self.use_offline = use_offline
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self):
        """Load embeddings model - tries HuggingFace first, then MiniMax, then TF-IDF."""
        import os

        # Set HuggingFace mirror for China
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # Try HuggingFace first (reliable and consistent dimension)
        if not self.use_offline:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                print(f"Loading HuggingFace model: {self.model_name}")
                return HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs,
                )
            except Exception as e:
                print(f"✗ HuggingFace failed: {e}")

        # Try MiniMax API (works in China, but may have balance issues)
        api_key = os.environ.get("MINIMAX_API_KEY")
        if api_key and not self.use_offline:
            try:
                print("Trying MiniMax embeddings API...")
                embedder = MiniMaxEmbedder(api_key=api_key)
                # Test it
                test_emb = embedder.embed_query("test")
                # Check if we got valid embeddings (non-zero)
                if test_emb is not None and len(test_emb) > 0 and np.any(test_emb != 0):
                    print(f"✓ MiniMax embeddings loaded! Dimension: {len(test_emb)}")
                    return embedder
                else:
                    print(f"✗ MiniMax embeddings returned invalid (zero) vectors")
            except Exception as e:
                print(f"✗ MiniMax embeddings failed: {e}")

        # Fallback to TF-IDF
        print("Using TFIDF embeddings (offline mode)")
        return TFIDFEmbedder()

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Embed a list of documents."""
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embeddings.embed_query(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        test_embedding = self.embed_query("test")
        return len(test_embedding)


class EmbeddingPipeline:
    """Pipeline for processing documents and generating embeddings."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def process_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
    ) -> List[Document]:
        """Process documents and add embeddings to metadata."""
        processed_docs = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.embedder.embed_documents(batch)

            for doc, embedding in zip(batch, embeddings):
                doc.metadata["embedding"] = embedding
                doc.metadata["embedding_model"] = self.embedder.model_name
                processed_docs.append(doc)

        return processed_docs

    def create_embeddings_matrix(
        self,
        documents: List[Document],
    ) -> np.ndarray:
        """Create a matrix of embeddings for all documents."""
        embeddings = self.embedder.embed_documents(documents)
        return np.array(embeddings)


def create_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    normalize: bool = True,
    use_offline: bool = True,
) -> Embedder:
    """
    Factory function to create an embedder.

    Args:
        model_name: Hugging Face model name
        device: Device to use (cpu, cuda)
        normalize: Whether to normalize embeddings
        use_offline: Use TFIDF instead of HuggingFace (for offline use)

    Returns:
        Embedder instance
    """
    return Embedder(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
        use_offline=use_offline,
    )
