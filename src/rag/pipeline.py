"""
End-to-end RAG pipeline.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from ..data_pipeline.loader import DirectoryLoader
from ..data_pipeline.chunker import create_chunker
from ..data_pipeline.embedder import create_embedder
from .retriever import create_retriever
from .generator import create_generator, GroundedGenerator


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            config_path: Path to YAML config file
            config: Configuration dictionary
        """
        if config:
            self.config = config
        elif config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        self.vectorstore = None
        self.documents = None
        self.retriever = None
        self.generator = None

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "model": {
                "generator_model": {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                },
                "embedding_model": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
            "rag": {
                "retrieval": {"k": 4},
                "generation": {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                "prompt": {
                    "system_prompt": """You are an expert assistant for Apple Battery Manufacturing domain.
Provide accurate, technical information about battery manufacturing processes, quality control, and safety procedures.""",
                },
            },
        }

    def initialize(
        self,
        data_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize pipeline components.

        Args:
            data_path: Path to documents directory
            persist_directory: Directory to persist vector store
        """
        # Set up embedder - use HuggingFace with mirror for China
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            use_offline=False,  # Use HuggingFace (with mirror site)
        )

        # Check for existing vector store
        has_existing_store = False
        if persist_directory and os.path.exists(persist_directory):
            # Check for Chroma-specific files
            chroma_files = [f for f in os.listdir(persist_directory) if f.endswith(('.sqlite3', '.parquet'))]
            has_existing_store = len(chroma_files) > 0

        # Decision: prefer loading from disk if exists
        if persist_directory and has_existing_store and not data_path:
            # Only persist_directory, load from disk
            print(f"Loading existing vector store from {persist_directory}")
            client_settings = Settings(
                is_persistent=True,
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
            chroma_client = chromadb.Client(client_settings)
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name="langchain",
                embedding_function=embedder.embeddings,
            )
        elif data_path and has_existing_store:
            # Both data and store exist - load from disk (faster)
            print(f"Loading existing vector store from {persist_directory}")
            client_settings = Settings(
                is_persistent=True,
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
            chroma_client = chromadb.Client(client_settings)
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name="langchain",
                embedding_function=embedder.embeddings,
            )
        elif data_path:
            # Only data_path, rebuild
            print(f"Loading documents from {data_path}")
            self._load_and_process_documents(data_path, embedder, persist_directory)
        else:
            raise ValueError("Either data_path or persist_directory must be provided")

        # Set up retriever
        retrieval_config = self.config.get("rag", {}).get("retrieval", {})
        k = retrieval_config.get("k", 4)
        # Disable hybrid for now - requires more setup
        use_hybrid = False  # Just use vector search
        use_reranking = False

        # Get documents for hybrid retrieval (only if using hybrid)
        if self.vectorstore and use_hybrid:
            all_docs = self.vectorstore.get()["documents"]
            docs_for_retriever = [
                Document(page_content=doc) for doc in all_docs if doc
            ]
        else:
            docs_for_retriever = []

        self.retriever = create_retriever(
            vectorstore=self.vectorstore,
            documents=docs_for_retriever,
            k=k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
        )

        # Set up generator
        gen_config = self.config["model"]["generator_model"]
        prompt_config = self.config.get("rag", {}).get("prompt", {})
        gen_settings = self.config["rag"]["generation"]

        # Check if local model is enabled
        use_local = gen_settings.get("use_local", False)
        lora_adapter_path = gen_settings.get("lora_adapter_path", "models/lora_adapter")

        if use_local:
            print(f"Using local LoRA model: {lora_adapter_path}")
            self.generator = create_generator(
                model_id=gen_settings.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                temperature=gen_settings["temperature"],
                max_tokens=gen_settings["max_tokens"],
                use_local=True,
                use_simple=False,
                use_minimax=False,
                lora_adapter_path=lora_adapter_path,
            )
        else:
            # Check if MiniMax API key is available
            minimax_key = os.environ.get("MINIMAX_API_KEY")

            if minimax_key:
                print("Using MiniMax LLM for generation")
                self.generator = create_generator(
                    model_id=gen_config["name"],
                    temperature=gen_settings["temperature"],
                    max_tokens=gen_settings["max_tokens"],
                    use_local=False,
                    use_simple=False,
                    use_minimax=True,
                )
            else:
                print("MINIMAX_API_KEY not found and use_local=false. Using simple generator.")
                self.generator = create_generator(
                    model_id=gen_config["name"],
                    temperature=gen_settings["temperature"],
                    max_tokens=gen_settings["max_tokens"],
                    use_local=False,
                    use_simple=True,  # Fallback to simple generator
                )

    def _load_and_process_documents(
        self,
        data_path: str,
        embedder,
        persist_directory: Optional[str] = None,
    ):
        """Load, process, and embed documents."""
        # Load documents
        loader = DirectoryLoader(data_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")

        # Chunk documents
        chunker = create_chunker(
            chunk_size=512,
            chunk_overlap=50,
            strategy="semantic",
        )
        chunks = chunker.split_documents(self.documents)
        print(f"Created {len(chunks)} chunks")

        # Create vector store with proper persistence settings for ChromaDB 1.x
        print(f"Creating vector store with persist_directory: {persist_directory}")

        # First, pre-compute all embeddings using our embedder
        print("Computing embeddings for chunks...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.embeddings.embed_documents(texts)
        print(f"Computed {len(embeddings)} embeddings")

        if persist_directory:
            # Create client settings for persistent storage
            client_settings = Settings(
                is_persistent=True,
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )

            # Use LangChain's Chroma with client settings - this handles persistence correctly
            # First delete existing data to avoid duplicates
            import shutil
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)

            # Create vector store - Chroma.from_documents with client_settings handles persistence
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedder.embeddings,
                persist_directory=persist_directory,
                client_settings=client_settings,
            )

            print(f"Vector store persisted to {persist_directory}")
        else:
            # In-memory mode
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedder.embeddings,
            )
            print(f"Saved vector store to {persist_directory}")

    def query(
        self,
        question: str,
        return_sources: bool = True,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.

        Args:
            question: User question
            return_sources: Whether to return source documents
            min_similarity: Minimum similarity score threshold (0.0-1.0)

        Returns:
            Dict with answer and optionally sources
        """
        if not self.retriever or not self.generator:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Retrieve relevant documents with scores
        if hasattr(self.retriever, 'retrieve_with_score'):
            retrieved_docs_with_scores = self.retriever.retrieve_with_score(question)
            # Filter by similarity threshold
            filtered_docs = [
                (doc, score) for doc, score in retrieved_docs_with_scores
                if score >= min_similarity
            ]
            retrieved_docs = [doc for doc, _ in filtered_docs]
            scores = {doc.page_content[:100]: score for doc, score in filtered_docs}
        else:
            retrieved_docs = self.retriever.get_relevant_documents(question)
            scores = {}

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate response
        answer = self.generator.generate(context, question)

        result = {"answer": answer}

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": scores.get(doc.page_content[:100], None),
                }
                for doc in retrieved_docs
            ]

        return result

    def get_relevant_documents(self, question: str) -> List[Document]:
        """Get relevant documents for a question without generating answer."""
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized")
        return self.retriever.get_relevant_documents(question)


class ConfigurableRAGPipeline(RAGPipeline):
    """RAG pipeline with more configuration options."""

    def __init__(
        self,
        retriever=None,
        generator=None,
        documents: Optional[List[Document]] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(config=config)
        self.retriever = retriever
        self.generator = generator
        self.documents = documents


def create_pipeline(
    config_path: str = "configs/rag_config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    data_path: Optional[str] = None,
    persist_directory: Optional[str] = "data/embeddings/chroma",
) -> RAGPipeline:
    """
    Factory function to create and initialize a RAG pipeline.

    Args:
        config_path: Path to RAG config
        model_config_path: Path to model config
        data_path: Path to documents
        persist_directory: Directory to persist vector store

    Returns:
        Initialized RAGPipeline
    """
    # Load configs
    with open(config_path, "r") as f:
        rag_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # Merge configs
    config = {**model_config, "rag": rag_config.get("rag", {})}

    # Create and initialize pipeline
    pipeline = RAGPipeline(config=config)

    if data_path or persist_directory:
        pipeline.initialize(
            data_path=data_path,
            persist_directory=persist_directory,
        )

    return pipeline
