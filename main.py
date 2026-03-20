#!/usr/bin/env python
"""
Main entry point for Battery LLM project.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Setup environment variables."""
    os.environ.setdefault("PYTHONPATH", str(project_root))
    os.environ.setdefault("DATA_DIR", str(project_root / "data"))


def run_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Run the FastAPI server."""
    from src.api.main import run_server
    run_server(host=host, port=port, reload=reload)


def initialize_pipeline(
    data_path: str = "data/raw",
    persist_dir: str = "data/embeddings/chroma",
):
    """Initialize the RAG pipeline."""
    from dotenv import load_dotenv
    load_dotenv()  # Load .env for MINIMAX_API_KEY
    from src.rag.pipeline import create_pipeline

    pipeline = create_pipeline(
        data_path=data_path,
        persist_directory=persist_dir,
    )

    print("Pipeline initialized successfully")
    return pipeline


def process_documents(
    data_path: str = "data/raw",
    output_path: str = "data/processed",
):
    """Process documents and create embeddings."""
    from src.data_pipeline.loader import DirectoryLoader
    from src.data_pipeline.chunker import create_chunker
    from src.data_pipeline.embedder import create_embedder
    from langchain_community.vectorstores import Chroma

    print(f"Loading documents from {data_path}")

    # Load documents
    loader = DirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    chunker = create_chunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Create embeddings
    embedder = create_embedder()
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder.embeddings,
        persist_directory="data/embeddings/chroma",
    )
    vectorstore.persist()

    print(f"Vector store saved to data/embeddings/chroma")


def prepare_finetune_dataset(
    data_path: str = "data/raw",
    output_path: str = "data/finetune/battery_dataset.jsonl",
):
    """Prepare fine-tuning dataset."""
    from src.data_pipeline.loader import DirectoryLoader
    from src.fine_tuning.dataset_prep import prepare_battery_dataset

    print(f"Loading documents from {data_path}")

    # Load documents
    loader = DirectoryLoader(data_path)
    documents = loader.load()

    print("Preparing fine-tuning dataset...")
    examples = prepare_battery_dataset(
        documents=documents,
        output_path=output_path,
    )

    print(f"Created {len(examples)} training examples")
    print(f"Dataset saved to {output_path}")


def run_tests():
    """Run tests."""
    import pytest
    pytest.main(["tests/", "-v", "--tb=short"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Battery LLM Project CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # API command
    api_parser = subparsers.add_parser("api", help="Run API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host")
    api_parser.add_argument("--port", type=int, default=8000, help="Port")
    api_parser.add_argument("--reload", action="store_true", help="Enable reload")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize pipeline")
    init_parser.add_argument(
        "--data-path",
        default="data/raw",
        help="Path to raw data",
    )
    init_parser.add_argument(
        "--persist-dir",
        default="data/embeddings/chroma",
        help="Path to persist embeddings",
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument(
        "--data-path",
        default="data/raw",
        help="Path to raw data",
    )

    # Finetune command
    ft_parser = subparsers.add_parser("finetune-data", help="Prepare fine-tuning data")
    ft_parser.add_argument(
        "--data-path",
        default="data/raw",
        help="Path to raw data",
    )
    ft_parser.add_argument(
        "--output",
        default="data/finetune/battery_dataset.jsonl",
        help="Output path",
    )

    # Test command
    subparsers.add_parser("test", help="Run tests")

    args = parser.parse_args()

    setup_environment()

    if args.command == "api":
        run_api(
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "init":
        initialize_pipeline(
            data_path=args.data_path,
            persist_dir=args.persist_dir,
        )
    elif args.command == "process":
        process_documents(data_path=args.data_path)
    elif args.command == "finetune-data":
        prepare_finetune_dataset(
            data_path=args.data_path,
            output_path=args.output,
        )
    elif args.command == "test":
        run_tests()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
