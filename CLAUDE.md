# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) intelligent assistant for Apple Battery Manufacturing domain. It provides technical Q&A about battery chemistry, manufacturing processes, quality control, and safety procedures using a hybrid retrieval system (vector + keyword search) with MiniMax LLM for generation.

## Common Commands

```bash
# Run the FastAPI server
python main.py api [--host HOST] [--port PORT] [--reload]

# Initialize the RAG pipeline (creates vector store)
python main.py init [--data-path PATH] [--persist-dir DIR]

# Process documents and create embeddings
python main.py process [--data-path PATH]

# Prepare fine-tuning dataset
python main.py finetune-data [--data-path PATH] [--output PATH]

# Run tests
python main.py test
# or directly:
pytest tests/ -v --tb=short
```

## Architecture

### Data Flow
1. Documents loaded from `data/raw/` → chunked (512 chars, 50 overlap) → embedded
2. Embeddings stored in ChromaDB at `data/embeddings/chroma/`
3. Queries use hybrid retrieval (0.3 keyword BM25 / 0.7 semantic) with k=4
4. Retrieved context + question → MiniMax LLM → response with citations

### Key Modules
- `src/rag/pipeline.py` - End-to-end RAG orchestration
- `src/rag/retriever.py` - Hybrid search (vector + BM25)
- `src/rag/generator.py` - MiniMax LLM generation, LocalGenerator (TinyLlama + LoRA), or SimpleGenerator fallback
- `src/data_pipeline/loader.py` - Multi-format document loading
- `src/data_pipeline/chunker.py` - Text chunking (recursive/semantic/markdown)
- `src/data_pipeline/embedder.py` - Embedding generation (HuggingFace → MiniMax → TF-IDF fallback chain). Uses HuggingFace mirror (hf-mirror.com) for China.
- `src/fine_tuning/loratuner.py` - LoRA fine-tuning via PEFT

### Entry Points
- `main.py` - CLI interface
- `src/api/main.py` - FastAPI server with endpoints at `/query`, `/query/batch`, `/documents/upload`, `/metrics`
- `src/api/gradio_app.py` - Alternative Gradio chat interface (port 7860)

## Configuration

**`configs/model_config.yaml`** - Generator, embedder, LoRA params, vector store settings
**`configs/rag_config.yaml`** - Retrieval (k=4, hybrid weights), generation (temp=0.7, max_tokens=1024), citation format

## Environment

Requires `MINIMAX_API_KEY` in `.env` for LLM access (MiniMax-M2.7 via `https://api.minimaxi.com/anthropic`).

**HuggingFace Mirror:** The embedder automatically uses `https://hf-mirror.com` as the HuggingFace endpoint for China regions.

## Local Inference with LoRA

For offline/local inference, use TinyLlama + LoRA adapter:

```bash
# 1. Prepare training data
python main.py finetune-data

# 2. Train LoRA adapter
python scripts/train_lora.py --epochs 3

# 3. Run API with local model
# Set use_local=true in configs/rag_config.yaml
python main.py api
```

The LoRA adapter is stored in `models/lora_adapter/` and uses TinyLlama/TinyLlama-1.1B-Chat-v1.0 as base model.

## Testing

Tests use markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`
Run specific markers: `pytest tests/ -m "not slow"`
