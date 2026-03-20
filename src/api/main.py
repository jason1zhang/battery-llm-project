"""
FastAPI application for Battery LLM.
"""

import os
import time
import logging
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
import yaml

# Load .env file if exists
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    HealthResponse,
    DocumentUploadResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    MetricsResponse,
)

# Try to import RAG components
try:
    from ..rag.pipeline import RAGPipeline, create_pipeline
    from ..data_pipeline.loader import DirectoryLoader
    from ..data_pipeline.chunker import create_chunker
    from ..data_pipeline.embedder import create_embedder
    from langchain_community.vectorstores import Chroma

    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"Warning: RAG components not available: {e}")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global state
class AppState:
    """Application state."""

    pipeline: Optional[RAGPipeline] = None
    config: dict = {}
    metrics = {
        "total_queries": 0,
        "total_response_time": 0.0,
        "total_answer_length": 0,
    }


app_state = AppState()


def load_config(config_path: str = "configs/rag_config.yaml") -> dict:
    """Load configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Battery LLM API...")

    # Load config
    app_state.config = load_config()

    # Try to initialize pipeline
    if RAG_AVAILABLE:
        try:
            data_path = "data/raw"
            persist_dir = "data/embeddings/chroma"

            # Always use persist_directory if data_path exists
            if os.path.exists(data_path):
                # Create directory if it doesn't exist
                os.makedirs(persist_dir, exist_ok=True)
                app_state.pipeline = create_pipeline(
                    data_path=data_path,
                    persist_directory=persist_dir,
                )
                logger.info("Pipeline initialized successfully")
            elif os.path.exists(persist_dir):
                app_state.pipeline = create_pipeline(
                    persist_directory=persist_dir,
                )
                logger.info("Pipeline loaded from existing vector store")
            else:
                logger.warning("No data found, pipeline not initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Battery LLM API...")


# Create FastAPI app
app = FastAPI(
    title="Battery Manufacturing LLM API",
    description="RAG-based intelligent assistant for Apple Battery Manufacturing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - serve the chat UI."""
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    pipeline_ready = app_state.pipeline is not None

    return HealthResponse(
        status="healthy" if pipeline_ready else "degraded",
        version="1.0.0",
        model_loaded=pipeline_ready,
        vectorstore_ready=pipeline_ready,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Query endpoint for asking questions."""
    start_time = time.time()

    if app_state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please ensure data is loaded.",
        )

    try:
        result = app_state.pipeline.query(
            question=request.question,
            return_sources=request.return_sources,
        )

        # Update metrics
        response_time = (time.time() - start_time) * 1000
        app_state.metrics["total_queries"] += 1
        app_state.metrics["total_response_time"] += response_time
        app_state.metrics["total_answer_length"] += len(result.get("answer", "").split())

        # Build response
        response = QueryResponse(
            answer=result.get("answer", ""),
            query=request.question,
        )

        if request.return_sources and "sources" in result:
            response.sources = [
                SourceDocument(
                    content=src.get("content", ""),
                    source=src.get("metadata", {}).get("source_file", "Unknown"),
                    metadata=src.get("metadata"),
                )
                for src in result["sources"]
            ]

        response.metadata = {
            "response_time_ms": response_time,
            "sources_retrieved": len(result.get("sources", [])),
        }

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/batch", response_model=BatchQueryResponse, tags=["Query"])
async def batch_query(request: BatchQueryRequest):
    """Batch query endpoint."""
    results = []

    for question in request.questions:
        if app_state.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Pipeline not initialized",
            )

        try:
            result = app_state.pipeline.query(
                question=question,
                return_sources=request.return_sources,
            )

            query_response = QueryResponse(
                answer=result.get("answer", ""),
                query=question,
            )

            if request.return_sources and "sources" in result:
                query_response.sources = [
                    SourceDocument(
                        content=src.get("content", ""),
                        source=src.get("metadata", {}).get("source_file", "Unknown"),
                    )
                    for src in result["sources"]
                ]

            results.append(query_response)

        except Exception as e:
            logger.error(f"Error in batch query: {e}")
            results.append(
                QueryResponse(
                    answer=f"Error: {str(e)}",
                    query=question,
                )
            )

    return BatchQueryResponse(
        results=results,
        total_questions=len(results),
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents():
    """Endpoint to reload/process documents."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=501, detail="RAG not available")

    try:
        data_path = "data/raw"

        if not os.path.exists(data_path):
            return DocumentUploadResponse(
                success=False,
                message="No data directory found",
                documents_processed=0,
            )

        # Reload pipeline
        app_state.pipeline = create_pipeline(
            data_path=data_path,
            persist_directory="data/embeddings/chroma",
        )

        # Count documents
        doc_count = len(app_state.pipeline.documents) if app_state.pipeline.documents else 0

        return DocumentUploadResponse(
            success=True,
            message="Documents processed successfully",
            documents_processed=doc_count,
        )

    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get API metrics."""
    total = app_state.metrics["total_queries"]

    if total == 0:
        return MetricsResponse(
            total_queries=0,
            avg_response_time=0.0,
            avg_answer_length=0.0,
        )

    return MetricsResponse(
        total_queries=total,
        avg_response_time=app_state.metrics["total_response_time"] / total,
        avg_answer_length=app_state.metrics["total_answer_length"] / total,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Run the FastAPI server."""
    # Load .env file before starting
    from dotenv import load_dotenv
    load_dotenv()

    # Print debug info
    import os
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    print(f"MiniMax API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key prefix: {api_key[:15]}...")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,  # Use single worker to ensure env vars are passed
    )


if __name__ == "__main__":
    run_server()
