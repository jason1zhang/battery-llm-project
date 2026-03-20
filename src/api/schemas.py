"""
API request and response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""

    question: str = Field(..., description="User question", min_length=1)
    return_sources: bool = Field(
        default=True,
        description="Whether to return source documents",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Generation temperature (overrides config)",
        ge=0.0,
        le=2.0,
    )


class SourceDocument(BaseModel):
    """Source document schema."""

    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, description="Source file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: Optional[List[SourceDocument]] = Field(
        None,
        description="Retrieved source documents",
    )
    query: str = Field(..., description="Original question")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional response metadata",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    vectorstore_ready: bool = Field(..., description="Whether vector store is ready")


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""

    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(..., description="Number of documents processed")


class BatchQueryRequest(BaseModel):
    """Request schema for batch queries."""

    questions: List[str] = Field(..., description="List of questions", min_length=1)
    return_sources: bool = Field(default=True)


class BatchQueryResponse(BaseModel):
    """Response for batch queries."""

    results: List[QueryResponse] = Field(..., description="Query results")
    total_questions: int = Field(..., description="Total questions processed")


class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""

    total_queries: int = Field(..., description="Total queries processed")
    avg_response_time: float = Field(..., description="Average response time (ms)")
    avg_answer_length: float = Field(..., description="Average answer length")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")
