"""
LangChain tracing and observability.
"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


class LangChainTracer:
    """Local LangChain tracer for debugging and monitoring."""

    def __init__(self, trace_dir: str = "logs/traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace = None

    def start_trace(self, name: str, metadata: Optional[Dict] = None):
        """Start a new trace."""
        self.current_trace = {
            "name": name,
            "metadata": metadata or {},
            "start_time": datetime.now().isoformat(),
            "spans": [],
        }

    def add_span(
        self,
        name: str,
        span_type: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict] = None,
    ):
        """Add a span to the current trace."""
        if self.current_trace is None:
            self.start_trace(name)

        span = {
            "name": name,
            "type": span_type,
            "timestamp": datetime.now().isoformat(),
            "input": self._serialize_data(input_data),
            "output": self._serialize_data(output_data),
            "metadata": metadata or {},
        }

        self.current_trace["spans"].append(span)

    def end_trace(self, status: str = "success", error: Optional[str] = None):
        """End the current trace."""
        if self.current_trace is None:
            return

        self.current_trace["end_time"] = datetime.now().isoformat()
        self.current_trace["status"] = status
        self.current_trace["error"] = error

        # Save trace to file
        self._save_trace()

        self.current_trace = None

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if data is None:
            return None

        if isinstance(data, (str, int, float, bool)):
            return data

        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]

        if isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}

        # Convert to string for other types
        return str(data)

    def _save_trace(self):
        """Save trace to file."""
        if self.current_trace is None:
            return

        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.trace_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.current_trace, f, indent=2)

    def get_recent_traces(self, limit: int = 10) -> list:
        """Get recent traces."""
        traces = sorted(
            self.trace_dir.glob("trace_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        recent_traces = []
        for trace_file in traces[:limit]:
            with open(trace_file, "r") as f:
                recent_traces.append(json.load(f))

        return recent_traces


class LLMObservability:
    """Observability for LLM operations."""

    def __init__(self, tracer: Optional[LangChainTracer] = None):
        self.tracer = tracer or LangChainTracer()

    def observe_retrieval(
        self,
        query: str,
        retrieved_docs: list,
        k: int,
    ):
        """Observe retrieval operation."""
        self.tracer.add_span(
            name="retrieval",
            span_type="retrieval",
            input_data={"query": query, "k": k},
            output_data={
                "num_docs": len(retrieved_docs),
                "docs_preview": [
                    doc.page_content[:100] for doc in retrieved_docs
                ],
            },
        )

    def observe_generation(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
    ):
        """Observe generation operation."""
        self.tracer.add_span(
            name="generation",
            span_type="generation",
            input_data={
                "prompt_length": len(prompt),
                "model": model,
            },
            output_data={
                "response_length": len(response),
                "latency_ms": latency_ms,
            },
        )

    def observe_rag_pipeline(
        self,
        query: str,
        answer: str,
        sources: list,
        latency_ms: float,
    ):
        """Observe full RAG pipeline."""
        self.tracer.add_span(
            name="rag_pipeline",
            span_type="pipeline",
            input_data={"query": query},
            output_data={
                "answer_length": len(answer),
                "num_sources": len(sources),
                "latency_ms": latency_ms,
            },
        )


def setup_langsmith_tracing(
    api_key: Optional[str] = None,
    project_name: str = "battery-llm",
):
    """
    Setup LangSmith tracing.

    Args:
        api_key: LangSmith API key
        project_name: Project name for LangSmith
    """
    # LangSmith tracing is enabled via environment variables
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    print(f"LangSmith tracing enabled for project: {project_name}")


def create_observability() -> LLMObservability:
    """Create observability instance."""
    return LLMObservability()
