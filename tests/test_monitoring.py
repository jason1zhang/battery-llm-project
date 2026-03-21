"""
Unit tests for monitoring and evaluation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.monitoring.logger import (
    StructuredLogger,
    setup_logging,
    MetricsCollector,
    log_execution_time,
)
from src.monitoring.tracer import (
    LangChainTracer,
    LLMObservability,
)


class TestStructuredLogger:
    """Test structured logger."""

    def test_logger_creation(self, tmp_path):
        """Test logger can be created."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
        )

        assert logger is not None

    def test_logging(self, tmp_path):
        """Test basic logging."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            level=10,  # DEBUG
        )

        logger.info("Test message", key="value")

        # Check log file was created
        log_files = list(tmp_path.glob("test_logger_*.log"))
        assert len(log_files) > 0


class TestMetricsCollector:
    """Test metrics collector."""

    def test_metrics_creation(self):
        """Test metrics collector can be created."""
        collector = MetricsCollector()

        assert collector is not None

    def test_record_request(self):
        """Test recording requests."""
        collector = MetricsCollector()

        collector.record_request(100.0, success=True)

        assert collector.metrics["requests"] == 1

    def test_get_stats(self):
        """Test getting statistics."""
        collector = MetricsCollector()

        collector.record_request(100.0)
        collector.record_request(200.0)

        stats = collector.get_stats()

        assert stats["total_requests"] == 2
        assert stats["avg_latency_ms"] == 150.0


class TestLangChainTracer:
    """Test LangChain tracer."""

    def test_tracer_creation(self, tmp_path):
        """Test tracer can be created."""
        tracer = LangChainTracer(trace_dir=str(tmp_path))

        assert tracer is not None

    def test_start_end_trace(self, tmp_path):
        """Test starting and ending trace."""
        tracer = LangChainTracer(trace_dir=str(tmp_path))

        tracer.start_trace("test_trace")
        tracer.add_span("test_span", "test_type", input_data={"key": "value"})
        tracer.end_trace()

        # Check trace file was created
        trace_files = list(tmp_path.glob("trace_*.json"))
        assert len(trace_files) > 0


class TestLLMObservability:
    """Test LLM observability."""

    def test_observability_creation(self):
        """Test observability can be created."""
        obs = LLMObservability()

        assert obs is not None

    def test_observe_retrieval(self):
        """Test observing retrieval."""
        tracer = LangChainTracer(trace_dir=tempfile.mkdtemp())
        obs = LLMObservability(tracer=tracer)

        docs = [
            type("Doc", (), {"page_content": "test content"})(),
        ]

        obs.observe_retrieval("test query", docs, k=4)

        # Trace should be saved (not None means it exists)
        assert tracer.current_trace is not None or len(tracer.traces) > 0


class TestDecorators:
    """Test decorators."""

    def test_log_execution_time(self, tmp_path):
        """Test execution time logging."""
        logger = StructuredLogger(
            name="test",
            log_dir=str(tmp_path),
            level=10,
        )

        @log_execution_time(logger)
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
