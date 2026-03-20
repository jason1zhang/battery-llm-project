"""
Logging and monitoring utilities.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import time


class StructuredLogger:
    """Structured logging for the application."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def log(
        self,
        level: int,
        message: str,
        **kwargs,
    ):
        """Log with structured data."""
        extra = {"structured_data": json.dumps(kwargs)} if kwargs else {}
        self.logger.log(level, message, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error."""
        self.log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug."""
        self.log(logging.DEBUG, message, **kwargs)


def setup_logging(
    name: str = "battery_llm",
    log_dir: str = "logs",
    level: str = "INFO",
) -> StructuredLogger:
    """
    Setup application logging.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Log level

    Returns:
        StructuredLogger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    return StructuredLogger(name, log_dir, log_level)


class RequestLogger:
    """Middleware for logging API requests."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_request(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict] = None,
    ):
        """Log incoming request."""
        self.logger.info(
            "API Request",
            endpoint=endpoint,
            method=method,
            params=params or {},
            timestamp=datetime.now().isoformat(),
        )

    def log_response(
        self,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        error: Optional[str] = None,
    ):
        """Log response."""
        self.logger.info(
            "API Response",
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=duration_ms,
            error=error,
            timestamp=datetime.now().isoformat(),
        )


def log_execution_time(logger: StructuredLogger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.info(
                    f"Function {func.__name__} completed",
                    duration_ms=duration,
                    status="success",
                )
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {func.__name__} failed",
                    duration_ms=duration,
                    status="error",
                    error=str(e),
                )
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """Collect application metrics."""

    def __init__(self):
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_latency": 0.0,
            "latencies": [],
            "model_inferences": 0,
            "retrievals": 0,
        }

    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request."""
        self.metrics["requests"] += 1
        self.metrics["total_latency"] += latency_ms
        self.metrics["latencies"].append(latency_ms)

        if not success:
            self.metrics["errors"] += 1

    def record_model_inference(self):
        """Record model inference."""
        self.metrics["model_inferences"] += 1

    def record_retrieval(self):
        """Record retrieval operation."""
        self.metrics["retrievals"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        latencies = self.metrics["latencies"]
        avg_latency = (
            self.metrics["total_latency"] / self.metrics["requests"]
            if self.metrics["requests"] > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests"],
            "total_errors": self.metrics["errors"],
            "error_rate": (
                self.metrics["errors"] / self.metrics["requests"]
                if self.metrics["requests"] > 0
                else 0
            ),
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": (
                sorted(latencies)[len(latencies) // 2] if latencies else 0
            ),
            "p95_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            ),
            "p99_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
            ),
            "total_model_inferences": self.metrics["model_inferences"],
            "total_retrievals": self.metrics["retrievals"],
        }

    def reset(self):
        """Reset metrics."""
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_latency": 0.0,
            "latencies": [],
            "model_inferences": 0,
            "retrievals": 0,
        }
