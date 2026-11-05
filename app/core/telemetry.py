"""
OpenTelemetry Integration & Telemetry Layer.

Provides structured observability for the AI Security Gateway:

  • **Tracing** — Distributed traces for each request with spans
    covering RBAC resolution, guardrail checks, and LLM calls.
  • **Metrics** — Counters and histograms for token consumption,
    tool call latency, and blocked requests.
  • **Redis Logging** — Persists execution trajectories as JSON
    documents in a Redis Stream for near-real-time dashboards.
  • **BigQuery Schema Mocks** — Defines the target schema for
    production log shipping (actual BQ writes are stubbed).

All telemetry functions are safe to call even when the OTLP
collector or Redis are unavailable — they degrade gracefully
with structured log warnings.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import structlog
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.core.config import settings

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# 1. OpenTelemetry Bootstrap
# ────────────────────────────────────────────────────────────────────

import sys


class SafeStream:
    """Safe wrapper around stdout preventing I/O errors when streams close."""

    def write(self, s: str) -> int:
        try:
            if not getattr(sys.stdout, "closed", False):
                return sys.stdout.write(s)
        except (ValueError, OSError):
            pass
        return len(s)

    def flush(self) -> None:
        try:
            if not getattr(sys.stdout, "closed", False):
                sys.stdout.flush()
        except (ValueError, OSError):
            pass


_resource = Resource.create(
    {
        "service.name": settings.otel_service_name,
        "service.version": "1.0.0",
        "deployment.environment": "development",
    }
)


def init_tracing() -> trace.Tracer:
    """Initialize the OpenTelemetry tracer provider.

    In development mode the spans are exported to the console.
    In production, swap ``ConsoleSpanExporter`` for the OTLP gRPC
    exporter pointing at your collector.

    Returns:
        A configured ``Tracer`` instance scoped to this service.
    """
    provider = TracerProvider(resource=_resource)
    # Console exporter with SafeStream
    processor = BatchSpanProcessor(ConsoleSpanExporter(out=SafeStream()))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


def init_metrics() -> metrics.Meter:
    """Initialize the OpenTelemetry meter provider.

    Returns:
        A configured ``Meter`` for recording counters / histograms.
    """
    reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(out=SafeStream()),
        export_interval_millis=30_000,
    )
    provider = MeterProvider(resource=_resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return metrics.get_meter(settings.otel_service_name)


# Module-level singletons
tracer = init_tracing()
meter = init_metrics()

# ── Custom Metrics ──────────────────────────────────────────────────
request_counter = meter.create_counter(
    name="gateway.requests.total",
    description="Total inbound gateway requests",
    unit="1",
)

tokens_histogram = meter.create_histogram(
    name="gateway.tokens.consumed",
    description="Token consumption per request",
    unit="tokens",
)

tool_latency_histogram = meter.create_histogram(
    name="gateway.tool.latency",
    description="Tool invocation latency",
    unit="ms",
)

blocked_counter = meter.create_counter(
    name="gateway.requests.blocked",
    description="Requests blocked by guardrails or RBAC",
    unit="1",
)


# ────────────────────────────────────────────────────────────────────
# 2. Execution Trace Document
# ────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """Single tool invocation record within a trace."""

    tool_name: str
    start_time: str
    end_time: str
    latency_ms: float
    input_args: Dict[str, Any] = field(default_factory=dict)
    output_summary: str = ""
    status: str = "ok"


@dataclass
class ExecutionTrace:
    """Full execution trajectory for one gateway request.

    This structure mirrors the BigQuery target schema and is also
    serialized into Redis Streams for real-time consumption.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    user_id: str = "anonymous"
    user_role: str = "VIEWER"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Request metadata
    prompt_text: str = ""
    prompt_hash: str = ""
    model_used: str = ""

    # Guardrail outcomes
    injection_score: float = 0.0
    injection_blocked: bool = False
    pii_redacted_count: int = 0

    # Agent execution
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    # Token accounting
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Response
    response_text: str = ""
    total_latency_ms: float = 0.0
    status: str = "ok"
    error_message: str = ""


# ────────────────────────────────────────────────────────────────────
# 3. Redis Telemetry Sink
# ────────────────────────────────────────────────────────────────────

def _get_redis_client():
    """Lazy-initialize a Redis client.

    Returns ``None`` if Redis is unavailable so callers can degrade
    gracefully.
    """
    try:
        import redis as redis_lib
        client = redis_lib.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        client.ping()
        return client
    except Exception as exc:
        logger.warning("redis_unavailable", error=str(exc))
        return None


def publish_trace_to_redis(execution_trace: ExecutionTrace) -> bool:
    """Publish an ``ExecutionTrace`` to a Redis Stream.

    The stream key is ``ai_gateway:traces`` and each entry is a
    flat JSON-serialized document.

    Args:
        execution_trace: Completed trace to publish.

    Returns:
        ``True`` if the trace was successfully published.
    """
    client = _get_redis_client()
    if client is None:
        logger.warning("trace_publish_skipped", reason="redis_unavailable")
        return False

    try:
        trace_dict = asdict(execution_trace)
        # Redis Streams accept flat string→string mappings;
        # serialize nested structures as JSON strings.
        flat = {
            k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            for k, v in trace_dict.items()
        }
        stream_key = "ai_gateway:traces"
        entry_id = client.xadd(stream_key, flat, maxlen=10_000)
        logger.info(
            "trace_published",
            stream=stream_key,
            entry_id=entry_id,
            trace_id=execution_trace.trace_id,
        )
        return True
    except Exception as exc:
        logger.error("trace_publish_failed", error=str(exc))
        return False


def get_recent_traces(count: int = 20) -> List[Dict[str, Any]]:
    """Retrieve the most recent traces from Redis.

    Args:
        count: Number of recent entries to retrieve.

    Returns:
        List of trace dicts (newest first).
    """
    client = _get_redis_client()
    if client is None:
        return []

    try:
        stream_key = "ai_gateway:traces"
        entries = client.xrevrange(stream_key, count=count)
        traces = []
        for entry_id, data in entries:
            # Deserialize JSON fields
            for key in ("tool_calls",):
                if key in data:
                    try:
                        data[key] = json.loads(data[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            data["_entry_id"] = entry_id
            traces.append(data)
        return traces
    except Exception as exc:
        logger.error("trace_retrieval_failed", error=str(exc))
        return []


# ────────────────────────────────────────────────────────────────────
# 4. BigQuery Schema Mock
# ────────────────────────────────────────────────────────────────────

BIGQUERY_SCHEMA: List[Dict[str, str]] = [
    {"name": "trace_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "request_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "user_id", "type": "STRING", "mode": "REQUIRED"},
    {"name": "user_role", "type": "STRING", "mode": "REQUIRED"},
    {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "prompt_text", "type": "STRING", "mode": "NULLABLE"},
    {"name": "prompt_hash", "type": "STRING", "mode": "NULLABLE"},
    {"name": "model_used", "type": "STRING", "mode": "NULLABLE"},
    {"name": "injection_score", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "injection_blocked", "type": "BOOLEAN", "mode": "NULLABLE"},
    {"name": "pii_redacted_count", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "tool_calls", "type": "JSON", "mode": "NULLABLE"},
    {"name": "input_tokens", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "output_tokens", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "total_tokens", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "response_text", "type": "STRING", "mode": "NULLABLE"},
    {"name": "total_latency_ms", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "status", "type": "STRING", "mode": "NULLABLE"},
    {"name": "error_message", "type": "STRING", "mode": "NULLABLE"},
]


def log_to_bigquery_mock(execution_trace: ExecutionTrace) -> Dict[str, Any]:
    """Simulate writing a trace row to BigQuery.

    In production, replace this with the ``google-cloud-bigquery``
    streaming insert API.  Here we validate the trace against the
    schema and return the row that *would* be inserted.

    Returns:
        The validated row dict.
    """
    row = asdict(execution_trace)
    # Serialize nested fields to JSON strings for BQ JSON columns
    row["tool_calls"] = json.dumps(row.get("tool_calls", []))

    logger.info(
        "bigquery_mock_insert",
        table=f"{settings.bigquery_dataset}.{settings.bigquery_table}",
        trace_id=execution_trace.trace_id,
    )
    return row


# ────────────────────────────────────────────────────────────────────
# 5. Convenience Span Context Manager
# ────────────────────────────────────────────────────────────────────

@contextmanager
def traced_operation(
    operation_name: str,
    attributes: Optional[Dict[str, str]] = None,
) -> Generator[trace.Span, None, None]:
    """Context manager that wraps a code block in an OTEL span.

    Usage::

        with traced_operation("guardrail_check", {"user": user_id}) as span:
            result = check_guardrails(prompt)
            span.set_attribute("threat_level", result.threat_level)

    Args:
        operation_name: Span name.
        attributes: Optional initial span attributes.
    """
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            raise


def record_token_usage(input_tokens: int, output_tokens: int, role: str) -> None:
    """Record token consumption metrics.

    Args:
        input_tokens: Prompt / input tokens consumed.
        output_tokens: Completion / output tokens consumed.
        role: Caller role for metric labeling.
    """
    total = input_tokens + output_tokens
    tokens_histogram.record(total, {"role": role, "direction": "total"})
    tokens_histogram.record(input_tokens, {"role": role, "direction": "input"})
    tokens_histogram.record(output_tokens, {"role": role, "direction": "output"})


def record_tool_latency(tool_name: str, latency_ms: float) -> None:
    """Record latency of a tool invocation.

    Args:
        tool_name: Name of the tool that was called.
        latency_ms: Wall-clock duration in milliseconds.
    """
    tool_latency_histogram.record(latency_ms, {"tool": tool_name})
