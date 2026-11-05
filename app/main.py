"""
FastAPI Application — Enterprise AI Security & Governance Gateway.

Exposes two primary endpoints:

  • ``POST /v1/chat/completions`` — RBAC-protected chat completion
    endpoint that routes requests through the ADK agent pipeline
    with guardrails and telemetry.

  • ``POST /v1/eval/run`` — Triggers an automated evaluation run
    against the golden benchmark dataset (restricted to Executive
    and Admin roles).

Supporting endpoints:
  • ``GET  /health``          — Liveness probe.
  • ``GET  /v1/traces``       — Retrieve recent telemetry traces.
  • ``GET  /v1/rbac/whoami``  — Inspect resolved identity/policy.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.adk_agent import build_agent, create_runner, run_agent_query
from app.core.config import settings
from app.core.guardrails import detect_prompt_injection, redact_pii
from app.core.rbac import (
    RBACMiddleware,
    Role,
    UserIdentity,
    enforce_eval_access,
    enforce_tool_access,
    filter_context_for_role,
    resolve_identity,
)
from app.core.telemetry import (
    ExecutionTrace,
    ToolCallRecord,
    blocked_counter,
    get_recent_traces,
    log_to_bigquery_mock,
    publish_trace_to_redis,
    record_token_usage,
    request_counter,
    traced_operation,
)
from app.eval.eval_harness import EvalHarness

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Request / Response Models
# ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Inbound chat completion request body."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="User's natural-language message.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity.",
    )
    context: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional context chunks (each with a 'topic' key).",
    )


class ChatResponse(BaseModel):
    """Outbound chat completion response."""

    request_id: str
    response_text: str
    model: str
    session_id: str
    user_role: str
    tools_invoked: List[Dict[str, Any]] = []
    guardrail_status: Dict[str, Any] = {}
    latency_ms: float = 0.0


class EvalRequest(BaseModel):
    """Request body for triggering an evaluation run."""

    dataset_path: Optional[str] = Field(
        default=None,
        description="Optional path to a custom golden dataset JSON.",
    )
    use_llm_judge: bool = Field(
        default=True,
        description="Whether to use Gemini as an evaluator judge.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = settings.otel_service_name
    version: str = "1.0.0"
    timestamp: float = 0.0


class WhoAmIResponse(BaseModel):
    """RBAC identity inspection response."""

    user_id: str
    role: str
    allowed_tools: List[str]
    allowed_topics: List[str]
    max_tokens: int
    can_run_eval: bool
    can_access_pii: bool


# ────────────────────────────────────────────────────────────────────
# Application Lifespan
# ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    logger.info(
        "gateway_starting",
        host=settings.host,
        port=settings.port,
        model=settings.gemini_model,
    )
    # Pre-warm the default agent & runner
    app.state.default_agent = build_agent()
    app.state.default_runner = create_runner(app.state.default_agent)
    logger.info("gateway_ready")
    yield
    logger.info("gateway_shutting_down")


# ────────────────────────────────────────────────────────────────────
# FastAPI Application
# ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise AI Security & Governance Gateway",
    description=(
        "Identity-aware security proxy, observability layer, and "
        "evaluation gateway for enterprise Gemini AI agents."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware Stack ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RBACMiddleware)


# ────────────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe for load balancers and orchestrators."""
    return HealthResponse(timestamp=time.time())


# ────────────────────────────────────────────────────────────────────
# RBAC Identity Inspection
# ────────────────────────────────────────────────────────────────────

@app.get("/v1/rbac/whoami", response_model=WhoAmIResponse, tags=["RBAC"])
async def whoami(request: Request):
    """Return the resolved identity and policy for the current caller.

    Useful for debugging RBAC headers and token claims.
    """
    identity: UserIdentity = request.state.identity
    return WhoAmIResponse(
        user_id=identity.user_id,
        role=identity.role.name,
        allowed_tools=sorted(identity.policy.allowed_tools),
        allowed_topics=sorted(identity.policy.allowed_topics),
        max_tokens=identity.policy.max_tokens,
        can_run_eval=identity.policy.can_run_eval,
        can_access_pii=identity.policy.can_access_pii,
    )


# ────────────────────────────────────────────────────────────────────
# Chat Completions Endpoint
# ────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="RBAC-protected chat completion",
)
async def chat_completions(body: ChatRequest, request: Request):
    """Process a chat completion request through the secured pipeline.

    Pipeline stages:
      1. RBAC identity resolution (via middleware).
      2. Context filtering based on caller role.
      3. Prompt injection detection.
      4. PII redaction.
      5. ADK agent execution with tool-level RBAC.
      6. Telemetry recording to Redis and BigQuery mock.
    """
    request_id = str(uuid.uuid4())
    identity: UserIdentity = request.state.identity
    start_ts = time.time()

    # Increment request counter
    request_counter.add(1, {"role": identity.role.name, "endpoint": "chat"})

    with traced_operation(
        "chat_completion",
        {"request_id": request_id, "role": identity.role.name},
    ) as span:

        # ── 1. Context Filtering ───────────────────────────────────
        filtered_context = []
        if body.context:
            filtered_context = filter_context_for_role(identity, body.context)
            span.set_attribute("context.original", len(body.context))
            span.set_attribute("context.filtered", len(filtered_context))

        # ── 2. Guardrail Checks ────────────────────────────────────
        injection_result = detect_prompt_injection(body.message)
        pii_result = redact_pii(body.message)

        guardrail_status = {
            "injection_threat_level": injection_result.threat_level.value,
            "injection_score": injection_result.score,
            "injection_blocked": injection_result.blocked,
            "pii_redacted_count": pii_result.redacted_count,
            "pii_redacted_types": pii_result.redacted_types,
        }

        if injection_result.blocked:
            blocked_counter.add(
                1, {"role": identity.role.name, "reason": "injection"}
            )
            # Build trace for blocked request
            trace = ExecutionTrace(
                request_id=request_id,
                user_id=identity.user_id,
                user_role=identity.role.name,
                prompt_text="[BLOCKED]",
                prompt_hash=hashlib.sha256(
                    body.message.encode()
                ).hexdigest()[:16],
                injection_score=injection_result.score,
                injection_blocked=True,
                status="blocked",
                total_latency_ms=round((time.time() - start_ts) * 1000, 2),
            )
            publish_trace_to_redis(trace)
            log_to_bigquery_mock(trace)

            return ChatResponse(
                request_id=request_id,
                response_text=(
                    "⚠️ Your request was blocked by the security gateway. "
                    f"Threat level: {injection_result.threat_level.value} "
                    f"(score: {injection_result.score}). "
                    "Please rephrase your query."
                ),
                model=settings.gemini_model,
                session_id=body.session_id or "none",
                user_role=identity.role.name,
                guardrail_status=guardrail_status,
                latency_ms=round((time.time() - start_ts) * 1000, 2),
            )

        # ── 3. Build RBAC-scoped Agent ─────────────────────────────
        allowed_tools = list(identity.policy.allowed_tools)
        agent = build_agent(allowed_tools=allowed_tools)
        runner = create_runner(agent)

        # ── 4. Execute Agent Pipeline ──────────────────────────────
        session_id = body.session_id or str(uuid.uuid4())
        sanitized_message = pii_result.cleaned_text

        # Prepend filtered context if available
        if filtered_context:
            context_text = "\n".join(
                f"[{c.get('topic', 'unknown')}]: {c.get('content', '')}"
                for c in filtered_context
            )
            sanitized_message = (
                f"Context:\n{context_text}\n\n"
                f"User Query: {sanitized_message}"
            )

        try:
            result = await run_agent_query(
                runner=runner,
                user_id=identity.user_id,
                session_id=session_id,
                message=sanitized_message,
            )
        except Exception as exc:
            logger.error(
                "agent_execution_failed",
                request_id=request_id,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {exc}",
            )

        # ── 5. Telemetry ───────────────────────────────────────────
        elapsed_ms = round((time.time() - start_ts) * 1000, 2)

        trace = ExecutionTrace(
            request_id=request_id,
            user_id=identity.user_id,
            user_role=identity.role.name,
            prompt_text=sanitized_message[:200],
            prompt_hash=hashlib.sha256(
                body.message.encode()
            ).hexdigest()[:16],
            model_used=settings.gemini_model,
            injection_score=injection_result.score,
            injection_blocked=False,
            pii_redacted_count=pii_result.redacted_count,
            tool_calls=[
                ToolCallRecord(
                    tool_name=tc.get("tool_name", "unknown"),
                    start_time="",
                    end_time="",
                    latency_ms=0,
                    input_args=tc.get("args", {}),
                )
                for tc in result.get("tool_calls", [])
            ],
            response_text=result.get("response_text", "")[:200],
            total_latency_ms=elapsed_ms,
            status="ok",
        )

        publish_trace_to_redis(trace)
        log_to_bigquery_mock(trace)

        span.set_attribute("latency_ms", elapsed_ms)
        span.set_attribute("tools_invoked", len(result.get("tool_calls", [])))

        return ChatResponse(
            request_id=request_id,
            response_text=result.get("response_text", ""),
            model=result.get("model", settings.gemini_model),
            session_id=session_id,
            user_role=identity.role.name,
            tools_invoked=result.get("tool_calls", []),
            guardrail_status=guardrail_status,
            latency_ms=elapsed_ms,
        )


# ────────────────────────────────────────────────────────────────────
# Evaluation Endpoint
# ────────────────────────────────────────────────────────────────────

@app.post("/v1/eval/run", tags=["Evaluation"], summary="Run evaluation suite")
async def run_evaluation(body: EvalRequest, request: Request):
    """Trigger an automated evaluation run against the golden dataset.

    Restricted to **Executive** and **Admin** roles.

    Returns:
        Full evaluation report with per-case scores and aggregates.
    """
    identity: UserIdentity = request.state.identity
    enforce_eval_access(identity)

    request_counter.add(1, {"role": identity.role.name, "endpoint": "eval"})

    with traced_operation(
        "eval_run",
        {"role": identity.role.name},
    ):
        harness = EvalHarness(
            dataset_path=body.dataset_path,
            use_llm_judge=body.use_llm_judge,
        )
        report = await harness.run_evaluation()

    return asdict(report)


# ────────────────────────────────────────────────────────────────────
# Traces Endpoint
# ────────────────────────────────────────────────────────────────────

@app.get("/v1/traces", tags=["Telemetry"], summary="Get recent traces")
async def get_traces(request: Request, count: int = 20):
    """Retrieve recent execution traces from Redis.

    Args:
        count: Number of recent traces to return (max 100).
    """
    count = min(count, 100)
    traces = get_recent_traces(count)
    return {"traces": traces, "count": len(traces)}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=True,
    )
