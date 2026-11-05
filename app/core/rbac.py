"""
IAM / Role-Based Access Control (RBAC) Middleware.

This module implements identity-aware security for the AI Gateway.
It extracts user identity from JWT bearer tokens or explicit
``X-User-Role`` headers and enforces per-role policies that control:

  • Which tools an agent is allowed to invoke.
  • Which context topics are visible to the caller.
  • Maximum token budgets per request.

Roles hierarchy (least → most privileged):
  Viewer → Analyst → Executive → Admin

Usage:
  Attach ``RBACMiddleware`` to the FastAPI app, or call
  ``enforce_rbac(request)`` from individual route handlers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import settings


# ────────────────────────────────────────────────────────────────────
# 1. Role Definitions
# ────────────────────────────────────────────────────────────────────

class Role(IntEnum):
    """Enumerated roles ordered by privilege level."""

    VIEWER = 0
    ANALYST = 1
    EXECUTIVE = 2
    ADMIN = 3


# Convenient string → enum mapping
ROLE_MAP: Dict[str, Role] = {r.name.lower(): r for r in Role}


# ────────────────────────────────────────────────────────────────────
# 2. Per-Role Policy Definitions
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RolePolicy:
    """Immutable policy attached to a role.

    Attributes:
        allowed_tools:      Set of tool names the role may invoke.
        allowed_topics:     Context topics the role may reference.
        max_tokens:         Maximum output token budget per request.
        can_run_eval:       Whether the role can trigger eval runs.
        can_access_pii:     Whether raw PII is visible (vs. redacted).
    """

    allowed_tools: Set[str] = field(default_factory=set)
    allowed_topics: Set[str] = field(default_factory=set)
    max_tokens: int = 1024
    can_run_eval: bool = False
    can_access_pii: bool = False


# ── Policy registry ────────────────────────────────────────────────
ROLE_POLICIES: Dict[Role, RolePolicy] = {
    Role.VIEWER: RolePolicy(
        allowed_tools={"search_documents"},
        allowed_topics={"public_reports", "faq"},
        max_tokens=512,
        can_run_eval=False,
        can_access_pii=False,
    ),
    Role.ANALYST: RolePolicy(
        allowed_tools={"search_documents", "query_database", "generate_chart"},
        allowed_topics={"public_reports", "faq", "internal_analytics", "market_data"},
        max_tokens=2048,
        can_run_eval=False,
        can_access_pii=False,
    ),
    Role.EXECUTIVE: RolePolicy(
        allowed_tools={
            "search_documents", "query_database", "generate_chart",
            "financial_forecast", "strategic_summary",
        },
        allowed_topics={
            "public_reports", "faq", "internal_analytics",
            "market_data", "executive_briefings", "m_and_a",
        },
        max_tokens=4096,
        can_run_eval=True,
        can_access_pii=False,
    ),
    Role.ADMIN: RolePolicy(
        allowed_tools={"*"},  # wildcard – all tools permitted
        allowed_topics={"*"},  # wildcard – all topics permitted
        max_tokens=8192,
        can_run_eval=True,
        can_access_pii=True,
    ),
}


# ────────────────────────────────────────────────────────────────────
# 3. Identity Extraction Helpers
# ────────────────────────────────────────────────────────────────────

@dataclass
class UserIdentity:
    """Resolved identity of the inbound caller."""

    user_id: str
    role: Role
    policy: RolePolicy
    raw_claims: Dict[str, Any] = field(default_factory=dict)


def _decode_jwt(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT bearer token.

    Returns:
        Dict of JWT claims including ``sub`` and ``role``.

    Raises:
        HTTPException 401 on invalid / expired tokens.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def resolve_identity(request: Request) -> UserIdentity:
    """Resolve a ``UserIdentity`` from the inbound request.

    Resolution order:
      1. ``Authorization: Bearer <jwt>`` header → decode JWT claims.
      2. ``X-User-Role`` header → trust upstream identity proxy.
      3. Fall back to ``VIEWER`` with anonymous user id.

    Returns:
        Fully resolved ``UserIdentity`` with the matching policy.
    """
    # ── Attempt JWT bearer token first ──────────────────────────────
    auth_header: Optional[str] = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
        claims = _decode_jwt(token)
        role_name = claims.get("role", "viewer").lower()
        role = ROLE_MAP.get(role_name, Role.VIEWER)
        return UserIdentity(
            user_id=claims.get("sub", "unknown"),
            role=role,
            policy=ROLE_POLICIES[role],
            raw_claims=claims,
        )

    # ── Fallback: explicit header from identity-aware proxy ─────────
    role_header: Optional[str] = request.headers.get("x-user-role")
    user_id_header: Optional[str] = request.headers.get("x-user-id", "anonymous")

    if role_header:
        role = ROLE_MAP.get(role_header.strip().lower(), Role.VIEWER)
        return UserIdentity(
            user_id=user_id_header,
            role=role,
            policy=ROLE_POLICIES[role],
        )

    # ── Default: unauthenticated viewer ─────────────────────────────
    return UserIdentity(
        user_id="anonymous",
        role=Role.VIEWER,
        policy=ROLE_POLICIES[Role.VIEWER],
    )


# ────────────────────────────────────────────────────────────────────
# 4. Policy Enforcement
# ────────────────────────────────────────────────────────────────────

def enforce_tool_access(identity: UserIdentity, tool_name: str) -> None:
    """Raise 403 if the caller's role does not permit ``tool_name``.

    The Admin role uses a wildcard (``*``) and always passes.
    """
    allowed = identity.policy.allowed_tools
    if "*" not in allowed and tool_name not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{identity.role.name}' is not authorized to use "
                f"tool '{tool_name}'. Allowed tools: {sorted(allowed)}"
            ),
        )


def enforce_topic_access(identity: UserIdentity, topic: str) -> None:
    """Raise 403 if the caller's role cannot access ``topic``."""
    allowed = identity.policy.allowed_topics
    if "*" not in allowed and topic not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{identity.role.name}' cannot access topic "
                f"'{topic}'. Allowed topics: {sorted(allowed)}"
            ),
        )


def filter_context_for_role(
    identity: UserIdentity,
    context_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter a list of context chunks to those the role may see.

    Each chunk is expected to carry a ``"topic"`` key.  Chunks whose
    topic is not in the policy's ``allowed_topics`` are silently
    dropped.  Admin (wildcard) sees everything.

    Returns:
        Filtered list of context dicts.
    """
    allowed = identity.policy.allowed_topics
    if "*" in allowed:
        return context_chunks

    return [
        chunk for chunk in context_chunks
        if chunk.get("topic", "public_reports") in allowed
    ]


def enforce_eval_access(identity: UserIdentity) -> None:
    """Raise 403 if the caller is not permitted to trigger evaluations."""
    if not identity.policy.can_run_eval:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Role '{identity.role.name}' is not authorized to "
                "trigger evaluation runs."
            ),
        )


# ────────────────────────────────────────────────────────────────────
# 5. FastAPI / Starlette Middleware
# ────────────────────────────────────────────────────────────────────

class RBACMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that resolves identity and injects it
    into ``request.state`` for downstream route handlers.

    Every request is stamped with:
      • ``request.state.identity``  – ``UserIdentity``
      • ``request.state.rbac_ts``   – epoch timestamp of resolution
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip health-check & docs endpoints
        if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        try:
            identity = resolve_identity(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        # Attach to request state for use by route handlers
        request.state.identity = identity
        request.state.rbac_ts = time.time()

        response = await call_next(request)
        # Inject identity headers into the response for audit trails
        response.headers["X-Resolved-Role"] = identity.role.name
        response.headers["X-Resolved-User"] = identity.user_id
        return response
