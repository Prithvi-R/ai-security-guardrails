"""
Unit tests for the RBAC module.

Covers identity resolution, policy enforcement, context filtering,
and the middleware integration.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from jose import jwt
from starlette.testclient import TestClient

from app.core.config import settings
from app.core.rbac import (
    Role,
    RolePolicy,
    ROLE_MAP,
    ROLE_POLICIES,
    UserIdentity,
    enforce_eval_access,
    enforce_tool_access,
    enforce_topic_access,
    filter_context_for_role,
    resolve_identity,
)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_jwt(sub: str = "user-123", role: str = "analyst") -> str:
    """Create a valid JWT token for testing."""
    payload = {"sub": sub, "role": role}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


class _MockRequest:
    """Minimal request mock for ``resolve_identity``."""

    def __init__(self, headers: dict):
        self.headers = headers


# ────────────────────────────────────────────────────────────────────
# Test: Role Mapping
# ────────────────────────────────────────────────────────────────────

class TestRoleMapping:
    """Verify the role enumeration and mapping."""

    def test_all_roles_in_map(self):
        for role in Role:
            assert role.name.lower() in ROLE_MAP

    def test_role_ordering(self):
        assert Role.VIEWER < Role.ANALYST < Role.EXECUTIVE < Role.ADMIN

    def test_every_role_has_policy(self):
        for role in Role:
            assert role in ROLE_POLICIES


# ────────────────────────────────────────────────────────────────────
# Test: Identity Resolution
# ────────────────────────────────────────────────────────────────────

class TestResolveIdentity:
    """Test identity resolution from headers and tokens."""

    def test_jwt_bearer_token(self):
        token = _make_jwt(sub="alice", role="executive")
        request = _MockRequest({"authorization": f"Bearer {token}"})
        identity = resolve_identity(request)
        assert identity.user_id == "alice"
        assert identity.role == Role.EXECUTIVE

    def test_x_user_role_header(self):
        request = _MockRequest({
            "x-user-role": "Admin",
            "x-user-id": "bob",
        })
        identity = resolve_identity(request)
        assert identity.user_id == "bob"
        assert identity.role == Role.ADMIN

    def test_fallback_anonymous_viewer(self):
        request = _MockRequest({})
        identity = resolve_identity(request)
        assert identity.user_id == "anonymous"
        assert identity.role == Role.VIEWER

    def test_invalid_role_defaults_to_viewer(self):
        request = _MockRequest({"x-user-role": "SuperAdmin"})
        identity = resolve_identity(request)
        assert identity.role == Role.VIEWER

    def test_invalid_jwt_raises_401(self):
        request = _MockRequest({"authorization": "Bearer invalid.token.here"})
        with pytest.raises(HTTPException) as exc_info:
            resolve_identity(request)
        assert exc_info.value.status_code == 401


# ────────────────────────────────────────────────────────────────────
# Test: Tool Access Enforcement
# ────────────────────────────────────────────────────────────────────

class TestToolAccess:
    """Test tool-level RBAC enforcement."""

    def test_viewer_allowed_tool(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.VIEWER,
            policy=ROLE_POLICIES[Role.VIEWER],
        )
        # Should not raise
        enforce_tool_access(identity, "search_documents")

    def test_viewer_denied_tool(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.VIEWER,
            policy=ROLE_POLICIES[Role.VIEWER],
        )
        with pytest.raises(HTTPException) as exc_info:
            enforce_tool_access(identity, "query_database")
        assert exc_info.value.status_code == 403

    def test_admin_wildcard_access(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.ADMIN,
            policy=ROLE_POLICIES[Role.ADMIN],
        )
        # Admin should access any tool
        enforce_tool_access(identity, "any_tool_name")

    def test_analyst_partial_access(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.ANALYST,
            policy=ROLE_POLICIES[Role.ANALYST],
        )
        enforce_tool_access(identity, "query_database")
        with pytest.raises(HTTPException):
            enforce_tool_access(identity, "financial_forecast")


# ────────────────────────────────────────────────────────────────────
# Test: Topic Access Enforcement
# ────────────────────────────────────────────────────────────────────

class TestTopicAccess:
    """Test topic-level RBAC enforcement."""

    def test_viewer_public_topic(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.VIEWER,
            policy=ROLE_POLICIES[Role.VIEWER],
        )
        enforce_topic_access(identity, "public_reports")

    def test_viewer_restricted_topic(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.VIEWER,
            policy=ROLE_POLICIES[Role.VIEWER],
        )
        with pytest.raises(HTTPException) as exc_info:
            enforce_topic_access(identity, "executive_briefings")
        assert exc_info.value.status_code == 403


# ────────────────────────────────────────────────────────────────────
# Test: Context Filtering
# ────────────────────────────────────────────────────────────────────

class TestContextFiltering:
    """Test context chunk filtering by role."""

    def test_viewer_sees_only_public(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.VIEWER,
            policy=ROLE_POLICIES[Role.VIEWER],
        )
        chunks = [
            {"topic": "public_reports", "content": "Public data"},
            {"topic": "executive_briefings", "content": "Secret data"},
            {"topic": "faq", "content": "FAQ content"},
        ]
        filtered = filter_context_for_role(identity, chunks)
        assert len(filtered) == 2
        topics = {c["topic"] for c in filtered}
        assert "executive_briefings" not in topics

    def test_admin_sees_everything(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.ADMIN,
            policy=ROLE_POLICIES[Role.ADMIN],
        )
        chunks = [
            {"topic": "public_reports", "content": "Public data"},
            {"topic": "executive_briefings", "content": "Secret data"},
            {"topic": "m_and_a", "content": "M&A data"},
        ]
        filtered = filter_context_for_role(identity, chunks)
        assert len(filtered) == 3


# ────────────────────────────────────────────────────────────────────
# Test: Eval Access
# ────────────────────────────────────────────────────────────────────

class TestEvalAccess:
    """Test evaluation trigger access control."""

    def test_analyst_cannot_run_eval(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.ANALYST,
            policy=ROLE_POLICIES[Role.ANALYST],
        )
        with pytest.raises(HTTPException) as exc_info:
            enforce_eval_access(identity)
        assert exc_info.value.status_code == 403

    def test_executive_can_run_eval(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.EXECUTIVE,
            policy=ROLE_POLICIES[Role.EXECUTIVE],
        )
        # Should not raise
        enforce_eval_access(identity)

    def test_admin_can_run_eval(self):
        identity = UserIdentity(
            user_id="test",
            role=Role.ADMIN,
            policy=ROLE_POLICIES[Role.ADMIN],
        )
        enforce_eval_access(identity)
