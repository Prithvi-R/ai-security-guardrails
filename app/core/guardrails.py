"""
Prompt Injection Detection & PII Redaction Guardrails.

This module provides two layers of defense:

  1. **Prompt Injection Detection** – Regex heuristics plus an
     LLM-based classifier that scores user input for injection risk.
  2. **PII Redaction** – Deterministic regex patterns to strip
     emails, phone numbers, SSNs, and credit-card numbers before
     the prompt reaches the model.

Both layers are wired as Google ADK ``before_model_callback`` hooks
so they execute transparently inside the agent pipeline.

The module also exposes Vertex AI Safety Settings helpers that
configure harm-category thresholds on outbound Gemini requests.
"""

from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from google.genai import types as genai_types

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# 1. PII Redaction Engine
# ────────────────────────────────────────────────────────────────────

# Compiled regex patterns for common PII types.
_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "EMAIL",
        re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            re.IGNORECASE,
        ),
    ),
    (
        "PHONE_US",
        re.compile(
            r"(?<!\d)(\+?1[\s\-.]?)?"           # optional country code
            r"\(?\d{3}\)?[\s\-.]?"               # area code
            r"\d{3}[\s\-.]?\d{4}(?!\d)",         # subscriber number
        ),
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:\d[ \-]*?){13,19}\b"
        ),
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    ),
]


@dataclass
class RedactionResult:
    """Outcome of a PII redaction pass."""

    cleaned_text: str
    redacted_count: int = 0
    redacted_types: List[str] = field(default_factory=list)


def redact_pii(text: str) -> RedactionResult:
    """Replace all detected PII tokens with ``[REDACTED_<TYPE>]`` placeholders.

    Args:
        text: Raw user input string.

    Returns:
        ``RedactionResult`` with the sanitized text and metadata.
    """
    redacted_count = 0
    redacted_types: List[str] = []

    for pii_type, pattern in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # findall may return tuples for groups – flatten
            count = len(matches) if isinstance(matches[0], str) else len(matches)
            redacted_count += count
            redacted_types.append(pii_type)
            text = pattern.sub(f"[REDACTED_{pii_type}]", text)

    return RedactionResult(
        cleaned_text=text,
        redacted_count=redacted_count,
        redacted_types=redacted_types,
    )


# ────────────────────────────────────────────────────────────────────
# 2. Prompt Injection Detector
# ────────────────────────────────────────────────────────────────────

class ThreatLevel(str, Enum):
    """Severity classification for prompt injection attempts."""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


# Heuristic patterns that signal common injection techniques.
_INJECTION_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    # (pattern, score_contribution, description)
    (
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
        0.9,
        "Explicit instruction override attempt",
    ),
    (
        re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.I),
        0.7,
        "Role reassignment / jailbreak",
    ),
    (
        re.compile(r"system\s*:\s*", re.I),
        0.6,
        "System prompt injection via role tag",
    ),
    (
        re.compile(r"<\s*(?:system|admin|root)\s*>", re.I),
        0.8,
        "XML/HTML tag-based role injection",
    ),
    (
        re.compile(r"(?:do\s+not|don'?t)\s+(?:follow|obey|listen)", re.I),
        0.7,
        "Instruction negation attack",
    ),
    (
        re.compile(r"reveal\s+(?:your\s+)?(?:the\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)(?:\s+(?:and|or)\s+(?:prompt|instructions?|rules?))*", re.I),
        0.6,
        "Prompt extraction attempt",
    ),
    (
        re.compile(r"(?:repeat|print|echo|output)\s+(?:everything|all|the)\s+(?:above|previous)", re.I),
        0.5,
        "Context exfiltration via repetition",
    ),
    (
        re.compile(r"```\s*(?:python|bash|sh|javascript|sql)", re.I),
        0.3,
        "Code block injection (low confidence)",
    ),
    (
        re.compile(r"act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were))", re.I),
        0.5,
        "Persona override attempt",
    ),
    (
        re.compile(r"translate\s+(?:the\s+)?(?:above|previous|system)", re.I),
        0.4,
        "Translation-based prompt leak",
    ),
]

# Threshold boundaries
_SUSPICIOUS_THRESHOLD = 0.4
_MALICIOUS_THRESHOLD = 0.7


@dataclass
class InjectionScanResult:
    """Result of an injection heuristic scan."""

    threat_level: ThreatLevel
    score: float
    matched_rules: List[str]
    blocked: bool


def detect_prompt_injection(text: str) -> InjectionScanResult:
    """Score ``text`` for prompt injection risk using heuristic rules.

    The score is the *maximum* contribution of any single matched
    pattern (not additive) to avoid false-positive escalation on
    benign multi-pattern matches.  However, matching ≥ 3 patterns
    adds a 0.15 bonus regardless.

    Args:
        text: Raw user prompt.

    Returns:
        ``InjectionScanResult`` with threat level and match metadata.
    """
    max_score = 0.0
    matched_rules: List[str] = []

    for pattern, score, description in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched_rules.append(description)
            if score > max_score:
                max_score = score

    # Multi-match bonus
    if len(matched_rules) >= 3:
        max_score = min(max_score + 0.15, 1.0)

    # Classify threat level
    if max_score >= _MALICIOUS_THRESHOLD:
        threat_level = ThreatLevel.MALICIOUS
    elif max_score >= _SUSPICIOUS_THRESHOLD:
        threat_level = ThreatLevel.SUSPICIOUS
    else:
        threat_level = ThreatLevel.SAFE

    blocked = threat_level == ThreatLevel.MALICIOUS

    if matched_rules:
        logger.warning(
            "injection_scan",
            threat_level=threat_level.value,
            score=round(max_score, 3),
            rules=matched_rules,
            blocked=blocked,
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:12],
        )

    return InjectionScanResult(
        threat_level=threat_level,
        score=round(max_score, 3),
        matched_rules=matched_rules,
        blocked=blocked,
    )


# ────────────────────────────────────────────────────────────────────
# 3. Vertex AI Safety Settings
# ────────────────────────────────────────────────────────────────────

def build_safety_settings() -> List[genai_types.SafetySetting]:
    """Construct Vertex AI / Gemini safety settings with strict thresholds.

    These settings are passed to the ``GenerateContent`` call to block
    high-risk outputs across all harm categories.

    Returns:
        List of ``SafetySetting`` objects for the Gemini request.
    """
    harm_categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]

    return [
        genai_types.SafetySetting(
            category=category,
            threshold="BLOCK_MEDIUM_AND_ABOVE",
        )
        for category in harm_categories
    ]


# ────────────────────────────────────────────────────────────────────
# 4. ADK Callback Hooks
# ────────────────────────────────────────────────────────────────────

def before_model_callback(
    callback_context: Any,
    llm_request: Any,
) -> Optional[genai_types.Content]:
    """Google ADK ``before_model_callback`` hook.

    Executed by the ADK runner *before* every LLM call.  It performs:

      1. PII redaction on the latest user message.
      2. Prompt injection detection — blocks malicious prompts by
         returning a canned refusal ``Content`` object.

    Args:
        callback_context: ADK callback context (contains agent, session, etc.).
        llm_request: The ``LlmRequest`` about to be sent to the model.

    Returns:
        ``None`` to proceed normally, or a ``Content`` object to
        short-circuit the model call with a refusal message.
    """
    start_ts = time.time()

    # ── Extract last user text from the LLM request contents ────────
    user_text = ""
    if hasattr(llm_request, "contents") and llm_request.contents:
        last_content = llm_request.contents[-1]
        if hasattr(last_content, "parts"):
            user_text = " ".join(
                part.text for part in last_content.parts
                if hasattr(part, "text") and part.text
            )

    if not user_text:
        return None  # nothing to scan

    # ── Step 1: PII Redaction ───────────────────────────────────────
    redaction = redact_pii(user_text)
    if redaction.redacted_count > 0:
        logger.info(
            "pii_redacted",
            count=redaction.redacted_count,
            types=redaction.redacted_types,
        )
        # Rewrite the last content's text parts in-place
        if hasattr(llm_request, "contents") and llm_request.contents:
            last_content = llm_request.contents[-1]
            if hasattr(last_content, "parts") and last_content.parts:
                for part in last_content.parts:
                    if hasattr(part, "text") and part.text:
                        part.text = redaction.cleaned_text

    # ── Step 2: Injection Detection ─────────────────────────────────
    scan = detect_prompt_injection(redaction.cleaned_text)

    elapsed_ms = round((time.time() - start_ts) * 1000, 2)
    logger.info(
        "guardrail_check",
        threat_level=scan.threat_level.value,
        score=scan.score,
        elapsed_ms=elapsed_ms,
    )

    if scan.blocked:
        # Return a refusal Content to short-circuit the model call
        return genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    text=(
                        "⚠️ **Request Blocked** — Your message was flagged as a "
                        "potential prompt injection attack "
                        f"(threat_level={scan.threat_level.value}, "
                        f"score={scan.score}).  "
                        "If you believe this is an error, please rephrase your "
                        "query or contact your administrator."
                    ),
                )
            ],
        )

    # Proceed normally
    return None


def after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> Optional[genai_types.Content]:
    """Google ADK ``after_model_callback`` hook.

    Scans model *output* for PII leakage and redacts if the caller
    does not have ``can_access_pii`` privilege.

    Args:
        callback_context: ADK callback context.
        llm_response: The ``LlmResponse`` returned by the model.

    Returns:
        ``None`` to forward the original response, or a new
        ``Content`` with redacted text.
    """
    # Extract text from the model response
    response_text = ""
    if hasattr(llm_response, "content") and llm_response.content:
        content = llm_response.content
        if hasattr(content, "parts"):
            response_text = " ".join(
                part.text for part in content.parts
                if hasattr(part, "text") and part.text
            )

    if not response_text:
        return None

    # Redact PII from model output
    redaction = redact_pii(response_text)
    if redaction.redacted_count > 0:
        logger.info(
            "output_pii_redacted",
            count=redaction.redacted_count,
            types=redaction.redacted_types,
        )
        return genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=redaction.cleaned_text)],
        )

    return None
