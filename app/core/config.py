"""
Centralized application configuration using pydantic-settings.

Loads values from environment variables and .env file with
validation and type coercion.  Every module imports `settings`
from here instead of reading os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the AI Security Gateway."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Cloud ────────────────────────────────────────────────
    google_cloud_project: str = "my-gcp-project-id"
    google_cloud_location: str = "us-central1"
    google_api_key: str = ""

    # ── Gemini Models ───────────────────────────────────────────────
    gemini_model: str = "gemini-2.0-flash"
    gemini_eval_model: str = "gemini-2.0-pro"

    # ── Redis ───────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── BigQuery (mock target) ──────────────────────────────────────
    bigquery_dataset: str = "ai_gateway_telemetry"
    bigquery_table: str = "agent_traces"

    # ── Security / JWT ──────────────────────────────────────────────
    jwt_secret_key: str = "replace-with-a-strong-random-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── OpenTelemetry ───────────────────────────────────────────────
    otel_service_name: str = "ai-security-gateway"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # ── Server ──────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()


# Module-level convenience alias
settings = get_settings()
