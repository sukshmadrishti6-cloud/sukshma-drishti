"""Centralized Configuration Module for SukshmaDrishti Backend.

P2-T3.1 & P2-T4 — Configuration & Supabase Settings.
Supports type-safe environment variable parsing, safe defaults, CORS configuration,
environment management, Supabase database settings, and validation.
"""
import json
import os

from pydantic import BaseModel, Field, field_validator, model_validator


class Settings(BaseModel):
    """Application Settings with environment variable overrides and validation."""

    app_name: str = Field(
        default_factory=lambda: os.getenv(
            "APP_NAME", "SukshmaDrishti — Hybrid Quantum ML Platform Backend API"
        )
    )
    app_version: str = Field(
        default_factory=lambda: os.getenv("APP_VERSION", "0.1.0")
    )
    app_env: str = Field(
        default_factory=lambda: os.getenv("APP_ENV", "development").lower()
    )
    api_prefix: str = Field(
        default_factory=lambda: os.getenv("API_PREFIX", "/api/v1")
    )
    api_host: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    api_port: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    random_seed: int = Field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )
    cors_origins: list[str] = Field(default_factory=list)

    # Database Configuration
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./sukshmadrishti.db")
    )

    # JWT Authentication Configuration
    jwt_secret_key: str = Field(
        default_factory=lambda: (
            os.getenv("JWT_SECRET_KEY")
            or os.getenv("SUPABASE_JWT_SECRET")
            or "sukshmadrishti-dev-jwt-secret-key-change-in-production-32bytes"
        )
    )
    jwt_algorithm: str = Field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    )

    # Supabase Integration (Optional Legacy / Cloud Config)
    supabase_url: str | None = Field(
        default_factory=lambda: (
            os.getenv("SUPABASE_URL")
            or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            or os.getenv("VITE_SUPABASE_URL")
        )
    )
    supabase_anon_key: str | None = Field(
        default_factory=lambda: (
            os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
            or os.getenv("VITE_SUPABASE_ANON_KEY")
        )
    )
    supabase_service_role_key: str | None = Field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    supabase_jwt_secret: str | None = Field(
        default_factory=lambda: os.getenv("SUPABASE_JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
    )

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        env = v.strip().lower()
        allowed = {"development", "testing", "production"}
        if env not in allowed:
            raise ValueError(
                f"Invalid APP_ENV: '{v}'. Must be one of {sorted(allowed)}"
            )
        return env

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Invalid API_PORT: {v}. Must be between 1 and 65535")
        return v

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Invalid RANDOM_SEED: {v}. Must be >= 0")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        level = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{v}'. Must be one of {sorted(allowed)}"
            )
        return level

    @model_validator(mode="after")
    def populate_and_validate_cors(self) -> "Settings":
        if not self.cors_origins:
            raw_cors = os.getenv("CORS_ORIGINS", "").strip()
            if raw_cors:
                if raw_cors.startswith("[") and raw_cors.endswith("]"):
                    try:
                        parsed = json.loads(raw_cors)
                        if isinstance(parsed, list):
                            self.cors_origins = [str(item).strip() for item in parsed if item]
                    except json.JSONDecodeError:
                        pass
                if not self.cors_origins:
                    self.cors_origins = [
                        item.strip() for item in raw_cors.split(",") if item.strip()
                    ]
            else:
                if self.app_env in {"development", "testing"}:
                    self.cors_origins = [
                        "http://localhost:5173",
                        "http://localhost:3000",
                        "http://127.0.0.1:5173",
                        "http://127.0.0.1:3000",
                    ]
                else:
                    self.cors_origins = []

        if self.app_env == "production":
            if "*" in self.cors_origins or self.cors_origins == ["*"]:
                raise ValueError(
                    "Security Violation: Wildcard CORS_ORIGINS ('*') is not allowed in production."
                )

        return self

    @property
    def allow_credentials(self) -> bool:
        """Determines if credentials are supported based on CORS configuration."""
        if "*" in self.cors_origins or not self.cors_origins:
            return False
        return True


def get_settings() -> Settings:
    """Returns the singleton application configuration settings instance."""
    return Settings()


settings = get_settings()
