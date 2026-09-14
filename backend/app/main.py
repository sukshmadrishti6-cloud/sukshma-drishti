from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import setup_secure_logging
from app.core.middleware import (
    RateLimiterMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.db.session import init_db

# Initialize secure filtered logging
setup_secure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for setup and teardown."""
    init_db()
    yield


import os

app = FastAPI(
    title=settings.app_name,
    description="FastAPI REST service for SukshmaDrishti classical & quantum machine learning early disease detection.",
    version=settings.app_version,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url="/redoc" if settings.app_env != "production" else None,
    lifespan=lifespan,
)

if settings.app_env == "testing" or "PYTEST_CURRENT_TEST" in os.environ:
    app.state.bypass_rate_limit = True


# Security Middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1048576)  # 1MB limit
app.add_middleware(RateLimiterMiddleware, max_requests_per_minute=120)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Register secure exception handlers
register_exception_handlers(app, is_production=(settings.app_env == "production"))

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "message": "SukshmaDrishti Backend API",
        "docs": "/docs" if settings.app_env != "production" else None,
        "health": f"{settings.api_prefix}/health",
        "environment": settings.app_env,
    }
