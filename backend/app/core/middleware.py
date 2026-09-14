"""Security headers, rate limiting, and request protection middlewares."""
import os
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import FastAPI, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing standard OWASP defense-in-depth security response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Standard OWASP defensive headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; object-src 'none'"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"

        if settings.app_env == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window IP rate limiter to protect against request floods."""

    def __init__(self, app: FastAPI, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self._history: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if rate limit enforcement is explicitly requested (e.g. in rate limit test)
        if not getattr(request.app.state, "enforce_rate_limit", False):
            if (
                settings.app_env == "testing"
                or "PYTEST_CURRENT_TEST" in os.environ
                or getattr(request.app.state, "bypass_rate_limit", False)
                or getattr(request.state, "bypass_rate_limit", False)
            ):
                return await call_next(request)

        client_ip = request.client.host if request.client else "unknown_client"
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old timestamps
        timestamps = [t for t in self._history[client_ip] if t > cutoff]
        self._history[client_ip] = timestamps

        if len(timestamps) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - timestamps[0])) if timestamps else 60
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Too many requests. Please slow down and retry later.",
                    "status": "rate_limit_exceeded",
                    "retry_after_seconds": max(1, retry_after),
                },
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        self._history[client_ip].append(now)
        remaining = self.max_requests - len(self._history[client_ip])

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing a maximum allowable request payload size (default 1 MB, 10 MB for uploads)."""

    def __init__(self, app: FastAPI, max_bytes: int = 1048576, max_upload_bytes: int = 10485760):
        super().__init__(app)
        self.max_bytes = max_bytes
        self.max_upload_bytes = max_upload_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length_int = int(content_length)
                effective_limit = self.max_upload_bytes if "/report" in request.url.path else self.max_bytes
                if length_int > effective_limit:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request payload size {length_int} bytes exceeds maximum limit of {effective_limit} bytes.",
                            "status": "payload_too_large",
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)
