"""Centralized exception classes and secure exception handlers for FastAPI."""
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for application-level errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class RateLimitExceededError(APIError):
    """Raised when an IP or user exceeds allowed request rate limits."""

    def __init__(self, message: str = "Rate limit exceeded. Please retry after some time.", retry_after: int = 60):
        super().__init__(message, status_code=status.HTTP_429_TOO_MANY_REQUESTS, details={"retry_after": retry_after})
        self.retry_after = retry_after


class AuthorizationError(APIError):
    """Raised on broken object-level authorization or privilege violation."""

    def __init__(self, message: str = "Forbidden: Insufficient permissions for requested resource."):
        super().__init__(message, status_code=status.HTTP_403_FORBIDDEN)


def register_exception_handlers(app: FastAPI, is_production: bool = False) -> None:
    """Registers secure exception handlers to prevent stack trace and secret leakage."""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        headers = {}
        if isinstance(exc, RateLimitExceededError):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "status": "error"},
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status": "error"},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            errors.append(f"{loc}: {err.get('msg', 'Invalid input')}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "; ".join(errors),
                "status": "validation_error",
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc!s}", exc_info=True)
        detail_msg = (
            "An internal server error occurred. Please contact the system administrator."
            if is_production
            else f"Internal server error: {exc!s}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": detail_msg, "status": "server_error"},
        )
