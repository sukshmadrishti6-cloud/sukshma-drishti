"""Security Module for Password Hashing and JWT Token Management."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.core.config import settings

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 8


def validate_password_strength(password: str) -> None:
    """Enforces minimum password strength requirements."""
    if not password or len(password.strip()) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.",
        )


def hash_password(password: str) -> str:
    """Hashes plaintext password using bcrypt."""
    validate_password_strength(password)
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plaintext password against stored bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception as e:
        logger.warning(f"Password verification exception: {e!s}")
        return False



def create_access_token(
    subject: str,
    email: str,
    role: str = "authenticated",
    expires_delta: timedelta | None = None,
) -> str:
    """Generates signed JWT access token containing subject (user_id) claim."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload = {
        "sub": str(subject),
        "user_id": str(subject),
        "email": email.strip().lower(),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "aud": "authenticated",
        "iss": settings.app_name,
    }

    encoded_jwt = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str, audience: str | None = "authenticated") -> dict[str, Any]:
    """Decodes and verifies signed JWT access token."""
    try:
        kwargs: dict[str, Any] = {
            "algorithms": [settings.jwt_algorithm],
            "options": {"verify_signature": True, "verify_exp": True},
        }
        if audience:
            kwargs["audience"] = audience
        else:
            kwargs["options"]["verify_aud"] = False

        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            **kwargs,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e!s}",
            headers={"WWW-Authenticate": "Bearer"},
        )

