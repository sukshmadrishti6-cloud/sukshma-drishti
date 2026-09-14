"""FastAPI dependency injection utilities for authentication, authorization, and database access."""
from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

import jwt
from app.core.config import settings
from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_db


def parse_jwt_claims(token: str) -> dict[str, Any]:
    """Parses and cryptographically verifies JWT claims payload.
    
    Verifies signature, expiration, issuer, and audience using PyJWT.
    """
    # 1. Primary verification using app JWT secret key
    try:
        return decode_access_token(token)
    except HTTPException as e:
        if "expired" in str(e.detail).lower():
            raise e
    except Exception:
        pass

    # 2. Fallback verification using Supabase JWT secret if configured
    secret = settings.supabase_jwt_secret
    if secret:
        try:
            options = {
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
            }
            kwargs: dict[str, Any] = {
                "algorithms": ["HS256"],
                "audience": "authenticated",
                "options": options,
            }
            if settings.supabase_url:
                expected_iss = f"{settings.supabase_url.rstrip('/')}/auth/v1"
                kwargs["issuer"] = expected_iss
                options["verify_iss"] = True

            payload = jwt.decode(
                token,
                secret,
                **kwargs,
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
        except jwt.InvalidAudienceError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience.")
        except jwt.InvalidIssuerError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer.")
        except jwt.InvalidSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature.")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e!s}")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )



async def get_current_user_optional(
    authorization: str | None = Header(None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User | None:
    """Dependency extracting user identity from Bearer token if present."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None

    try:
        claims = parse_jwt_claims(token)
    except HTTPException:
        return None

    user_id = claims.get("sub") or claims.get("user_id")
    user = None
    if user_id:
        user = db.query(User).filter(User.id == str(user_id)).first()

    if not user and claims.get("email"):
        email = claims.get("email", "").strip().lower()
        if email:
            user = db.query(User).filter(User.email == email).first()

    return user


async def get_current_user(
    user: User | None = Depends(get_current_user_optional),
) -> User:
    """Dependency enforcing that an authenticated user is present."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _get_user_id(user: Any) -> str | None:
    if isinstance(user, dict):
        return user.get("id")
    return getattr(user, "id", None)


def _get_user_role(user: Any) -> str | None:
    if isinstance(user, dict):
        return user.get("role")
    return getattr(user, "role", None)


async def require_authenticated(
    user: Any = Depends(get_current_user),
) -> Any:
    """Authorization dependency verifying user authentication."""
    if not _get_user_id(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation forbidden: User identity unverified.",
        )
    return user


def require_role(required_role: str) -> Callable:
    """Role-based Access Control (RBAC) dependency factory."""

    async def role_checker(user: Any = Depends(get_current_user)) -> Any:
        user_role = _get_user_role(user) or ""
        if user_role != required_role and user_role != "admin" and user_role != "service_role":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation forbidden: Requires '{required_role}' role (current: '{user_role}').",
            )
        return user

    return role_checker


def verify_user_resource_access(requested_user_id: str, current_user: Any) -> bool:
    """Broken Object Level Authorization (BOLA/IDOR) verification helper.

    Guarantees regular users can only read or modify their own resources.
    """
    user_id = _get_user_id(current_user)
    user_role = _get_user_role(current_user) or ""

    # Admin or service role can access across tenants/users
    if user_role in ["admin", "service_role"]:
        return True

    if user_id != requested_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to access another user's resources.",
        )
    return True

