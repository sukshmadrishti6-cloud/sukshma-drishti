"""Authentication API Router for user registration, login, and profile management."""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.api import Token, UserCreate, UserLogin, UserOut
from app.services.auth_service import register_user, authenticate_user
from app.core.security import create_access_token

router = APIRouter()


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers a new user with email, name, and password. Password must be at least 8 characters.",
)
def register(user_in: UserCreate, db: Session = Depends(get_db)) -> UserOut:
    """User registration endpoint."""
    user = register_user(db, user_in)
    return UserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        created_at=user.created_at.isoformat() if hasattr(user.created_at, "isoformat") else str(user.created_at),
    )


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and obtain JWT token",
    description="Validates user credentials and returns a signed JWT access token.",
)
def login(credentials: UserLogin, db: Session = Depends(get_db)) -> Token:
    """User authentication endpoint."""
    user = authenticate_user(db, credentials.email, credentials.password)
    access_token = create_access_token(
        subject=user.id,
        email=user.email,
        role=user.role,
    )
    user_out = UserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        created_at=user.created_at.isoformat() if hasattr(user.created_at, "isoformat") else str(user.created_at),
    )
    return Token(access_token=access_token, token_type="bearer", user=user_out)


@router.get(
    "/me",
    response_model=UserOut,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user profile",
    description="Returns profile details of the authenticated user.",
)
def get_me(current_user: User = Depends(get_current_user)) -> UserOut:
    """Retrieve current user profile."""
    return UserOut(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at.isoformat() if hasattr(current_user.created_at, "isoformat") else str(current_user.created_at),
    )
