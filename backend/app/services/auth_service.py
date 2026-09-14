"""Authentication Service managing user registration, authentication, and JWT token issuance."""
import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.db.models import User
from app.schemas.api import UserCreate, UserLogin, UserOut, Token
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    validate_password_strength,
)

logger = logging.getLogger(__name__)


def normalize_email(email: str) -> str:
    """Normalizes email to lower case stripped string."""
    return email.strip().lower()


def register_user(db: Session, user_in: UserCreate) -> User:
    """Registers a new user account with secure password hashing and duplicate check."""
    email_norm = normalize_email(user_in.email)

    existing_user = db.query(User).filter(User.email == email_norm).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    validate_password_strength(user_in.password)

    hashed_pwd = hash_password(user_in.password)
    db_user = User(
        name=user_in.name.strip(),
        email=email_norm,
        password_hash=hashed_pwd,
        role="authenticated",
    )

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        logger.info(f"User account registered successfully: '{db_user.email}' (ID: {db_user.id})")
        return db_user
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to register user account: {e!s}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete user registration.",
        )


def authenticate_user(db: Session, email: str, password: str) -> User:
    """Authenticates user credentials and returns DB User ORM instance."""
    email_norm = normalize_email(email)
    user = db.query(User).filter(User.email == email_norm).first()

    generic_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user:
        raise generic_error

    if not verify_password(password, user.password_hash):
        raise generic_error

    return user


class AuthService:
    """Service class wrapper handling user account operations."""

    @staticmethod
    def normalize_email(email: str) -> str:
        return normalize_email(email)

    def register_user(self, db: Session, user_in: UserCreate) -> UserOut:
        user = register_user(db, user_in)
        return UserOut(**user.to_dict())

    def authenticate_user(self, db: Session, credentials: UserLogin) -> Token:
        user = authenticate_user(db, credentials.email, credentials.password)
        token_str = create_access_token(
            subject=user.id,
            email=user.email,
            role=user.role,
        )
        return Token(
            access_token=token_str,
            token_type="bearer",
            user=UserOut(**user.to_dict()),
        )


auth_service = AuthService()

