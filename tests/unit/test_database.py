"""Unit tests for Database ORM models, session, password hashing, and JWT security."""
import os
import sys
from datetime import timedelta
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.db.session import Base
from app.db.models import User
from app.core.security import (
    hash_password,
    verify_password,
    validate_password_strength,
    create_access_token,
    decode_access_token,
)


@pytest.fixture
def in_memory_db():
    """Fixture initializing an in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_password_hashing_and_verification():
    """Unit Test: Password hashing produces valid bcrypt hashes and verifies correctly."""
    password = "SecurePassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert hashed.startswith("$2")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


def test_password_strength_validation():
    """Unit Test: Enforces minimum 8-character password constraint."""
    # Valid passwords
    validate_password_strength("12345678")
    validate_password_strength("  12345678  ")

    # Invalid passwords raise HTTP 400
    with pytest.raises(HTTPException) as exc_info:
        validate_password_strength("short")
    assert exc_info.value.status_code == 400
    assert "at least 8 characters" in exc_info.value.detail

    with pytest.raises(HTTPException):
        validate_password_strength("")


def test_jwt_token_generation_and_decoding():
    """Unit Test: JWT token encoding and decoding cycle."""
    subject = "user-12345-abc"
    email = "researcher@sukshmadrishti.org"
    role = "authenticated"

    token = create_access_token(subject=subject, email=email, role=role)
    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_access_token(token)
    assert payload["sub"] == subject
    assert payload["user_id"] == subject
    assert payload["email"] == email
    assert payload["role"] == role


def test_jwt_token_expiration():
    """Unit Test: Expired JWT tokens raise HTTP 401 Unauthorized."""
    token = create_access_token(
        subject="test-user",
        email="expired@test.com",
        expires_delta=timedelta(seconds=-10),  # expired 10s ago
    )

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_user_orm_model_crud(in_memory_db):
    """Unit Test: User ORM model CRUD operations in SQLite database."""
    db = in_memory_db
    user = User(
        name="Test Researcher",
        email="researcher@test.com",
        password_hash=hash_password("Password123!"),
        role="authenticated",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    assert user.id is not None
    assert user.name == "Test Researcher"
    assert user.email == "researcher@test.com"

    # Query from DB
    queried = db.query(User).filter(User.email == "researcher@test.com").first()
    assert queried is not None
    assert queried.id == user.id

    # Test to_dict helper excludes password_hash
    user_dict = queried.to_dict()
    assert "password_hash" not in user_dict
    assert user_dict["email"] == "researcher@test.com"
