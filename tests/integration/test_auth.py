"""Integration test suite for User Authentication and Authorization API Endpoints."""
import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.db.session import Base, get_db

# Setup test SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sukshmadrishti.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Recreate test database tables before each integration test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_sukshmadrishti.db"):
        try:
            os.remove("./test_sukshmadrishti.db")
        except PermissionError:
            pass




def test_user_registration_success():
    """Integration Test: Successfully registers a new user via POST /api/v1/auth/register."""
    payload = {
        "name": "Dr. Alan Turing",
        "email": "alan.turing@sukshmadrishti.org",
        "password": "Password123!",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Dr. Alan Turing"
    assert data["email"] == "alan.turing@sukshmadrishti.org"
    assert data["role"] == "authenticated"
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_user_registration_duplicate_email():
    """Integration Test: Rejects duplicate email registration with 400 Bad Request."""
    payload = {
        "name": "Dr. Alan Turing",
        "email": "alan.turing@sukshmadrishti.org",
        "password": "Password123!",
    }
    res1 = client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = client.post("/api/v1/auth/register", json=payload)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"].lower()



def test_user_registration_short_password():
    """Integration Test: Rejects password shorter than 8 characters."""
    payload = {
        "name": "Short Pass",
        "email": "short@sukshmadrishti.org",
        "password": "123",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422 or response.status_code == 400


def test_user_login_success():
    """Integration Test: Valid credentials return JWT access token and user info via POST /api/v1/auth/login."""
    # 1. Register user
    reg_payload = {
        "name": "Ada Lovelace",
        "email": "ada.lovelace@sukshmadrishti.org",
        "password": "Algorithm456!",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201

    # 2. Login user
    login_payload = {
        "email": "ada.lovelace@sukshmadrishti.org",
        "password": "Algorithm456!",
    }
    login_res = client.post("/api/v1/auth/login", json=login_payload)
    assert login_res.status_code == 200
    data = login_res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "ada.lovelace@sukshmadrishti.org"


def test_user_login_invalid_password():
    """Integration Test: Invalid password returns 401 Unauthorized without exposing user presence."""
    # Register user
    client.post(
        "/api/v1/auth/register",
        json={
            "name": "Ada Lovelace",
            "email": "ada.lovelace@sukshmadrishti.org",
            "password": "Algorithm456!",
        },
    )

    # Wrong password
    login_res = client.post(
        "/api/v1/auth/login",
        json={
            "email": "ada.lovelace@sukshmadrishti.org",
            "password": "WrongPassword123!",
        },
    )
    assert login_res.status_code == 401
    assert "invalid email or password" in login_res.json()["detail"].lower()


def test_user_login_nonexistent_email():
    """Integration Test: Nonexistent email returns 401 Unauthorized with generic error message."""
    login_res = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@sukshmadrishti.org",
            "password": "Password123!",
        },
    )
    assert login_res.status_code == 401
    assert "invalid email or password" in login_res.json()["detail"].lower()


def test_get_me_authenticated():
    """Integration Test: GET /api/v1/auth/me returns profile of authenticated user."""
    # Register & Login
    email = "profile.user@sukshmadrishti.org"
    password = "MySecretPassword123!"
    client.post(
        "/api/v1/auth/register",
        json={"name": "Profile User", "email": email, "password": password},
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    token = login_res.json()["access_token"]

    # Fetch /me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == email
    assert me_data["name"] == "Profile User"


def test_get_me_unauthenticated():
    """Integration Test: GET /api/v1/auth/me without token returns 401 Unauthorized."""
    me_res = client.get("/api/v1/auth/me")
    assert me_res.status_code == 401


def test_get_me_invalid_token():
    """Integration Test: GET /api/v1/auth/me with invalid token returns 401 Unauthorized."""
    headers = {"Authorization": "Bearer invalid.jwt.token"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 401
