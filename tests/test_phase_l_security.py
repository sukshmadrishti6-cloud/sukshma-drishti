"""Security & Production Hardening Test Suite for Phase L.

Tests authentication, JWT token verification, BOLA/IDOR ownership authorization,
input validation (NaN, Infinity, target injection), path traversal protection,
security response headers, and rate/size limits.
"""
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import jwt

from app.main import app
from app.core.config import settings
from app.core.security import create_access_token
from app.db.models import User, PredictionRecord
from app.db.session import get_db

client = TestClient(app)


def test_unauthenticated_protected_endpoint():
    """Unauthenticated requests to protected endpoints must return 401."""
    res_history = client.get("/api/v1/history")
    assert res_history.status_code == 401

    res_recs = client.get("/api/v1/recommendations")
    assert res_recs.status_code == 401


def test_invalid_token():
    """Malformed or invalid JWT tokens must return 401."""
    headers = {"Authorization": "Bearer malformed.token.string"}
    res = client.get("/api/v1/history", headers=headers)
    assert res.status_code == 401


def test_expired_token():
    """Expired JWT tokens must be rejected with 401."""
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    payload = {
        "sub": "test_user_id",
        "user_id": "test_user_id",
        "email": "test@example.com",
        "role": "authenticated",
        "iat": int(past.timestamp()) - 3600,
        "exp": int(past.timestamp()),
        "aud": "authenticated",
        "iss": settings.app_name,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    headers = {"Authorization": f"Bearer {token}"}
    res = client.get("/api/v1/history", headers=headers)
    assert res.status_code == 401


def test_tampered_token_signature():
    """JWT tokens with invalid signatures must return 401."""
    valid_token = create_access_token("user_123", "user123@example.com")
    parts = valid_token.split(".")
    tampered = f"{parts[0]}.{parts[1]}.tampered_signature"
    headers = {"Authorization": f"Bearer {tampered}"}
    res = client.get("/api/v1/history", headers=headers)
    assert res.status_code == 401


def test_cross_user_history_denial():
    """User A must NOT be able to access or delete User B's prediction history (BOLA/IDOR)."""
    token_a = create_access_token("user_a_id", "usera@example.com")
    token_b = create_access_token("user_b_id", "userb@example.com")

    db = next(get_db())
    try:
        user_a = User(id="user_a_id", name="User A", email="usera@example.com", password_hash="hash")
        user_b = User(id="user_b_id", name="User B", email="userb@example.com", password_hash="hash")
        db.merge(user_a)
        db.merge(user_b)
        db.commit()

        rec_b = PredictionRecord(
            id="pred_record_b_123",
            user_id="user_b_id",
            disease="diabetes",
            model_name="Logistic Regression",
            model_version="v3.0",
            prediction=1,
            risk_score=0.85,
            risk_category="High Risk",
            input_data='{"Glucose": 150}',
        )
        db.merge(rec_b)
        db.commit()

        # User A attempts to view User B's record
        headers_a = {"Authorization": f"Bearer {token_a}"}
        res_view = client.get("/api/v1/history/pred_record_b_123", headers=headers_a)
        assert res_view.status_code == 404  # Returns 404 Not Found to prevent IDOR enumeration

        # User A attempts to delete User B's record
        res_del = client.delete("/api/v1/history/pred_record_b_123", headers=headers_a)
        assert res_del.status_code == 404

        # User B can view their own record
        headers_b = {"Authorization": f"Bearer {token_b}"}
        res_b_view = client.get("/api/v1/history/pred_record_b_123", headers=headers_b)
        assert res_b_view.status_code == 200
        assert res_b_view.json()["id"] == "pred_record_b_123"

    finally:
        db.query(PredictionRecord).filter(PredictionRecord.id == "pred_record_b_123").delete()
        db.query(User).filter(User.id.in_(["user_a_id", "user_b_id"])).delete()
        db.commit()


def test_cross_user_recommendation_denial():
    """User A must NOT be able to access User B's health recommendations."""
    token_a = create_access_token("user_a_id", "usera@example.com")
    db = next(get_db())
    try:
        user_a = User(id="user_a_id", name="User A", email="usera@example.com", password_hash="hash")
        user_b = User(id="user_b_id", name="User B", email="userb@example.com", password_hash="hash")
        db.merge(user_a)
        db.merge(user_b)
        rec_b = PredictionRecord(
            id="pred_record_b_456",
            user_id="user_b_id",
            disease="diabetes",
            model_name="Logistic Regression",
            model_version="v3.0",
            prediction=1,
            risk_score=0.85,
            risk_category="High Risk",
            input_data='{"Glucose": 150}',
        )
        db.merge(rec_b)
        db.commit()

        headers_a = {"Authorization": f"Bearer {token_a}"}
        res = client.get("/api/v1/recommendations/pred_record_b_456", headers=headers_a)
        assert res.status_code == 404
    finally:
        db.query(PredictionRecord).filter(PredictionRecord.id == "pred_record_b_456").delete()
        db.query(User).filter(User.id.in_(["user_a_id", "user_b_id"])).delete()
        db.commit()


def test_model_path_traversal_blocked():
    """Path traversal strings in model_id must be rejected."""
    payload = {
        "model_id": "../../etc/passwd",
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 33,
        },
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 404
    assert "not found in registry" in res.json()["detail"].lower()


def test_unregistered_model_id_blocked():
    """Unregistered model IDs must be rejected with 404."""
    payload = {
        "model_id": "malicious_custom_model",
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 33,
        },
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 404


def test_nan_and_inf_inputs_blocked():
    """NaN and Infinity feature inputs must be rejected by server validation with 422."""
    raw_nan_json = '{"model_id": "logistic_regression_v3", "features": {"Pregnancies": NaN, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 33}}'
    res_nan = client.post("/api/v1/predict", content=raw_nan_json, headers={"Content-Type": "application/json"})
    assert res_nan.status_code == 422

    raw_inf_json = '{"model_id": "logistic_regression_v3", "features": {"Pregnancies": 2, "Glucose": Infinity, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 33}}'
    res_inf = client.post("/api/v1/predict", content=raw_inf_json, headers={"Content-Type": "application/json"})
    assert res_inf.status_code == 422


def test_target_injection_blocked():
    """Target variable 'Outcome' injection must be rejected with 422."""
    payload = {
        "model_id": "logistic_regression_v3",
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 33,
            "Outcome": 1,
        },
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422
    assert "must not be provided" in res.json()["detail"].lower()


def test_extra_unauthorized_features_blocked():
    """Unauthorized extra feature keys must be rejected with 422."""
    payload = {
        "model_id": "logistic_regression_v3",
        "features": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 33,
            "extra_malicious_key": 999,
        },
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422


def test_account_enumeration_prevention():
    """Login failures must return safe generic messages to prevent email enumeration."""
    payload = {"email": "nonexistent_user_12345@example.com", "password": "WrongPassword123!"}
    res = client.post("/api/v1/auth/login", json=payload)
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid email or password."


def test_security_headers_present():
    """HTTP responses must contain standard OWASP defensive headers."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    headers = res.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert "strict-origin-when-cross-origin" in headers.get("Referrer-Policy", "")
    assert "Content-Security-Policy" in headers
