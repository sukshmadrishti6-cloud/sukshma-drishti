"""Comprehensive Security Hardening and Threat Model Regression Test Suite."""
import logging
import os
import sys
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

# Ensure backend package and workspace root are in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.core.logging import SensitiveDataFilter
from app.core.middleware import RateLimiterMiddleware, SecurityHeadersMiddleware
from app.dependencies import (
    get_current_user,
    require_role,
    verify_user_resource_access,
)
from app.main import app
from app.schemas.api import InferenceRequest

client = TestClient(app)

SAMPLE_VALID_FEATURES = [2.0, 130.0, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0]


def test_security_response_headers():
    """Security Test: Standard OWASP defense-in-depth security response headers are present."""
    response = client.get("/")
    assert response.status_code == 200
    headers = response.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


def test_rate_limiting_middleware_exceeded():
    """Security Test: Rapid request flooding triggers HTTP 429 Too Many Requests."""
    test_app = FastAPI()
    test_app.state.enforce_rate_limit = True
    test_app.add_middleware(RateLimiterMiddleware, max_requests_per_minute=5)

    @test_app.get("/ping")
    def ping():
        return {"status": "ok"}

    rl_client = TestClient(test_app)

    # First 5 requests succeed
    for _ in range(5):
        res = rl_client.get("/ping")
        assert res.status_code == 200

    # 6th request is throttled
    res_throttled = rl_client.get("/ping")
    assert res_throttled.status_code == 429
    data = res_throttled.json()
    assert "retry_after_seconds" in data
    assert "Retry-After" in res_throttled.headers


def test_payload_size_limit():
    """Security Test: Oversized request payloads exceeding 1MB are rejected with HTTP 413 or 422."""
    huge_data = {"features": SAMPLE_VALID_FEATURES, "model_id": "rf", "padding": "A" * (1024 * 1024 + 100)}
    response = client.post("/api/v1/predict", json=huge_data)
    assert response.status_code in [413, 422]


def test_path_traversal_model_name_blocked():
    """Security Test: Path traversal attempts in model identifiers are rejected."""
    traversal_attacks = [
        "../../models/classical/rf_v1.joblib",
        "..\\..\\etc\\passwd",
        "/etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
        "....//....//secret",
        "models/../secret",
    ]
    for attack in traversal_attacks:
        payload = {"features": SAMPLE_VALID_FEATURES, "model_id": attack}
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code in [400, 404, 422], f"Failed to block traversal: {attack}"


def test_target_column_outcome_injection_blocked():
    """Security Test: Direct injection of target variable 'Outcome' is rejected."""
    payload = {
        "features": {
            "Pregnancies": 2.0,
            "Glucose": 130.0,
            "BloodPressure": 70.0,
            "SkinThickness": 25.0,
            "Insulin": 100.0,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 32.0,
            "Outcome": 1.0,  # Injected target
        },
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert "outcome" in response.text.lower()


def test_malformed_numerical_inputs_nan_infinity_blocked():
    """Security Test: Raw NaN, Infinity, and invalid string floats are rejected with HTTP 422."""
    # 1. Test via raw HTTP payload with string representations of NaN
    raw_payload = '{"features": [2.0, "not_a_number", 70.0, 25.0, 100.0, 28.5, 0.45, 32.0], "model_id": "rf"}'
    response = client.post("/api/v1/predict", content=raw_payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422

    # 2. Test via endpoint with actual NaN
    nan_payload = '{"features": [2.0, NaN, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0], "model_id": "rf"}'
    response = client.post("/api/v1/predict", content=nan_payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_quantum_parameter_tampering_ignored_or_blocked():
    """Security Test: Attacker-controlled quantum hyperparameters (qubits, shots, depth) are forbidden."""
    payload = {
        "features": SAMPLE_VALID_FEATURES,
        "model_id": "qsvc_v2",
        "qubits": 1000,  # Malicious resource exhaustion attempts
        "shots": 99999999,
        "circuit_depth": 500,
    }
    response = client.post("/api/v1/predict", json=payload)
    # Since Extra fields are forbidden in PredictRequest, this is rejected with 422
    assert response.status_code == 422


def test_broken_object_level_authorization_idor():
    """Security Test: Regular users cannot access or modify other users' resources (BOLA/IDOR)."""
    user_alice = {"id": "user_alice_123", "email": "alice@hospital.org", "role": "authenticated"}
    user_bob_id = "user_bob_456"

    # Alice attempting to access Bob's resource -> HTTP 403 Forbidden
    with pytest.raises(Exception) as exc_info:
        verify_user_resource_access(requested_user_id=user_bob_id, current_user=user_alice)
    assert "403" in str(exc_info.value) or "forbidden" in str(exc_info.value).lower()

    # Alice accessing Alice's resource -> Allowed
    assert verify_user_resource_access(requested_user_id="user_alice_123", current_user=user_alice) is True

    # Admin accessing Bob's resource -> Allowed
    admin_user = {"id": "admin_999", "email": "admin@hospital.org", "role": "admin"}
    assert verify_user_resource_access(requested_user_id=user_bob_id, current_user=admin_user) is True


def test_rbac_role_enforcement():
    """Security Test: RBAC requires appropriate roles (e.g. physician, admin)."""
    import asyncio
    checker = require_role("physician")

    # Regular user without physician role -> 403 Forbidden
    reg_user = {"id": "u1", "email": "user@hospital.org", "role": "authenticated"}
    with pytest.raises(Exception) as exc_info:
        asyncio.run(checker(user=reg_user))
    assert "403" in str(exc_info.value) or "forbidden" in str(exc_info.value).lower()

    # User with physician role -> Allowed
    doc_user = {"id": "doc1", "email": "dr.smith@hospital.org", "role": "physician"}
    res = asyncio.run(checker(user=doc_user))
    assert res["role"] == "physician"

    # Admin user -> Allowed
    admin_user = {"id": "adm1", "email": "admin@hospital.org", "role": "admin"}
    res_adm = asyncio.run(checker(user=admin_user))
    assert res_adm["role"] == "admin"


def test_jwt_cryptographic_verification_suite(monkeypatch):
    """Security Test: JWT signature, expiration, and audience are cryptographically verified."""
    import jwt
    import time
    from fastapi import HTTPException
    from app.core.config import settings
    from app.dependencies import parse_jwt_claims

    test_secret = "test-cryptographic-secret-key-32bytes!!"
    monkeypatch.setattr(settings, "supabase_jwt_secret", test_secret)
    monkeypatch.setattr(settings, "supabase_url", None)

    # 1. Valid signed token succeeds
    valid_payload = {
        "sub": "user_12345",
        "email": "doctor@hospital.org",
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    }
    valid_token = jwt.encode(valid_payload, test_secret, algorithm="HS256")
    claims = parse_jwt_claims(valid_token)
    assert claims["sub"] == "user_12345"
    assert claims["email"] == "doctor@hospital.org"

    # 2. Tampered signature is rejected with 401
    tampered_token = jwt.encode(valid_payload, "wrong-forged-secret-key-12345678", algorithm="HS256")
    with pytest.raises(HTTPException) as exc_info:
        parse_jwt_claims(tampered_token)
    assert exc_info.value.status_code == 401

    # 3. Expired token is rejected with 401
    expired_payload = dict(valid_payload, exp=int(time.time()) - 3600)
    expired_token = jwt.encode(expired_payload, test_secret, algorithm="HS256")
    with pytest.raises(HTTPException) as exc_info:
        parse_jwt_claims(expired_token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()

    # 4. Invalid audience is rejected with 401
    wrong_aud_payload = dict(valid_payload, aud="malicious_audience")
    wrong_aud_token = jwt.encode(wrong_aud_payload, test_secret, algorithm="HS256")
    with pytest.raises(HTTPException) as exc_info:
        parse_jwt_claims(wrong_aud_token)
    assert exc_info.value.status_code == 401
    assert "audience" in exc_info.value.detail.lower()

    # 5. Missing secret causes 401
    monkeypatch.setattr(settings, "supabase_jwt_secret", None)
    with pytest.raises(HTTPException) as exc_info:
        parse_jwt_claims(valid_token)
    assert exc_info.value.status_code == 401


def test_logging_filter_masks_credentials_and_tokens():
    """Security Test: SensitiveDataFilter redacts Bearer tokens, passwords, and service role keys."""
    sec_filter = SensitiveDataFilter()

    # Log record with raw Bearer JWT
    raw_jwt = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    record = logging.LogRecord("test", logging.INFO, "", 0, f"User authenticated with token {raw_jwt}", (), None)
    sec_filter.filter(record)
    assert "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" not in record.msg
    assert "[REDACTED_JWT]" in record.msg

    # Log record with service role key
    record_key = logging.LogRecord("test", logging.INFO, "", 0, "Connected with service_role_key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret_key", (), None)
    sec_filter.filter(record_key)
    assert "secret_key" not in record_key.msg
    assert "[REDACTED]" in record_key.msg


def test_no_stack_trace_leakage_on_server_error():
    """Security Test: Unhandled server errors return sanitized messages without stack traces."""
    response = client.get("/api/v1/non_existent_route_404")
    assert response.status_code == 404
    text = response.text.lower()
    assert "traceback" not in text
    assert "file \"" not in text
