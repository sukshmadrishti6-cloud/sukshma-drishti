"""Integration Test Suite for Phase E Cardiovascular Model Pipeline and Inference."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.db.session import Base, get_db
from app.db.models import PredictionRecord
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Isolated test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sukshmadrishti_cardio.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


client = TestClient(app)

SAMPLE_CARDIO_LIST = [63.0, 1.0, 3.0, 145.0, 233.0, 1.0, 0.0, 150.0, 0.0, 2.3, 0.0, 0.0, 1.0]

SAMPLE_CARDIO_DICT = {
    "age": 63.0,
    "sex": 1.0,
    "cp": 3.0,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 0.0,
    "ca": 0.0,
    "thal": 1.0,
}


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Recreate test database schema before each integration test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_sukshmadrishti_cardio.db"):
        try:
            os.remove("./test_sukshmadrishti_cardio.db")
        except PermissionError:
            pass


def _register_and_login(name: str, email: str, password: str = "Password123!") -> tuple[dict[str, str], str]:
    """Helper registering user and returning authorization headers and user ID."""
    reg_res = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    assert reg_res.status_code == 201
    user_id = reg_res.json()["id"]

    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    return headers, user_id


def test_cardiovascular_predict_success_list_input():
    """Integration Test: POST /api/v1/predict/cardiovascular with list input executes inference."""
    payload = {
        "features": SAMPLE_CARDIO_LIST,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["disease"] == "cardiovascular"
    assert data["model_used"] == "Cardiovascular Random Forest"
    assert 0.0 <= data["probability"] <= 1.0


def test_cardiovascular_predict_success_dict_input():
    """Integration Test: POST /api/v1/predict/cardiovascular with dict input executes inference."""
    payload = {
        "features": SAMPLE_CARDIO_DICT,
        "model_id": "cardio_lr",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["disease"] == "cardiovascular"
    assert data["model_used"] == "Cardiovascular Logistic Regression"


def test_cardiovascular_validation_missing_feature():
    """Validation Test: Reject request missing required clinical features."""
    incomplete = dict(SAMPLE_CARDIO_DICT)
    del incomplete["chol"]
    payload = {
        "features": incomplete,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 422


def test_cardiovascular_validation_extra_feature():
    """Validation Test: Reject request containing unauthorized extra features."""
    extra_dict = dict(SAMPLE_CARDIO_DICT)
    extra_dict["unauthorized_key"] = 99.0
    payload = {
        "features": extra_dict,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 422


def test_cardiovascular_validation_target_injection():
    """Security Test: Reject request attempting target column injection ('target')."""
    bad_dict = dict(SAMPLE_CARDIO_DICT)
    bad_dict["target"] = 1.0
    payload = {
        "features": bad_dict,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 422


def test_cardiovascular_validation_nan_inf():
    """Security Test: Reject NaN or Infinite values in input features."""
    nan_payload = '{"features": [NaN, 1.0, 3.0, 145.0, 233.0, 1.0, 0.0, 150.0, 0.0, 2.3, 0.0, 0.0, 1.0], "model_id": "cardio_rf"}'
    response = client.post("/api/v1/predict/cardiovascular", content=nan_payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_cardiovascular_prediction_persistence():
    """Integration Test: Authenticated cardiovascular prediction is saved in SQLite history with disease='cardiovascular'."""
    headers, user_id = _register_and_login("Dr. Heart Physician", "cardio.doc@hospital.org")

    payload = {
        "features": SAMPLE_CARDIO_LIST,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload, headers=headers)
    assert response.status_code == 200

    # Query history
    hist_res = client.get("/api/v1/history?disease=cardiovascular", headers=headers)
    assert hist_res.status_code == 200
    hdata = hist_res.json()
    assert hdata["total"] == 1
    item = hdata["items"][0]
    assert item["disease"] == "cardiovascular"
    assert item["model_name"] == "Cardiovascular Random Forest"


def test_cardiovascular_prediction_isolation():
    """Non-Regression Test: Cardiovascular endpoint returns disease='cardiovascular' and correct model identity."""
    payload = {
        "features": SAMPLE_CARDIO_LIST,
        "model_id": "cardio_rf",
    }
    response = client.post("/api/v1/predict/cardiovascular", json=payload)
    assert response.status_code == 200
    assert response.json()["disease"] == "cardiovascular"

