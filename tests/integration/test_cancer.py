"""Integration Test Suite for Phase F Cancer Model Pipeline and Inference."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.db.session import Base, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Isolated test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sukshmadrishti_cancer.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


client = TestClient(app)

SAMPLE_CANCER_LIST = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

SAMPLE_CANCER_DICT = {
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0,
    "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
    "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399, "compactness_error": 0.04904,
    "concavity_error": 0.05373, "concave_points_error": 0.01587, "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
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
    if os.path.exists("./test_sukshmadrishti_cancer.db"):
        try:
            os.remove("./test_sukshmadrishti_cancer.db")
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


def test_cancer_predict_success_list_input():
    """Integration Test: POST /api/v1/predict/cancer with list input executes inference."""
    payload = {
        "features": SAMPLE_CANCER_LIST,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/cancer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["disease"] == "cancer"
    assert data["model_used"] == "Breast Cancer Random Forest"
    assert 0.0 <= data["probability"] <= 1.0


def test_cancer_predict_success_dict_input():
    """Integration Test: POST /api/v1/predict/cancer with dict input executes inference."""
    payload = {
        "features": SAMPLE_CANCER_DICT,
        "model_id": "cancer_lr",
    }
    response = client.post("/api/v1/predict/cancer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["disease"] == "cancer"
    assert data["model_used"] == "Breast Cancer Logistic Regression"


def test_cancer_validation_missing_feature():
    """Validation Test: Reject request missing required clinical features."""
    incomplete = dict(SAMPLE_CANCER_DICT)
    del incomplete["mean_radius"]
    payload = {
        "features": incomplete,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/cancer", json=payload)
    assert response.status_code == 422


def test_cancer_validation_extra_feature():
    """Validation Test: Reject request containing unauthorized extra features."""
    extra_dict = dict(SAMPLE_CANCER_DICT)
    extra_dict["unauthorized_key"] = 99.0
    payload = {
        "features": extra_dict,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/cancer", json=payload)
    assert response.status_code == 422


def test_cancer_validation_target_injection():
    """Security Test: Reject request attempting target column injection ('target')."""
    bad_dict = dict(SAMPLE_CANCER_DICT)
    bad_dict["target"] = 1.0
    payload = {
        "features": bad_dict,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/cancer", json=payload)
    assert response.status_code == 422


def test_cancer_validation_nan_inf():
    """Security Test: Reject NaN or Infinite values in input features."""
    nan_payload = '{"features": [NaN, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189], "model_id": "cancer_rf"}'
    response = client.post("/api/v1/predict/cancer", content=nan_payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422



def test_cancer_prediction_persistence():
    """Integration Test: Authenticated cancer prediction is saved in SQLite history with disease='cancer'."""
    headers, user_id = _register_and_login("Dr. Oncologist", "oncology.doc@hospital.org")

    payload = {
        "features": SAMPLE_CANCER_LIST,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/cancer", json=payload, headers=headers)
    assert response.status_code == 200

    # Query history
    hist_res = client.get("/api/v1/history?disease=cancer", headers=headers)
    assert hist_res.status_code == 200
    hdata = hist_res.json()
    assert hdata["total"] == 1
    item = hdata["items"][0]
    assert item["disease"] == "cancer"
    assert item["model_name"] == "Breast Cancer Random Forest"


def test_cancer_user_isolation():
    """Security Test: User B cannot access User A's cancer prediction history."""
    headers_a, _ = _register_and_login("User A", "userA@hospital.org")
    headers_b, _ = _register_and_login("User B", "userB@hospital.org")

    payload = {
        "features": SAMPLE_CANCER_LIST,
        "model_id": "cancer_rf",
    }
    res_a = client.post("/api/v1/predict/cancer", json=payload, headers=headers_a)
    assert res_a.status_code == 200

    hist_b = client.get("/api/v1/history?disease=cancer", headers=headers_b)
    assert hist_b.status_code == 200
    assert hist_b.json()["total"] == 0
