"""Integration and authorization security tests for Recommendations API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


import uuid

def _register_and_login_user(email: str, name: str = "Test User") -> dict[str, str]:
    """Helper registering and logging in a user, returning auth header dict."""
    unique_email = f"{uuid.uuid4().hex[:8]}_{email}"
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": unique_email, "password": "SecurePassword123!"},
    )
    assert reg_resp.status_code == 201

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": "SecurePassword123!"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}



def test_end_to_end_prediction_recommendation_flow():
    """Test full flow: predict diabetes -> receive prediction + recommendation -> query recommendation by prediction ID."""
    auth_headers = _register_and_login_user("rec_e2e_user@example.com", "Rec E2E User")

    # 1. Execute Diabetes Prediction
    predict_payload = {
        "disease": "diabetes",
        "model_id": "diabetes_rf_opt_v1",
        "features": {
            "Pregnancies": 6,
            "Glucose": 168.0,
            "BloodPressure": 84.0,
            "SkinThickness": 32.0,
            "Insulin": 0.0,
            "BMI": 34.2,
            "DiabetesPedigreeFunction": 0.67,
            "Age": 48.0,
        },
        "explain": True,
    }

    pred_resp = client.post("/api/v1/predict/diabetes", json=predict_payload, headers=auth_headers)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()

    assert "recommendation" in pred_data
    assert pred_data["recommendation"] is not None
    rec = pred_data["recommendation"]
    assert rec["disease"] == "diabetes"
    assert rec["rule_set_version"] == "diabetes-v1.0.0"
    assert len(rec["priority_actions"]) > 0

    # 2. Query History to get prediction record ID
    hist_resp = client.get("/api/v1/history", headers=auth_headers)
    assert hist_resp.status_code == 200
    hist_items = hist_resp.json()["items"]
    assert len(hist_items) > 0
    prediction_id = hist_items[0]["id"]

    # 3. Query GET /recommendations/{prediction_id}
    rec_detail_resp = client.get(f"/api/v1/recommendations/{prediction_id}", headers=auth_headers)
    assert rec_detail_resp.status_code == 200
    rec_detail = rec_detail_resp.json()
    assert rec_detail["prediction_record_id"] == prediction_id
    assert rec_detail["disease"] == "diabetes"
    assert rec_detail["rule_set_version"] == "diabetes-v1.0.0"

    # 4. Query GET /recommendations list
    rec_list_resp = client.get("/api/v1/recommendations", headers=auth_headers)
    assert rec_list_resp.status_code == 200
    rec_list = rec_list_resp.json()
    assert rec_list["total"] >= 1
    assert rec_list["items"][0]["disease"] == "diabetes"


def test_recommendation_ownership_authorization():
    """Test security isolation: User A cannot retrieve User B's recommendation."""
    headers_user_a = _register_and_login_user("usera_rec@example.com", "User A")
    headers_user_b = _register_and_login_user("userb_rec@example.com", "User B")

    # User A executes prediction
    predict_payload = {
        "disease": "cardiovascular",
        "model_id": "cardiovascular_extra_trees_opt_v1",
        "features": {
            "age": 60.0,
            "sex": 1.0,
            "cp": 2.0,
            "trestbps": 145.0,
            "chol": 255.0,
            "fbs": 1.0,
            "restecg": 0.0,
            "thalach": 140.0,
            "exang": 1.0,
            "oldpeak": 2.0,
            "slope": 1.0,
            "ca": 1.0,
            "thal": 2.0,
        },
    }

    pred_a = client.post("/api/v1/predict/cardiovascular", json=predict_payload, headers=headers_user_a)
    assert pred_a.status_code == 200

    hist_a = client.get("/api/v1/history", headers=headers_user_a)
    prediction_id_a = hist_a.json()["items"][0]["id"]

    # User A can retrieve their own recommendation
    rec_a = client.get(f"/api/v1/recommendations/{prediction_id_a}", headers=headers_user_a)
    assert rec_a.status_code == 200

    # User B ATTEMPTS to retrieve User A's recommendation -> MUST BE DENIED (HTTP 404/403)
    rec_b_attack = client.get(f"/api/v1/recommendations/{prediction_id_a}", headers=headers_user_b)
    assert rec_b_attack.status_code in (404, 403)


def test_unauthenticated_recommendation_access_blocked():
    """Test unauthenticated requests to recommendations endpoints are rejected with HTTP 401."""
    resp_list = client.get("/api/v1/recommendations")
    assert resp_list.status_code == 401

    resp_detail = client.get("/api/v1/recommendations/some-dummy-id")
    assert resp_detail.status_code == 401


def test_cancer_recommendation_integration():
    """Test integration flow for Breast Cancer prediction recommendations."""
    headers = _register_and_login_user("cancer_rec_user@example.com", "Cancer Rec User")

    predict_payload = {
        "disease": "cancer",
        "model_id": "cancer_stacking_opt_v1",
        "features": {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "mean_smoothness": 0.1184,
            "mean_compactness": 0.2776,
            "mean_concavity": 0.3001,
            "mean_concave_points": 0.1471,
            "mean_symmetry": 0.2419,
            "mean_fractal_dimension": 0.07871,
            "radius_error": 1.095,
            "texture_error": 0.9053,
            "perimeter_error": 8.589,
            "area_error": 153.4,
            "smoothness_error": 0.006399,
            "compactness_error": 0.04904,
            "concavity_error": 0.05373,
            "concave_points_error": 0.01587,
            "symmetry_error": 0.03003,
            "fractal_dimension_error": 0.006193,
            "worst_radius": 25.38,
            "worst_texture": 17.33,
            "worst_perimeter": 184.6,
            "worst_area": 2019.0,
            "worst_smoothness": 0.1622,
            "worst_compactness": 0.6656,
            "worst_concavity": 0.7119,
            "worst_concave_points": 0.2654,
            "worst_symmetry": 0.4601,
            "worst_fractal_dimension": 0.1189,
        },
    }

    resp = client.post("/api/v1/predict/cancer", json=predict_payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["recommendation"] is not None
    rec = data["recommendation"]
    assert rec["disease"] == "cancer"
    assert rec["rule_set_version"] == "cancer-v1.0.0"
