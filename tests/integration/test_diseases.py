"""Integration Test Suite for Phase D Multi-Disease Architecture and Registry."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.diseases.registry import disease_registry

client = TestClient(app)
SAMPLE_FEATURES = [2.0, 130.0, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0]


def test_list_diseases_endpoint():
    """Integration Test: GET /api/v1/diseases returns registered diseases and deployment statuses."""
    response = client.get("/api/v1/diseases")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total"] == 3

    disease_map = {item["id"]: item for item in data["items"]}
    assert "diabetes" in disease_map
    assert "cardiovascular" in disease_map
    assert "cancer" in disease_map

    assert disease_map["diabetes"]["status"] == "available"
    assert disease_map["cardiovascular"]["status"] == "available"
    assert disease_map["cancer"]["status"] == "available"


def test_get_disease_detail_cancer():
    """Integration Test: GET /api/v1/diseases/cancer returns available metadata with 30 features and models."""
    response = client.get("/api/v1/diseases/cancer")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "cancer"
    assert data["status"] == "available"
    assert data["input_features_count"] == 30
    assert len(data["supported_models"]) >= 2



def test_get_disease_detail_diabetes():
    """Integration Test: GET /api/v1/diseases/diabetes returns metadata and model details."""
    response = client.get("/api/v1/diseases/diabetes")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "diabetes"
    assert data["status"] == "available"
    assert data["input_features_count"] == 8
    assert len(data["supported_models"]) >= 4


def test_get_disease_detail_cardiovascular():
    """Integration Test: GET /api/v1/diseases/cardiovascular returns available metadata with models."""
    response = client.get("/api/v1/diseases/cardiovascular")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "cardiovascular"
    assert data["status"] == "available"
    assert data["input_features_count"] == 13
    assert len(data["supported_models"]) >= 2


def test_get_disease_unknown_returns_404():
    """Validation Test: Requesting unknown disease ID returns controlled HTTP 404 error."""
    response = client.get("/api/v1/diseases/unknown_disease_xyz")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_disease_path_traversal_blocked():
    """Security Test: Path traversal attempts in disease resolution are rejected with HTTP 400."""
    response = client.get("/api/v1/diseases/..%2F..%2Fetc")
    assert response.status_code in [400, 404]


def test_predict_disease_diabetes_success():
    """Integration Test: POST /api/v1/predict/diabetes executes valid inference."""
    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict/diabetes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["disease"] == "diabetes"


def test_predict_disease_alias_resolution():
    """Integration Test: POST /api/v1/predict/diabetes_mellitus resolves alias to canonical diabetes."""
    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "lr",
    }
    response = client.post("/api/v1/predict/diabetes_mellitus", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "diabetes"


def test_predict_cancer_alias_resolution():
    """Integration Test: POST /api/v1/predict/oncology resolves alias to canonical cancer."""
    sample_cancer = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    payload = {
        "features": sample_cancer,
        "model_id": "cancer_rf",
    }
    response = client.post("/api/v1/predict/oncology", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "cancer"



def test_backward_compatible_predict_endpoint():
    """Backward Compatibility Test: POST /api/v1/predict without disease path defaults to diabetes."""
    payload = {
        "features": SAMPLE_FEATURES,
        "model_id": "xgb",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "diabetes"
