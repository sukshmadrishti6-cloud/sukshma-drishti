"""Tests for Phase K — Explainability & Evaluation Dashboards."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_evaluation_diabetes_endpoint():
    response = client.get("/api/v1/evaluation/diabetes")
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "diabetes"
    assert data["total_models"] > 0
    assert "models" in data
    
    # Check that at least one model has valid metrics
    eval_models = [m for m in data["models"] if m["evaluated"]]
    assert len(eval_models) > 0
    m = eval_models[0]
    assert "accuracy" in m["metrics"]
    assert "precision" in m["metrics"]
    assert "recall" in m["metrics"]
    assert "f1_score" in m["metrics"]
    assert "roc_auc" in m["metrics"]
    assert "confusion_matrix" in m


def test_evaluation_cardiovascular_endpoint():
    response = client.get("/api/v1/evaluation/cardiovascular")
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "cardiovascular"
    assert data["total_models"] > 0
    eval_models = [m for m in data["models"] if m["evaluated"]]
    assert len(eval_models) > 0


def test_evaluation_cancer_endpoint():
    response = client.get("/api/v1/evaluation/cancer")
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "cancer"
    assert data["total_models"] > 0
    eval_models = [m for m in data["models"] if m["evaluated"]]
    assert len(eval_models) > 0


def test_evaluation_all_diseases():
    response = client.get("/api/v1/evaluation/all")
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "all"
    assert data["total_models"] >= 30


def test_evaluation_invalid_disease():
    response = client.get("/api/v1/evaluation/invalid_disease_name")
    assert response.status_code == 404


def test_circuit_only_models_handling():
    response = client.get("/api/v1/evaluation/all")
    assert response.status_code == 200
    data = response.json()
    circuit_only = [m for m in data["models"] if not m["evaluated"]]
    for m in circuit_only:
        assert m["status"] == "Not Evaluated / Circuit Only"
        assert m["metrics"] is None


def test_comparison_filtered_by_disease():
    response = client.get("/api/v1/comparison/?disease=cardiovascular")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["disease"].lower() == "cardiovascular"


def test_global_explanation_models():
    # Test global explanation on a known primary model
    response = client.get("/api/v1/models/logistic_regression_v3/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) > 0
    assert "method" in data
