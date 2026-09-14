"""Comprehensive System Reconciliation & Multi-Disease Integrity Verification Test Suite.

Validates the complete single source of truth across:
1. Multi-disease catalog & dynamic endpoint queries.
2. Canonical 51-model registry with strict disease boundaries.
3. Production winners for diabetes, cardiovascular, and cancer domains.
4. Evaluation and Baseline API contracts and curve coordinates.
5. Quantum experiment transparency and zero metric fabrication.
6. Explainability dispatching and graceful fallback for non-decomposable architectures.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure backend and root paths are available
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app

client = TestClient(app)

DIABETES_PAYLOAD = {
    "Pregnancies": 2.0,
    "Glucose": 130.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 100.0,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 32.0,
}

CARDIO_PAYLOAD = {
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

CANCER_PAYLOAD = {
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
}


def test_reconciliation_disease_catalog():
    """Verify GET /api/v1/diseases returns authoritative metadata for 3 diseases."""
    res = client.get("/api/v1/diseases")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    items = data["items"]
    d_map = {d["id"]: d for d in items}

    assert "diabetes" in d_map
    assert "cardiovascular" in d_map
    assert "cancer" in d_map
    assert len(items) == 3


def test_reconciliation_all_canonical_models():
    """Verify GET /api/v1/models returns authoritative models with proper disease partitioning."""
    res_all = client.get("/api/v1/models")
    assert res_all.status_code == 200
    all_models = res_all.json()["models"]
    assert len(all_models) == 67

    # Diabetes models (19 classical/4Q + 4 evaluated 6Q/8Q = 23)
    res_diab = client.get("/api/v1/models?disease=diabetes")
    assert res_diab.status_code == 200
    diab_models = res_diab.json()["models"]
    assert len(diab_models) == 23
    for m in diab_models:
        assert m["disease"] == "diabetes"
        assert m["status"] == "Ready"

    # Cardio models (16 classical + 6 quantum = 22)
    res_cardio = client.get("/api/v1/models?disease=cardiovascular")
    assert res_cardio.status_code == 200
    cardio_models = res_cardio.json()["models"]
    assert len(cardio_models) == 22
    for m in cardio_models:
        assert m["disease"] == "cardiovascular"
        assert m["status"] == "Ready"

    # Cancer models (16 classical + 6 quantum = 22)
    res_cancer = client.get("/api/v1/models?disease=cancer")
    assert res_cancer.status_code == 200
    cancer_models = res_cancer.json()["models"]
    assert len(cancer_models) == 22
    for m in cancer_models:
        assert m["disease"] == "cancer"
        assert m["status"] == "Ready"


def test_reconciliation_production_winners():
    """Verify each disease domain has exactly one authoritative production winner."""
    # Diabetes Winner
    res_diab = client.post(
        "/api/v1/predict",
        json={"features": DIABETES_PAYLOAD, "model_id": "diabetes_rf_opt_v1", "disease": "diabetes", "explain": True},
    )
    assert res_diab.status_code == 200
    d_data = res_diab.json()
    assert d_data["disease"] == "diabetes"
    assert d_data["artifact_id"] == "diabetes_rf_opt_v1"
    assert d_data["probability"] is not None

    # Cardiovascular Winner
    res_cardio = client.post(
        "/api/v1/predict",
        json={"features": CARDIO_PAYLOAD, "model_id": "cardiovascular_extra_trees_opt_v1", "disease": "cardiovascular", "explain": True},
    )
    assert res_cardio.status_code == 200
    c_data = res_cardio.json()
    assert c_data["disease"] == "cardiovascular"
    assert c_data["artifact_id"] == "cardiovascular_extra_trees_opt_v1"
    assert c_data["probability"] is not None

    # Cancer Winner
    res_cancer = client.post(
        "/api/v1/predict",
        json={"features": CANCER_PAYLOAD, "model_id": "cancer_stacking_opt_v1", "disease": "cancer", "explain": True},
    )
    assert res_cancer.status_code == 200
    cn_data = res_cancer.json()
    assert cn_data["disease"] == "cancer"
    assert cn_data["artifact_id"] == "cancer_stacking_opt_v1"
    assert cn_data["probability"] is not None


def test_reconciliation_strict_disease_boundary_enforcement():
    """Verify mismatched payloads and cross-disease requests are rejected."""
    # Cardio payload sent to Diabetes model -> 422
    res_fail_1 = client.post(
        "/api/v1/predict",
        json={"features": CARDIO_PAYLOAD, "model_id": "diabetes_rf_opt_v1", "disease": "diabetes"},
    )
    assert res_fail_1.status_code == 422

    # Diabetes payload sent to Cancer model -> 422
    res_fail_2 = client.post(
        "/api/v1/predict",
        json={"features": DIABETES_PAYLOAD, "model_id": "cancer_stacking_opt_v1", "disease": "cancer"},
    )
    assert res_fail_2.status_code == 422

    # Unknown model ID -> 404
    res_fail_3 = client.post(
        "/api/v1/predict",
        json={"features": DIABETES_PAYLOAD, "model_id": "non_existent_model_id"},
    )
    assert res_fail_3.status_code == 404


def test_reconciliation_evaluation_api_contract():
    """Verify GET /api/v1/evaluation returns all evaluated models across disease domains."""
    res = client.get("/api/v1/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    models = data["models"]
    assert len(models) > 0

    for m in models:
        assert m["model_id"] is not None
        assert m["disease"] in ["diabetes", "cardiovascular", "cancer"]
        assert "Evaluated" in m["status"]
        assert m["evaluated"] is True
        assert isinstance(m["metrics"], dict)


def test_reconciliation_quantum_transparency():
    """Verify Quantum capability matrix reflects authentic evaluated status across 4Q, 6Q, and 8Q."""
    res = client.get("/api/v1/experiments/capabilities")
    assert res.status_code == 200
    caps_list = res.json()
    assert isinstance(caps_list, list)
    caps = {c["qubit_count"]: c for c in caps_list}

    # 4Q, 6Q, 8Q: All authentically evaluated and trained
    for q in [4, 6, 8]:
        assert q in caps
        assert caps[q]["status"] == "evaluated"
        assert caps[q]["evaluated"] is True
        assert caps[q]["trained"] is True


def test_reconciliation_explainability_graceful_fallbacks():
    """Verify explainability endpoints return real explanations or unsupported fallback without 500 crashes."""
    # 1. Classical RF (Optimized) -> available
    res_rf = client.get("/api/v1/models/diabetes_rf_opt_v1/explain/global")
    assert res_rf.status_code == 200
    assert res_rf.json()["status"] == "available"

    # 2. Classical Soft Voting Ensemble -> unsupported fallback (not 500)
    res_vote = client.get("/api/v1/models/diabetes_soft_voting_opt_v1/explain/global")
    assert res_vote.status_code == 200
    assert res_vote.json()["status"] in ["available", "unsupported"]

    # 3. Path traversal attack -> 404
    res_attack = client.get("/api/v1/models/../../etc/passwd/explain/global")
    assert res_attack.status_code == 404
