"""Integration tests for Phase G — Advanced ML Models + Ensembles across Diabetes, Cardiovascular, and Cancer."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from app.main import app
from app.services.ml_service import ml_service, MODEL_REGISTRY

client = TestClient(app)

SAMPLE_DIABETES_FEATURES = {
    "Pregnancies": 2.0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 33.0,
}

SAMPLE_CARDIO_FEATURES = {
    "age": 54.0,
    "sex": 1.0,
    "cp": 0.0,
    "trestbps": 130.0,
    "chol": 233.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 2.3,
    "slope": 0.0,
    "ca": 0.0,
    "thal": 2.0,
}

SAMPLE_CANCER_FEATURES = {
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


def test_registered_models_discovery():
    """Verify that all Phase G models are registered and discoverable."""
    dib_models = ml_service.get_registered_models_summary("diabetes")
    cardio_models = ml_service.get_registered_models_summary("cardiovascular")
    cancer_models = ml_service.get_registered_models_summary("cancer")

    dib_ids = [m["artifact_id"] for m in dib_models]
    cardio_ids = [m["artifact_id"] for m in cardio_models]
    cancer_ids = [m["artifact_id"] for m in cancer_models]

    assert "linear_svm_v1" in dib_ids
    assert "soft_voting_ensemble_v1" in dib_ids
    assert "cardiovascular_linear_svm_v1" in cardio_ids
    assert "cardiovascular_soft_voting_ensemble_v1" in cardio_ids
    assert "cancer_linear_svm_v1" in cancer_ids
    assert "cancer_soft_voting_ensemble_v1" in cancer_ids


def test_diabetes_phase_g_models_inference():
    """Verify inference across new Phase G Diabetes models."""
    models_to_test = [
        "linear_svm_v1",
        "knn_v1",
        "decision_tree_v1",
        "extra_trees_v1",
        "gradient_boosting_v1",
        "hist_gradient_boosting_v1",
        "soft_voting_ensemble_v1",
        "hard_voting_ensemble_v1",
        "stacking_ensemble_v1",
    ]
    for model_id in models_to_test:
        resp = client.post(
            "/api/v1/predict",
            json={"model_id": model_id, "features": SAMPLE_DIABETES_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["model_output"] in ["High Risk", "Low Risk"]
        assert "score_semantics" in data


def test_cardiovascular_phase_g_models_inference():
    """Verify inference across new Phase G Cardiovascular models."""
    models_to_test = [
        "cardiovascular_linear_svm_v1",
        "cardiovascular_rbf_svm_v1",
        "cardiovascular_knn_v1",
        "cardiovascular_decision_tree_v1",
        "cardiovascular_extra_trees_v1",
        "cardiovascular_gradient_boosting_v1",
        "cardiovascular_hist_gradient_boosting_v1",
        "cardiovascular_xgboost_v1",
        "cardiovascular_soft_voting_ensemble_v1",
        "cardiovascular_hard_voting_ensemble_v1",
        "cardiovascular_stacking_ensemble_v1",
    ]
    for model_id in models_to_test:
        resp = client.post(
            "/api/v1/predict/cardiovascular",
            json={"model_id": model_id, "features": SAMPLE_CARDIO_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for cardio model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["disease"] == "cardiovascular"


def test_cancer_phase_g_models_inference():
    """Verify inference across new Phase G Cancer models."""
    models_to_test = [
        "cancer_linear_svm_v1",
        "cancer_rbf_svm_v1",
        "cancer_knn_v1",
        "cancer_decision_tree_v1",
        "cancer_extra_trees_v1",
        "cancer_gradient_boosting_v1",
        "cancer_hist_gradient_boosting_v1",
        "cancer_xgboost_v1",
        "cancer_soft_voting_ensemble_v1",
        "cancer_hard_voting_ensemble_v1",
        "cancer_stacking_ensemble_v1",
    ]
    for model_id in models_to_test:
        resp = client.post(
            "/api/v1/predict/cancer",
            json={"model_id": model_id, "features": SAMPLE_CANCER_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for cancer model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["disease"] == "cancer"


def test_score_semantics_verification():
    """Verify correct score semantics classification for soft vs hard voting vs classical."""
    soft_resp = client.post(
        "/api/v1/predict",
        json={"model_id": "soft_voting_ensemble_v1", "features": SAMPLE_DIABETES_FEATURES},
    ).json()
    assert soft_resp["score_semantics"] == "ensemble_vote_probability"
    assert soft_resp["probability"] is not None

    hard_resp = client.post(
        "/api/v1/predict",
        json={"model_id": "hard_voting_ensemble_v1", "features": SAMPLE_DIABETES_FEATURES},
    ).json()
    assert hard_resp["score_semantics"] == "hard_voting_majority_class"

    stack_resp = client.post(
        "/api/v1/predict",
        json={"model_id": "stacking_ensemble_v1", "features": SAMPLE_DIABETES_FEATURES},
    ).json()
    assert stack_resp["score_semantics"] == "ensemble_vote_probability"


def test_security_unregistered_model_rejection():
    """Verify rejection of malicious path traversal and unregistered model IDs."""
    bad_ids = [
        "../models/classical/logistic_regression_v3.joblib",
        "non_existent_model_v999",
        "../../etc/passwd",
    ]
    for bad_id in bad_ids:
        resp = client.post(
            "/api/v1/predict",
            json={"model_id": bad_id, "features": SAMPLE_DIABETES_FEATURES},
        )
        assert resp.status_code == 404
