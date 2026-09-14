"""Integration tests for Phase H — Evaluation, Hyperparameter Optimization, Calibration, and Threshold Tuning."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from app.main import app
from app.services.ml_service import ml_service

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


def test_optimized_models_discovery():
    """Verify that Phase H optimized models are discoverable in registry summaries."""
    dib_models = ml_service.get_registered_models_summary("diabetes")
    cardio_models = ml_service.get_registered_models_summary("cardiovascular")
    cancer_models = ml_service.get_registered_models_summary("cancer")

    dib_ids = [m["artifact_id"] for m in dib_models]
    cardio_ids = [m["artifact_id"] for m in cardio_models]
    cancer_ids = [m["artifact_id"] for m in cancer_models]

    assert "diabetes_rf_opt_v1" in dib_ids
    assert "cardiovascular_stacking_opt_v1" in cardio_ids
    assert "cancer_stacking_opt_v1" in cancer_ids


def test_diabetes_optimized_inference():
    """Verify inference execution and custom threshold application for optimized Diabetes models."""
    opt_models = [
        "diabetes_rf_opt_v1",
        "diabetes_gb_opt_v1",
        "diabetes_xgb_opt_v1",
        "diabetes_soft_voting_opt_v1",
    ]
    for model_id in opt_models:
        resp = client.post(
            "/api/v1/predict",
            json={"model_id": model_id, "features": SAMPLE_DIABETES_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["probability"] is not None
        assert "score_semantics" in data


def test_cardiovascular_optimized_inference():
    """Verify inference execution for optimized Cardiovascular models."""
    opt_models = [
        "cardiovascular_extra_trees_opt_v1",
        "cardiovascular_rbf_svm_opt_v1",
        "cardiovascular_stacking_opt_v1",
    ]
    for model_id in opt_models:
        resp = client.post(
            "/api/v1/predict/cardiovascular",
            json={"model_id": model_id, "features": SAMPLE_CARDIO_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for cardio model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["disease"] == "cardiovascular"


def test_cancer_optimized_inference():
    """Verify inference execution for optimized Cancer models."""
    opt_models = [
        "cancer_stacking_opt_v1",
        "cancer_rf_opt_v1",
        "cancer_soft_voting_opt_v1",
    ]
    for model_id in opt_models:
        resp = client.post(
            "/api/v1/predict/cancer",
            json={"model_id": model_id, "features": SAMPLE_CANCER_FEATURES},
        )
        assert resp.status_code == 200, f"Failed for cancer model '{model_id}': {resp.text}"
        data = resp.json()
        assert data["predicted_class"] in [0, 1]
        assert data["disease"] == "cancer"


def test_custom_threshold_and_calibration_provenance():
    """Verify artifact metadata retains frozen threshold and calibration status."""
    artifact = ml_service.get_artifact("diabetes_rf_opt_v1")
    prov = getattr(artifact, "configuration_provenance", {})
    assert "frozen_threshold" in prov
    assert "optimization" in prov
    eval_prov = getattr(artifact, "evaluation_provenance", {}).get("metrics", {})
    assert "brier_score" in eval_prov
    assert "is_calibrated" in eval_prov
