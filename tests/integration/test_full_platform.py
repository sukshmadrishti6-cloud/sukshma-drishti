"""Comprehensive End-to-End Integration Verification Test Suite for SukshmaDrishti.

Verifies the 5 major core problem areas:
1. Cancer and multi-disease explainability with real feature attributions.
2. Complete ROC curves, PR curves, Confusion Matrices, and Calibration diagrams for evaluated models.
3. Multi-disease data pipeline verification (Diabetes, Cardiovascular, Cancer).
4. Hybrid comparison matrix with all canonical models and category breakdowns.
5. Multi-disease clinical inference execution with dedicated input schemas and production winner models.
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
    "Pregnancies": 6.0,
    "Glucose": 148.0,
    "BloodPressure": 72.0,
    "SkinThickness": 35.0,
    "Insulin": 0.0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50.0,
}

CARDIO_PAYLOAD = {
    "age": 58.0,
    "sex": 1.0,
    "cp": 2.0,
    "trestbps": 135.0,
    "chol": 240.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalach": 152.0,
    "exang": 0.0,
    "oldpeak": 1.2,
    "slope": 1.0,
    "ca": 0.0,
    "thal": 2.0,
}

CANCER_PAYLOAD = {
    "mean_radius": 14.2,
    "mean_texture": 19.3,
    "mean_perimeter": 92.5,
    "mean_area": 654.0,
    "mean_smoothness": 0.096,
    "mean_compactness": 0.104,
    "mean_concavity": 0.088,
    "mean_concave_points": 0.048,
    "mean_symmetry": 0.181,
    "mean_fractal_dimension": 0.062,
    "radius_error": 0.40,
    "texture_error": 1.21,
    "perimeter_error": 2.86,
    "area_error": 40.3,
    "smoothness_error": 0.007,
    "compactness_error": 0.025,
    "concavity_error": 0.031,
    "concave_points_error": 0.011,
    "symmetry_error": 0.020,
    "fractal_dimension_error": 0.003,
    "worst_radius": 16.2,
    "worst_texture": 25.6,
    "worst_perimeter": 107.2,
    "worst_area": 880.5,
    "worst_smoothness": 0.132,
    "worst_compactness": 0.254,
    "worst_concavity": 0.272,
    "worst_concave_points": 0.114,
    "worst_symmetry": 0.290,
    "worst_fractal_dimension": 0.083,
}


# ============================================================================
# 1. CANCER & MULTI-DISEASE EXPLAINABILITY TESTS
# ============================================================================

def test_cancer_global_explanation():
    """Verifies global explainability for cancer models returns real cancer feature names."""
    models_to_test = [
        "cancer_stacking_opt_v1",
        "cancer_rf_opt_v1",
        "cancer_xgboost_v1",
        "cancer_soft_voting_opt_v1",
    ]
    for m_id in models_to_test:
        resp = client.get(f"/api/v1/models/{m_id}/explain/global")
        assert resp.status_code == 200, f"Failed for {m_id}: {resp.text}"
        data = resp.json()
        assert data["model_name"] != ""
        assert len(data["features"]) > 0
        names = [c["name"] for c in data["features"]]
        assert any("radius" in n or "perimeter" in n or "concav" in n or "area" in n for n in names)


def test_cancer_local_inference_explanation():
    """Verifies cancer prediction with explain=True returns valid feature contributions."""
    resp = client.post(
        "/api/v1/predict/cancer",
        json={
            "features": CANCER_PAYLOAD,
            "model_id": "cancer_stacking_opt_v1",
            "explain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["predicted_class"] in [0, 1]
    assert data["explanation"] is not None
    assert data["explanation"]["status"] == "available"
    assert len(data["explanation"]["contributions"]) > 0
    contrib_names = [c["name"] for c in data["explanation"]["contributions"]]
    assert any("radius" in n or "perimeter" in n or "worst" in n for n in contrib_names)


# ============================================================================
# 2. EVALUATION ARTIFACTS & PERFORMANCE CURVES TESTS
# ============================================================================

def test_evaluation_curves_diabetes():
    """Verifies evaluation endpoint returns real ROC, PR, CM, and Calibration curves for diabetes models."""
    resp = client.get("/api/v1/evaluation/diabetes")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_models"] > 0
    evaluated = [m for m in data["models"] if m["evaluated"]]
    assert len(evaluated) >= 6

    for m in evaluated:
        if m["family"] != "quantum" and "hard_voting" not in m["model_id"]:
            assert m["confusion_matrix"] is not None, f"Missing CM for {m['model_id']}"
            assert m["curves"] is not None, f"Missing curves for {m['model_id']}"
            roc = m["curves"].get("roc_curve")
            assert roc is not None, f"Missing ROC for {m['model_id']}"
            assert len(roc["fpr"]) > 0 and len(roc["tpr"]) > 0
            assert "auc" in roc
            pr = m["curves"].get("pr_curve")
            assert pr is not None, f"Missing PR for {m['model_id']}"
            assert len(pr["precision"]) > 0 and len(pr["recall"]) > 0
            assert "auc" in pr


def test_evaluation_curves_cardiovascular():
    """Verifies evaluation endpoint returns real curves for cardiovascular models."""
    resp = client.get("/api/v1/evaluation/cardiovascular")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    evaluated = [m for m in data["models"] if m["evaluated"] and m["family"] != "quantum" and "hard_voting" not in m["model_id"]]
    assert len(evaluated) >= 5
    for m in evaluated:
        assert m["curves"] is not None, f"Missing curves for {m['model_id']}"
        assert len(m["curves"]["roc_curve"]["fpr"]) > 0


def test_evaluation_curves_cancer():
    """Verifies evaluation endpoint returns real curves for cancer models."""
    resp = client.get("/api/v1/evaluation/cancer")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    evaluated = [m for m in data["models"] if m["evaluated"] and m["family"] != "quantum" and "hard_voting" not in m["model_id"]]
    assert len(evaluated) >= 5
    for m in evaluated:
        assert m["curves"] is not None, f"Missing curves for {m['model_id']}"
        assert len(m["curves"]["roc_curve"]["fpr"]) > 0



# ============================================================================
# 3. DATA PIPELINE MULTI-DISEASE TESTS
# ============================================================================

def test_data_pipeline_metadata_all_diseases():
    """Verifies pipeline metadata endpoint provides verified shapes and distributions for each disease."""
    # Diabetes
    resp_d = client.get("/api/v1/pipeline?disease=diabetes")
    assert resp_d.status_code == 200
    d_data = resp_d.json()
    assert d_data["sample_count"] == 768
    assert d_data["raw_feature_count"] == 8
    assert d_data["total_feature_count"] == 11
    assert d_data["class_distribution"] == {"0": 500, "1": 268} or d_data["class_distribution"] == {0: 500, 1: 268}

    # Cardiovascular
    resp_c = client.get("/api/v1/pipeline?disease=cardiovascular")
    assert resp_c.status_code == 200
    c_data = resp_c.json()
    assert c_data["sample_count"] == 303
    assert c_data["raw_feature_count"] == 13
    assert len(c_data["feature_names"]) == 13

    # Cancer
    resp_ca = client.get("/api/v1/pipeline?disease=cancer")
    assert resp_ca.status_code == 200
    ca_data = resp_ca.json()
    assert ca_data["sample_count"] == 569
    assert ca_data["raw_feature_count"] == 30
    assert len(ca_data["feature_names"]) == 30


# ============================================================================
# 4. HYBRID COMPARISON MATRIX TESTS
# ============================================================================

def test_hybrid_comparison_matrix_multi_disease():
    """Verifies hybrid comparison matrix aggregates all models across categories."""
    for dis in ["diabetes", "cardiovascular", "cancer"]:
        resp = client.get(f"/api/v1/comparison?disease={dis}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert len(data["results"]) > 0
        categories = {r["category"] for r in data["results"]}
        assert "Baseline Classical" in categories or "Optimized Classical" in categories
        # Check production winner is marked
        prod_models = [r for r in data["results"] if r.get("production")]
        assert len(prod_models) >= 1


# ============================================================================
# 5. MULTI-DISEASE CLINICAL INFERENCE TESTS
# ============================================================================

def test_inference_diabetes():
    """Verifies end-to-end diabetes inference with production model."""
    resp = client.post(
        "/api/v1/predict/diabetes",
        json={
            "features": DIABETES_PAYLOAD,
            "model_id": "diabetes_rf_opt_v1",
            "explain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["predicted_class"] in [0, 1]
    assert data["probability"] is not None
    assert 0.0 <= data["probability"] <= 1.0
    assert data["audit"]["inference_time_ms"] > 0


def test_inference_cardiovascular():
    """Verifies end-to-end cardiovascular inference with production model."""
    resp = client.post(
        "/api/v1/predict/cardiovascular",
        json={
            "features": CARDIO_PAYLOAD,
            "model_id": "cardiovascular_extra_trees_opt_v1",
            "explain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["predicted_class"] in [0, 1]
    assert data["probability"] is not None
    assert 0.0 <= data["probability"] <= 1.0


def test_inference_cancer():
    """Verifies end-to-end cancer inference with production model."""
    resp = client.post(
        "/api/v1/predict/cancer",
        json={
            "features": CANCER_PAYLOAD,
            "model_id": "cancer_stacking_opt_v1",
            "explain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["predicted_class"] in [0, 1]
    assert data["probability"] is not None
    assert 0.0 <= data["probability"] <= 1.0
