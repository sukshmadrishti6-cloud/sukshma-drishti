"""Comprehensive API Contract & Integration Hardening Test Suite for Phase C."""
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

# Ensure backend package and workspace root are in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app

client = TestClient(app)

SAMPLE_VALID_FEATURES = [2.0, 130.0, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0]
SAMPLE_VALID_DICT = {
    "Pregnancies": 2.0,
    "Glucose": 130.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 100.0,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 32.0,
}


def test_health_endpoint_contract():
    """Verify GET /api/v1/health contract, status, and loaded production artifacts."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["backend_engine"] == "FastAPI"
    assert "models_available" in data
    models = data["models_available"]
    assert len(models) >= 6
    artifact_ids = [m["artifact_id"] for m in models]
    for expected in ["logistic_regression_v3", "svm_v3", "random_forest_v3", "xgboost_v3", "qsvc_v3", "vqc_v3"]:
        assert expected in artifact_ids


def test_models_endpoint_contract():
    """Verify GET /api/v1/models contract, provenance, feature counts, and score semantics."""
    res = client.get("/api/v1/models?disease=diabetes")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    models = data["models"]
    assert len(models) >= 6

    for m in models:
        assert m["status"] == "Ready"
        assert m["artifact_version"] in ["v1.0", "v2.0", "v2.0_optimized", "v3.0"]
        assert m["pipeline_version"] in ["v1.0", "v2.0"]
        assert m["raw_feature_count"] == 8

        assert m["classical_feature_count"] == 11
        assert m["score_semantics"] in [
            "calibrated_probability",
            "ensemble_vote_probability",
            "hard_voting_majority_class",
            "uncalibrated_quantum_decision_score",
        ]

        if m["is_quantum"]:
            assert m["qubit_count"] in [4, 6, 8]
            assert m["quantum_feature_count"] in [4, 6, 8]
            assert m["score_semantics"] == "uncalibrated_quantum_decision_score"
        else:
            assert m["qubit_count"] is None
            assert m["quantum_feature_count"] is None

    # Test cardiovascular models contract
    res_cardio = client.get("/api/v1/models?disease=cardiovascular")
    assert res_cardio.status_code == 200
    cardio_models = res_cardio.json()["models"]
    assert len(cardio_models) >= 2
    for cm in cardio_models:
        assert cm["status"] == "Ready"
        assert cm["raw_feature_count"] == 13
        assert cm["classical_feature_count"] == 13



def test_pipeline_endpoint_contract():
    """Verify GET /api/v1/pipeline reflects frozen Phase A & B canonical methodology."""
    res = client.get("/api/v1/pipeline")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "Ready"
    assert data["pipeline_version"] == "v2.0"
    assert data["raw_feature_count"] == 8
    assert data["engineered_feature_count"] == 3
    assert data["total_feature_count"] == 11
    assert "80/20 Stratified" in data["split_strategy"]
    assert data["target_variable"] == "Outcome"

    stages = [s["stage_name"] for s in data["stages"]]
    assert "Missing Value Imputation" in stages
    assert "Outlier Handling" in stages
    assert "Feature Engineering" in stages
    assert "Scaling" in stages
    assert "Quantum Reduction" in stages


def test_baselines_endpoint_contract():
    """Verify GET /api/v1/baselines returns classical holdout benchmark results."""
    res = client.get("/api/v1/baselines")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert len(data["results"]) >= 4
    for r in data["results"]:
        assert r["family"] == "classical"
        assert r["is_quantum"] is False
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["f1_score"] <= 1.0
        if r["roc_auc"] is not None:
            assert 0.0 <= r["roc_auc"] <= 1.0


def test_comparison_endpoint_contract():
    """Verify GET /api/v1/comparison returns unified classical & quantum metrics."""
    res = client.get("/api/v1/comparison")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert len(data["results"]) >= 6
    families = {r["family"] for r in data["results"]}
    assert "classical" in families
    assert "quantum" in families


def test_benchmarks_latest_endpoint_contract():
    """Verify GET /api/v1/benchmarks/latest exposes canonical Phase B benchmark with integrity."""
    res = client.get("/api/v1/benchmarks/latest")
    assert res.status_code == 200
    data = res.json()
    assert data["integrity_status"] == "verified"
    assert data["artifact_version"] == "v3.0"
    assert data["dataset"]["hash"] == "b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac"
    assert data["preprocessing"]["pipeline_version"] == "v2.0"
    assert "cross_validation_protocol" in data
    assert "evaluation_protocol" in data
    assert len(data["models"]) >= 6


def test_benchmarks_by_id_endpoint():
    """Verify GET /api/v1/benchmarks/{id} returns benchmark or 404."""
    # 1. Latest alias succeeds
    res_latest = client.get("/api/v1/benchmarks/latest")
    assert res_latest.status_code == 200
    b_id = res_latest.json().get("benchmark_id")

    # 2. Specific benchmark ID succeeds
    res_id = client.get(f"/api/v1/benchmarks/{b_id}")
    assert res_id.status_code == 200
    assert res_id.json()["benchmark_id"] == b_id

    # 3. Unknown benchmark ID returns 404
    res_404 = client.get("/api/v1/benchmarks/non_existent_benchmark_id_999")
    assert res_404.status_code == 404


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "svm", "qsvc", "vqc"])
def test_predict_endpoint_contract_all_models(model_key):
    """Verify POST /api/v1/predict returns valid inference response with audit and score semantics."""
    payload = {
        "features": SAMPLE_VALID_DICT,
        "model_id": model_key,
        "explain": True,
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["predicted_class"] in [0, 1]
    assert data["model_output"] in ["High Risk", "Low Risk"]
    assert data["artifact_version"] == "v3.0"
    assert data["pipeline_version"] == "v2.0"
    assert data["inference_id"] is not None
    assert data["score_semantics"] is not None
    assert data["audit"]["inference_time_ms"] > 0
    assert data["audit"]["request_id"] == data["inference_id"]

    # Verify explanation linkage
    if data["explanation"]:
        expl = data["explanation"]
        assert expl["status"] in ["available", "partial"]
        assert len(expl["contributions"]) > 0
        assert "shap" not in expl["explanation_method"].lower()


def test_predict_validation_errors():
    """Verify POST /api/v1/predict rejects invalid models, extra keys, missing keys, and traversal."""
    # 1. Unknown model ID -> 404
    res = client.post("/api/v1/predict", json={"features": SAMPLE_VALID_FEATURES, "model_id": "non_existent_model"})
    assert res.status_code == 404

    # 2. Path traversal in model_id -> 400
    res_trav = client.post("/api/v1/predict", json={"features": SAMPLE_VALID_FEATURES, "model_id": "../../etc/passwd"})
    assert res_trav.status_code in [400, 404]

    # 3. Missing required dictionary feature -> 422
    incomplete_dict = dict(SAMPLE_VALID_DICT)
    del incomplete_dict["Glucose"]
    res_missing = client.post("/api/v1/predict", json={"features": incomplete_dict, "model_id": "rf"})
    assert res_missing.status_code == 422

    # 4. Extra unapproved feature -> 422
    extra_dict = dict(SAMPLE_VALID_DICT, MaliciousField=999.0)
    res_extra = client.post("/api/v1/predict", json={"features": extra_dict, "model_id": "rf", "unexpected_param": True})
    assert res_extra.status_code == 422

    # 5. Outcome injection -> 422
    outcome_dict = dict(SAMPLE_VALID_DICT, Outcome=1.0)
    res_outcome = client.post("/api/v1/predict", json={"features": outcome_dict, "model_id": "rf"})
    assert res_outcome.status_code == 422


def test_global_explainability_all_models():
    """Verify GET /api/v1/models/{id}/explain/global returns real global feature importance."""
    for model_key in ["rf", "xgb", "lr", "svm", "qsvc", "vqc"]:
        res = client.get(f"/api/v1/models/{model_key}/explain/global")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["available", "partial"]
        assert len(data["features"]) > 0
        assert "shap" not in data["method"].lower()

    # Unknown model -> 404
    res_unknown = client.get("/api/v1/models/non_existent_model_id/explain/global")
    assert res_unknown.status_code == 404


def test_quantum_experiment_unknown_id_returns_404():
    """Verify GET /api/v1/experiments/{id} returns 404 for non-existent experiments."""
    res = client.get("/api/v1/experiments/exp_non_existent_id_404")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()

    res_stat = client.get("/api/v1/experiments/exp_non_existent_id_404/status")
    assert res_stat.status_code == 404


def test_cross_service_end_to_end_consistency():
    """E2E Consistency Test: GET /models -> POST /predict -> GET /explain/global share single truth."""
    # 1. Fetch models from registry
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    selected_model = res_models.json()["models"][0]
    m_id = selected_model["model_id"]
    expected_art_ver = selected_model["artifact_version"]
    expected_pipe_ver = selected_model["pipeline_version"]

    # 2. Run prediction with selected model
    pred_res = client.post(
        "/api/v1/predict",
        json={"features": SAMPLE_VALID_DICT, "model_id": m_id, "explain": True},
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["artifact_id"] == m_id
    assert pred_data["artifact_version"] == expected_art_ver
    assert pred_data["pipeline_version"] == expected_pipe_ver

    # 3. Retrieve global explanation
    expl_res = client.get(f"/api/v1/models/{m_id}/explain/global")
    assert expl_res.status_code == 200
    expl_data = expl_res.json()
    assert expl_data["status"] == "available"
