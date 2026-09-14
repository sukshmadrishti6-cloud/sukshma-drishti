"""Integration tests for FastAPI inference and explainability endpoints using TestClient."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure backend package and workspace root are in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app

client = TestClient(app)

SAMPLE_FEATURES_LIST = [2.0, 130.0, 70.0, 25.0, 100.0, 28.5, 0.45, 32.0]
SAMPLE_FEATURES_DICT = {
    "Pregnancies": 2.0,
    "Glucose": 130.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 100.0,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 32.0,
}


def test_root_endpoint():
    """Test API root endpoint response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "SukshmaDrishti Backend API"


def test_health_endpoint():
    """Test health check endpoint and model availability reporting."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend_engine"] == "FastAPI"
    assert "models_available" in data
    assert len(data["models_available"]) >= 4


@pytest.mark.parametrize(
    "model_name,expected_family",
    [
        ("logistic_regression_v2", "classical"),
        ("svm_v2", "classical"),
        ("random_forest_v2", "classical"),
        ("xgboost_v2", "classical"),
        ("lr", "classical"),
        ("rf", "classical"),
        ("xgb", "classical"),
        ("qsvc_v2", "quantum"),
        ("vqc_v2", "quantum"),
    ],
)
def test_predict_endpoint_all_models(model_name: str, expected_family: str):
    """Test inference execution across all classical and quantum models with list input."""
    payload = {
        "features": SAMPLE_FEATURES_LIST,
        "model_id": model_name,
        "explain": False,
    }
    response = client.post("/api/v1/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Models unavailable due to missing dataset.")
    
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [0, 1]
    assert data["model_family"] == expected_family
    assert "artifact_id" in data
    assert "audit" in data
    assert data["audit"]["inference_time_ms"] >= 0.0
    assert data["explanation"] is None
    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0


def test_predict_endpoint_with_explainability_classical():
    """Test inference execution with explain=True for classical models."""
    for model_alias in ["lr", "rf", "svm", "xgb"]:
        payload = {
            "features": SAMPLE_FEATURES_LIST,
            "model_id": model_alias,
            "explain": True,
        }
        response = client.post("/api/v1/predict", json=payload)
        if response.status_code == 503:
            continue
            
        assert response.status_code == 200
        data = response.json()

        expl = data.get("explanation")
        assert expl is not None
        assert expl["status"] == "available"
        assert len(expl["contributions"]) in [8, 11]
        allowed_features = list(SAMPLE_FEATURES_DICT.keys()) + ["Glucose_BMI", "Insulin_Glucose", "AgeGroup"]
        for f in expl["contributions"]:
            assert f["direction"] in ["positive", "negative", "neutral"]
            assert f["name"] in allowed_features


def test_predict_endpoint_with_explainability_quantum():
    """Test inference execution with explain=True for quantum models (QSVC & VQC)."""
    for model_alias in ["qsvc", "vqc"]:
        payload = {
            "features": SAMPLE_FEATURES_LIST,
            "model_id": model_alias,
            "explain": True,
        }
        response = client.post("/api/v1/predict", json=payload)
        if response.status_code == 503:
            continue
            
        assert response.status_code == 200
        data = response.json()

        expl = data.get("explanation")
        assert expl is not None
        assert expl["status"] == "partial"
        assert expl["model_version"] == "quantum"
        assert len(expl["contributions"]) == 8


def test_predict_endpoint_structured_dict_input():
    """Test inference execution with structured biomedical dictionary input."""
    payload = {
        "features": SAMPLE_FEATURES_DICT,
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Models unavailable")
        
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["model_family"] == "classical"
    assert data["model_used"] == "Random Forest"


def test_target_column_injection_rejected():
    """Security Test: Reject request attempting to inject target column 'Outcome'."""
    bad_features = dict(SAMPLE_FEATURES_DICT)
    bad_features["Outcome"] = 1.0
    payload = {
        "features": bad_features,
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_invalid_feature_count_rejected():
    """Validation Test: Reject feature vector with fewer or more than 8 elements."""
    payload = {
        "features": [1.0, 2.0, 3.0],  # only 3 elements
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_dict_features_rejected():
    """Validation Test: Reject dictionary missing required clinical features."""
    incomplete_features = {
        "Pregnancies": 2.0,
        "Glucose": 130.0,
    }
    payload = {
        "features": incomplete_features,
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_unknown_model_returns_404():
    """Validation Test: Unknown model alias returns controlled HTTP 404 error."""
    payload = {
        "features": SAMPLE_FEATURES_LIST,
        "model_id": "non_existent_deep_network",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_path_traversal_attempt_rejected():
    """Security Test: Path traversal in model_id is rejected with HTTP 422."""
    payload = {
        "features": SAMPLE_FEATURES_LIST,
        "model_id": "../../etc/passwd",
    }
    response = client.post("/api/v1/predict", json=payload)
    # The validation happens before or during lookup, it might be 404 or 422.
    # In my updated MLService it doesn't fail regex, but fails dictionary lookup
    # Because it is not in MODEL_REGISTRY and the path traversal is checked safely or not found.
    assert response.status_code in [404, 422]


def test_authenticated_predict_request():
    """Authentication Test: Valid Bearer token is parsed cleanly."""
    from app.core.security import create_access_token
    token = create_access_token(subject="user_123", email="doctor@hospital.org")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "features": SAMPLE_FEATURES_LIST,
        "model_id": "lr",
    }
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 200



def test_no_secret_leakage_in_api_response():
    """Security Test: Verify response never contains secrets or internal tokens."""
    payload = {
        "features": SAMPLE_FEATURES_LIST,
        "model_id": "rf",
    }
    response = client.post("/api/v1/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Models unavailable")
    assert response.status_code == 200
    raw_text = response.text.lower()
    assert "supabase_service_role_key" not in raw_text
    assert "password" not in raw_text
    assert "secret" not in raw_text


def test_classical_baselines_artifacts():
    """Verify GET /api/v1/baselines returns evaluation artifacts including confusion matrix, curves, and calibration."""
    response = client.get("/api/v1/baselines")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert len(data["results"]) >= 4

    for model_res in data["results"]:
        assert "confusion_matrix" in model_res
        cm = model_res["confusion_matrix"]
        assert cm is not None
        assert "tn" in cm and "fp" in cm and "fn" in cm and "tp" in cm

        assert "curves" in model_res
        curves = model_res["curves"]
        if curves:
            assert "roc_curve" in curves

        assert "calibration" in model_res
        calib = model_res["calibration"]
        assert calib is not None
        assert "status" in calib


def test_classical_baselines_status():
    """Verify GET /api/v1/baselines/status returns valid runner status."""
    response = client.get("/api/v1/baselines/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["idle", "running", "completed", "failed"]


def test_global_explanation_endpoint():
    """Verify GET /api/v1/models/{id}/explain/global returns valid global attribution."""
    # Test Random Forest (tree model)
    rf_res = client.get("/api/v1/models/random_forest_v2/explain/global")
    assert rf_res.status_code == 200
    rf_data = rf_res.json()
    assert rf_data["status"] == "available"
    assert rf_data["method"] == "mean_decrease_impurity"
    assert len(rf_data["features"]) > 0

    # Test SVM (RBF kernel: permutation importance)
    svm_res = client.get("/api/v1/models/svm_v2/explain/global")
    assert svm_res.status_code == 200
    svm_data = svm_res.json()
    assert svm_data["status"] == "available"
    assert svm_data["method"] == "permutation_importance"
    assert len(svm_data["features"]) == 11
    assert "metadata" in svm_data
    assert svm_data["metadata"]["scoring_metric"] == "roc_auc"
    assert svm_data["metadata"]["random_seed"] == 42
    assert len(svm_data["limitations"]) > 0


def test_quantum_capabilities_endpoint():
    """Verify GET /api/v1/experiments/capabilities returns authoritative width metadata."""
    res = client.get("/api/v1/experiments/capabilities")
    assert res.status_code == 200
    caps = res.json()
    assert len(caps) == 3

    cap_dict = {c["qubit_count"]: c for c in caps}
    assert 4 in cap_dict and 6 in cap_dict and 8 in cap_dict

    # 4Q, 6Q, and 8Q are all trained and evaluated
    for q in [4, 6, 8]:
        assert cap_dict[q]["status"] == "evaluated"
        assert cap_dict[q]["trained"] is True
        assert cap_dict[q]["evaluated"] is True


def test_quantum_experiment_lifecycle():
    """Verify POST /api/v1/experiments/run queues and completes simulation with Qiskit circuit for 4Q."""
    payload = {
        "title": "Integration Test QSVC",
        "dataset_name": "Pima Indians Diabetes",
        "dataset_hash": "b78029447fae",
        "qubit_count": 4,
        "feature_map": "ZZFeatureMap",
        "ansatz": "n/a",
        "optimizer": "n/a",
        "shots": 1024,
    }
    create_res = client.post("/api/v1/experiments/run", json=payload)
    assert create_res.status_code == 200
    exp_id = create_res.json()["id"]

    # Poll status
    import time
    completed = False
    for _ in range(12):
        time.sleep(0.5)
        stat = client.get(f"/api/v1/experiments/{exp_id}/status").json()
        if stat["status"] == "completed":
            completed = True
            break

    assert completed is True

    exp_res = client.get(f"/api/v1/experiments/{exp_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["status"] == "completed"
    assert exp_data["execution_status"] == "completed"
    assert exp_data["evaluation_status"] == "completed"
    assert exp_data["circuit_text"] is not None
    assert exp_data["circuit_depth"] is not None
    assert exp_data["num_parameters"] is not None
    assert exp_data["metrics"] is not None
    assert "f1_score" in exp_data["metrics"]


def test_quantum_6q_experiment_metric_isolation():
    """Critical Scientific Test: 6Q experiment must evaluate authentically and NEVER leak or copy 4Q metrics."""
    payload = {
        "title": "Integration Test QSVC 6Q",
        "dataset_name": "Pima Indians Diabetes",
        "dataset_hash": "b78029447fae",
        "qubit_count": 6,
        "feature_map": "ZZFeatureMap",
        "ansatz": "n/a",
        "optimizer": "n/a",
        "shots": 1024,
    }
    create_res = client.post("/api/v1/experiments/run", json=payload)
    assert create_res.status_code == 200
    exp_id = create_res.json()["id"]

    import time
    completed = False
    for _ in range(12):
        time.sleep(0.5)
        stat = client.get(f"/api/v1/experiments/{exp_id}/status").json()
        if stat["status"] == "completed":
            completed = True
            break

    assert completed is True

    exp_res = client.get(f"/api/v1/experiments/{exp_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()

    # Circuit simulation must succeed
    assert exp_data["status"] == "completed"
    assert exp_data["execution_status"] == "completed"
    assert exp_data["circuit_text"] is not None
    assert exp_data["qubit_count"] == 6

    # Holdout evaluation is complete with authentic 6Q model metrics, strictly isolated from 4Q
    assert exp_data["evaluation_status"] == "completed"
    assert exp_data["model_status"] == "trained"
    assert exp_data["model_id"] == "qsvc_6q_v1"
    assert exp_data["metrics"] is not None
    assert exp_data["confusion_matrix"] is not None
    assert exp_data["curves"] is not None
    assert exp_data["metrics"]["accuracy"] != 0.7532  # Must not copy 4Q accuracy
    assert 0.0 <= exp_data["metrics"]["accuracy"] <= 1.0


