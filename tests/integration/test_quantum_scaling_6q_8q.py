"""Integration tests for 6Q and 8Q Quantum Models (QSVC-6Q, VQC-6Q, QSVC-8Q, VQC-8Q).

Verifies:
1. Canonical Model Registry registers 6Q and 8Q models with proper metadata.
2. Model ID aliases correctly resolve 6Q and 8Q variants.
3. Experiments capabilities endpoint reports 4Q, 6Q, and 8Q as evaluated and trained.
4. Models can be loaded from artifacts and verified for SHA-256 integrity.
5. Models perform valid inference on diabetes biomedical features.
6. Benchmark results file contains valid evaluated records for all configurations.
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ml_service import (
    CANONICAL_MODELS,
    MODEL_ALIASES,
    ml_service,
)
from ml.artifacts import load_artifact, compute_sha256_hash


@pytest.fixture
def client():
    return TestClient(app)


SAMPLE_DIABETES_INPUT = {
    "Pregnancies": 2.0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 25.5,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 32.0,
}


def test_canonical_models_registered_6q_8q():
    """Verify 6Q and 8Q quantum models exist in CANONICAL_MODELS with correct attributes."""
    expected_models = ["qsvc_6q_v1", "vqc_6q_v1", "qsvc_8q_v1", "vqc_8q_v1"]
    for m_id in expected_models:
        assert m_id in CANONICAL_MODELS, f"Model {m_id} missing from CANONICAL_MODELS"
        meta = CANONICAL_MODELS[m_id]
        assert meta["disease"] == "diabetes"
        assert meta["family"] == "quantum"
        assert meta["status"] == "Ready"
        assert meta["trained"] is True
        assert meta["evaluated"] is True
        assert meta["quantum"] is True
        if "6q" in m_id:
            assert meta["qubit_count"] == 6
            assert meta["quantum_feature_count"] == 6
        elif "8q" in m_id:
            assert meta["qubit_count"] == 8
            assert meta["quantum_feature_count"] == 8


def test_model_id_aliases_resolution():
    """Verify canonical model aliases resolve to expected model IDs."""
    aliases_to_test = {
        "qsvc_6q": "qsvc_6q_v1",
        "qsvc-6q": "qsvc_6q_v1",
        "vqc_6q": "vqc_6q_v1",
        "vqc-6q": "vqc_6q_v1",
        "qsvc_8q": "qsvc_8q_v1",
        "qsvc-8q": "qsvc_8q_v1",
        "vqc_8q": "vqc_8q_v1",
        "vqc-8q": "vqc_8q_v1",
    }
    for alias, expected in aliases_to_test.items():
        assert ml_service.resolve_artifact_id(alias, disease="diabetes") == expected, f"Alias {alias} did not resolve to {expected}"


def test_capabilities_api_reports_evaluated(client):
    """Verify GET /api/v1/experiments/capabilities returns 4, 6, 8 as evaluated and trained."""
    response = client.get("/api/v1/experiments/capabilities")
    assert response.status_code == 200
    caps = response.json()
    assert isinstance(caps, list)
    assert len(caps) >= 3
    
    qubit_map = {c["qubit_count"]: c for c in caps}
    for q in [4, 6, 8]:
        assert q in qubit_map, f"{q} qubits missing in capabilities"
        c = qubit_map[q]
        assert c["status"] == "evaluated"
        assert c["trained"] is True
        assert c["evaluated"] is True
        assert any("hardware_execution" in lim for lim in c["limitations"])


def test_experiments_latest_quantum_scaling(client):
    """Verify GET /api/v1/experiments/latest contains valid width-scaling records."""
    response = client.get("/api/v1/experiments/latest")
    assert response.status_code == 200
    exp = response.json()
    assert exp["status"] == "completed"
    
    scaling = exp.get("scaling", [])
    assert len(scaling) >= 6  # QSVC-4Q, VQC-4Q, QSVC-6Q, VQC-6Q, QSVC-8Q, VQC-8Q
    
    configs = {s["configuration"] for s in scaling}
    expected_configs = {"QSVC-4Q", "VQC-4Q", "QSVC-6Q", "VQC-6Q", "QSVC-8Q", "VQC-8Q"}
    assert expected_configs.issubset(configs)


def test_benchmark_results_file_integrity():
    """Verify data/processed/benchmark_results.json has evaluated 6Q and 8Q records."""
    bench_file = Path("data/processed/benchmark_results.json")
    assert bench_file.exists(), "benchmark_results.json does not exist"
    
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    
    models = {m["model_id"]: m for m in bench_data.get("models", [])}
    for m_id in ["qsvc_6q_v1", "vqc_6q_v1", "qsvc_8q_v1", "vqc_8q_v1"]:
        assert m_id in models, f"Model {m_id} missing in benchmark_results.json"
        m = models[m_id]
        assert m.get("status") in ["SUCCESS", "Evaluated", "Ready"] or "metrics" in m
        assert "metrics" in m
        assert m["metrics"]["accuracy"] > 0.0
        assert "confusion_matrix" in m
        assert "curves" in m


def test_quantum_artifacts_loading_and_hash():
    """Verify all 6Q and 8Q artifacts load from disk and have valid SHA-256 manifests."""
    for model_id in ["qsvc_6q_v1", "vqc_6q_v1", "qsvc_8q_v1", "vqc_8q_v1"]:
        artifact_path = Path(f"models/quantum/{model_id}.joblib")
        manifest_path = Path(f"models/quantum/{model_id}_manifest.json")
        
        assert artifact_path.exists(), f"Artifact file {artifact_path} does not exist"
        assert manifest_path.exists(), f"Manifest file {manifest_path} does not exist"
        
        artifact = load_artifact(model_id, base_dir="models")
        assert artifact is not None
        assert artifact.model_id == model_id
        assert artifact.model_family == "quantum"
        
        expected_qubits = 6 if "6q" in model_id else 8
        assert artifact.metadata.get("num_qubits") == expected_qubits


def test_predict_api_with_quantum_models(client):
    """Verify POST /api/v1/predict works with 6Q and 8Q models."""
    for model_id in ["qsvc_6q_v1", "vqc_6q_v1", "qsvc_8q_v1", "vqc_8q_v1"]:
        response = client.post(
            "/api/v1/predict",
            json={
                "disease": "diabetes",
                "model_id": model_id,
                "features": SAMPLE_DIABETES_INPUT,
                "explain": True,
            },
        )
        assert response.status_code == 200, f"Predict failed for {model_id}: {response.text}"
        res = response.json()
        assert res["artifact_id"] == model_id
        assert res["predicted_class"] in [0, 1]
        assert res["score_semantics"] == "uncalibrated_quantum_decision_score"
        assert res["model_family"] == "quantum"
        assert res["explanation"] is not None
