"""Integration tests verifying full 67-model canonical registry preservation and quantum optimization integrity."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.ml_service import CANONICAL_MODELS

client = TestClient(app)


def test_canonical_models_total_count():
    """Verify CANONICAL_MODELS contains all 67 models across 3 diseases."""
    assert len(CANONICAL_MODELS) == 67, f"Expected 67 models, found {len(CANONICAL_MODELS)}"

    dib_models = [m for m in CANONICAL_MODELS.values() if m["disease"] == "diabetes"]
    cardio_models = [m for m in CANONICAL_MODELS.values() if m["disease"] == "cardiovascular"]
    cancer_models = [m for m in CANONICAL_MODELS.values() if m["disease"] == "cancer"]

    assert len(dib_models) == 23, f"Expected 23 diabetes models, found {len(dib_models)}"
    assert len(cardio_models) == 22, f"Expected 22 cardio models, found {len(cardio_models)}"
    assert len(cancer_models) == 22, f"Expected 22 cancer models, found {len(cancer_models)}"


def test_models_api_counts_per_disease():
    """Verify /api/v1/models endpoint returns exact counts per disease and globally."""
    # Diabetes
    res_dib = client.get("/api/v1/models?disease=diabetes")
    assert res_dib.status_code == 200
    data_dib = res_dib.json()
    assert data_dib["total"] == 23, f"Expected 23 diabetes models, got {data_dib['total']}"

    # Cardiovascular
    res_cardio = client.get("/api/v1/models?disease=cardiovascular")
    assert res_cardio.status_code == 200
    data_cardio = res_cardio.json()
    assert data_cardio["total"] == 22, f"Expected 22 cardio models, got {data_cardio['total']}"

    # Cancer
    res_cancer = client.get("/api/v1/models?disease=cancer")
    assert res_cancer.status_code == 200
    data_cancer = res_cancer.json()
    assert data_cancer["total"] == 22, f"Expected 22 cancer models, got {data_cancer['total']}"

    # All
    res_all = client.get("/api/v1/models")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["total"] == 67, f"Expected 67 total models, got {data_all['total']}"


def test_evaluation_api_counts_per_disease():
    """Verify /api/v1/evaluation/{disease} returns exact counts."""
    # Diabetes
    res_dib = client.get("/api/v1/evaluation/diabetes")
    assert res_dib.status_code == 200
    data_dib = res_dib.json()
    assert data_dib["total_models"] == 23, f"Expected 23 diabetes models in eval, got {data_dib['total_models']}"

    # Cardiovascular
    res_cardio = client.get("/api/v1/evaluation/cardiovascular")
    assert res_cardio.status_code == 200
    data_cardio = res_cardio.json()
    assert data_cardio["total_models"] == 22, f"Expected 22 cardio models in eval, got {data_cardio['total_models']}"

    # Cancer
    res_cancer = client.get("/api/v1/evaluation/cancer")
    assert res_cancer.status_code == 200
    data_cancer = res_cancer.json()
    assert data_cancer["total_models"] == 22, f"Expected 22 cancer models in eval, got {data_cancer['total_models']}"

    # All
    res_all = client.get("/api/v1/evaluation/all")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["total_models"] == 67, f"Expected 67 total models in eval, got {data_all['total_models']}"


def test_comparison_api_counts():
    """Verify /api/v1/comparison/{disease} returns exact counts."""
    res_dib = client.get("/api/v1/comparison/diabetes")
    assert res_dib.status_code == 200
    assert len(res_dib.json()["results"]) == 23

    res_cardio = client.get("/api/v1/comparison/cardiovascular")
    assert res_cardio.status_code == 200
    assert len(res_cardio.json()["results"]) == 22

    res_cancer = client.get("/api/v1/comparison/cancer")
    assert res_cancer.status_code == 200
    assert len(res_cancer.json()["results"]) == 22

    res_all = client.get("/api/v1/comparison/all")
    assert res_all.status_code == 200
    assert len(res_all.json()["results"]) == 67


def test_classical_baselines_api_counts():
    """Verify /api/v1/baselines/{disease} returns exact classical model counts."""
    # Classical only: 17 diabetes, 16 cardio, 16 cancer = 49 total
    res_dib = client.get("/api/v1/baselines/diabetes")
    assert res_dib.status_code == 200
    assert len(res_dib.json()["results"]) == 17

    res_cardio = client.get("/api/v1/baselines/cardiovascular")
    assert res_cardio.status_code == 200
    assert len(res_cardio.json()["results"]) == 16

    res_cancer = client.get("/api/v1/baselines/cancer")
    assert res_cancer.status_code == 200
    assert len(res_cancer.json()["results"]) == 16

    res_all = client.get("/api/v1/baselines/all")
    assert res_all.status_code == 200
    assert len(res_all.json()["results"]) == 49


def test_diagnostics_api():
    """Verify /api/v1/diagnostics returns healthy registry verification."""
    res = client.get("/api/v1/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert data["integrity_verified"] is True
    assert data["total_canonical_models"] == 67
    assert data["diseases"]["diabetes"]["total_registered"] == 23
    assert data["diseases"]["cardiovascular"]["total_registered"] == 22
    assert data["diseases"]["cancer"]["total_registered"] == 22


def test_baseline_run_invariance():
    """Verify running classical baselines does not decrease available model count."""
    count_before = len(client.get("/api/v1/evaluation/all").json()["models"])
    assert count_before == 67

    # Trigger baseline run
    res_run = client.post("/api/v1/baselines/run")
    assert res_run.status_code == 200

    # Verify model count invariant (after_count >= before_count)
    count_after = len(client.get("/api/v1/evaluation/all").json()["models"])
    assert count_after >= count_before, f"Model count decreased! Before: {count_before}, After: {count_after}"
    assert count_after == 67
