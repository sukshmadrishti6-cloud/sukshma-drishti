"""Integration tests verifying real PDF report scanning and inference pipeline."""
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app

client = TestClient(app)

SAMPLE_PDF_PATH = "x:/SukshmaDrishti_Sample_Diabetes_Lab_Report.pdf"


def test_real_sample_diabetes_pdf_scan_api():
    """Verifies that the actual SukshmaDrishti_Sample_Diabetes_Lab_Report.pdf extracts 8/8 features."""
    if not os.path.exists(SAMPLE_PDF_PATH):
        pytest.skip(f"Sample PDF '{SAMPLE_PDF_PATH}' not found on filesystem.")

    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("SukshmaDrishti_Sample_Diabetes_Lab_Report.pdf", pdf_bytes, "application/pdf")}
    data = {"disease": "diabetes"}

    response = client.post("/api/v1/report/scan", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["success"] is True
    assert res_data["disease"] == "diabetes"
    assert res_data["compatible"] is True
    assert res_data["total_features_required"] == 8
    assert res_data["total_features_detected"] == 8
    assert res_data["missing_features"] == []
    assert len(res_data["extracted_features"]) == 8

    # Verify each individual feature value
    extracted = res_data["extracted_features"]

    assert "Pregnancies" in extracted
    assert extracted["Pregnancies"]["value"] == 0.0

    assert "Glucose" in extracted
    assert extracted["Glucose"]["value"] == 142.0
    assert extracted["Glucose"]["unit"] == "mg/dL"

    assert "BloodPressure" in extracted
    assert extracted["BloodPressure"]["value"] == 86.0  # Diastolic BP
    assert extracted["BloodPressure"]["unit"] == "mmHg"

    assert "SkinThickness" in extracted
    assert extracted["SkinThickness"]["value"] == 29.0
    assert extracted["SkinThickness"]["unit"] == "mm"

    assert "Insulin" in extracted
    assert extracted["Insulin"]["value"] == 118.0

    assert "BMI" in extracted
    assert extracted["BMI"]["value"] == 27.4

    assert "DiabetesPedigreeFunction" in extracted
    assert extracted["DiabetesPedigreeFunction"]["value"] == 0.72

    assert "Age" in extracted
    assert extracted["Age"]["value"] == 46.0

    # Step 2: Feed extracted features to existing diabetes prediction model
    verified_features = {k: v["value"] for k, v in extracted.items()}
    predict_payload = {
        "features": verified_features,
        "model_id": "diabetes_rf_opt_v1",
        "explain": True,
        "disease": "diabetes",
    }
    pred_resp = client.post("/api/v1/predict/diabetes", json=predict_payload)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()

    assert pred_data["predicted_class"] in [0, 1]
    assert pred_data["model_family"] == "classical"
    assert "probability" in pred_data
    assert "explanation" in pred_data
    assert len(pred_data["explanation"]["contributions"]) == 8


def test_real_sample_pdf_scanned_for_cancer_rejection():
    """Verifies that uploading a generic diabetes lab report for Cancer returns compatible=False and 0/30 features."""
    if not os.path.exists(SAMPLE_PDF_PATH):
        pytest.skip(f"Sample PDF '{SAMPLE_PDF_PATH}' not found on filesystem.")

    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("SukshmaDrishti_Sample_Diabetes_Lab_Report.pdf", pdf_bytes, "application/pdf")}
    data = {"disease": "cancer"}

    response = client.post("/api/v1/report/scan", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["success"] is True
    assert res_data["disease"] == "cancer"
    assert res_data["compatible"] is False
    assert res_data["total_features_detected"] == 0
    assert res_data["total_features_required"] == 30
    assert len(res_data["missing_features"]) == 30
    assert "cell nucleus morphometric measurements" in res_data["message"]
