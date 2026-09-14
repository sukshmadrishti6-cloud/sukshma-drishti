"""Integration tests for /report/scan API endpoint."""
import io
import os
import sys
from fastapi.testclient import TestClient
import pypdf
import pytest

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.main import app
from app.schemas.report import ReportScanResponse

client = TestClient(app)


def _create_mock_pdf(text_lines: list[str]) -> bytes:
    """Creates an in-memory PDF with specified text lines."""
    # Since pypdf is mainly a reader/manipulator, we can use a basic valid PDF structure
    # or write a stream
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    
    # Write page content stream
    text_content = "\n".join(text_lines)
    # Basic PDF text string representation
    buffer = io.BytesIO()
    writer.write(buffer)
    # For testing, we can also test text extraction directly or with mock
    return buffer.getvalue()


def test_scan_diabetes_report_api(monkeypatch):
    """Integration Test: Uploading a medical report for Diabetes extraction."""
    from app.report.parser import ExtractionResult, document_parser

    # Mock parse_pdf to return realistic extracted text
    def mock_parse_pdf(file_bytes, filename):
        lines = [
            "CLINICAL LABORATORY DIAGNOSTIC REPORT",
            "Patient Name: John Doe",
            "Age: 52 Yrs",
            "Fasting Blood Sugar: 154 mg/dL",
            "Blood Pressure: 138/84 mmHg",
            "BMI: 31.8 kg/m2",
            "Fasting Insulin: 110.0 uIU/mL",
            "Pregnancies: 0",
        ]
        return ExtractionResult(
            raw_text="\n".join(lines),
            lines=lines,
            file_type="pdf",
            filename=filename,
            page_count=1,
            ocr_engine_used="pypdf_text_engine",
        )

    monkeypatch.setattr(document_parser, "parse_pdf", mock_parse_pdf)

    files = {"file": ("diabetes_report.pdf", b"%PDF-1.4 Mock Data", "application/pdf")}
    data = {"disease": "diabetes"}

    response = client.post("/api/v1/report/scan", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["success"] is True
    assert res_data["disease"] == "diabetes"
    assert res_data["compatible"] is True
    assert "Glucose" in res_data["extracted_features"]
    assert res_data["extracted_features"]["Glucose"]["value"] == 154.0
    assert "BloodPressure" in res_data["extracted_features"]
    assert res_data["extracted_features"]["BloodPressure"]["value"] == 84.0
    assert "BMI" in res_data["extracted_features"]
    assert res_data["extracted_features"]["BMI"]["value"] == 31.8
    assert "Age" in res_data["extracted_features"]
    assert res_data["extracted_features"]["Age"]["value"] == 52.0


def test_scan_cardio_report_api(monkeypatch):
    """Integration Test: Uploading a cardiology report for Cardiovascular extraction."""
    from app.report.parser import ExtractionResult, document_parser

    def mock_parse_pdf(file_bytes, filename):
        lines = [
            "CARDIOLOGY CONSULTATION & STRESS TEST REPORT",
            "Age / Gender: 62 / Male",
            "Resting Blood Pressure: 145 mmHg",
            "Total Cholesterol: 235 mg/dL",
            "Fasting Blood Sugar: 105 mg/dL",
            "Max Heart Rate: 160 bpm",
            "Exercise Induced Angina: No",
            "ST Depression: 1.4 mm",
        ]
        return ExtractionResult(
            raw_text="\n".join(lines),
            lines=lines,
            file_type="pdf",
            filename=filename,
            page_count=1,
            ocr_engine_used="pypdf_text_engine",
        )

    monkeypatch.setattr(document_parser, "parse_pdf", mock_parse_pdf)

    files = {"file": ("cardio_report.pdf", b"%PDF-1.4 Mock Data", "application/pdf")}
    data = {"disease": "cardiovascular"}

    response = client.post("/api/v1/report/scan", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["success"] is True
    assert res_data["disease"] == "cardiovascular"
    assert "trestbps" in res_data["extracted_features"]
    assert res_data["extracted_features"]["trestbps"]["value"] == 145.0
    assert "chol" in res_data["extracted_features"]
    assert res_data["extracted_features"]["chol"]["value"] == 235.0
    assert "thalach" in res_data["extracted_features"]
    assert res_data["extracted_features"]["thalach"]["value"] == 160.0


def test_scan_invalid_file_extension_rejected():
    """Security Integration Test: Executables and unsupported formats are rejected with 400."""
    files = {"file": ("malicious_payload.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")}
    data = {"disease": "diabetes"}

    response = client.post("/api/v1/report/scan", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_end_to_end_scan_and_predict_flow(monkeypatch):
    """End-to-end Test: Upload Report -> Extract -> Complete Missing -> Run Existing Prediction."""
    from app.report.parser import ExtractionResult, document_parser

    def mock_parse_pdf(file_bytes, filename):
        lines = [
            "CLINICAL LABORATORY REPORT",
            "Patient Age: 45 Yrs",
            "Fasting Blood Sugar: 165 mg/dL",
            "Blood Pressure: 135/85 mmHg",
            "Body Mass Index: 29.5 kg/m2",
        ]
        return ExtractionResult(
            raw_text="\n".join(lines),
            lines=lines,
            file_type="pdf",
            filename=filename,
            page_count=1,
            ocr_engine_used="pypdf_text_engine",
        )

    monkeypatch.setattr(document_parser, "parse_pdf", mock_parse_pdf)

    # 1. Step 1: Scan report
    files = {"file": ("lab_report.pdf", b"%PDF-1.4 Mock", "application/pdf")}
    scan_resp = client.post("/api/v1/report/scan", files=files, data={"disease": "diabetes"})
    assert scan_resp.status_code == 200
    scan_data = scan_resp.json()

    extracted = {k: v["value"] for k, v in scan_data["extracted_features"].items()}
    missing = scan_data["missing_features"]

    # 2. Step 2: Verification / Complete missing fields
    feature_payload = dict(extracted)
    defaults_for_missing = {
        "Pregnancies": 2.0,
        "SkinThickness": 28.0,
        "Insulin": 80.0,
        "DiabetesPedigreeFunction": 0.52,
    }
    for m in missing:
        feature_payload[m] = defaults_for_missing.get(m, 0.0)

    # 3. Step 3: Send to existing prediction API
    predict_payload = {
        "features": feature_payload,
        "model_id": "diabetes_rf_opt_v1",
        "explain": True,
        "disease": "diabetes",
    }
    pred_resp = client.post("/api/v1/predict/diabetes", json=predict_payload)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()

    assert pred_data["predicted_class"] in [0, 1]
    assert pred_data["disease"] == "diabetes"
    assert "probability" in pred_data
    assert "explanation" in pred_data
    assert pred_data["audit"]["inference_time_ms"] > 0
