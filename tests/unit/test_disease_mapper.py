"""Unit tests for DiseaseFeatureMapper."""
import pytest
from app.report.disease_mapper import DiseaseFeatureMapper
from app.report.matcher import MatchedParameter


def test_map_diabetes_partial_features():
    matches = [
        MatchedParameter(
            canonical_name="Glucose",
            raw_name="Fasting Blood Glucose",
            raw_value="142",
            raw_unit="mg/dL",
            reference_range="70-99",
            confidence=0.96,
            confidence_level="high",
            source_snippet="Fasting Blood Glucose: 142 mg/dL",
        ),
        MatchedParameter(
            canonical_name="BloodPressure",
            raw_name="Blood Pressure",
            raw_value="130/80",
            raw_unit="mmHg",
            reference_range=None,
            confidence=0.95,
            confidence_level="high",
            source_snippet="Blood Pressure: 130/80 mmHg",
        ),
        MatchedParameter(
            canonical_name="BMI",
            raw_name="BMI",
            raw_value="28.4",
            raw_unit="kg/m2",
            reference_range=None,
            confidence=0.95,
            confidence_level="high",
            source_snippet="BMI: 28.4",
        ),
        MatchedParameter(
            canonical_name="Age",
            raw_name="Age",
            raw_value="46",
            raw_unit="years",
            reference_range=None,
            confidence=0.95,
            confidence_level="high",
            source_snippet="Age: 46",
        ),
    ]

    res = DiseaseFeatureMapper.map_extracted_data(
        disease="diabetes",
        matches=matches,
        raw_text="Patient Report: Glucose 142, BP 130/80, BMI 28.4, Age 46",
        filename="report.pdf",
        file_type="pdf",
    )

    assert res.success is True
    assert res.disease == "diabetes"
    assert res.total_features_detected == 4
    assert res.total_features_required == 8
    assert "Glucose" in res.extracted_features
    assert res.extracted_features["Glucose"].value == 142.0
    assert "BloodPressure" in res.extracted_features
    assert res.extracted_features["BloodPressure"].value == 80.0
    assert "BMI" in res.extracted_features
    assert "Age" in res.extracted_features
    # Missing features
    assert "Pregnancies" in res.missing_features
    assert "Insulin" in res.missing_features
    assert "SkinThickness" in res.missing_features
    assert "DiabetesPedigreeFunction" in res.missing_features


def test_map_cancer_incompatible_blood_report():
    # A generic blood report should be marked as incompatible for Breast Cancer
    matches = [
        MatchedParameter(
            canonical_name="Glucose",
            raw_name="Glucose",
            raw_value="95",
            raw_unit="mg/dL",
            reference_range=None,
            confidence=0.9,
            confidence_level="high",
            source_snippet="Glucose: 95 mg/dL",
        )
    ]
    raw_text = "Routine Blood Test & Complete Blood Count (CBC) Report. Hemoglobin: 13.5 g/dL, WBC: 6500 /uL, Glucose: 95 mg/dL."

    res = DiseaseFeatureMapper.map_extracted_data(
        disease="cancer",
        matches=matches,
        raw_text=raw_text,
        filename="cbc_report.pdf",
        file_type="pdf",
    )

    assert res.success is True
    assert res.compatible is False
    assert res.total_features_detected == 0
    assert len(res.missing_features) == 30
    assert "morphometric" in res.message.lower() or "morphology" in res.message.lower()
