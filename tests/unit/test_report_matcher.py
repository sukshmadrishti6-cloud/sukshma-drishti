"""Unit tests for ParameterMatcher clinical extraction."""
import pytest
from app.report.matcher import ParameterMatcher


def test_matcher_diabetes_sample_lines():
    lines = [
        "METABOLIC PANEL & BIOCHEMISTRY REPORT",
        "Patient: Jane Doe | Age: 48 Yrs | Gender: Female",
        "Fasting Blood Sugar: 148 mg/dL (Ref: 70 - 99)",
        "Blood Pressure: 130/80 mmHg",
        "Body Mass Index: 31.4 kg/m2",
        "Fasting Serum Insulin: 95.0 uIU/mL",
        "Skinfold Thickness: 32 mm",
        "Pregnancies: 3",
        "Diabetes Pedigree Function: 0.65",
    ]

    matches = ParameterMatcher.extract_from_lines(lines)
    match_dict = {m.canonical_name: m for m in matches}

    assert "Glucose" in match_dict
    assert match_dict["Glucose"].raw_value == "148"
    assert match_dict["Glucose"].confidence_level == "high"

    assert "BloodPressure" in match_dict
    assert match_dict["BloodPressure"].raw_value == "130/80"

    assert "BMI" in match_dict
    assert match_dict["BMI"].raw_value == "31.4"

    assert "Insulin" in match_dict
    assert match_dict["Insulin"].raw_value == "95.0"

    assert "Age" in match_dict
    assert match_dict["Age"].raw_value == "48"

    assert "Pregnancies" in match_dict
    assert match_dict["Pregnancies"].raw_value == "3"


def test_matcher_hba1c_not_matched_as_glucose():
    # HbA1c should NEVER be falsely extracted as fasting glucose
    lines = [
        "Glycated Hemoglobin (HbA1c): 7.2 %",
        "HbA1c: 6.8%",
    ]
    matches = ParameterMatcher.extract_from_lines(lines)
    glucose_matches = [m for m in matches if m.canonical_name == "Glucose"]
    assert len(glucose_matches) == 0


def test_matcher_cardio_sample_lines():
    lines = [
        "CARDIOLOGY STRESS TEST & CLINICAL EVALUATION",
        "Age / Sex: 58 / Male",
        "Resting Blood Pressure: 140 mmHg",
        "Serum Total Cholesterol: 245 mg/dL",
        "Fasting Blood Sugar: 130 mg/dL",
        "Maximum Heart Rate Achieved: 155 bpm",
        "ST Depression (Oldpeak): 1.8 mm",
        "Exercise Induced Angina: No",
    ]

    matches = ParameterMatcher.extract_from_lines(lines)
    match_dict = {m.canonical_name: m for m in matches}

    assert "age" in match_dict
    assert match_dict["age"].raw_value == "58"

    assert "sex" in match_dict
    assert match_dict["sex"].raw_value == "1"

    assert "trestbps" in match_dict
    assert match_dict["trestbps"].raw_value == "140"

    assert "chol" in match_dict
    assert match_dict["chol"].raw_value == "245"

    assert "thalach" in match_dict
    assert match_dict["thalach"].raw_value == "155"

    assert "fbs" in match_dict
    assert match_dict["fbs"].raw_value == "1"

    assert "exang" in match_dict
    assert match_dict["exang"].raw_value == "0"

    assert "oldpeak" in match_dict
    assert match_dict["oldpeak"].raw_value == "1.8"


def test_classify_report_type():
    lab_text = "Clinical Biochemistry Laboratory Report - Fasting Blood Glucose, Serum Creatinine, Complete Blood Count"
    assert ParameterMatcher.classify_report_type(lab_text) == "laboratory_report"

    cardio_text = "Department of Cardiology - Treadmill Stress Test (TMT) and Electrocardiogram (ECG) Report"
    assert ParameterMatcher.classify_report_type(cardio_text) == "cardiology_report"

    fna_text = "Pathology Report: Fine Needle Aspiration (FNA) Biopsy Cytology - Tumor Cell Nucleus Morphology"
    assert ParameterMatcher.classify_report_type(fna_text) == "pathology_report"
