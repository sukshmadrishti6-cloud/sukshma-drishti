"""Unit tests for UnitNormalizer in report extraction pipeline."""
import pytest
from app.report.normalizer import UnitNormalizer


def test_glucose_normalization_mmol():
    # 7.0 mmol/L should convert to ~126.1 mg/dL
    norm = UnitNormalizer.normalize_glucose(7.0, "mmol/L")
    assert norm is not None
    assert norm.unit == "mg/dL"
    assert norm.conversion_applied is True
    assert pytest.approx(norm.value, 0.5) == 126.1
    assert norm.original_value == 7.0


def test_glucose_normalization_mg_dl():
    # 142 mg/dL should remain 142.0 mg/dL
    norm = UnitNormalizer.normalize_glucose("142", "mg/dL")
    assert norm is not None
    assert norm.value == 142.0
    assert norm.unit == "mg/dL"
    assert norm.conversion_applied is False


def test_cholesterol_normalization_mmol():
    # 5.5 mmol/L should convert to ~212.7 mg/dL
    norm = UnitNormalizer.normalize_cholesterol(5.5, "mmol/L")
    assert norm is not None
    assert norm.unit == "mg/dL"
    assert norm.conversion_applied is True
    assert pytest.approx(norm.value, 0.5) == 212.7


def test_cholesterol_normalization_mg_dl():
    norm = UnitNormalizer.normalize_cholesterol("230", "mg/dL")
    assert norm is not None
    assert norm.value == 230.0
    assert norm.unit == "mg/dL"
    assert norm.conversion_applied is False


def test_blood_pressure_compound_extraction():
    sys_norm, dia_norm = UnitNormalizer.normalize_blood_pressure("135/85 mmHg")
    assert sys_norm is not None
    assert sys_norm.value == 135.0
    assert sys_norm.unit == "mmHg"
    assert dia_norm is not None
    assert dia_norm.value == 85.0
    assert dia_norm.unit == "mmHg"


def test_blood_pressure_single_value():
    sys_norm, dia_norm = UnitNormalizer.normalize_blood_pressure(120)
    assert sys_norm is not None
    assert sys_norm.value == 120.0
    assert dia_norm is None


def test_insulin_normalization():
    # 85 µIU/mL
    norm = UnitNormalizer.normalize_insulin("85.5", "µIU/mL")
    assert norm is not None
    assert norm.value == 85.5
    assert norm.unit == "µIU/mL"
    assert norm.conversion_applied is False

    # pmol/L conversion
    norm_pmol = UnitNormalizer.normalize_insulin("100.0", "pmol/L")
    assert norm_pmol is not None
    assert pytest.approx(norm_pmol.value, 0.2) == 14.4
    assert norm_pmol.conversion_applied is True
