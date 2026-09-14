"""Disease-Specific Feature Mapping, Schema Alignment, and Validation Layer."""
import logging
from typing import Any

from app.report.matcher import MatchedParameter, ParameterMatcher
from app.report.normalizer import UnitNormalizer
from app.schemas.report import ExtractedFeatureItem, MissingFeatureItem, ReportScanResponse
from ml.cancer.pipeline import CANCER_FEATURE_NAMES
from ml.cardiovascular.pipeline import CARDIO_FEATURE_NAMES
from ml.preprocessing.pipeline import RAW_FEATURE_NAMES as DIABETES_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Feature descriptions and display names for missing items
FEATURE_METADATA: dict[str, dict[str, str]] = {
    # Diabetes
    "Pregnancies": {"display": "Pregnancies", "desc": "Number of times pregnant (0 for males/nulliparous)"},
    "Glucose": {"display": "Fasting Glucose", "desc": "Plasma glucose concentration (mg/dL)"},
    "BloodPressure": {"display": "Blood Pressure", "desc": "Diastolic blood pressure (mmHg)"},
    "SkinThickness": {"display": "Skin Thickness", "desc": "Triceps skin fold thickness (mm)"},
    "Insulin": {"display": "2-Hour Serum Insulin", "desc": "Serum insulin level (µIU/mL)"},
    "BMI": {"display": "Body Mass Index", "desc": "Weight in kg/(height in m)^2"},
    "DiabetesPedigreeFunction": {"display": "Diabetes Pedigree", "desc": "Genetic risk pedigree score"},
    "Age": {"display": "Age", "desc": "Patient age in years"},
    # Cardiovascular
    "age": {"display": "Age", "desc": "Age in years"},
    "sex": {"display": "Sex", "desc": "Biological sex (1 = male, 0 = female)"},
    "cp": {"display": "Chest Pain Type", "desc": "Chest pain type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)"},
    "trestbps": {"display": "Resting Blood Pressure", "desc": "Resting blood pressure (systolic) in mm Hg on admission"},
    "chol": {"display": "Serum Cholesterol", "desc": "Serum cholesterol in mg/dL"},
    "fbs": {"display": "Fasting Blood Sugar > 120", "desc": "Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)"},
    "restecg": {"display": "Resting ECG", "desc": "Resting electrocardiographic results (0: Normal, 1: ST-T abnormality, 2: LV hypertrophy)"},
    "thalach": {"display": "Maximum Heart Rate", "desc": "Maximum heart rate achieved during exercise/stress test"},
    "exang": {"display": "Exercise Induced Angina", "desc": "Exercise induced angina (1 = yes, 0 = no)"},
    "oldpeak": {"display": "ST Depression (Oldpeak)", "desc": "ST depression induced by exercise relative to rest"},
    "slope": {"display": "ST Slope", "desc": "Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)"},
    "ca": {"display": "Major Vessels (Fluoroscopy)", "desc": "Number of major vessels (0-3) colored by fluoroscopy"},
    "thal": {"display": "Thalassemia", "desc": "Thalassemia category (1: Normal, 2: Fixed defect, 3: Reversible defect)"},
}


class DiseaseFeatureMapper:
    """Maps extracted document parameters to authoritative disease model feature schemas."""

    @classmethod
    def map_extracted_data(
        cls,
        disease: str,
        matches: list[MatchedParameter],
        raw_text: str,
        filename: str,
        file_type: str,
        extraction_method: str = "pdf_text",
        text_detected: bool = True,
        extracted_text_length: int = 0,
    ) -> ReportScanResponse:
        """Processes matches, executes normalizations, and creates an aligned ReportScanResponse."""
        disease_clean = disease.lower().strip()
        report_category = ParameterMatcher.classify_report_type(raw_text)

        if disease_clean == "diabetes":
            return cls._map_diabetes(
                matches, raw_text, filename, file_type, report_category,
                extraction_method, text_detected, extracted_text_length
            )
        elif disease_clean == "cardiovascular":
            return cls._map_cardiovascular(
                matches, raw_text, filename, file_type, report_category,
                extraction_method, text_detected, extracted_text_length
            )
        elif disease_clean == "cancer":
            return cls._map_cancer(
                matches, raw_text, filename, file_type, report_category,
                extraction_method, text_detected, extracted_text_length
            )
        else:
            return cls._map_diabetes(
                matches, raw_text, filename, file_type, report_category,
                extraction_method, text_detected, extracted_text_length
            )

    @classmethod
    def _map_diabetes(
        cls,
        matches: list[MatchedParameter],
        raw_text: str,
        filename: str,
        file_type: str,
        report_category: str,
        extraction_method: str,
        text_detected: bool,
        extracted_text_length: int,
    ) -> ReportScanResponse:
        required_features = list(DIABETES_FEATURE_NAMES)
        extracted: dict[str, ExtractedFeatureItem] = {}
        warnings: list[str] = []

        match_map = {m.canonical_name: m for m in matches}

        # 1. Glucose
        if "Glucose" in match_map:
            m = match_map["Glucose"]
            norm = UnitNormalizer.normalize_glucose(m.raw_value, m.raw_unit)
            if norm:
                extracted["Glucose"] = ExtractedFeatureItem(
                    name="Glucose",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    conversion_applied=norm.conversion_applied,
                    requires_verification=norm.value < 40 or norm.value > 400,
                )

        # 2. BloodPressure (In Pima Diabetes dataset, this is Diastolic Blood Pressure)
        if "diastolic_bp" in match_map:
            m = match_map["diastolic_bp"]
            norm = UnitNormalizer.normalize_generic(m.raw_value, "mmHg", m.raw_unit)
            if norm:
                extracted["BloodPressure"] = ExtractedFeatureItem(
                    name="BloodPressure",
                    value=norm.value,
                    unit=norm.unit or "mmHg",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    requires_verification=norm.value < 40 or norm.value > 160,
                )
        elif "BloodPressure" in match_map:
            m = match_map["BloodPressure"]
            sys_norm, dia_norm = UnitNormalizer.normalize_blood_pressure(m.raw_value)
            chosen = dia_norm if dia_norm else sys_norm
            if chosen:
                extracted["BloodPressure"] = ExtractedFeatureItem(
                    name="BloodPressure",
                    value=chosen.value,
                    unit=chosen.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    conversion_applied=chosen.conversion_applied,
                    requires_verification=chosen.value < 40 or chosen.value > 160,
                )


        # 3. BMI
        if "BMI" in match_map:
            m = match_map["BMI"]
            norm = UnitNormalizer.normalize_generic(m.raw_value, "kg/m²", m.raw_unit)
            if norm:
                extracted["BMI"] = ExtractedFeatureItem(
                    name="BMI",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    requires_verification=norm.value < 12.0 or norm.value > 65.0,
                )

        # 4. Insulin
        if "Insulin" in match_map:
            m = match_map["Insulin"]
            norm = UnitNormalizer.normalize_insulin(m.raw_value, m.raw_unit)
            if norm:
                extracted["Insulin"] = ExtractedFeatureItem(
                    name="Insulin",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    conversion_applied=norm.conversion_applied,
                )

        # 5. Age
        if "Age" in match_map or "age" in match_map:
            m = match_map.get("Age") or match_map.get("age")
            norm = UnitNormalizer.normalize_generic(m.raw_value, "years")
            if norm:
                extracted["Age"] = ExtractedFeatureItem(
                    name="Age",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 6. Pregnancies
        if "Pregnancies" in match_map:
            m = match_map["Pregnancies"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["Pregnancies"] = ExtractedFeatureItem(
                    name="Pregnancies",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 7. SkinThickness
        if "SkinThickness" in match_map:
            m = match_map["SkinThickness"]
            norm = UnitNormalizer.normalize_generic(m.raw_value, "mm", m.raw_unit)
            if norm:
                extracted["SkinThickness"] = ExtractedFeatureItem(
                    name="SkinThickness",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 8. DiabetesPedigreeFunction
        if "DiabetesPedigreeFunction" in match_map:
            m = match_map["DiabetesPedigreeFunction"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["DiabetesPedigreeFunction"] = ExtractedFeatureItem(
                    name="DiabetesPedigreeFunction",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        missing = [f for f in required_features if f not in extracted]
        missing_details = [
            MissingFeatureItem(
                name=f,
                display_name=FEATURE_METADATA.get(f, {}).get("display", f),
                description=FEATURE_METADATA.get(f, {}).get("desc", ""),
                reason="Not found in document; manual verification required.",
            )
            for f in missing
        ]

        det_count = len(extracted)
        req_count = len(required_features)

        if det_count == req_count:
            msg = f"Report successfully scanned. All {req_count} diabetes clinical parameters detected with high confidence."
        elif det_count > 0:
            msg = f"Report scanned successfully. Detected {det_count} of {req_count} parameters. Please complete the missing fields below."
        else:
            msg = "Document scanned, but no compatible diabetes parameters were detected. Please enter values manually."

        preview = (raw_text[:250] + "...") if len(raw_text) > 250 else raw_text

        return ReportScanResponse(
            success=True,
            filename=filename,
            file_type=file_type,
            disease="diabetes",
            report_category=report_category,
            compatible=True,
            extracted_features=extracted,
            missing_features=missing,
            missing_features_detail=missing_details,
            total_features_required=req_count,
            total_features_detected=det_count,
            raw_text_summary=preview,
            extraction_method=extraction_method,
            text_detected=text_detected,
            extracted_text_length=extracted_text_length,
            message=msg,
            validation_warnings=warnings,
        )

    @classmethod
    def _map_cardiovascular(
        cls,
        matches: list[MatchedParameter],
        raw_text: str,
        filename: str,
        file_type: str,
        report_category: str,
        extraction_method: str,
        text_detected: bool,
        extracted_text_length: int,
    ) -> ReportScanResponse:
        required_features = list(CARDIO_FEATURE_NAMES)
        extracted: dict[str, ExtractedFeatureItem] = {}
        warnings: list[str] = []

        match_map = {m.canonical_name: m for m in matches}

        # 1. age
        if "age" in match_map or "Age" in match_map:
            m = match_map.get("age") or match_map.get("Age")
            norm = UnitNormalizer.normalize_generic(m.raw_value, "years")
            if norm:
                extracted["age"] = ExtractedFeatureItem(
                    name="age",
                    value=norm.value,
                    unit="years",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 2. sex
        if "sex" in match_map:
            m = match_map["sex"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["sex"] = ExtractedFeatureItem(
                    name="sex",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 3. trestbps (Resting Blood Pressure - Systolic)
        if "systolic_bp" in match_map:
            m = match_map["systolic_bp"]
            norm = UnitNormalizer.normalize_generic(m.raw_value, "mmHg", m.raw_unit)
            if norm:
                extracted["trestbps"] = ExtractedFeatureItem(
                    name="trestbps",
                    value=norm.value,
                    unit="mmHg",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit or "mmHg",
                    source_snippet=m.source_snippet,
                )
        elif "trestbps" in match_map or "BloodPressure" in match_map:
            m = match_map.get("trestbps") or match_map.get("BloodPressure")
            sys_norm, _ = UnitNormalizer.normalize_blood_pressure(m.raw_value)
            if sys_norm:
                extracted["trestbps"] = ExtractedFeatureItem(
                    name="trestbps",
                    value=sys_norm.value,
                    unit="mmHg",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit="mmHg",
                    source_snippet=m.source_snippet,
                )


        # 4. chol (Total Cholesterol)
        if "chol" in match_map:
            m = match_map["chol"]
            norm = UnitNormalizer.normalize_cholesterol(m.raw_value, m.raw_unit)
            if norm:
                extracted["chol"] = ExtractedFeatureItem(
                    name="chol",
                    value=norm.value,
                    unit=norm.unit,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                    conversion_applied=norm.conversion_applied,
                )

        # 5. fbs (Fasting Blood Sugar > 120 mg/dL)
        if "fbs" in match_map:
            m = match_map["fbs"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["fbs"] = ExtractedFeatureItem(
                    name="fbs",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )
        elif "Glucose" in match_map:
            m = match_map["Glucose"]
            g_norm = UnitNormalizer.normalize_glucose(m.raw_value, m.raw_unit)
            if g_norm:
                is_fbs_high = 1.0 if g_norm.value > 120.0 else 0.0
                extracted["fbs"] = ExtractedFeatureItem(
                    name="fbs",
                    value=is_fbs_high,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=f"Computed from {m.raw_name} ({g_norm.value} mg/dL)",
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 6. thalach (Max Heart Rate)
        if "thalach" in match_map:
            m = match_map["thalach"]
            norm = UnitNormalizer.normalize_generic(m.raw_value, "bpm", m.raw_unit)
            if norm:
                extracted["thalach"] = ExtractedFeatureItem(
                    name="thalach",
                    value=norm.value,
                    unit="bpm",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 7. exang (Exercise Induced Angina)
        if "exang" in match_map:
            m = match_map["exang"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["exang"] = ExtractedFeatureItem(
                    name="exang",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 8. oldpeak (ST depression)
        if "oldpeak" in match_map:
            m = match_map["oldpeak"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["oldpeak"] = ExtractedFeatureItem(
                    name="oldpeak",
                    value=norm.value,
                    unit="mm",
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=m.raw_unit,
                    source_snippet=m.source_snippet,
                )

        # 9. cp (Chest Pain Type)
        if "cp" in match_map:
            m = match_map["cp"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["cp"] = ExtractedFeatureItem(
                    name="cp",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 10. restecg
        if "restecg" in match_map:
            m = match_map["restecg"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["restecg"] = ExtractedFeatureItem(
                    name="restecg",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 11. slope
        if "slope" in match_map:
            m = match_map["slope"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["slope"] = ExtractedFeatureItem(
                    name="slope",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 12. ca
        if "ca" in match_map:
            m = match_map["ca"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["ca"] = ExtractedFeatureItem(
                    name="ca",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        # 13. thal
        if "thal" in match_map:
            m = match_map["thal"]
            norm = UnitNormalizer.normalize_generic(m.raw_value)
            if norm:
                extracted["thal"] = ExtractedFeatureItem(
                    name="thal",
                    value=norm.value,
                    unit=None,
                    confidence=m.confidence,
                    confidence_level=m.confidence_level,
                    original_name=m.raw_name,
                    original_value=m.raw_value,
                    original_unit=None,
                    source_snippet=m.source_snippet,
                )

        missing = [f for f in required_features if f not in extracted]
        missing_details = [
            MissingFeatureItem(
                name=f,
                display_name=FEATURE_METADATA.get(f, {}).get("display", f),
                description=FEATURE_METADATA.get(f, {}).get("desc", ""),
                reason="Clinical parameter requiring manual verification.",
            )
            for f in missing
        ]

        det_count = len(extracted)
        req_count = len(required_features)

        msg = f"Cardiovascular report processed. Detected {det_count} of {req_count} parameters. Clinical categoricals require manual verification."
        preview = (raw_text[:250] + "...") if len(raw_text) > 250 else raw_text

        return ReportScanResponse(
            success=True,
            filename=filename,
            file_type=file_type,
            disease="cardiovascular",
            report_category=report_category,
            compatible=True,
            extracted_features=extracted,
            missing_features=missing,
            missing_features_detail=missing_details,
            total_features_required=req_count,
            total_features_detected=det_count,
            raw_text_summary=preview,
            extraction_method=extraction_method,
            text_detected=text_detected,
            extracted_text_length=extracted_text_length,
            message=msg,
            validation_warnings=warnings,
        )

    @classmethod
    def _map_cancer(
        cls,
        matches: list[MatchedParameter],
        raw_text: str,
        filename: str,
        file_type: str,
        report_category: str,
        extraction_method: str,
        text_detected: bool,
        extracted_text_length: int,
    ) -> ReportScanResponse:
        required_features = list(CANCER_FEATURE_NAMES)
        extracted: dict[str, ExtractedFeatureItem] = {}
        warnings: list[str] = []

        match_map = {m.canonical_name: m for m in matches}

        for fn in required_features:
            if fn in match_map:
                m = match_map[fn]
                norm = UnitNormalizer.normalize_generic(m.raw_value)
                if norm:
                    extracted[fn] = ExtractedFeatureItem(
                        name=fn,
                        value=norm.value,
                        unit=None,
                        confidence=m.confidence,
                        confidence_level=m.confidence_level,
                        original_name=m.raw_name,
                        original_value=m.raw_value,
                        original_unit=None,
                        source_snippet=m.source_snippet,
                    )

        det_count = len(extracted)
        req_count = len(required_features)

        # Check compatibility: If report contains general blood test markers but lacks cancer morphometry
        lower = raw_text.lower()
        has_generic_blood = any(kw in lower for kw in ["cbc", "glucose", "hemoglobin", "cholesterol", "platelets", "wbc", "rbc", "fasting blood glucose"])

        if det_count < 5 and (has_generic_blood or report_category == "laboratory_report"):
            # Mandatory rejection/fallback for incompatible general medical reports
            compatible = False
            msg = (
                "Report scanned successfully, but the uploaded document does not contain the 30 cell nucleus "
                "morphometric measurements (e.g., radius, texture, perimeter, concavity) required by the Breast Cancer model. "
                "Please enter the values manually or upload a compatible FNA cytology / pathology analysis report."
            )
            warnings.append("Document identified as general laboratory report rather than specialized tumor pathology.")
        elif det_count == req_count:
            compatible = True
            msg = f"Pathology report scanned successfully. All {req_count} cell nucleus morphometric features detected."
        elif det_count > 0:
            compatible = True
            msg = f"Pathology report scanned. Detected {det_count} of {req_count} morphology features. Please verify and complete the missing measurements."
        else:
            compatible = False
            msg = (
                "No compatible breast tumor morphology features were detected in the uploaded document. "
                "Please provide the required values manually or upload a compatible pathology report."
            )

        missing = [f for f in required_features if f not in extracted]
        missing_details = [
            MissingFeatureItem(
                name=f,
                display_name=f.replace("_", " ").title(),
                description=f"Morphometric measurement: {f}",
                reason="Required tumor morphology measurement not present in document.",
            )
            for f in missing
        ]

        preview = (raw_text[:250] + "...") if len(raw_text) > 250 else raw_text

        return ReportScanResponse(
            success=True,
            filename=filename,
            file_type=file_type,
            disease="cancer",
            report_category="pathology_report" if det_count >= 5 else report_category,
            compatible=compatible,
            extracted_features=extracted,
            missing_features=missing,
            missing_features_detail=missing_details,
            total_features_required=req_count,
            total_features_detected=det_count,
            raw_text_summary=preview,
            extraction_method=extraction_method,
            text_detected=text_detected,
            extracted_text_length=extracted_text_length,
            message=msg,
            validation_warnings=warnings,
        )
