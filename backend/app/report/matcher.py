"""Parameter Identification and Medical Table/Text Parsing Engine."""
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.report.normalizer import UnitNormalizer

logger = logging.getLogger(__name__)


@dataclass
class MatchedParameter:
    """Represents a raw identified parameter match in document text."""
    canonical_name: str
    raw_name: str
    raw_value: str
    raw_unit: str | None
    reference_range: str | None
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    source_snippet: str
    line_number: int | None = None


class ParameterMatcher:
    """Extracts clinical parameters from medical report text lines using multi-line table & layout parsing."""

    # Explicit exclusions (e.g. HbA1c should never match as standard glucose)
    EXCLUSIONS: dict[str, list[str]] = {
        "glucose": ["hba1c", "a1c", "glycated", "glycohemoglobin", "estimated average", "eag", "urine", "postprandial", "ppbs"],
        "blood_pressure": ["pulse pressure", "mean arterial pressure", "map"],
        "bloodpressure_compound": ["systolic", "diastolic", "sbp", "dbp", "pulse pressure"],
        "cholesterol": ["hdl", "ldl", "vldl", "triglycerides", "ratio", "non-hdl"],
        "age": ["report date", "sample date", "page", "date of birth", "dob"],
        "pregnancies": ["testing note", "disclaimer", "model", "parameter", "feature", "sample"],
    }


    # Document type indicator keywords
    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "laboratory_report": [
            "biochemistry", "hematology", "complete blood count", "cbc", "lipid profile",
            "metabolic panel", "serum", "plasma", "blood test", "laboratory report", "pathology lab"
        ],
        "cardiology_report": [
            "electrocardiogram", "ecg", "ekg", "echocardiogram", "cardiology", "treadmill test",
            "tmt", "stress test", "coronary", "cardiac", "angina", "holter", "catheterization"
        ],
        "pathology_report": [
            "fine needle aspiration", "fna", "biopsy", "histopathology", "cytology",
            "tumor morphology", "cell nucleus", "malignancy", "carcinoma", "neoplasm"
        ],
    }

    # Diabetes Model Feature Aliases
    DIABETES_ALIASES: dict[str, list[str]] = {
        "Glucose": [
            "fasting blood glucose", "fasting plasma glucose", "fasting blood sugar",
            "fasting glucose", "blood glucose fasting", "glucose fasting",
            "plasma glucose", "serum glucose", "blood glucose", "blood sugar",
            "fbs", "glucose"
        ],
        "BloodPressure_Diastolic": [
            "blood pressure - diastolic", "blood pressure diastolic",
            "diastolic blood pressure", "diastolic bp", "dbp", "diastolic"
        ],
        "BloodPressure_Systolic": [
            "blood pressure - systolic", "blood pressure systolic",
            "systolic blood pressure", "systolic bp", "sbp", "systolic"
        ],
        "BloodPressure_Compound": [
            "resting blood pressure", "resting bp", "blood pressure", "bp"
        ],
        "SkinThickness": [
            "triceps skinfold thickness", "triceps skin fold thickness",
            "skinfold thickness", "skin fold thickness", "triceps skinfold",
            "skin thickness", "tsft"
        ],
        "Insulin": [
            "2-hour serum insulin", "two-hour serum insulin", "serum fasting insulin",
            "fasting serum insulin", "serum insulin", "fasting insulin",
            "plasma insulin", "insulin"
        ],
        "BMI": [
            "body mass index (bmi)", "body mass index", "body mass idx",
            "quetelet index", "bmi"
        ],
        "DiabetesPedigreeFunction": [
            "diabetes pedigree function", "diabetes pedigree score",
            "diabetes pedigree", "pedigree function", "dpf"
        ],
        "Pregnancies": [
            "number of pregnancies", "pregnancy history", "pregnancy count",
            "pregnancies count", "pregnancies", "gravida", "parity"
        ],
        "Age": [
            "patient age", "patient's age", "age of patient", "age"
        ],
    }

    # Cardiovascular Model Feature Aliases
    CARDIO_ALIASES: dict[str, list[str]] = {
        "chol": ["total cholesterol", "serum cholesterol", "cholesterol, total", "cholesterol total", "chol", "serum total cholesterol"],
        "thalach": ["maximum heart rate", "max heart rate", "peak heart rate", "thalach", "max hr", "peak hr", "achieved heart rate", "heart rate", "pulse rate", "pulse"],
        "oldpeak": ["st depression", "st segment depression", "exercise st depression", "oldpeak"],
        "ca": ["number of major vessels", "major vessels", "colored vessels", "ca"],
        "cp": ["chest pain type", "chest pain", "cp"],
        "restecg": ["resting ecg", "resting electrocardiographic", "restecg", "ecg"],
        "exang": ["exercise induced angina", "angina induced by exercise", "exang", "exercise angina"],
        "slope": ["st slope", "slope of peak exercise st", "slope"],
        "thal": ["thalassemia", "thal"],
    }

    # Known Valid Clinical Units
    KNOWN_UNITS: set[str] = {
        "mg/dl", "mmol/l", "mmhg", "kg/m2", "kg/m^2", "kg/m²", "uiu/ml", "µiu/ml", "miu/l",
        "pmol/l", "mm", "cm", "years", "yrs", "%", "bpm", "beats/min", "iu/ml"
    }

    @classmethod
    def classify_report_type(cls, text: str) -> str:
        """Classifies document category based on medical keywords."""
        lower = text.lower()
        scores: dict[str, int] = {}
        for cat, keywords in cls.CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in lower)
            scores[cat] = count

        top_cat = max(scores, key=scores.get) if scores else "unknown"
        if scores.get(top_cat, 0) >= 1:
            return top_cat
        return "general_medical_report" if len(text.strip()) > 50 else "unknown"

    @classmethod
    def extract_from_lines(cls, lines: list[str]) -> list[MatchedParameter]:
        """Scans document text lines and extracts clinical measurements using multi-line & layout strategies."""
        cleaned_lines = [UnitNormalizer.clean_text(l).strip() for l in lines if l and l.strip()]
        matches: list[MatchedParameter] = []

        # 1. Extract diabetes & general parameters (multi-line lookahead + same line)
        cls._extract_alias_matches(cleaned_lines, cls.DIABETES_ALIASES, matches)

        # 2. Extract cardio parameters
        cls._extract_alias_matches(cleaned_lines, cls.CARDIO_ALIASES, matches)

        # 3. Extract sex / gender
        cls._extract_sex(cleaned_lines, matches)

        # 4. Extract cancer 30 morphology features
        cls._extract_cancer_morphology(cleaned_lines, matches)

        return cls._deduplicate_matches(matches)

    @classmethod
    def _is_excluded(cls, target_key: str, context: str) -> bool:
        lower = context.lower()
        for key, excl_list in cls.EXCLUSIONS.items():
            if key in target_key.lower():
                for excl in excl_list:
                    if excl in lower:
                        return True
        return False

    @classmethod
    def _is_unit(cls, text: str) -> bool:
        clean = text.strip().lower()
        return clean in cls.KNOWN_UNITS

    @classmethod
    def _validate_number_for_target(cls, num_str: str, target: str) -> bool:
        """Enforces realistic biological bounds for target parameters."""
        try:
            val = float(num_str)
        except ValueError:
            return False

        t = target.lower()
        if "glucose" in t:
            return 10.0 <= val <= 900.0
        elif "systolic" in t:
            return 50.0 <= val <= 280.0
        elif "diastolic" in t:
            return 30.0 <= val <= 180.0
        elif "bloodpressure" in t or "trestbps" in t:
            return 30.0 <= val <= 280.0
        elif "skin" in t:
            return 1.0 <= val <= 100.0
        elif "insulin" in t:
            return 0.1 <= val <= 1200.0
        elif "bmi" in t:
            return 10.0 <= val <= 90.0
        elif "pedigree" in t or "dpf" in t:
            return 0.01 <= val <= 5.0
        elif "pregnanc" in t:
            return 0.0 <= val <= 25.0
        elif "age" in t:
            return 1.0 <= val <= 125.0
        elif "chol" in t:
            return 50.0 <= val <= 600.0
        elif "thalach" in t:
            return 40.0 <= val <= 240.0
        elif "oldpeak" in t:
            return 0.0 <= val <= 10.0
        elif "ca" in t:
            return 0.0 <= val <= 4.0
        elif "cp" in t or "restecg" in t or "slope" in t or "thal" in t:
            return 0.0 <= val <= 4.0
        return True

    @classmethod
    def _extract_alias_matches(
        cls,
        lines: list[str],
        alias_dict: dict[str, list[str]],
        matches: list[MatchedParameter],
    ):
        num_lines = len(lines)
        all_targets = []
        for target, aliases in alias_dict.items():
            for alias in aliases:
                all_targets.append((target, alias))
        all_targets.sort(key=lambda x: len(x[1]), reverse=True)

        matched_targets: set[str] = set()

        for i in range(num_lines):
            line = lines[i]
            line_lower = line.lower()

            # Ignore disclaimer / testing note lines
            if any(kw in line_lower for kw in ["testing note:", "clinical disclaimer:", "synthetic test document", "not a real patient"]):
                continue

            for target, alias in all_targets:
                if target in matched_targets:
                    continue

                pattern = rf"\b{re.escape(alias)}\b"
                alias_match = re.search(pattern, line_lower)
                if not alias_match:
                    continue

                if cls._is_excluded(target, line):
                    continue

                # Strategy 1: Check same line
                val_data = cls._find_value_in_same_line(line, alias_match.end(), target)
                if val_data:
                    val_str, unit_str, ref_str, conf = val_data
                    cls._record_match(target, line, val_str, unit_str, ref_str, conf, line, i + 1, matches, matched_targets)
                    break

                # Strategy 2: Lookahead next 1-4 lines (for table layouts)
                lookahead_data = cls._find_value_in_lookahead(lines, i + 1, min(i + 5, num_lines), target)
                if lookahead_data:
                    val_str, unit_str, ref_str, conf, snippet = lookahead_data
                    cls._record_match(target, line, val_str, unit_str, ref_str, conf, snippet, i + 1, matches, matched_targets)
                    break

    @classmethod
    def _find_value_in_same_line(
        cls, line: str, alias_end_idx: int, target: str
    ) -> tuple[str, str | None, str | None, float] | None:
        after_alias = line[alias_end_idx:].strip(" :=-|\t")
        t_lower = target.lower()
        # Check compound BP e.g. "138/86 mmHg" (must use slash to avoid matching reference ranges like 90 - 120)
        if "compound" in t_lower:
            bp_m = re.search(r"(\d{2,3})\s*\/\s*(\d{2,3})\s*(mm\s*hg)?", after_alias, re.IGNORECASE)
            if bp_m:
                return (f"{bp_m.group(1)}/{bp_m.group(2)}", bp_m.group(3) or "mmHg", None, 0.96)


        # Check yes/no categoricals for exang
        if "exang" in t_lower:
            yes_no_m = re.search(r"\b(yes|no|positive|negative|present|absent|true|false)\b", after_alias, re.IGNORECASE)
            if yes_no_m:
                word = yes_no_m.group(1).lower()
                is_pos = "1" if word in ["yes", "positive", "present", "true"] else "0"
                return (is_pos, None, None, 0.95)

        # Standard number starting after alias
        num_m = re.search(r"^(\d+(?:\.\d+)?)\s*([a-zA-Zµ/%^2]+)?(?:\s*[\(\[]?([0-9\.\-\s]+)[\)\]]?)?", after_alias)
        if num_m:
            val_str = num_m.group(1)
            unit_str = num_m.group(2)
            ref_str = num_m.group(3)
            if cls._validate_number_for_target(val_str, target):
                return (val_str, unit_str if (unit_str and cls._is_unit(unit_str)) else None, ref_str, 0.95 if unit_str else 0.88)

        # Number appearing anywhere after alias on same line
        num_m2 = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Zµ/%^2]+)?(?:\s*[\(\[]?([0-9\.\-\s]+)[\)\]]?)?", after_alias)
        if num_m2:
            val_str = num_m2.group(1)
            unit_str = num_m2.group(2)
            ref_str = num_m2.group(3)
            if cls._validate_number_for_target(val_str, target):
                return (val_str, unit_str if (unit_str and cls._is_unit(unit_str)) else None, ref_str, 0.92 if unit_str else 0.85)

        return None

    @classmethod
    def _find_value_in_lookahead(
        cls, lines: list[str], start_idx: int, end_idx: int, target: str
    ) -> tuple[str, str | None, str | None, float, str] | None:
        ignored_headers = {"result", "unit", "units", "value", "reference range", "ref range", "status", "test", "parameter"}
        snippet_lines = []
        found_num: str | None = None
        found_unit: str | None = None
        found_ref: str | None = None

        for j in range(start_idx, end_idx):
            next_line = lines[j].strip()
            if not next_line:
                continue
            if next_line.lower() in ignored_headers:
                continue

            # If next line is a major section title, stop lookahead
            if next_line.isupper() and len(next_line) > 5 and not re.search(r"\d", next_line):
                break

            snippet_lines.append(next_line)

            # Check exang yes/no in lookahead
            if "exang" in target.lower():
                yes_no_m = re.search(r"\b(yes|no|positive|negative|present|absent|true|false)\b", next_line, re.IGNORECASE)
                if yes_no_m:
                    word = yes_no_m.group(1).lower()
                    is_pos = "1" if word in ["yes", "positive", "present", "true"] else "0"
                    return (is_pos, None, None, 0.95, " -> ".join(snippet_lines))

            # Check compound BP in lookahead
            if "compound" in target.lower():
                bp_m = re.search(r"(\d{2,3})\s*\/\s*(\d{2,3})\s*(mm\s*hg)?", next_line, re.IGNORECASE)
                if bp_m:
                    return (f"{bp_m.group(1)}/{bp_m.group(2)}", bp_m.group(3) or "mmHg", None, 0.95, " -> ".join(snippet_lines))

            # Match number on next line
            num_m = re.search(r"^(\d+(?:\.\d+)?)\s*([a-zA-Zµ/%^2]+)?", next_line)
            if num_m and found_num is None:
                cand_num = num_m.group(1)
                cand_unit = num_m.group(2)
                if cls._validate_number_for_target(cand_num, target):
                    found_num = cand_num
                    found_unit = cand_unit if (cand_unit and cls._is_unit(cand_unit)) else None
                    if not found_unit and j + 1 < end_idx:
                        unit_line = lines[j + 1].strip()
                        if cls._is_unit(unit_line):
                            found_unit = unit_line
                            snippet_lines.append(unit_line)
                            if j + 2 < end_idx:
                                ref_line = lines[j + 2].strip()
                                if re.search(r"\d+\s*-\s*\d+", ref_line):
                                    found_ref = ref_line
                                    snippet_lines.append(ref_line)
                    return (found_num, found_unit, found_ref, 0.92, " -> ".join(snippet_lines))

        return None

    @classmethod
    def _record_match(
        cls,
        target: str,
        raw_name: str,
        val_str: str,
        unit_str: str | None,
        ref_str: str | None,
        conf: float,
        snippet: str,
        line_no: int,
        matches: list[MatchedParameter],
        matched_targets: set[str],
    ):
        matched_targets.add(target)

        if target == "BloodPressure_Diastolic":
            # For diabetes model, BloodPressure is diastolic
            matches.append(MatchedParameter(
                canonical_name="BloodPressure",
                raw_name=f"{raw_name} (Diastolic)",
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
            matches.append(MatchedParameter(
                canonical_name="diastolic_bp",
                raw_name=f"{raw_name} (Diastolic)",
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
        elif target == "BloodPressure_Systolic":
            # For cardio model, trestbps is systolic
            matches.append(MatchedParameter(
                canonical_name="systolic_bp",
                raw_name=f"{raw_name} (Systolic)",
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
            matches.append(MatchedParameter(
                canonical_name="trestbps",
                raw_name=f"{raw_name} (Systolic)",
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
        elif target == "BloodPressure_Compound":
            matches.append(MatchedParameter(
                canonical_name="BloodPressure",
                raw_name=raw_name,
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
            matches.append(MatchedParameter(
                canonical_name="trestbps",
                raw_name=raw_name,
                raw_value=val_str,
                raw_unit=unit_str or "mmHg",
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
        elif target == "Glucose":
            matches.append(MatchedParameter(
                canonical_name="Glucose",
                raw_name=raw_name,
                raw_value=val_str,
                raw_unit=unit_str,
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
            try:
                g_val = float(val_str)
                is_fbs_high = "1" if (g_val > 120.0 or (g_val < 25.0 and g_val > 6.6)) else "0"
                matches.append(MatchedParameter(
                    canonical_name="fbs",
                    raw_name=f"Fasting Blood Sugar > 120 ({val_str} {unit_str or 'mg/dL'})",
                    raw_value=is_fbs_high,
                    raw_unit=None,
                    reference_range="1=True (>120 mg/dL), 0=False",
                    confidence=conf,
                    confidence_level="high" if conf > 0.9 else "medium",
                    source_snippet=snippet,
                    line_number=line_no,
                ))
            except ValueError:
                pass
        else:
            canonical = target
            matches.append(MatchedParameter(
                canonical_name=canonical,
                raw_name=raw_name,
                raw_value=val_str,
                raw_unit=unit_str,
                reference_range=ref_str,
                confidence=conf,
                confidence_level="high" if conf > 0.9 else "medium",
                source_snippet=snippet,
                line_number=line_no,
            ))
            if canonical == "Age":
                matches.append(MatchedParameter(
                    canonical_name="age",
                    raw_name=raw_name,
                    raw_value=val_str,
                    raw_unit=unit_str,
                    reference_range=ref_str,
                    confidence=conf,
                    confidence_level="high" if conf > 0.9 else "medium",
                    source_snippet=snippet,
                    line_number=line_no,
                ))


    @classmethod
    def _extract_sex(cls, lines: list[str], matches: list[MatchedParameter]):
        for idx, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in ["testing note:", "clinical disclaimer:", "synthetic test document"]):
                continue

            if re.search(r"\bmale\b", lower) and not re.search(r"\bfemale\b", lower):
                matches.append(MatchedParameter(
                    canonical_name="sex",
                    raw_name="Sex (Male)",
                    raw_value="1",
                    raw_unit=None,
                    reference_range="1=Male, 0=Female",
                    confidence=0.98,
                    confidence_level="high",
                    source_snippet=line,
                    line_number=idx + 1,
                ))
                return
            elif re.search(r"\bfemale\b", lower):
                matches.append(MatchedParameter(
                    canonical_name="sex",
                    raw_name="Sex (Female)",
                    raw_value="0",
                    raw_unit=None,
                    reference_range="1=Male, 0=Female",
                    confidence=0.98,
                    confidence_level="high",
                    source_snippet=line,
                    line_number=idx + 1,
                ))
                return

    @classmethod
    def _extract_cancer_morphology(cls, lines: list[str], matches: list[MatchedParameter]):
        """Detects explicit 30 cell nucleus morphology features from FNA/biopsy cytology reports."""
        cancer_features = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
            'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
            'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
            'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
            'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
            'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
        ]
        full_text_lower = " \n ".join(lines).lower()

        for fn in cancer_features:
            readable = fn.replace("_", " ")
            patterns = [
                rf"\b{re.escape(readable)}\b[^\d\n\r]*?(\d+(?:\.\d+)?)",
                rf"\b{re.escape(fn)}\b[^\d\n\r]*?(\d+(?:\.\d+)?)",
            ]
            for p in patterns:
                m = re.search(p, full_text_lower)
                if m:
                    matches.append(MatchedParameter(
                        canonical_name=fn,
                        raw_name=readable.title(),
                        raw_value=m.group(1),
                        raw_unit=None,
                        reference_range=None,
                        confidence=0.94,
                        confidence_level="high",
                        source_snippet=m.group(0),
                    ))
                    break

    @classmethod
    def _deduplicate_matches(cls, matches: list[MatchedParameter]) -> list[MatchedParameter]:
        """Deduplicates matches by canonical name, keeping the highest confidence instance."""
        best: dict[str, MatchedParameter] = {}
        for m in matches:
            if m.canonical_name not in best or m.confidence > best[m.canonical_name].confidence:
                best[m.canonical_name] = m
        return list(best.values())
