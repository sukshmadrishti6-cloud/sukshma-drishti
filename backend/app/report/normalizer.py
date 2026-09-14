"""Unit Conversion and Numerical Value Normalization Engine."""
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NormalizedValue:
    """Represents a normalized numerical measurement with provenance."""
    value: float
    unit: str | None
    original_value: float | str
    original_unit: str | None
    conversion_applied: bool = False
    notes: str | None = None


class UnitNormalizer:
    """Provides medically valid unit conversions and numerical normalization."""

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalizes unicode characters, dashes, spaces, and clinical symbols."""
        if not text:
            return ""
        # Normalize dash variations (en-dash, em-dash, minus, horizontal bar)
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
        # Normalize smart quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)
        text = re.sub(r"[\u201c\u201d]", '"', text)
        # Non-breaking and zero-width spaces
        text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
        # Normalize BMI units
        text = re.sub(r"kg/m[\u00b2\u00b3\ufffd\^2]*", "kg/m2", text, flags=re.IGNORECASE)
        # Normalize micro units
        text = re.sub(r"[\u00b5\u03bc\ufffd]iu", "uiu", text, flags=re.IGNORECASE)
        # Replace stray replacement characters with space or dash as appropriate
        text = text.replace("\ufffd", "-")
        return text

    @staticmethod
    def clean_number(raw_val: str | float | int) -> float | None:
        """Parses a numerical value from text, removing commas and stray chars."""
        if isinstance(raw_val, (int, float)):
            return float(raw_val)
        if not raw_val:
            return None
        # Remove commas, trailing periods, stray characters
        cleaned = re.sub(r"[^\d\.\-]", "", str(raw_val).strip())
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None


    @classmethod
    def normalize_glucose(cls, value: float | str, unit: str | None) -> NormalizedValue | None:
        """Normalizes Blood Glucose to standard mg/dL units."""
        num = cls.clean_number(value)
        if num is None:
            return None

        clean_unit = (unit or "").strip().lower()

        # Check if mmol/L (typical physiological range ~3.0 - 35.0 mmol/L)
        if "mmol" in clean_unit or (num < 25.0 and "mg" not in clean_unit):
            # Convert mmol/L -> mg/dL (1 mmol/L = 18.0182 mg/dL)
            converted = round(num * 18.0182, 1)
            return NormalizedValue(
                value=converted,
                unit="mg/dL",
                original_value=num,
                original_unit=unit or "mmol/L",
                conversion_applied=True,
                notes="Converted from mmol/L to mg/dL (factor 18.0182)",
            )

        return NormalizedValue(
            value=round(num, 1),
            unit="mg/dL",
            original_value=num,
            original_unit=unit or "mg/dL",
            conversion_applied=False,
        )

    @classmethod
    def normalize_cholesterol(cls, value: float | str, unit: str | None) -> NormalizedValue | None:
        """Normalizes Serum Total Cholesterol to standard mg/dL units."""
        num = cls.clean_number(value)
        if num is None:
            return None

        clean_unit = (unit or "").strip().lower()

        # Check if mmol/L (typical cholesterol ~2.5 - 15.0 mmol/L)
        if "mmol" in clean_unit or (num < 20.0 and "mg" not in clean_unit):
            # Convert mmol/L -> mg/dL (1 mmol/L = 38.67 mg/dL)
            converted = round(num * 38.67, 1)
            return NormalizedValue(
                value=converted,
                unit="mg/dL",
                original_value=num,
                original_unit=unit or "mmol/L",
                conversion_applied=True,
                notes="Converted from mmol/L to mg/dL (factor 38.67)",
            )

        return NormalizedValue(
            value=round(num, 1),
            unit="mg/dL",
            original_value=num,
            original_unit=unit or "mg/dL",
            conversion_applied=False,
        )

    @classmethod
    def normalize_blood_pressure(cls, raw_text: str | float) -> tuple[NormalizedValue | None, NormalizedValue | None]:
        """Extracts Systolic and Diastolic Blood Pressure components from BP text.

        Returns (systolic_norm, diastolic_norm).
        """
        if isinstance(raw_text, (int, float)):
            # Single numeric BP assumed to be systolic
            val = float(raw_text)
            return (
                NormalizedValue(value=val, unit="mmHg", original_value=val, original_unit="mmHg"),
                None,
            )

        text_str = str(raw_text).strip()
        # Look for compound pattern with slash like "120/80", "120 / 80"
        bp_match = re.search(r"(\d{2,3})\s*\/\s*(\d{2,3})", text_str)
        if bp_match:
            sys_val = float(bp_match.group(1))
            dia_val = float(bp_match.group(2))
            return (
                NormalizedValue(
                    value=sys_val,
                    unit="mmHg",
                    original_value=f"{int(sys_val)}/{int(dia_val)}",
                    original_unit="mmHg",
                    notes="Extracted systolic from compound BP",
                ),
                NormalizedValue(
                    value=dia_val,
                    unit="mmHg",
                    original_value=f"{int(sys_val)}/{int(dia_val)}",
                    original_unit="mmHg",
                    notes="Extracted diastolic from compound BP",
                ),
            )

        # Fallback single number
        num = cls.clean_number(text_str)
        if num is not None and 30 <= num <= 280:
            return (
                NormalizedValue(value=num, unit="mmHg", original_value=num, original_unit="mmHg"),
                None,
            )

        return (None, None)


    @classmethod
    def normalize_insulin(cls, value: float | str, unit: str | None) -> NormalizedValue | None:
        """Normalizes Serum Insulin to standard µIU/mL (or pmol/L conversion)."""
        num = cls.clean_number(value)
        if num is None:
            return None

        clean_unit = (unit or "").strip().lower()
        if "pmol" in clean_unit:
            # 1 µIU/mL = 6.945 pmol/L
            converted = round(num / 6.945, 1)
            return NormalizedValue(
                value=converted,
                unit="µIU/mL",
                original_value=num,
                original_unit=unit,
                conversion_applied=True,
                notes="Converted from pmol/L to µIU/mL (divided by 6.945)",
            )

        return NormalizedValue(
            value=round(num, 1),
            unit="µIU/mL",
            original_value=num,
            original_unit=unit or "µIU/mL",
            conversion_applied=False,
        )

    @classmethod
    def normalize_generic(
        cls, value: float | str, expected_unit: str | None = None, unit: str | None = None
    ) -> NormalizedValue | None:
        """Standard numerical normalization without conversion."""
        num = cls.clean_number(value)
        if num is None:
            return None
        return NormalizedValue(
            value=float(num),
            unit=unit or expected_unit,
            original_value=num,
            original_unit=unit,
            conversion_applied=False,
        )
