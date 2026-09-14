"""Pydantic schema definitions for AI-assisted Medical Report Scanning."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ExtractedFeatureItem(BaseModel):
    """Structured representation of a single extracted clinical parameter."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Canonical model feature name")
    value: float = Field(..., description="Normalized numerical value ready for model input")
    unit: str | None = Field(default=None, description="Normalized clinical unit (e.g. mg/dL, mmHg)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score from 0.0 to 1.0")
    confidence_level: Literal["high", "medium", "low"] = Field(
        default="high", description="Categorical confidence tier"
    )
    original_name: str = Field(..., description="Original parameter label found in document text")
    original_value: float | str = Field(..., description="Raw extracted value prior to normalization")
    original_unit: str | None = Field(default=None, description="Original unit found in document text")
    source_snippet: str | None = Field(default=None, description="Extracted line or context snippet")
    conversion_applied: bool = Field(
        default=False, description="Whether unit or metric conversion was applied"
    )
    requires_verification: bool = Field(
        default=False, description="Flag indicating user verification is strongly recommended"
    )


class MissingFeatureItem(BaseModel):
    """Details on required features not identified in the uploaded report."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Canonical model feature name")
    display_name: str = Field(..., description="Human-readable feature name")
    description: str | None = Field(default=None, description="Clinical feature description")
    reason: str = Field(
        default="Requires Manual Input",
        description="Reason why value was not extracted from document",
    )


class ReportScanResponse(BaseModel):
    """Authoritative API response schema returned by /report/scan."""

    model_config = ConfigDict(extra="ignore")

    success: bool = Field(..., description="Whether document was successfully processed")
    filename: str = Field(..., description="Original name of uploaded document")
    file_type: str = Field(..., description="Detected MIME or file format extension")
    disease: str = Field(..., description="Target disease domain")
    report_category: str = Field(
        default="unknown",
        description="Classified document category: laboratory_report, cardiology_report, pathology_report, general_medical_report, or unknown",
    )
    compatible: bool = Field(
        default=True,
        description="Whether uploaded report type matches target disease requirements",
    )
    extracted_features: dict[str, ExtractedFeatureItem] = Field(
        default_factory=dict,
        description="Map of canonical feature name to extracted and normalized parameter",
    )
    missing_features: list[str] = Field(
        default_factory=list,
        description="List of canonical feature names that could not be extracted",
    )
    missing_features_detail: list[MissingFeatureItem] = Field(
        default_factory=list,
        description="Detailed items for missing parameters requiring manual entry",
    )
    total_features_required: int = Field(
        ..., description="Total number of features required by the target disease model"
    )
    total_features_detected: int = Field(
        ..., description="Total number of compatible features extracted from document"
    )
    raw_text_summary: str | None = Field(
        default=None,
        description="Sanitized non-PII preview of extracted document text",
    )
    extraction_method: str | None = Field(
        default="pdf_text",
        description="Method used for extraction: pdf_text, pdf_ocr, image_ocr, or unavailable",
    )
    text_detected: bool = Field(
        default=True,
        description="Whether readable text streams or OCR tokens were identified in document",
    )
    extracted_text_length: int = Field(
        default=0,
        description="Length in characters of extracted raw text content",
    )
    message: str = Field(..., description="User-facing summary message describing extraction result")
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="List of warnings or boundary caveats found during extraction",
    )

