"""Report Extraction Service coordinating document parsing, parameter extraction, and disease mapping."""
import logging
from fastapi import HTTPException, UploadFile, status

from app.diseases.registry import disease_registry
from app.report.disease_mapper import DiseaseFeatureMapper
from app.report.matcher import ParameterMatcher
from app.report.parser import document_parser
from app.schemas.report import ReportScanResponse

logger = logging.getLogger(__name__)


class ReportService:
    """Service orchestrating AI medical report ingestion and feature extraction."""

    async def scan_report(self, file: UploadFile, disease: str) -> ReportScanResponse:
        """Processes an uploaded medical report and returns structured extraction metadata."""
        # 1. Resolve and validate target disease
        try:
            canonical_disease = disease_registry.resolve_disease_id(disease)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid disease parameter '{disease}': {e!s}",
            )

        # 2. Parse uploaded document (PDF or Image)
        try:
            extraction_result = await document_parser.parse_upload(file)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document parsing error for file '{file.filename}': {e!s}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from document: {e!s}",
            )

        # 3. Match clinical parameters from extracted text lines
        raw_text = extraction_result.raw_text
        lines = extraction_result.lines

        if not raw_text or len(lines) == 0:
            # Handle scanned image with empty text or unreadable PDF
            return ReportScanResponse(
                success=True,
                filename=extraction_result.filename,
                file_type=extraction_result.file_type,
                disease=canonical_disease,
                report_category="unknown",
                compatible=False,
                extracted_features={},
                missing_features=[],
                total_features_required=8 if canonical_disease == "diabetes" else (13 if canonical_disease == "cardiovascular" else 30),
                total_features_detected=0,
                raw_text_summary="Document contained no selectable text streams or OCR text.",
                extraction_method=extraction_result.extraction_method,
                text_detected=False,
                extracted_text_length=0,
                message="We couldn't read text in this document. Please upload a clearer digital report or enter values manually.",
                validation_warnings=["No text or numerical tokens identified in document."],
            )

        matches = ParameterMatcher.extract_from_lines(lines)

        # 4. Map, normalize, and validate against disease-specific model schema
        response = DiseaseFeatureMapper.map_extracted_data(
            disease=canonical_disease,
            matches=matches,
            raw_text=raw_text,
            filename=extraction_result.filename,
            file_type=extraction_result.file_type,
            extraction_method=extraction_result.extraction_method,
            text_detected=extraction_result.text_detected,
            extracted_text_length=extraction_result.extracted_text_length,
        )

        logger.info(
            f"Report '{extraction_result.filename}' processed for '{canonical_disease}': "
            f"{response.total_features_detected}/{response.total_features_required} features detected "
            f"via {extraction_result.extraction_method} ({extraction_result.extracted_text_length} chars)."
        )


        return response


# Singleton service instance
report_service = ReportService()
