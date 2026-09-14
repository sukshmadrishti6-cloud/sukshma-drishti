"""Medical Report Scanning and Ingestion API endpoints."""
import logging
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.report import ReportScanResponse
from app.services.report_service import report_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/scan",
    response_model=ReportScanResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and scan medical report for AI parameter extraction",
    description="Processes uploaded PDF or image medical report, extracting clinical indicators and normalizing them for disease assessment.",
)
async def scan_medical_report(
    file: UploadFile = File(..., description="Uploaded medical report file (PDF, PNG, JPG, JPEG, WEBP)"),
    disease: str = Form("diabetes", description="Target disease domain (diabetes, cardiovascular, cancer)"),
) -> ReportScanResponse:
    """Scans and extracts medical parameters from an uploaded clinical document.

    Does NOT run inference immediately; returns structured extracted features and missing fields for user verification.
    """
    return await report_service.scan_report(file=file, disease=disease)
