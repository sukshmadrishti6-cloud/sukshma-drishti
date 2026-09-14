"""Unit tests for DocumentParser in report scanning pipeline."""
import io
import pytest
from fastapi import HTTPException
from app.report.parser import DocumentParser, MAX_FILE_SIZE_BYTES
import pypdf


def test_file_validation_valid_pdf():
    parser = DocumentParser()
    ext = parser.validate_file("patient_lab_report.pdf", "application/pdf", 1024 * 50)
    assert ext == ".pdf"


def test_file_validation_valid_image():
    parser = DocumentParser()
    ext = parser.validate_file("scan.PNG", "image/png", 1024 * 100)
    assert ext == ".png"


def test_file_validation_invalid_extension():
    parser = DocumentParser()
    with pytest.raises(HTTPException) as exc_info:
        parser.validate_file("script.exe", "application/x-msdownload", 500)
    assert exc_info.value.status_code == 400
    assert "Unsupported file type" in exc_info.value.detail


def test_file_validation_oversized_file():
    parser = DocumentParser()
    with pytest.raises(HTTPException) as exc_info:
        parser.validate_file("report.pdf", "application/pdf", MAX_FILE_SIZE_BYTES + 100)
    assert exc_info.value.status_code == 413
    assert "too large" in exc_info.value.detail


def test_file_validation_empty_file():
    parser = DocumentParser()
    with pytest.raises(HTTPException) as exc_info:
        parser.validate_file("report.pdf", "application/pdf", 0)
    assert exc_info.value.status_code == 400


def test_pdf_parsing_synthetic_pdf():
    # Generate a simple in-memory PDF using pypdf writer
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    
    # We can write text or metadata
    buffer = io.BytesIO()
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()

    parser = DocumentParser()
    res = parser.parse_pdf(pdf_bytes, "test.pdf")
    assert res.file_type == "pdf"
    assert res.page_count == 1
