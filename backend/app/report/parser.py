"""Document Parser and OCR Text Extraction Engine for Medical Reports."""
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image
import pypdf

logger = logging.getLogger(__name__)

# Allowed file extensions and corresponding MIME types
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME_TYPES: set[str] = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "application/octet-stream",  # Sometimes sent by clients for binary uploads
}

# Maximum allowable file size in bytes (10 MB)
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024


@dataclass
class ExtractionResult:
    """Encapsulates raw text extraction output from document parsing."""
    raw_text: str
    lines: list[str]
    file_type: str
    filename: str
    page_count: int = 1
    extraction_method: str = "pdf_text"  # 'pdf_text', 'pdf_ocr', 'image_ocr', 'unavailable'
    ocr_engine_used: str | None = None
    text_detected: bool = True
    extracted_text_length: int = 0
    metadata: dict = field(default_factory=dict)


class DocumentParser:
    """Secure document parser extracting text from PDFs and images."""

    def __init__(self, max_file_size: int = MAX_FILE_SIZE_BYTES):
        self.max_file_size = max_file_size

    def validate_file(self, filename: str, content_type: str | None, file_size: int) -> str:
        """Validates uploaded file against MIME, extension, and size security policies.

        Returns normalized extension (e.g. '.pdf', '.png').
        """
        # 1. Sanitize filename and extract extension
        safe_name = os.path.basename(filename.strip())
        if not safe_name or ".." in safe_name or "/" in safe_name or "\\" in safe_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security violation: Invalid or malicious filename detected.",
            )

        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{ext}'. Please upload a PDF, JPG, JPEG, PNG, or WEBP medical report.",
            )

        # 2. Validate MIME type if provided
        if content_type:
            clean_mime = content_type.lower().split(";")[0].strip()
            if clean_mime not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid content type '{content_type}'. Please upload a valid PDF or image file.",
                )

        # 3. Validate size limit
        if file_size > self.max_file_size:
            max_mb = self.max_file_size // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"The selected file is too large ({file_size} bytes). Maximum allowed size is {max_mb}MB.",
            )

        if file_size <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty. Please provide a valid medical report document.",
            )

        return ext

    def parse_pdf(self, file_bytes: bytes, filename: str) -> ExtractionResult:
        """Extracts text content and layout lines from a PDF document with OCR fallback."""
        try:
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)
            num_pages = len(reader.pages)

            if num_pages == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Uploaded PDF contains no pages.",
                )

            # Strategy 1: Direct PDF text extraction (plain and layout)
            plain_pages: list[str] = []
            layout_pages: list[str] = []

            for page in reader.pages:
                t_plain = page.extract_text() or ""
                plain_pages.append(t_plain)
                try:
                    t_layout = page.extract_text(extraction_mode="layout") or ""
                    layout_pages.append(t_layout)
                except Exception:
                    layout_pages.append(t_plain)

            full_plain_text = "\n".join(plain_pages).strip()
            full_layout_text = "\n".join(layout_pages).strip()

            # Check if meaningful text was found (>= 20 alphanumeric characters)
            has_meaningful_text = len(re.findall(r"[a-zA-Z0-9]", full_plain_text)) >= 20

            if has_meaningful_text:
                # Combine plain and layout lines into a comprehensive collection
                lines: list[str] = []
                seen_lines: set[str] = set()

                for raw_l in full_plain_text.splitlines() + full_layout_text.splitlines():
                    clean_l = raw_l.strip()
                    if clean_l and clean_l not in seen_lines:
                        lines.append(clean_l)
                        seen_lines.add(clean_l)

                return ExtractionResult(
                    raw_text=full_plain_text,
                    lines=lines,
                    file_type="pdf",
                    filename=filename,
                    page_count=num_pages,
                    extraction_method="pdf_text",
                    ocr_engine_used=None,
                    text_detected=True,
                    extracted_text_length=len(full_plain_text),
                    metadata={"pages": num_pages, "has_layout": bool(full_layout_text)},
                )

            # Strategy 2: Scanned PDF fallback via OCR on embedded page images
            logger.info(f"PDF '{filename}' contained no selectable text stream; attempting OCR extraction on page images.")
            ocr_text_parts: list[str] = []

            try:
                import pytesseract

                for page_idx, page in enumerate(reader.pages):
                    if hasattr(page, "images") and page.images:
                        for img_obj in page.images:
                            try:
                                pil_img = Image.open(io.BytesIO(img_obj.data))
                                if pil_img.mode not in ("L", "RGB"):
                                    pil_img = pil_img.convert("RGB")
                                img_text = pytesseract.image_to_string(pil_img)
                                if img_text.strip():
                                    ocr_text_parts.append(img_text.strip())
                            except Exception as img_err:
                                logger.warning(f"Failed to OCR image on page {page_idx} of '{filename}': {img_err}")
            except Exception as ocr_lib_err:
                logger.warning(f"Tesseract OCR unavailable for scanned PDF '{filename}': {ocr_lib_err}")

            full_ocr_text = "\n".join(ocr_text_parts).strip()
            if full_ocr_text and len(re.findall(r"[a-zA-Z0-9]", full_ocr_text)) >= 10:
                ocr_lines = [l.strip() for l in full_ocr_text.splitlines() if l.strip()]
                return ExtractionResult(
                    raw_text=full_ocr_text,
                    lines=ocr_lines,
                    file_type="pdf",
                    filename=filename,
                    page_count=num_pages,
                    extraction_method="pdf_ocr",
                    ocr_engine_used="tesseract_pdf_images",
                    text_detected=True,
                    extracted_text_length=len(full_ocr_text),
                    metadata={"scanned_pdf": True, "ocr_extracted": True},
                )

            # No readable text found
            return ExtractionResult(
                raw_text="",
                lines=[],
                file_type="pdf",
                filename=filename,
                page_count=num_pages,
                extraction_method="unavailable",
                ocr_engine_used=None,
                text_detected=False,
                extracted_text_length=0,
                metadata={"scanned_only": True, "no_text_found": True},
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process PDF document '{filename}': {e!s}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"We couldn't read this PDF document. The file might be corrupted or malformed. ({e!s})",
            )

    def parse_image(self, file_bytes: bytes, filename: str, ext: str) -> ExtractionResult:
        """Extracts text content from an image file using OCR with graceful fallbacks."""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # Verify image integrity
            # Reopen after verify since verify alters stream position
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Malformed image file '{filename}'. Please upload a valid image (PNG, JPG, WEBP).",
            )

        ocr_text = ""
        ocr_engine = None
        extraction_method = "image_ocr"

        # Attempt pytesseract OCR extraction
        try:
            import pytesseract
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")
            ocr_text = pytesseract.image_to_string(image)
            ocr_engine = "tesseract_image_ocr"
        except Exception as ocr_err:
            logger.warning(
                f"Tesseract OCR is not configured or failed on host for '{filename}': {ocr_err}. "
                "Falling back to manual entry."
            )
            ocr_text = ""
            ocr_engine = "unavailable"
            extraction_method = "unavailable"

        lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        has_text = len(re.findall(r"[a-zA-Z0-9]", ocr_text)) >= 5

        return ExtractionResult(
            raw_text=ocr_text,
            lines=lines,
            file_type=ext.replace(".", ""),
            filename=filename,
            page_count=1,
            extraction_method=extraction_method,
            ocr_engine_used=ocr_engine,
            text_detected=has_text,
            extracted_text_length=len(ocr_text),
            metadata={"width": image.width, "height": image.height, "mode": image.mode},
        )

    async def parse_upload(self, file: UploadFile) -> ExtractionResult:
        """Asynchronously reads and parses an UploadFile from a FastAPI request."""
        # Read content
        content = await file.read()
        file_size = len(content)
        filename = file.filename or "medical_report.pdf"

        # Validate
        ext = self.validate_file(filename, file.content_type, file_size)

        if ext == ".pdf":
            return self.parse_pdf(content, filename)
        else:
            return self.parse_image(content, filename, ext)


# Singleton parser instance
document_parser = DocumentParser()

