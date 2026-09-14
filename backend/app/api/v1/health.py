"""Health check endpoint for platform monitoring and model readiness."""
from app.core.config import settings
from app.schemas.api import HealthResponse
from app.services.ml_service import ml_service
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for platform monitoring and model readiness."""
    models_summary = ml_service.get_registered_models_summary()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.app_env,
        backend_engine="FastAPI",
        models_available=models_summary,
    )
