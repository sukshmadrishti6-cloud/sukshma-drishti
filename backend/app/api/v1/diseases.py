"""Disease discovery and metadata API endpoints."""
from fastapi import APIRouter, status
from app.diseases.registry import disease_registry
from app.schemas.api import DiseaseDetail, DiseaseListResponse

router = APIRouter()


@router.get(
    "",
    response_model=DiseaseListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all registered diseases and deployment availability status",
    description="Returns metadata for all platform-supported disease domains, distinguishing between active models ('available') and future targets ('coming_soon').",
)
async def list_diseases() -> DiseaseListResponse:
    """Lists registered disease domains."""
    items = disease_registry.list_diseases()
    return DiseaseListResponse(items=items, total=len(items))


@router.get(
    "/{disease_id}",
    response_model=DiseaseDetail,
    status_code=status.HTTP_200_OK,
    summary="Get metadata and supported model details for a specific disease",
    description="Returns detailed metadata, feature list, and registered inference models for a canonical disease identifier.",
)
async def get_disease(disease_id: str) -> DiseaseDetail:
    """Returns detailed metadata for a specified disease identifier."""
    return disease_registry.get_disease_detail(disease_id)
