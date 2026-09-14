from typing import Any
from fastapi import APIRouter, HTTPException, Query, status
from app.services.ml_service import ml_service, CANONICAL_MODELS
from app.schemas.api import ModelSummary, ModelsResponse

router = APIRouter()


@router.get("", response_model=ModelsResponse)
@router.get("/", response_model=ModelsResponse, include_in_schema=False)
def get_models_status(
    disease: str | None = Query(None, description="Optional target disease identifier: 'diabetes', 'cardiovascular', 'cancer', or 'all'")
) -> ModelsResponse:
    """Returns authoritative metadata and deployment status for registered models (strictly filtered by disease)."""
    target_disease = disease.lower().strip() if disease and disease.lower() != "all" else None
    raw_models = ml_service.get_registered_models_summary(disease=target_disease)
    
    models_list = [ModelSummary(**m) for m in raw_models]
    return ModelsResponse(
        models=models_list,
        total=len(models_list),
        disease=target_disease or "all"
    )


@router.get("/{model_id}")
def get_model_detail(model_id: str) -> ModelSummary:
    """Returns full authoritative metadata for an individual model."""
    summary_list = ml_service.get_registered_models_summary()
    matched = next((m for m in summary_list if m["model_id"].lower() == model_id.lower()), None)
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in authoritative registry.",
        )
    return ModelSummary(**matched)


@router.get("/{model_id}/explain/global")
def get_global_explanation(model_id: str) -> dict[str, Any]:
    """Returns population-level global feature importance for the requested model artifact."""
    from ml.explainability.service import explanation_service

    try:
        art_id = ml_service.resolve_artifact_id(model_id)
        artifact = ml_service.get_artifact(art_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact '{model_id}' not found in registry: {e!s}",
        )

    res = explanation_service.explain_global(artifact)
    return res.to_dict()
