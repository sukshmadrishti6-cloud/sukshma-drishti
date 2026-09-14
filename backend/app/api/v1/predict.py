"""Inference prediction API route for classical and quantum ML models across supported diseases."""
from typing import Any
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies import get_current_user_optional
from app.diseases.registry import disease_registry
from app.schemas.api import InferenceRequest, InferenceResponse
from app.services.prediction_service import prediction_service

router = APIRouter()


@router.post(
    "/predict/{disease}",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute disease risk prediction for specified disease domain",
    description="Accepts clinical predictor features and a model identifier, routing request to disease adapter for inference.",
)
async def predict_risk_for_disease(
    disease: str,
    request: InferenceRequest,
    current_user: Any = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> InferenceResponse:
    """Disease risk prediction endpoint for specific disease.

    Resolves disease adapter via DiseaseRegistry. Rejects unavailable diseases with HTTP 501.
    Persists prediction record if user is authenticated.
    """
    adapter = disease_registry.get_adapter(disease)
    response = adapter.predict(request)

    if current_user:
        user_id = getattr(current_user, "id", None) or (current_user.get("id") if isinstance(current_user, dict) else None)
        if user_id:
            prediction_service.save_prediction_record(
                db=db,
                user_id=str(user_id),
                request=request,
                response=response,
                disease=adapter.disease_id,
            )

    return response


@router.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute disease risk prediction (defaults to Diabetes)",
    description="Backward-compatible endpoint routing inference to default diabetes prediction pipeline.",
)
async def predict_risk(
    request: InferenceRequest,
    current_user: Any = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> InferenceResponse:
    """Backward-compatible endpoint delegating to predict_risk_for_disease with 'diabetes'."""
    target_disease = request.disease or "diabetes"
    return await predict_risk_for_disease(
        disease=target_disease,
        request=request,
        current_user=current_user,
        db=db,
    )
