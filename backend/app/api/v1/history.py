"""Prediction History API Router for per-user health record management."""
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies import get_current_user
from app.schemas.api import (
    PredictionHistoryDetail,
    PredictionHistoryItem,
    PredictionHistoryListResponse,
)
from app.services.prediction_service import prediction_service

router = APIRouter()


def _extract_user_id(current_user: Any) -> str:
    """Extracts internal string user ID from authenticated identity object."""
    user_id = getattr(current_user, "id", None) or (
        current_user.get("id") if isinstance(current_user, dict) else None
    )
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or invalid.",
        )
    return str(user_id)


@router.get(
    "",
    response_model=PredictionHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get authenticated user prediction history",
    description="Returns paginated list of disease prediction records belonging exclusively to the authenticated user.",
)
def get_user_history(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    disease: str | None = Query(None, description="Optional disease filter (e.g. 'diabetes')"),
    model_name: str | None = Query(None, description="Optional model name filter"),
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionHistoryListResponse:
    """Paginated user prediction history endpoint."""
    user_id = _extract_user_id(current_user)

    records, total = prediction_service.get_user_history(
        db=db,
        user_id=user_id,
        disease=disease,
        model_name=model_name,
        page=page,
        page_size=page_size,
    )

    items = [PredictionHistoryItem(**r.to_summary_dict()) for r in records]
    return PredictionHistoryListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionHistoryDetail,
    status_code=status.HTTP_200_OK,
    summary="Get individual prediction record details",
    description="Returns full prediction record details including biomedical input payload and explanation for a record owned by the authenticated user.",
)
def get_prediction_detail(
    prediction_id: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionHistoryDetail:
    """Individual prediction record detail endpoint."""
    user_id = _extract_user_id(current_user)

    record = prediction_service.get_prediction_detail(db, prediction_id, user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction history record not found.",
        )

    return PredictionHistoryDetail(**record.to_detail_dict())


@router.delete(
    "/{prediction_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete individual prediction record",
    description="Permanently deletes a prediction record owned by the authenticated user.",
)
def delete_prediction_record(
    prediction_id: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Prediction record deletion endpoint."""
    user_id = _extract_user_id(current_user)

    deleted = prediction_service.delete_prediction_record(db, prediction_id, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction history record not found.",
        )

    return {
        "status": "success",
        "message": "Prediction history record deleted successfully.",
        "id": prediction_id,
    }
