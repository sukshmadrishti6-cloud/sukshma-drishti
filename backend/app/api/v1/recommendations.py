"""Recommendations API Router for authenticated user health guidance."""
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.dependencies import get_current_user
from app.schemas.api import (
    RecommendationListResponse,
    RecommendationResponse,
)
from app.services.recommendation_service import recommendation_service

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
    response_model=RecommendationListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get authenticated user recommendations",
    description="Returns paginated list of structured health recommendations belonging exclusively to the authenticated user.",
)
def get_user_recommendations(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    disease: str | None = Query(None, description="Optional disease filter (e.g. 'diabetes')"),
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationListResponse:
    """Paginated user recommendations endpoint."""
    user_id = _extract_user_id(current_user)

    items, total = recommendation_service.get_user_recommendations(
        db=db,
        user_id=user_id,
        disease=disease,
        page=page,
        page_size=page_size,
    )

    return RecommendationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{prediction_id}",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get recommendation for individual prediction record",
    description="Returns structured health recommendation for a prediction record owned by the authenticated user.",
)
def get_recommendation_by_prediction_id(
    prediction_id: str,
    current_user: Any = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    """Individual recommendation lookup by prediction record ID endpoint."""
    user_id = _extract_user_id(current_user)

    recommendation = recommendation_service.get_recommendation_by_prediction_id(
        db=db,
        prediction_record_id=prediction_id,
        user_id=user_id,
    )
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation record not found or access denied.",
        )

    return recommendation
