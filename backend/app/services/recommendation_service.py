"""Recommendation Service for persistence and user authorization management."""
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException, status

from app.db.models import RecommendationRecord, PredictionRecord
from app.schemas.api import RecommendationResponse, RecommendationFactor

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service handling recommendation database persistence and ownership-verified retrieval."""

    def save_recommendation(
        self,
        db: Session,
        user_id: str,
        prediction_record_id: str,
        disease: str,
        recommendation: RecommendationResponse,
    ) -> RecommendationRecord | None:
        """Persists a recommendation record associated with a prediction record."""
        try:
            priority_actions_json = json.dumps(recommendation.priority_actions)
            monitoring_json = json.dumps(recommendation.monitoring)
            factors_json = json.dumps([f.model_dump() for f in recommendation.factors])

            record = RecommendationRecord(
                prediction_record_id=str(prediction_record_id),
                user_id=str(user_id),
                disease=disease,
                summary=recommendation.summary,
                priority_actions=priority_actions_json,
                monitoring=monitoring_json,
                professional_follow_up=recommendation.professional_follow_up,
                when_to_seek_care=recommendation.when_to_seek_care,
                factors=factors_json,
                disclaimer=recommendation.disclaimer,
                rule_set_version=recommendation.rule_set_version,
            )

            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(
                f"Saved recommendation '{record.id}' for prediction '{prediction_record_id}' (User: '{user_id}')"
            )
            return record
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist recommendation for prediction '{prediction_record_id}': {e!s}", exc_info=True)
            return None

    def get_recommendation_by_prediction_id(
        self,
        db: Session,
        prediction_record_id: str,
        user_id: str,
    ) -> RecommendationResponse | None:
        """Retrieves a recommendation strictly verifying ownership by the authenticated user."""
        record = (
            db.query(RecommendationRecord)
            .filter(
                RecommendationRecord.prediction_record_id == str(prediction_record_id),
                RecommendationRecord.user_id == str(user_id),
            )
            .first()
        )
        if not record:
            return None

        factors = [RecommendationFactor(**f) for f in record.get_factors_list()]

        return RecommendationResponse(
            id=record.id,
            prediction_record_id=record.prediction_record_id,
            disease=record.disease,
            summary=record.summary,
            priority_actions=record.get_priority_actions_list(),
            monitoring=record.get_monitoring_list(),
            professional_follow_up=record.professional_follow_up,
            when_to_seek_care=record.when_to_seek_care,
            factors=factors,
            disclaimer=record.disclaimer,
            rule_set_version=record.rule_set_version,
            created_at=record.created_at.isoformat() if hasattr(record.created_at, "isoformat") else str(record.created_at),
        )

    def get_user_recommendations(
        self,
        db: Session,
        user_id: str,
        disease: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[RecommendationResponse], int]:
        """Retrieves paginated recommendations strictly belonging to the authenticated user."""
        query = db.query(RecommendationRecord).filter(RecommendationRecord.user_id == str(user_id))

        if disease:
            query = query.filter(RecommendationRecord.disease == disease.strip().lower())

        total = query.count()

        records = (
            query.order_by(desc(RecommendationRecord.created_at), desc(RecommendationRecord.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        items = []
        for r in records:
            factors = [RecommendationFactor(**f) for f in r.get_factors_list()]
            items.append(
                RecommendationResponse(
                    id=r.id,
                    prediction_record_id=r.prediction_record_id,
                    disease=r.disease,
                    summary=r.summary,
                    priority_actions=r.get_priority_actions_list(),
                    monitoring=r.get_monitoring_list(),
                    professional_follow_up=r.professional_follow_up,
                    when_to_seek_care=r.when_to_seek_care,
                    factors=factors,
                    disclaimer=r.disclaimer,
                    rule_set_version=r.rule_set_version,
                    created_at=r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
                )
            )

        return items, total


recommendation_service = RecommendationService()
