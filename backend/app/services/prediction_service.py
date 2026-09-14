"""Prediction Persistence and History Service managing persistent patient records."""
import json
import logging
from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException, status

from app.db.models import PredictionRecord
from app.schemas.api import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class PredictionService:
    """Service handling prediction persistence and user history operations."""

    def sanitize_input_features(self, features: dict[str, float] | list[float]) -> dict[str, float]:
        """Converts feature input payload into a clean, sanitized feature dictionary."""
        if isinstance(features, dict):
            # Only retain recognized/valid keys or numeric values
            sanitized = {}
            for k, v in features.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sanitized[str(k)] = float(v)
            return sanitized
        elif isinstance(features, list):
            sanitized = {}
            for idx, val in enumerate(features):
                col_name = FEATURE_COLUMNS[idx] if idx < len(FEATURE_COLUMNS) else f"feature_{idx}"
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    sanitized[col_name] = float(val)
            return sanitized
        return {}

    def save_prediction_record(
        self,
        db: Session,
        user_id: str,
        request: InferenceRequest,
        response: InferenceResponse,
        disease: str = "diabetes",
    ) -> PredictionRecord | None:
        """Persists a successful inference prediction result and generates/persists associated recommendation."""
        from app.db.models import User
        from app.recommendations.engine import recommendation_engine
        from app.services.recommendation_service import recommendation_service

        user = db.query(User).filter(User.id == str(user_id)).first()
        if not user:
            logger.warning(f"Skipping prediction persistence: User ID '{user_id}' not found in database.")
            return None

        try:
            sanitized_input = self.sanitize_input_features(request.features)
            input_json_str = json.dumps(sanitized_input)

            explanation_json_str = None
            if response.explanation:
                explanation_json_str = json.dumps(response.explanation.model_dump())

            risk_score = response.probability if response.probability is not None else response.decision_score

            # 1. Fetch user history context for history-aware recommendations
            past_records, _ = self.get_user_history(db=db, user_id=str(user.id), disease=disease, page=1, page_size=10)
            history_context = [r.to_summary_dict() for r in past_records]

            # 2. Generate recommendation
            rec_context = recommendation_engine.build_context_from_inference(
                response=response,
                input_data=sanitized_input,
                disease=disease,
                history_context=history_context,
            )
            recommendation_resp = recommendation_engine.generate_recommendation(rec_context)

            recommendation_json_str = json.dumps(recommendation_resp.model_dump())

            # 3. Create PredictionRecord
            record = PredictionRecord(
                user_id=str(user.id),
                disease=disease,
                model_name=response.model_used,
                model_version=response.artifact_version,
                prediction=response.predicted_class,
                risk_score=risk_score,
                risk_category=response.model_output,
                score_semantics=response.score_semantics,
                input_data=input_json_str,
                explanation=explanation_json_str,
                recommendation=recommendation_json_str,
                source="user_inference",
            )

            db.add(record)
            db.commit()
            db.refresh(record)

            # 4. Create RecommendationRecord
            recommendation_resp.prediction_record_id = record.id
            recommendation_service.save_recommendation(
                db=db,
                user_id=str(user.id),
                prediction_record_id=record.id,
                disease=disease,
                recommendation=recommendation_resp,
            )

            # Attach recommendation to response object
            response.recommendation = recommendation_resp

            logger.info(
                f"Saved prediction record '{record.id}' and recommendation for user '{user.id}' (Disease: {disease}, Model: {response.model_used})"
            )
            return record
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist prediction record for user '{user_id}': {e!s}", exc_info=True)
            return None



    def get_user_history(
        self,
        db: Session,
        user_id: str,
        disease: str | None = None,
        model_name: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[PredictionRecord], int]:
        """Retrieves paginated prediction records strictly belonging to the authenticated user."""
        query = db.query(PredictionRecord).filter(PredictionRecord.user_id == str(user_id))

        if disease:
            query = query.filter(PredictionRecord.disease == disease.strip().lower())
        if model_name:
            query = query.filter(PredictionRecord.model_name == model_name.strip())

        total = query.count()

        # Deterministic ordering: newest records first
        records = (
            query.order_by(desc(PredictionRecord.created_at), desc(PredictionRecord.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return records, total

    def get_prediction_detail(
        self,
        db: Session,
        prediction_id: str,
        user_id: str,
    ) -> PredictionRecord | None:
        """Retrieves an individual prediction record only if owned by the authenticated user."""
        record = (
            db.query(PredictionRecord)
            .filter(
                PredictionRecord.id == str(prediction_id),
                PredictionRecord.user_id == str(user_id),
            )
            .first()
        )
        return record

    def delete_prediction_record(
        self,
        db: Session,
        prediction_id: str,
        user_id: str,
    ) -> bool:
        """Deletes a prediction record owned by the authenticated user."""
        record = self.get_prediction_detail(db, prediction_id, user_id)
        if not record:
            return False

        try:
            db.delete(record)
            db.commit()
            logger.info(f"Deleted prediction record '{prediction_id}' for user '{user_id}'")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete prediction record '{prediction_id}': {e!s}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prediction history record.",
            )


prediction_service = PredictionService()
