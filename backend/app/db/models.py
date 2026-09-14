"""SQLAlchemy Database Models for User Accounts and Application Storage."""
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.db.session import Base


class User(Base):
    """User Account Database Model."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), default="authenticated", nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    prediction_records = relationship(
        "PredictionRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(PredictionRecord.created_at)",
    )
    recommendation_records = relationship(
        "RecommendationRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(RecommendationRecord.created_at)",
    )

    def to_dict(self) -> dict:
        """Returns safe user attribute dictionary (password_hash explicitly excluded)."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PredictionRecord(Base):
    """Persistent Patient Disease Prediction Record Model."""

    __tablename__ = "prediction_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    disease = Column(String(50), nullable=False, default="diabetes", index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(30), nullable=False, default="v3.0", index=True)
    prediction = Column(Integer, nullable=False)  # 0: Low Risk, 1: High Risk
    risk_score = Column(Float, nullable=True)
    risk_category = Column(String(50), nullable=True)
    score_semantics = Column(String(50), nullable=True)
    input_data = Column(Text, nullable=False)  # Sanitized JSON string of biomedical features
    explanation = Column(Text, nullable=True)  # Serialized ExplanationResult JSON
    recommendation = Column(Text, nullable=True)  # Serialized RecommendationResponse JSON
    source = Column(String(50), nullable=False, default="user_inference")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="prediction_records")
    recommendation_record = relationship(
        "RecommendationRecord",
        back_populates="prediction_record",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_user_created_at", "user_id", "created_at"),
    )

    def get_input_dict(self) -> dict:
        """Parses input_data JSON text column into Python dictionary."""
        try:
            return json.loads(self.input_data) if self.input_data else {}
        except Exception:
            return {}

    def get_explanation_dict(self) -> dict | None:
        """Parses explanation JSON text column into Python dictionary if present."""
        try:
            return json.loads(self.explanation) if self.explanation else None
        except Exception:
            return None

    def get_recommendation_dict(self) -> dict | None:
        """Parses recommendation JSON text column into Python dictionary if present."""
        try:
            return json.loads(self.recommendation) if self.recommendation else None
        except Exception:
            return None

    def to_summary_dict(self) -> dict:
        """Lightweight summary dictionary for list endpoint responses."""
        return {
            "id": self.id,
            "disease": self.disease,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction": self.prediction,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "score_semantics": self.score_semantics,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
        }

    def to_detail_dict(self) -> dict:
        """Comprehensive detail dictionary including input payload, explanation, and recommendation."""
        summary = self.to_summary_dict()
        summary.update({
            "input_data": self.get_input_dict(),
            "explanation": self.get_explanation_dict(),
            "recommendation": self.get_recommendation_dict(),
            "source": self.source,
        })
        return summary


class RecommendationRecord(Base):
    """Persistent Patient Recommendation Model associated with a Prediction Record."""

    __tablename__ = "recommendations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    prediction_record_id = Column(
        String(36),
        ForeignKey("prediction_records.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    disease = Column(String(50), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    priority_actions = Column(Text, nullable=False)  # Serialized JSON list of priority actions
    monitoring = Column(Text, nullable=False)  # Serialized JSON list of monitoring suggestions
    professional_follow_up = Column(Text, nullable=False)
    when_to_seek_care = Column(Text, nullable=False)
    factors = Column(Text, nullable=True)  # Serialized JSON list of relevant factor dictionaries
    disclaimer = Column(Text, nullable=False)
    rule_set_version = Column(String(30), nullable=False, default="v1.0.0")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="recommendation_records")
    prediction_record = relationship("PredictionRecord", back_populates="recommendation_record")

    __table_args__ = (
        Index("idx_rec_user_created_at", "user_id", "created_at"),
    )

    def get_priority_actions_list(self) -> list[str]:
        """Parses priority_actions JSON string into Python list."""
        try:
            return json.loads(self.priority_actions) if self.priority_actions else []
        except Exception:
            return []

    def get_monitoring_list(self) -> list[str]:
        """Parses monitoring JSON string into Python list."""
        try:
            return json.loads(self.monitoring) if self.monitoring else []
        except Exception:
            return []

    def get_factors_list(self) -> list[dict]:
        """Parses factors JSON string into Python list of factor dictionaries."""
        try:
            return json.loads(self.factors) if self.factors else []
        except Exception:
            return []

    def to_dict(self) -> dict:
        """Returns structured recommendation dictionary."""
        return {
            "id": self.id,
            "prediction_record_id": self.prediction_record_id,
            "user_id": self.user_id,
            "disease": self.disease,
            "summary": self.summary,
            "priority_actions": self.get_priority_actions_list(),
            "monitoring": self.get_monitoring_list(),
            "professional_follow_up": self.professional_follow_up,
            "when_to_seek_care": self.when_to_seek_care,
            "factors": self.get_factors_list(),
            "disclaimer": self.disclaimer,
            "rule_set_version": self.rule_set_version,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
        }


