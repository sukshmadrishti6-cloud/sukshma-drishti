"""Internal Pydantic Schemas for Recommendation Engine Processing."""
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class RecommendationCategory(str, Enum):
    """Categorization of individual recommendation actions."""
    PRIORITY_ACTION = "priority_action"
    MONITORING = "monitoring"
    PROFESSIONAL_FOLLOW_UP = "professional_follow_up"
    WHEN_TO_SEEK_CARE = "when_to_seek_care"


class RecommendationInputContext(BaseModel):
    """Contextual container passed to disease recommendation strategies."""
    disease: str
    predicted_class: int  # 0: Low Risk, 1: High Risk / Malignant / Positive
    score: float | None = None
    score_semantics: str | None = None  # calibrated_probability | ensemble_vote_probability | uncalibrated_quantum_decision_score
    risk_category: str | None = None
    input_data: dict[str, float] = Field(default_factory=dict)
    explanation: dict[str, Any] | None = None
    model_name: str
    model_version: str
    history_context: list[dict[str, Any]] = Field(default_factory=list)


class RecommendationRuleResult(BaseModel):
    """Result of evaluating an individual recommendation rule."""
    rule_id: str
    category: RecommendationCategory
    priority: int  # Lower integer = higher priority
    message: str
    factor_name: str | None = None
    factor_value: float | None = None
    factor_contribution: float | None = None
    factor_direction: str | None = None
