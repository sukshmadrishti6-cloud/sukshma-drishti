"""Recommendation Strategy Registry mapping disease domains to rule strategies."""
import logging
from fastapi import HTTPException, status
from app.recommendations.base import BaseDiseaseRecommendationStrategy
from app.recommendations.rules.diabetes import DiabetesRecommendationStrategy
from app.recommendations.rules.cardiovascular import CardiovascularRecommendationStrategy
from app.recommendations.rules.cancer import CancerRecommendationStrategy

logger = logging.getLogger(__name__)

# Alias mapping matching backend disease registry
RECOMMANDATION_ALIAS_MAP: dict[str, str] = {
    "diabetes": "diabetes",
    "diabetes_mellitus": "diabetes",
    "dm": "diabetes",
    "cardiovascular": "cardiovascular",
    "cardio": "cardiovascular",
    "heart": "cardiovascular",
    "heart_disease": "cardiovascular",
    "cancer": "cancer",
    "oncology": "cancer",
}


class RecommendationRegistry:
    """Registry managing disease-specific recommendation strategies."""

    def __init__(self):
        self._strategies: dict[str, BaseDiseaseRecommendationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Registers default platform recommendation strategies."""
        self.register_strategy(DiabetesRecommendationStrategy())
        self.register_strategy(CardiovascularRecommendationStrategy())
        self.register_strategy(CancerRecommendationStrategy())

    def register_strategy(self, strategy: BaseDiseaseRecommendationStrategy) -> None:
        """Registers a disease recommendation strategy instance."""
        disease_id = strategy.disease_id.lower()
        self._strategies[disease_id] = strategy
        logger.info(
            f"Registered recommendation strategy for '{disease_id}' (Rule set: {strategy.rule_set_version})"
        )

    def resolve_disease_id(self, raw_identifier: str) -> str:
        """Resolves raw disease string or alias to canonical disease ID."""
        clean_id = raw_identifier.strip().lower()
        canonical_id = RECOMMANDATION_ALIAS_MAP.get(clean_id)
        if not canonical_id or canonical_id not in self._strategies:
            supported = sorted(list(self._strategies.keys()))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation strategy for disease '{raw_identifier}' not found. Supported strategies: {supported}",
            )
        return canonical_id

    def get_strategy(self, disease_identifier: str) -> BaseDiseaseRecommendationStrategy:
        """Resolves disease identifier and returns registered strategy instance."""
        canonical_id = self.resolve_disease_id(disease_identifier)
        return self._strategies[canonical_id]


recommendation_registry = RecommendationRegistry()
