"""Recommendation Provider Abstraction Layer.

Defines the common interface for recommendation generation.
Currently implements RuleBasedRecommendationProvider.
MLRecommendationProvider is provided as a stub for future learned recommendation models when legitimate datasets exist.
"""
from abc import ABC, abstractmethod
from typing import Any
from app.recommendations.schemas import RecommendationInputContext
from app.schemas.api import RecommendationResponse


class BaseRecommendationProvider(ABC):
    """Abstract interface defining the recommendation provider contract."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Type tag identifying provider implementation ('rule_based' or 'ml_learned')."""
        pass

    @abstractmethod
    def generate_recommendation(
        self,
        context: RecommendationInputContext,
    ) -> RecommendationResponse:
        """Generates a structured recommendation response for a given prediction context."""
        pass


class MLRecommendationProvider(BaseRecommendationProvider):
    """Stub implementation for future ML-learned recommendation provider.

    NOTE: The SukshmaDrishti platform currently does NOT claim to use a separately trained ML recommendation model.
    This stub exists to enable future learned model integration without changing application architecture.
    """

    @property
    def provider_type(self) -> str:
        return "ml_learned"

    def generate_recommendation(
        self,
        context: RecommendationInputContext,
    ) -> RecommendationResponse:
        raise NotImplementedError(
            "ML-learned recommendation provider is not currently active. "
            "Recommendation engine is currently deterministic and rule-based."
        )
