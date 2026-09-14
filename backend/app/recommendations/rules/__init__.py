"""Disease-specific recommendation rule strategies."""
from app.recommendations.rules.diabetes import DiabetesRecommendationStrategy
from app.recommendations.rules.cardiovascular import CardiovascularRecommendationStrategy
from app.recommendations.rules.cancer import CancerRecommendationStrategy

__all__ = [
    "DiabetesRecommendationStrategy",
    "CardiovascularRecommendationStrategy",
    "CancerRecommendationStrategy",
]
