"""Abstract base interface and rule primitive definitions for disease recommendation strategies."""
from abc import ABC, abstractmethod
from typing import Callable
from app.recommendations.schemas import (
    RecommendationCategory,
    RecommendationInputContext,
    RecommendationRuleResult,
)


class Rule:
    """Reusable deterministic recommendation rule primitive."""

    def __init__(
        self,
        rule_id: str,
        category: RecommendationCategory,
        priority: int,
        message_builder: Callable[[RecommendationInputContext], str | None],
        condition: Callable[[RecommendationInputContext], bool],
        factor_extractor: Callable[[RecommendationInputContext], dict | None] | None = None,
    ):
        self.rule_id = rule_id
        self.category = category
        self.priority = priority
        self.message_builder = message_builder
        self.condition = condition
        self.factor_extractor = factor_extractor

    def evaluate(self, context: RecommendationInputContext) -> RecommendationRuleResult | None:
        """Evaluates rule against context; returns RecommendationRuleResult if condition passes, else None."""
        try:
            if not self.condition(context):
                return None

            message = self.message_builder(context)
            if not message:
                return None

            factor_info = self.factor_extractor(context) if self.factor_extractor else None

            return RecommendationRuleResult(
                rule_id=self.rule_id,
                category=self.category,
                priority=self.priority,
                message=message,
                factor_name=factor_info.get("name") if factor_info else None,
                factor_value=factor_info.get("value") if factor_info else None,
                factor_contribution=factor_info.get("contribution") if factor_info else None,
                factor_direction=factor_info.get("direction") if factor_info else None,
            )
        except Exception:
            return None


class BaseDiseaseRecommendationStrategy(ABC):
    """Abstract base strategy class for disease-specific recommendation rules."""

    @property
    @abstractmethod
    def disease_id(self) -> str:
        """Canonical disease identifier (e.g., 'diabetes', 'cardiovascular', 'cancer')."""
        pass

    @property
    @abstractmethod
    def rule_set_version(self) -> str:
        """Version tag for rule set auditability (e.g., 'diabetes-v1.0.0')."""
        pass

    @property
    @abstractmethod
    def disclaimer(self) -> str:
        """Disease-specific medical disclaimer."""
        pass

    @property
    @abstractmethod
    def default_when_to_seek_care(self) -> str:
        """Default guidance on when to seek urgent professional care."""
        pass

    @property
    @abstractmethod
    def rules(self) -> list[Rule]:
        """List of deterministic rules registered for this disease strategy."""
        pass

    @abstractmethod
    def generate_summary(self, context: RecommendationInputContext) -> str:
        """Generates concise overall recommendation summary based on prediction output and score semantics."""
        pass

    @abstractmethod
    def generate_professional_follow_up(self, context: RecommendationInputContext) -> str:
        """Generates tailored professional follow-up guidance."""
        pass
