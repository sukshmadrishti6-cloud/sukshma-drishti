"""Recommendation Engine processing disease prediction results into structured, explainable guidance."""
import logging
from typing import Any
from app.recommendations.provider import BaseRecommendationProvider
from app.recommendations.registry import recommendation_registry, RecommendationRegistry
from app.recommendations.schemas import (
    RecommendationCategory,
    RecommendationInputContext,
    RecommendationRuleResult,
)
from app.schemas.api import (
    ExplanationResult,
    InferenceResponse,
    RecommendationFactor,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)


class RecommendationEngine(BaseRecommendationProvider):
    """Deterministic, rule-based recommendation engine for multi-disease platform."""

    def __init__(self, registry: RecommendationRegistry | None = None):
        self.registry = registry or recommendation_registry

    @property
    def provider_type(self) -> str:
        return "rule_based"

    def build_context_from_inference(
        self,
        response: InferenceResponse,
        input_data: dict[str, float],
        disease: str | None = None,
        history_context: list[dict[str, Any]] | None = None,
    ) -> RecommendationInputContext:
        """Helper building RecommendationInputContext from InferenceResponse and payload."""
        target_disease = disease or response.disease or "diabetes"
        score = response.probability if response.probability is not None else response.decision_score

        explanation_dict = response.explanation.model_dump() if response.explanation else None

        return RecommendationInputContext(
            disease=target_disease,
            predicted_class=response.predicted_class,
            score=score,
            score_semantics=response.score_semantics,
            risk_category=response.model_output,
            input_data=input_data,
            explanation=explanation_dict,
            model_name=response.model_used,
            model_version=response.artifact_version,
            history_context=history_context or [],
        )

    def extract_factors_from_explanation(
        self,
        context: RecommendationInputContext,
        rule_factors: list[dict[str, Any]],
    ) -> list[RecommendationFactor]:
        """Extracts and formats factor contributions using non-causal language."""
        factors: list[RecommendationFactor] = []
        seen_names: set[str] = set()

        # 1. Include factor info extracted directly from triggered rules
        for rf in rule_factors:
            name = rf.get("name")
            if name and name not in seen_names:
                seen_names.add(name)
                val = rf.get("value")
                factors.append(
                    RecommendationFactor(
                        name=name,
                        value=float(val) if val is not None else None,
                        contribution=rf.get("contribution"),
                        direction=rf.get("direction"),
                        summary=f"{name} (value: {val}) contributed to the model's risk assessment.",
                    )
                )

        # 2. Include top contributing factors from explanation result
        if context.explanation and isinstance(context.explanation.get("contributions"), list):
            contributions = context.explanation["contributions"]
            # Sort by absolute contribution weight descending
            sorted_contribs = sorted(
                contributions,
                key=lambda c: abs(c.get("normalized_weight", c.get("contribution", 0.0))),
                reverse=True,
            )
            for c in sorted_contribs[:4]:  # Top 4 factors
                name = c.get("name")
                if name and name not in seen_names:
                    seen_names.add(name)
                    val = c.get("value")
                    weight = c.get("contribution", c.get("normalized_weight", 0.0))
                    direction = c.get("direction", "positive" if weight > 0 else "neutral")
                    factors.append(
                        RecommendationFactor(
                            name=name,
                            value=float(val) if val is not None else None,
                            contribution=float(weight),
                            direction=direction,
                            summary=f"{name} contributed to the model's prediction.",
                        )
                    )

        return factors

    def evaluate_history_context(self, context: RecommendationInputContext) -> str | None:
        """Evaluates past user assessments for the same disease and formats non-diagnostic trend context."""
        if not context.history_context:
            return None

        same_disease_records = [
            r for r in context.history_context
            if r.get("disease", "").lower() == context.disease.lower()
        ]

        if not same_disease_records:
            return None

        total_past = len(same_disease_records)
        past_positive = sum(1 for r in same_disease_records if r.get("prediction") == 1)

        if past_positive > 0 and context.predicted_class == 1:
            return (
                f"Historical Context: Repeated elevated model risk predictions have been observed across "
                f"{past_positive + 1} of your recent {total_past + 1} health assessments for this disease domain. "
                "Consistent elevated risk predictions suggest sharing your assessment history with your physician."
            )
        elif past_positive > 0 and context.predicted_class == 0:
            return (
                f"Historical Context: Current assessment indicates lower risk, whereas elevated risk predictions were "
                f"observed in {past_positive} of your prior {total_past} assessments. Discussing physiological trends with a provider is advised."
            )
        elif past_positive == 0 and context.predicted_class == 1:
            return (
                "Historical Context: This is the first elevated risk prediction observed across your recent health assessments. "
                "Consider discussing this new finding during your next medical checkup."
            )

        return None

    def generate_recommendation(
        self,
        context: RecommendationInputContext,
    ) -> RecommendationResponse:
        """Main entry point generating structured RecommendationResponse from RecommendationInputContext."""
        strategy = self.registry.get_strategy(context.disease)

        rule_results: list[RecommendationRuleResult] = []
        rule_factors: list[dict[str, Any]] = []

        # Evaluate rules
        for rule in strategy.rules:
            res = rule.evaluate(context)
            if res:
                rule_results.append(res)
                if res.factor_name:
                    rule_factors.append({
                        "name": res.factor_name,
                        "value": res.factor_value,
                        "contribution": res.factor_contribution,
                        "direction": res.factor_direction,
                    })

        # Separate actions and monitoring items
        priority_actions_raw = [
            r for r in rule_results if r.category == RecommendationCategory.PRIORITY_ACTION
        ]
        monitoring_raw = [
            r for r in rule_results if r.category == RecommendationCategory.MONITORING
        ]

        # Sort by priority integer (ascending)
        priority_actions_raw.sort(key=lambda r: r.priority)
        monitoring_raw.sort(key=lambda r: r.priority)

        # Deduplicate while preserving order
        priority_actions: list[str] = []
        for r in priority_actions_raw:
            if r.message not in priority_actions:
                priority_actions.append(r.message)

        monitoring: list[str] = []
        for r in monitoring_raw:
            if r.message not in monitoring:
                monitoring.append(r.message)

        # Evaluate history context
        history_note = self.evaluate_history_context(context)
        if history_note:
            monitoring.append(history_note)

        # Generate summary & professional follow-up
        summary = strategy.generate_summary(context)
        professional_follow_up = strategy.generate_professional_follow_up(context)

        # Extract factor contributions
        factors = self.extract_factors_from_explanation(context, rule_factors)

        return RecommendationResponse(
            disease=strategy.disease_id,
            summary=summary,
            priority_actions=priority_actions,
            monitoring=monitoring,
            professional_follow_up=professional_follow_up,
            when_to_seek_care=strategy.default_when_to_seek_care,
            factors=factors,
            disclaimer=strategy.disclaimer,
            rule_set_version=strategy.rule_set_version,
        )


recommendation_engine = RecommendationEngine()
