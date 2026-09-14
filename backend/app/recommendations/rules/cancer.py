"""Breast Cancer Disease Recommendation Strategy (Rule Set Version: cancer-v1.0.0)."""
from app.recommendations.base import BaseDiseaseRecommendationStrategy, Rule
from app.recommendations.schemas import (
    RecommendationCategory,
    RecommendationInputContext,
)


class CancerRecommendationStrategy(BaseDiseaseRecommendationStrategy):
    """Deterministic, rule-based recommendation strategy for Breast Cancer cell morphometry assessments."""

    @property
    def disease_id(self) -> str:
        return "cancer"

    @property
    def rule_set_version(self) -> str:
        return "cancer-v1.0.0"

    @property
    def disclaimer(self) -> str:
        return (
            "These recommendations are generated deterministically based on statistical risk model outputs "
            "analyzing cell mass morphometry features (UCI Breast Cancer Wisconsin dataset). They are for "
            "informational and screening purposes only and do NOT constitute a definitive cancer diagnosis or clinical biopsy report."
        )

    @property
    def default_when_to_seek_care(self) -> str:
        return (
            "If you notice a new breast mass, skin dimpling, nipple inversion, unusual discharge, or persistent localized pain, "
            "consult a qualified healthcare professional promptly for clinical breast examination and diagnostic imaging."
        )

    def generate_summary(self, context: RecommendationInputContext) -> str:
        score_info = ""
        if context.score_semantics == "calibrated_probability" and context.score is not None:
            pct = round(context.score * 100, 1)
            score_info = f" (calibrated model risk probability: {pct}%)"
        elif context.score_semantics == "ensemble_vote_probability" and context.score is not None:
            pct = round(context.score * 100, 1)
            score_info = f" (ensemble consensus risk score: {pct}%)"
        elif context.score_semantics == "uncalibrated_quantum_decision_score":
            score_info = " (quantum model decision margin classification)"

        if context.predicted_class == 1:
            return (
                f"The assessment model detected cell morphometry patterns consistent with elevated risk{score_info}. "
                "Prompt clinical evaluation by a physician or breast health specialist for diagnostic imaging or biopsy confirmation is strongly recommended."
            )
        else:
            return (
                f"The assessment model detected cell morphometry patterns consistent with lower risk / benign characteristics{score_info}. "
                "Continuing routine age-appropriate clinical breast screening and self-awareness is recommended."
            )

    def generate_professional_follow_up(self, context: RecommendationInputContext) -> str:
        if context.predicted_class == 1:
            return (
                "Schedule a follow-up consultation with an oncologist, surgeon, or primary care physician to review "
                "diagnostic imaging (mammography, ultrasound) and discuss whether tissue biopsy evaluation is indicated."
            )
        return (
            "Follow standard clinical guidelines for age-appropriate screening mammograms and routine clinical breast exams."
        )

    @property
    def rules(self) -> list[Rule]:
        return [
            # Elevated Risk Class Action Rule
            Rule(
                rule_id="CA-ACT-POS-CLASS",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=1,
                condition=lambda ctx: ctx.predicted_class == 1,
                message_builder=lambda ctx: (
                    "Model prediction indicates elevated risk cell morphometry features; consult a medical specialist promptly for diagnostic verification."
                ),
            ),
            # Diagnostic Imaging Action Rule
            Rule(
                rule_id="CA-ACT-IMAGING-REFERRAL",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=2,
                condition=lambda ctx: ctx.predicted_class == 1,
                message_builder=lambda ctx: (
                    "Request clinical breast examination and diagnostic imaging (digital mammography or targeted ultrasound) from your healthcare provider."
                ),
            ),
            # Benign Routine Screening Rule
            Rule(
                rule_id="CA-MON-ROUTINE-SCREENING",
                category=RecommendationCategory.MONITORING,
                priority=5,
                condition=lambda ctx: ctx.predicted_class == 0,
                message_builder=lambda ctx: (
                    "Maintain routine annual clinical breast examinations and age-recommended screening mammography."
                ),
            ),
            # Cell Morphometry Feature Factor Rule
            Rule(
                rule_id="CA-MON-MORPHOMETRY-FACTORS",
                category=RecommendationCategory.MONITORING,
                priority=6,
                condition=lambda ctx: bool(ctx.explanation and ctx.explanation.get("contributions")),
                message_builder=lambda ctx: (
                    "Model prediction was influenced by specific nuclear morphometry parameters (e.g. concave points, perimeter, area). "
                    "Share complete diagnostic findings with your physician."
                ),
            ),
        ]
