"""Cardiovascular Disease Recommendation Strategy (Rule Set Version: cardiovascular-v1.0.0)."""
from app.recommendations.base import BaseDiseaseRecommendationStrategy, Rule
from app.recommendations.schemas import (
    RecommendationCategory,
    RecommendationInputContext,
)


class CardiovascularRecommendationStrategy(BaseDiseaseRecommendationStrategy):
    """Deterministic, rule-based recommendation strategy for Cardiovascular risk assessments."""

    @property
    def disease_id(self) -> str:
        return "cardiovascular"

    @property
    def rule_set_version(self) -> str:
        return "cardiovascular-v1.0.0"

    @property
    def disclaimer(self) -> str:
        return (
            "These recommendations are generated deterministically based on statistical risk model outputs "
            "and clinical guidelines for informational and screening purposes only. They do not constitute a "
            "medical diagnosis, emergency triage, or clinical treatment plan."
        )

    @property
    def default_when_to_seek_care(self) -> str:
        return (
            "If you experience sudden chest pain, pressure, shortness of breath, pain radiating to the jaw/arm, "
            "or severe lightheadedness, seek immediate emergency medical care."
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
                f"The assessment model indicates elevated cardiovascular risk markers{score_info}. "
                "Consulting a cardiologist or primary care physician for comprehensive cardiovascular evaluation is recommended."
            )
        else:
            return (
                f"The assessment model indicates lower predicted cardiovascular risk{score_info}. "
                "Maintaining heart-healthy lifestyle habits and routine blood pressure/lipid tracking is recommended."
            )

    def generate_professional_follow_up(self, context: RecommendationInputContext) -> str:
        if context.predicted_class == 1:
            return (
                "Schedule a cardiovascular risk assessment with a physician or cardiologist to review lipid panels, "
                "blood pressure controls, and ECG or stress testing if indicated."
            )
        return (
            "Continue routine cardiovascular monitoring during annual health checkups with your primary care provider."
        )

    @property
    def rules(self) -> list[Rule]:
        return [
            # High Cholesterol Action Rule
            Rule(
                rule_id="CV-ACT-CHOL-HIGH",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=1,
                condition=lambda ctx: ctx.input_data.get("chol", 0) >= 240,
                message_builder=lambda ctx: (
                    f"Serum cholesterol ({ctx.input_data.get('chol')} mg/dl) is high. "
                    "Discuss full lipid panel evaluation, dietary modifications, and lipid management with a physician."
                ),
                factor_extractor=lambda ctx: {
                    "name": "chol",
                    "value": ctx.input_data.get("chol"),
                    "summary": "Elevated serum cholesterol",
                },
            ),
            # High Blood Pressure Action Rule
            Rule(
                rule_id="CV-ACT-BP-HIGH",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=2,
                condition=lambda ctx: ctx.input_data.get("trestbps", 0) >= 140,
                message_builder=lambda ctx: (
                    f"Resting blood pressure ({ctx.input_data.get('trestbps')} mm Hg) is elevated (Stage 1/2 Hypertension range). "
                    "Consult a healthcare provider for blood pressure evaluation and lifestyle/treatment planning."
                ),
                factor_extractor=lambda ctx: {
                    "name": "trestbps",
                    "value": ctx.input_data.get("trestbps"),
                    "summary": "Elevated resting blood pressure",
                },
            ),
            # Chest Pain / Angina Action Rule
            Rule(
                rule_id="CV-ACT-ANGINA",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=3,
                condition=lambda ctx: ctx.input_data.get("exang", 0) == 1 or ctx.input_data.get("cp", 0) > 0,
                message_builder=lambda ctx: (
                    "Reported chest discomfort or exercise-induced angina symptoms warrant prompt medical review and cardiac evaluation."
                ),
                factor_extractor=lambda ctx: {
                    "name": "exang/cp",
                    "value": ctx.input_data.get("exang", ctx.input_data.get("cp")),
                    "summary": "Reported angina or chest pain symptoms",
                },
            ),
            # Positive Class Follow-up Rule
            Rule(
                rule_id="CV-ACT-POS-CLASS",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=4,
                condition=lambda ctx: ctx.predicted_class == 1,
                message_builder=lambda ctx: (
                    "Model prediction indicates elevated cardiovascular risk markers; consult a healthcare provider for comprehensive screening."
                ),
            ),
            # Borderline Cholesterol Monitoring Rule
            Rule(
                rule_id="CV-MON-CHOL-BORDERLINE",
                category=RecommendationCategory.MONITORING,
                priority=5,
                condition=lambda ctx: 200 <= ctx.input_data.get("chol", 0) < 240,
                message_builder=lambda ctx: (
                    f"Serum cholesterol ({ctx.input_data.get('chol')} mg/dl) is in the borderline high range. "
                    "Adopt a heart-healthy Mediterranean or low-saturated-fat diet."
                ),
                factor_extractor=lambda ctx: {
                    "name": "chol",
                    "value": ctx.input_data.get("chol"),
                    "summary": "Borderline serum cholesterol level",
                },
            ),
            # Prehypertension Monitoring Rule
            Rule(
                rule_id="CV-MON-BP-PREHYPERTENSION",
                category=RecommendationCategory.MONITORING,
                priority=6,
                condition=lambda ctx: 120 <= ctx.input_data.get("trestbps", 0) < 140,
                message_builder=lambda ctx: (
                    f"Resting blood pressure ({ctx.input_data.get('trestbps')} mm Hg) is in the prehypertension range. "
                    "Monitor blood pressure regularly and practice dietary sodium restriction."
                ),
                factor_extractor=lambda ctx: {
                    "name": "trestbps",
                    "value": ctx.input_data.get("trestbps"),
                    "summary": "Prehypertension blood pressure range",
                },
            ),
            # Fasting Blood Sugar Monitoring Rule
            Rule(
                rule_id="CV-MON-FBS",
                category=RecommendationCategory.MONITORING,
                priority=7,
                condition=lambda ctx: ctx.input_data.get("fbs", 0) == 1,
                message_builder=lambda ctx: (
                    "Fasting blood sugar > 120 mg/dl is a metabolic risk factor for cardiovascular disease. Discuss glycemic control with your physician."
                ),
                factor_extractor=lambda ctx: {
                    "name": "fbs",
                    "value": 1.0,
                    "summary": "Elevated fasting blood sugar (> 120 mg/dl)",
                },
            ),
            # General Health Maintenance Rule (Low Risk)
            Rule(
                rule_id="CV-MON-MAINTENANCE",
                category=RecommendationCategory.MONITORING,
                priority=10,
                condition=lambda ctx: ctx.predicted_class == 0,
                message_builder=lambda ctx: (
                    "Maintain regular aerobic physical activity (150 mins/week), heart-healthy dietary habits, and routine lipid monitoring."
                ),
            ),
        ]
