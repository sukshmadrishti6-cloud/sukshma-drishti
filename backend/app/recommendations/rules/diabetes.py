"""Diabetes Disease Recommendation Strategy (Rule Set Version: diabetes-v1.0.0)."""
from app.recommendations.base import BaseDiseaseRecommendationStrategy, Rule
from app.recommendations.schemas import (
    RecommendationCategory,
    RecommendationInputContext,
)


class DiabetesRecommendationStrategy(BaseDiseaseRecommendationStrategy):
    """Deterministic, rule-based recommendation strategy for Diabetes risk assessments."""

    @property
    def disease_id(self) -> str:
        return "diabetes"

    @property
    def rule_set_version(self) -> str:
        return "diabetes-v1.0.0"

    @property
    def disclaimer(self) -> str:
        return (
            "These recommendations are generated deterministically based on statistical risk model outputs "
            "and clinical guidelines for informational and screening purposes only. They do not constitute a "
            "medical diagnosis, clinical treatment plan, or prescription."
        )

    @property
    def default_when_to_seek_care(self) -> str:
        return (
            "If you experience symptoms of extreme thirst, frequent urination, unexplained weight loss, fatigue, "
            "or severe dizziness, consult a qualified healthcare professional promptly."
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
                f"The assessment model indicates elevated predicted diabetes risk{score_info}. "
                "Consulting a healthcare provider for comprehensive clinical screening and HbA1c testing is recommended."
            )
        else:
            return (
                f"The assessment model indicates lower predicted diabetes risk{score_info}. "
                "Maintaining healthy lifestyle habits and routine annual health screenings is recommended."
            )

    def generate_professional_follow_up(self, context: RecommendationInputContext) -> str:
        if context.predicted_class == 1:
            return (
                "Schedule a appointment with a primary care physician or endocrinologist to review blood glucose parameters, "
                "HbA1c tests, and personalized metabolic risk management."
            )
        return (
            "Continue routine annual wellness exams with a primary care provider and monitor key physiological metrics."
        )

    @property
    def rules(self) -> list[Rule]:
        return [
            # High Glucose Action Rule
            Rule(
                rule_id="DM-ACT-GLUCOSE-HIGH",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=1,
                condition=lambda ctx: ctx.input_data.get("Glucose", 0) >= 140,
                message_builder=lambda ctx: (
                    f"Fasting blood glucose ({ctx.input_data.get('Glucose')} mg/dL) is elevated. "
                    "Discuss HbA1c and oral glucose tolerance evaluation with a physician."
                ),
                factor_extractor=lambda ctx: {
                    "name": "Glucose",
                    "value": ctx.input_data.get("Glucose"),
                    "summary": "Elevated fasting blood glucose level",
                },
            ),
            # Elevated BMI Priority Action Rule
            Rule(
                rule_id="DM-ACT-BMI-HIGH",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=2,
                condition=lambda ctx: ctx.input_data.get("BMI", 0) >= 30.0,
                message_builder=lambda ctx: (
                    f"Body Mass Index ({ctx.input_data.get('BMI')} kg/m²) indicates obesity range. "
                    "Discuss weight management strategies and dietary modifications with a healthcare provider."
                ),
                factor_extractor=lambda ctx: {
                    "name": "BMI",
                    "value": ctx.input_data.get("BMI"),
                    "summary": "Elevated Body Mass Index",
                },
            ),
            # Positive Class Follow-up Rule
            Rule(
                rule_id="DM-ACT-POS-CLASS",
                category=RecommendationCategory.PRIORITY_ACTION,
                priority=3,
                condition=lambda ctx: ctx.predicted_class == 1,
                message_builder=lambda ctx: (
                    "Model prediction indicates elevated metabolic risk; schedule a clinical consultation for formal diagnostic evaluation."
                ),
            ),
            # Moderate BMI Monitoring Rule
            Rule(
                rule_id="DM-MON-BMI-OVERWEIGHT",
                category=RecommendationCategory.MONITORING,
                priority=4,
                condition=lambda ctx: 25.0 <= ctx.input_data.get("BMI", 0) < 30.0,
                message_builder=lambda ctx: (
                    f"Body Mass Index ({ctx.input_data.get('BMI')} kg/m²) is in the overweight range. "
                    "Incorporate regular physical activity and balanced nutritional habits."
                ),
                factor_extractor=lambda ctx: {
                    "name": "BMI",
                    "value": ctx.input_data.get("BMI"),
                    "summary": "Overweight BMI range",
                },
            ),
            # Blood Pressure Monitoring Rule
            Rule(
                rule_id="DM-MON-BP-ELEVATED",
                category=RecommendationCategory.MONITORING,
                priority=5,
                condition=lambda ctx: ctx.input_data.get("BloodPressure", 0) >= 80,
                message_builder=lambda ctx: (
                    f"Resting blood pressure ({ctx.input_data.get('BloodPressure')} mm Hg) is elevated. "
                    "Track resting blood pressure regularly and discuss cardiovascular risk during routine visits."
                ),
                factor_extractor=lambda ctx: {
                    "name": "BloodPressure",
                    "value": ctx.input_data.get("BloodPressure"),
                    "summary": "Elevated resting blood pressure",
                },
            ),
            # Glucose Routine Monitoring Rule
            Rule(
                rule_id="DM-MON-GLUCOSE-ROUTINE",
                category=RecommendationCategory.MONITORING,
                priority=6,
                condition=lambda ctx: 100 <= ctx.input_data.get("Glucose", 0) < 140,
                message_builder=lambda ctx: (
                    f"Fasting blood glucose ({ctx.input_data.get('Glucose')} mg/dL) is in the pre-diabetic range. "
                    "Monitor blood glucose levels periodically and reduce refined sugar intake."
                ),
                factor_extractor=lambda ctx: {
                    "name": "Glucose",
                    "value": ctx.input_data.get("Glucose"),
                    "summary": "Pre-diabetic blood glucose range",
                },
            ),
            # Age Screening Rule
            Rule(
                rule_id="DM-MON-AGE-SCREENING",
                category=RecommendationCategory.MONITORING,
                priority=7,
                condition=lambda ctx: ctx.input_data.get("Age", 0) >= 45,
                message_builder=lambda ctx: (
                    f"Age ({int(ctx.input_data.get('Age'))} years) meets clinical recommendations for annual diabetes screening."
                ),
                factor_extractor=lambda ctx: {
                    "name": "Age",
                    "value": ctx.input_data.get("Age"),
                    "summary": "Age-appropriate metabolic screening guideline",
                },
            ),
            # General Health Maintenance Rule (Low Risk)
            Rule(
                rule_id="DM-MON-MAINTENANCE",
                category=RecommendationCategory.MONITORING,
                priority=10,
                condition=lambda ctx: ctx.predicted_class == 0,
                message_builder=lambda ctx: (
                    "Maintain routine annual physiological screenings, balanced nutrition, and consistent physical exercise."
                ),
            ),
        ]
