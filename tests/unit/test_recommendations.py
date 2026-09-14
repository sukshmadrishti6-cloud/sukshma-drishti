"""Unit tests for Phase I Recommendation Engine, Strategies, Score Semantics, and Safety Rules."""
import pytest
from app.recommendations.engine import recommendation_engine, RecommendationEngine
from app.recommendations.registry import recommendation_registry
from app.recommendations.schemas import RecommendationInputContext
from app.schemas.api import ExplanationContribution, ExplanationResult, InferenceResponse, AuditMetadata


def test_registry_strategy_resolution():
    """Verify registry resolves strategies for diabetes, cardiovascular, cancer, and aliases."""
    diabetes_strat = recommendation_registry.get_strategy("diabetes")
    assert diabetes_strat.disease_id == "diabetes"
    assert diabetes_strat.rule_set_version == "diabetes-v1.0.0"

    cardio_strat = recommendation_registry.get_strategy("cardio")
    assert cardio_strat.disease_id == "cardiovascular"
    assert cardio_strat.rule_set_version == "cardiovascular-v1.0.0"

    cancer_strat = recommendation_registry.get_strategy("oncology")
    assert cancer_strat.disease_id == "cancer"
    assert cancer_strat.rule_set_version == "cancer-v1.0.0"


def test_diabetes_positive_recommendation():
    """Test recommendation generation for positive Diabetes prediction."""
    ctx = RecommendationInputContext(
        disease="diabetes",
        predicted_class=1,
        score=0.82,
        score_semantics="calibrated_probability",
        risk_category="High Risk",
        input_data={
            "Pregnancies": 6,
            "Glucose": 168.0,
            "BloodPressure": 84.0,
            "SkinThickness": 32.0,
            "Insulin": 0.0,
            "BMI": 34.2,
            "DiabetesPedigreeFunction": 0.67,
            "Age": 48.0,
        },
        explanation={
            "contributions": [
                {"name": "Glucose", "value": 168.0, "contribution": 0.35, "normalized_weight": 0.35, "direction": "positive"},
                {"name": "BMI", "value": 34.2, "contribution": 0.25, "normalized_weight": 0.25, "direction": "positive"},
            ]
        },
        model_name="diabetes_rf_opt_v1",
        model_version="v2.0_optimized",
    )

    rec = recommendation_engine.generate_recommendation(ctx)

    assert rec.disease == "diabetes"
    assert rec.rule_set_version == "diabetes-v1.0.0"
    assert "calibrated model risk probability: 82.0%" in rec.summary
    assert len(rec.priority_actions) > 0
    assert any("Fasting blood glucose (168.0 mg/dL) is elevated" in act for act in rec.priority_actions)
    assert any("Body Mass Index (34.2 kg/m²) indicates obesity range" in act for act in rec.priority_actions)
    assert "These recommendations are generated deterministically" in rec.disclaimer
    assert "extreme thirst" in rec.when_to_seek_care
    assert len(rec.factors) > 0
    assert any(f.name == "Glucose" for f in rec.factors)


def test_quantum_score_semantics_handling():
    """Test quantum decision score semantics (uncalibrated_quantum_decision_score) handling.

    CRITICAL RULE: Quantum raw decision score MUST NOT be scaled (score * 100) or treated as a probability.
    """
    ctx = RecommendationInputContext(
        disease="diabetes",
        predicted_class=1,
        score=0.412,  # Raw SVM decision margin
        score_semantics="uncalibrated_quantum_decision_score",
        risk_category="Model-predicted positive class",
        input_data={"Glucose": 150.0, "BMI": 28.0, "Age": 45.0},
        model_name="qsvc_v3",
        model_version="v3.0",
    )

    rec = recommendation_engine.generate_recommendation(ctx)

    assert "quantum model decision margin classification" in rec.summary
    # Ensure raw margin 0.412 is NOT falsely formatted as probability percentage (e.g. 41.2%)
    assert "41.2%" not in rec.summary
    assert "calibrated model risk probability" not in rec.summary
    assert rec.rule_set_version == "diabetes-v1.0.0"


def test_cardiovascular_recommendation():
    """Test recommendation generation for Cardiovascular prediction."""
    ctx = RecommendationInputContext(
        disease="cardiovascular",
        predicted_class=1,
        score=0.75,
        score_semantics="calibrated_probability",
        risk_category="High Risk",
        input_data={
            "age": 60.0,
            "sex": 1.0,
            "cp": 2.0,
            "trestbps": 145.0,
            "chol": 255.0,
            "fbs": 1.0,
            "restecg": 0.0,
            "thalach": 140.0,
            "exang": 1.0,
            "oldpeak": 2.0,
            "slope": 1.0,
            "ca": 1.0,
            "thal": 2.0,
        },
        model_name="cardiovascular_extra_trees_opt_v1",
        model_version="v2.0_optimized",
    )

    rec = recommendation_engine.generate_recommendation(ctx)

    assert rec.disease == "cardiovascular"
    assert rec.rule_set_version == "cardiovascular-v1.0.0"
    assert any("Serum cholesterol (255.0 mg/dl) is high" in act for act in rec.priority_actions)
    assert any("Resting blood pressure (145.0 mm Hg) is elevated" in act for act in rec.priority_actions)
    assert "chest pain" in rec.when_to_seek_care.lower()


def test_cancer_recommendation():
    """Test recommendation generation for Breast Cancer cell mass prediction."""
    ctx = RecommendationInputContext(
        disease="cancer",
        predicted_class=1,
        score=0.96,
        score_semantics="calibrated_probability",
        risk_category="Malignant",
        input_data={
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "worst_concave_points": 0.265,
        },
        model_name="cancer_stacking_opt_v1",
        model_version="v2.0_optimized",
    )

    rec = recommendation_engine.generate_recommendation(ctx)

    assert rec.disease == "cancer"
    assert rec.rule_set_version == "cancer-v1.0.0"
    assert "cell morphometry patterns consistent with elevated risk" in rec.summary
    assert any("oncologist" in rec.professional_follow_up.lower() or "physician" in rec.professional_follow_up.lower() for _ in [1])
    assert "do not constitute a definitive cancer diagnosis" in rec.disclaimer.lower()


def test_history_context_evaluation():
    """Test history-aware context in recommendations."""
    history = [
        {"disease": "diabetes", "prediction": 1, "created_at": "2026-09-01T10:00:00Z"},
        {"disease": "diabetes", "prediction": 1, "created_at": "2026-09-05T10:00:00Z"},
    ]

    ctx = RecommendationInputContext(
        disease="diabetes",
        predicted_class=1,
        score=0.85,
        score_semantics="calibrated_probability",
        input_data={"Glucose": 150.0, "BMI": 31.0},
        model_name="diabetes_rf_opt_v1",
        model_version="v2.0_optimized",
        history_context=history,
    )

    rec = recommendation_engine.generate_recommendation(ctx)

    assert any("Repeated elevated model risk predictions have been observed" in m for m in rec.monitoring)
    # Ensure safety: no claim of worsening or disease progression
    assert "worsening" not in " ".join(rec.monitoring).lower()
    assert "progressing" not in " ".join(rec.monitoring).lower()


def test_recommendation_determinism():
    """Verify identical context produces exact same recommendation output (determinism)."""
    ctx = RecommendationInputContext(
        disease="diabetes",
        predicted_class=0,
        score=0.15,
        score_semantics="calibrated_probability",
        input_data={"Glucose": 95.0, "BMI": 22.5, "Age": 30.0},
        model_name="logistic_regression_v3",
        model_version="v3.0",
    )

    rec1 = recommendation_engine.generate_recommendation(ctx)
    rec2 = recommendation_engine.generate_recommendation(ctx)

    assert rec1.summary == rec2.summary
    assert rec1.priority_actions == rec2.priority_actions
    assert rec1.monitoring == rec2.monitoring
    assert rec1.rule_set_version == rec2.rule_set_version
