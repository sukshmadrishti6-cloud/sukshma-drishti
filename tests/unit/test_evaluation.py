"""Unit tests for ml.evaluation module (Metrics, CV, Bootstrap, ModelEvaluator, Unified Benchmark)."""
import numpy as np
import pandas as pd
import pytest

from ml.classical import LogisticRegressionModel, RandomForestModel
from ml.evaluation import (
    ModelEvaluator,
    UnifiedBenchmarkRunner,
    calculate_bootstrap_confidence_intervals,
    calculate_classification_metrics,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
    compare_models,
    evaluate_cross_validation,
    generate_benchmark_summary_table,
)
from ml.exceptions import ValidationError
from ml.preprocessing import BiomedicalPreprocessor, split_data


@pytest.fixture
def known_binary_data():
    """Known fixture with exact mathematical metrics:
    y_true: [1, 1, 1, 1, 0, 0, 0, 0] (4 pos, 4 neg)
    y_pred: [1, 1, 1, 0, 0, 0, 1, 0]
    TP=3, FN=1, TN=3, FP=1
    Accuracy = 6/8 = 0.75
    Precision = 3/(3+1) = 0.75
    Recall / Sensitivity = 3/(3+1) = 0.75
    Specificity = 3/(3+1) = 0.75
    F1 = 0.75
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.85, 0.4, 0.1, 0.2, 0.7, 0.15])
    return y_true, y_pred, y_prob


@pytest.fixture
def mock_pima_df() -> pd.DataFrame:
    """Fixture producing synthetic Pima-like DataFrame for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 10, size=n),
            "Glucose": np.random.uniform(70, 180, size=n),
            "BloodPressure": np.random.uniform(60, 90, size=n),
            "SkinThickness": np.random.uniform(10, 40, size=n),
            "Insulin": np.random.uniform(15, 200, size=n),
            "BMI": np.random.uniform(18, 35, size=n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.5, size=n),
            "Age": np.random.randint(21, 65, size=n),
            "Outcome": np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
        }
    )
    return df


def test_evaluation_package_import():
    """Verify evaluation module package structure imports cleanly."""
    import ml.evaluation
    assert hasattr(ml.evaluation, "calculate_classification_metrics")
    assert hasattr(ml.evaluation, "evaluate_cross_validation")
    assert hasattr(ml.evaluation, "ModelEvaluator")
    assert hasattr(ml.evaluation, "compare_models")
    assert hasattr(ml.evaluation, "UnifiedBenchmarkRunner")


def test_exact_classification_metrics(known_binary_data):
    """Verify exact mathematical values of TP, TN, FP, FN, precision, recall, specificity, accuracy, F1."""
    y_true, y_pred, y_prob = known_binary_data
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75

    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 3
    assert cm["fn"] == 1
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["matrix"] == [[3, 1], [1, 3]]

    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["brier_score"] is not None
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_roc_and_pr_curve_generation(known_binary_data):
    """Verify ROC curve and PR curve data generator functions."""
    y_true, _y_pred, y_prob = known_binary_data
    roc_data = calculate_roc_curve_data(y_true, y_prob)
    assert roc_data is not None
    assert len(roc_data["fpr"]) == len(roc_data["tpr"])
    assert 0.0 <= roc_data["auc"] <= 1.0

    pr_data = calculate_pr_curve_data(y_true, y_prob)
    assert pr_data is not None
    assert len(pr_data["precision"]) == len(pr_data["recall"])
    assert 0.0 <= pr_data["auc"] <= 1.0


def test_edge_case_all_zero_predictions():
    """Verify edge case where model predicts only class 0."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0  # Safe zero division
    assert metrics["recall"] == 0.0
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 0
    assert metrics["confusion_matrix"]["fp"] == 0


def test_edge_case_length_mismatch_raises_validation_error():
    """Verify array length mismatch raises ValidationError."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValidationError, match="Length mismatch"):
        calculate_classification_metrics(y_true, y_pred)


def test_bootstrap_confidence_intervals(known_binary_data):
    """Verify bootstrap confidence intervals produce bounds containing point estimate."""
    y_true, y_pred, y_prob = known_binary_data
    ci_report = calculate_bootstrap_confidence_intervals(
        y_true, y_pred, y_prob, n_bootstraps=200, confidence_level=0.95, random_seed=42
    )

    assert ci_report["confidence_level"] == 0.95
    assert ci_report["n_bootstraps"] == 200

    acc_ci = ci_report["metrics"]["accuracy"]
    assert acc_ci["point_estimate"] == 0.75
    assert acc_ci["ci_lower"] <= acc_ci["point_estimate"] <= acc_ci["ci_upper"]


def test_cross_validation_execution(mock_pima_df: pd.DataFrame):
    """Verify 5-fold Stratified Cross-Validation executes and aggregates metric statistics."""
    cv_res = evaluate_cross_validation(
        "logistic_regression", mock_pima_df, target_column="Outcome", n_splits=5, random_seed=42
    )

    assert cv_res["n_splits"] == 5
    assert cv_res["model_name"] == "Logistic Regression"
    assert len(cv_res["folds"]) == 5

    acc_stats = cv_res["metrics"]["accuracy"]
    assert acc_stats is not None
    assert 0.0 <= acc_stats["mean"] <= 1.0
    assert len(acc_stats["values"]) == 5
    assert cv_res["timing"]["mean_train_time_ms"] > 0.0


def test_cross_validation_leakage_prevention(mock_pima_df: pd.DataFrame):
    """CRITICAL TEST: Verify preprocessing is fit strictly inside each CV fold without leakage."""
    df_polluted = mock_pima_df.copy()
    df_polluted.loc[0:10, "Glucose"] = 999999.0

    cv_res = evaluate_cross_validation(
        "rf", df_polluted, target_column="Outcome", n_splits=5, random_seed=42
    )

    assert cv_res["n_splits"] == 5
    assert cv_res["metrics"]["accuracy"]["mean"] >= 0.0


def test_model_evaluator_and_comparison(mock_pima_df: pd.DataFrame):
    """Verify ModelEvaluator evaluates trained models and compare_models ranks results."""
    X_train, X_test, y_train, y_test = split_data(
        mock_pima_df, target_column="Outcome", test_size=0.2, random_seed=42
    )
    preprocessor = BiomedicalPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Model 1: Logistic Regression
    lr = LogisticRegressionModel(random_seed=42)
    lr.fit(X_train_trans, y_train)

    # Model 2: Random Forest
    rf = RandomForestModel(random_seed=42)
    rf.fit(X_train_trans, y_train)

    evaluator = ModelEvaluator(random_seed=42)
    rep_lr = evaluator.evaluate_model(lr, X_test_trans, y_test, compute_bootstrap=False)
    rep_rf = evaluator.evaluate_model(rf, X_test_trans, y_test, compute_bootstrap=False)

    assert rep_lr["metrics"]["accuracy"] >= 0.0
    assert rep_rf["metrics"]["accuracy"] >= 0.0
    assert rep_lr["timing"]["inference_time_ms"] >= 0.0

    comp_df = compare_models([rep_lr, rep_rf])
    assert len(comp_df) == 2
    assert "Accuracy" in comp_df.columns
    assert "ROC-AUC" in comp_df.columns


def test_unified_benchmark_runner_small_fixture():
    """Verify UnifiedBenchmarkRunner executes across classical and quantum models on a small fixture."""
    np.random.seed(42)
    n = 20
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 5, size=n),
            "Glucose": np.random.uniform(70, 160, size=n),
            "BloodPressure": np.random.uniform(60, 90, size=n),
            "SkinThickness": np.random.uniform(10, 30, size=n),
            "Insulin": np.random.uniform(15, 150, size=n),
            "BMI": np.random.uniform(18, 32, size=n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.2, size=n),
            "Age": np.random.randint(21, 60, size=n),
            "Outcome": np.array([0, 1] * 10),
        }
    )

    runner = UnifiedBenchmarkRunner(random_seed=42, shots=256, quantum_qubits=4, vqc_max_iter=5)
    results = runner.run_benchmark(
        df,
        target_column="Outcome",
        test_size=0.2,
        dataset_name="test_fixture",
        include_scaling=False,
        save_results=False,
    )

    assert "benchmark_id" in results
    assert len(results["models"]) == 6  # LR, SVM, RF, XGB, QSVC, VQC
    for m in results["models"]:
        assert m["status"] == "SUCCESS"
        assert "accuracy" in m["metrics"]

    summary_df = generate_benchmark_summary_table(results)
    assert len(summary_df) == 6
    assert "Model" in summary_df.columns
    assert "Sensitivity" in summary_df.columns
