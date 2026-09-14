"""Model evaluation metrics, cross-validation runners, and benchmark reporting."""
from ml.evaluation.benchmark import UnifiedBenchmarkRunner, generate_benchmark_summary_table
from ml.evaluation.bootstrap import calculate_bootstrap_confidence_intervals
from ml.evaluation.cross_validation import evaluate_cross_validation
from ml.evaluation.evaluator import ModelEvaluator, compare_models
from ml.evaluation.metrics import (
    calculate_calibration_data,
    calculate_classification_metrics,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
)

__all__ = [
    "ModelEvaluator",
    "UnifiedBenchmarkRunner",
    "calculate_bootstrap_confidence_intervals",
    "calculate_calibration_data",
    "calculate_classification_metrics",
    "calculate_pr_curve_data",
    "calculate_roc_curve_data",
    "compare_models",
    "evaluate_cross_validation",
    "generate_benchmark_summary_table",
]
