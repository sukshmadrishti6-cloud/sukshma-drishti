"""Model-Agnostic Evaluator and Comparison Engine for SIH26139."""
import time
from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.bootstrap import calculate_bootstrap_confidence_intervals
from ml.evaluation.metrics import calculate_classification_metrics
from ml.exceptions import ModelError, ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Model-agnostic evaluation engine for Classical and Quantum ML models."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.logger = logger

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        compute_bootstrap: bool = True,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Evaluates a trained model on a held-out test set.

        Args:
            model: Fitted model exposing predict() and optionally predict_proba().
            X_test: Preprocessed test feature matrix.
            y_test: True test binary labels.
            compute_bootstrap: Whether to compute bootstrap confidence intervals.
            n_bootstraps: Number of bootstrap resamples (default 1000).
            confidence_level: Confidence interval level (default 0.95).

        Returns:
            Dict[str, Any]: Structured evaluation result.
        """
        if not hasattr(model, "predict"):
            raise ModelError(f"Model object of type {type(model)} does not implement predict().")

        y_true_arr = np.asarray(y_test).ravel()

        t_eval_start = time.perf_counter()

        # Inference timing
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception as e:
                self.logger.warning(f"Failed to obtain probability predictions: {e!s}")

        # Metrics calculation
        metrics_dict = calculate_classification_metrics(y_true_arr, y_pred, y_prob)

        # Bootstrap confidence intervals
        bootstrap_results = None
        if compute_bootstrap:
            bootstrap_results = calculate_bootstrap_confidence_intervals(
                y_true_arr,
                y_pred,
                y_prob,
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                random_seed=self.random_seed,
            )

        evaluation_time_ms = (time.perf_counter() - t_eval_start) * 1000.0

        model_name = getattr(model, "model_name", type(model).__name__)
        model_type = getattr(model, "model_type", "unknown")

        report: dict[str, Any] = {
            "model_name": model_name,
            "model_type": model_type,
            "sample_count": len(y_true_arr),
            "metrics": metrics_dict,
            "confusion_matrix": metrics_dict["confusion_matrix"],
            "confidence_intervals": bootstrap_results,
            "timing": {
                "inference_time_ms": float(inference_time_ms),
                "evaluation_time_ms": float(evaluation_time_ms),
            },
        }

        self.logger.info(
            f"Evaluation Complete for '{model_name}': "
            f"Accuracy={metrics_dict['accuracy']:.4f}, "
            f"F1={metrics_dict['f1_score']:.4f}, "
            f"ROC-AUC={metrics_dict.get('roc_auc')}, "
            f"Inference Time={inference_time_ms:.2f}ms"
        )
        return report


def compare_models(eval_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Compiles a list of evaluation reports into a tabular comparison DataFrame.

    Args:
        eval_results: List of result dictionaries produced by evaluate_model().

    Returns:
        pd.DataFrame: Tabular summary ranking models across key performance metrics.
    """
    if not eval_results:
        raise ValidationError("eval_results list cannot be empty.")

    rows = []
    for r in eval_results:
        metrics = r.get("metrics", {})
        timing = r.get("timing", {})
        rows.append(
            {
                "Model": r.get("model_name", "Unknown"),
                "Type": r.get("model_type", "Unknown"),
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-Score": metrics.get("f1_score"),
                "Sensitivity": metrics.get("sensitivity"),
                "Specificity": metrics.get("specificity"),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
                "Brier Score": metrics.get("brier_score"),
                "Inference (ms)": timing.get("inference_time_ms"),
            }
        )

    df_comp = pd.DataFrame(rows)
    return df_comp.sort_values(by="ROC-AUC", ascending=False, na_position="last").reset_index(
        drop=True
    )
