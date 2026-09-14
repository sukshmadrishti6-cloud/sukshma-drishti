"""Classification Metrics Suite for SIH26139 Biomedical Evaluation.

Provides rigorous calculation of accuracy, precision, recall, F1, sensitivity,
specificity, ROC-AUC, PR-AUC, Brier score, log loss, confusion matrix, ROC curves, and PR curves.
"""
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.exceptions import ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)


def calculate_classification_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None,
) -> dict[str, Any]:
    """Calculates comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability estimates (shape (N, 2) or (N,)).

    Returns:
        Dict[str, Any]: Structured dictionary with classification metrics and confusion matrix.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if y_t.size == 0 or y_p.size == 0:
        raise ValidationError("y_true and y_pred cannot be empty arrays.")

    if y_t.shape[0] != y_p.shape[0]:
        raise ValidationError(
            f"Length mismatch: y_true has {y_t.shape[0]} samples, y_pred has {y_p.shape[0]} samples."
        )

    # Validate binary labels
    unique_true = np.unique(y_t)
    if not set(unique_true).issubset({0, 1}):
        raise ValidationError(f"y_true contains non-binary values: {unique_true}")

    # Standard metrics
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Confusion matrix & Specificity
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    sensitivity = rec  # Sensitivity is mathematically equivalent to recall for positive class
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
    }

    # Probability-based metrics
    if y_prob is not None:
        y_pr = np.asarray(y_prob)
        if y_pr.shape[0] != y_t.shape[0]:
            raise ValidationError(
                f"Probability shape mismatch: y_prob has {y_pr.shape[0]} samples, y_true has {y_t.shape[0]}."
            )

        # Extract positive class probability
        if y_pr.ndim == 2 and y_pr.shape[1] >= 2:
            y_prob_pos = y_pr[:, 1]
            y_prob_2d = y_pr
        else:
            y_prob_pos = y_pr.ravel()
            y_prob_2d = np.column_stack([1.0 - y_prob_pos, y_prob_pos])

        # ROC-AUC
        if len(unique_true) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_t, y_prob_pos))
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed: {e!s}")
                metrics["roc_auc"] = None

            try:
                metrics["pr_auc"] = float(average_precision_score(y_t, y_prob_pos))
            except Exception as e:
                logger.warning(f"PR-AUC calculation failed: {e!s}")
                metrics["pr_auc"] = None
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

        # Brier Score & Log Loss
        try:
            metrics["brier_score"] = float(brier_score_loss(y_t, y_prob_pos))
        except Exception:
            metrics["brier_score"] = None

        try:
            metrics["log_loss"] = float(log_loss(y_t, y_prob_2d, labels=[0, 1]))
        except Exception:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None
        metrics["log_loss"] = None

    return metrics


def calculate_roc_curve_data(
    y_true: np.ndarray | list,
    y_prob_pos: np.ndarray | list,
) -> dict[str, Any] | None:
    """Generates ROC curve plotting data (FPR, TPR, Thresholds)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob_pos).ravel()

    if len(np.unique(y_t)) < 2:
        return None

    try:
        fpr, tpr, thresholds = roc_curve(y_t, y_p)
        auc_val = float(roc_auc_score(y_t, y_p))
        return {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in thresholds],
            "auc": auc_val,
        }
    except Exception as e:
        logger.warning(f"Failed to generate ROC curve data: {e!s}")
        return None


def calculate_pr_curve_data(
    y_true: np.ndarray | list,
    y_prob_pos: np.ndarray | list,
) -> dict[str, Any] | None:
    """Generates Precision-Recall curve plotting data (Precision, Recall, Thresholds)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob_pos).ravel()

    if len(np.unique(y_t)) < 2:
        return None

    try:
        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        auc_val = float(average_precision_score(y_t, y_p))
        return {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "thresholds": [float(x) for x in thresholds],
            "auc": auc_val,
        }
    except Exception as e:
        logger.warning(f"Failed to generate PR curve data: {e!s}")
        return None


def calculate_calibration_data(
    y_true: np.ndarray | list,
    y_prob_pos: np.ndarray | list,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Generates reliability diagram calibration data (prob_true, prob_pred, bin_counts)."""
    from sklearn.calibration import calibration_curve

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob_pos).ravel()

    if len(np.unique(y_t)) < 2:
        return {
            "status": "unavailable",
            "reason": "Calibration unavailable: dataset contains fewer than 2 distinct classes.",
        }

    try:
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy="uniform")
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(y_p, bins) - 1
        bin_counts = [int(np.sum(bin_assignments == b)) for b in range(n_bins)]

        return {
            "status": "available",
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred],
            "bin_counts": bin_counts,
            "n_bins": n_bins,
        }
    except Exception as e:
        logger.warning(f"Failed to generate calibration data: {e!s}")
        return {
            "status": "unavailable",
            "reason": f"Calibration calculation error: {e!s}",
        }

