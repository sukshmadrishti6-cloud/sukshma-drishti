"""Bootstrap Confidence Interval Engine for SIH26139 Evaluation."""
from typing import Any

import numpy as np

from ml.evaluation.metrics import calculate_classification_metrics
from ml.exceptions import ValidationError


def calculate_bootstrap_confidence_intervals(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Estimates empirical bootstrap confidence intervals for classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Optional predicted probabilities.
        n_bootstraps: Number of bootstrap iterations (default 1000).
        confidence_level: Confidence level (default 0.95 for 95% CI).
        random_seed: Random seed for deterministic resampling.

    Returns:
        Dict[str, Any]: Metric names mapped to point estimates and confidence intervals.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    y_pr = np.asarray(y_prob) if y_prob is not None else None

    n_samples = len(y_t)
    if n_samples < 2:
        raise ValidationError("Bootstrap requires at least 2 samples.")

    # Calculate point estimates
    point_estimates = calculate_classification_metrics(y_t, y_p, y_pr)

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "sensitivity",
        "specificity",
    ]
    if y_pr is not None and point_estimates.get("roc_auc") is not None:
        metric_keys.extend(["roc_auc", "pr_auc", "brier_score", "log_loss"])

    bootstrap_scores: dict[str, list] = {key: [] for key in metric_keys}

    rng = np.random.default_rng(random_seed)

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_y_t = y_t[indices]
        sample_y_p = y_p[indices]
        sample_y_pr = y_pr[indices] if y_pr is not None else None

        # Check if bootstrap sample has both classes for probability metrics
        try:
            sample_metrics = calculate_classification_metrics(sample_y_t, sample_y_p, sample_y_pr)
            for key in metric_keys:
                val = sample_metrics.get(key)
                if val is not None and not np.isnan(val):
                    bootstrap_scores[key].append(val)
        except Exception:
            continue

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - (alpha / 2.0))

    results: dict[str, Any] = {
        "confidence_level": confidence_level,
        "n_bootstraps": n_bootstraps,
        "random_seed": random_seed,
        "metrics": {},
    }

    for key in metric_keys:
        scores = bootstrap_scores[key]
        point = point_estimates.get(key)

        if scores and len(scores) >= 10:
            ci_lower = float(np.percentile(scores, lower_pct))
            ci_upper = float(np.percentile(scores, upper_pct))
        else:
            ci_lower = float(point) if point is not None else None
            ci_upper = float(point) if point is not None else None

        results["metrics"][key] = {
            "point_estimate": point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return results
