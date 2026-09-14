"""Stratified Cross-Validation Engine with Strict Leakage Prevention."""
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml.classical.models import ClassicalModel, ClassicalModelFactory
from ml.data.validator import DataValidator
from ml.evaluation.metrics import calculate_classification_metrics
from ml.exceptions import ValidationError
from ml.logger import get_logger
from ml.preprocessing.pipeline import BiomedicalPreprocessor

logger = get_logger(__name__)


def evaluate_cross_validation(
    model_or_name: str | Callable[[], ClassicalModel],
    df: pd.DataFrame,
    target_column: str = "Outcome",
    n_splits: int = 5,
    random_seed: int = 42,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs leakage-safe stratified cross-validation on a biomedical dataset.

    Guarantees that imputation medians and scaling parameters are fit STRICTLY
    within each fold's training partition.

    Args:
        model_or_name: Model identifier string (e.g. 'svm', 'rf') or factory callable.
        df: Raw or validated DataFrame containing features and target column.
        target_column: Target binary column name.
        n_splits: Number of cross-validation folds (default 5).
        random_seed: Random seed for deterministic fold generation.
        model_params: Optional hyperparameter dictionary for model instantiation.

    Returns:
        Dict[str, Any]: Structured cross-validation summary and per-fold metrics.
    """
    if df is None or df.empty:
        raise ValidationError("DataFrame cannot be empty for cross-validation.")

    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' missing from DataFrame.")

    # Validate dataset quality first
    validator = DataValidator(target_column=target_column)
    validator.inspect_quality(df)

    X_full = df.drop(columns=[target_column]).copy()
    y_full = df[target_column].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
    ]

    fold_results: list[dict[str, Any]] = []
    training_times: list[float] = []
    inference_times: list[float] = []

    model_display_name = (
        model_or_name if isinstance(model_or_name, str) else "custom_model"
    )

    logger.info(
        f"Starting {n_splits}-fold Stratified CV for '{model_display_name}' (seed={random_seed})..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train_fold = X_full.iloc[train_idx].copy()
        y_train_fold = y_full[train_idx]
        X_val_fold = X_full.iloc[val_idx].copy()
        y_val_fold = y_full[val_idx]

        # Fit preprocessor STRICTLY on fold training partition (zero leakage)
        preprocessor = BiomedicalPreprocessor()
        X_train_trans = preprocessor.fit_transform(X_train_fold)
        X_val_trans = preprocessor.transform(X_val_fold)

        # Instantiate fresh model instance for fold
        if isinstance(model_or_name, str):
            model_instance = ClassicalModelFactory.create_model(
                model_or_name, params=model_params, random_seed=random_seed
            )
        else:
            model_instance = model_or_name()

        model_display_name = model_instance.model_name

        # Train model & record timing
        t0 = time.perf_counter()
        model_instance.fit(X_train_trans, y_train_fold)
        t_train = (time.perf_counter() - t0) * 1000.0  # ms
        training_times.append(t_train)

        # Predict & record inference timing
        t1 = time.perf_counter()
        y_pred_val = model_instance.predict(X_val_trans)
        t_infer = (time.perf_counter() - t1) * 1000.0  # ms
        inference_times.append(t_infer)

        y_prob_val = None
        try:
            y_prob_val = model_instance.predict_proba(X_val_trans)
        except Exception:
            pass

        fold_metrics = calculate_classification_metrics(y_val_fold, y_pred_val, y_prob_val)
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["train_time_ms"] = t_train
        fold_metrics["inference_time_ms"] = t_infer
        fold_results.append(fold_metrics)

    # Aggregate metric statistics across folds
    cv_summary: dict[str, Any] = {
        "model_name": model_display_name,
        "n_splits": n_splits,
        "random_seed": random_seed,
        "metrics": {},
        "timing": {
            "mean_train_time_ms": float(np.mean(training_times)),
            "mean_inference_time_ms": float(np.mean(inference_times)),
        },
        "folds": fold_results,
    }

    for metric in metric_names:
        values = [
            f[metric]
            for f in fold_results
            if f.get(metric) is not None and not np.isnan(f[metric])
        ]
        if values:
            cv_summary["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
        else:
            cv_summary["metrics"][metric] = None

    logger.info(
        f"Completed {n_splits}-fold CV for '{model_display_name}'. "
        f"Mean Accuracy: {cv_summary['metrics']['accuracy']['mean']:.4f} ± {cv_summary['metrics']['accuracy']['std']:.4f}, "
        f"Mean F1: {cv_summary['metrics']['f1_score']['mean']:.4f}"
    )

    return cv_summary
