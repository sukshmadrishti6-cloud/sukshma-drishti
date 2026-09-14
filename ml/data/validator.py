"""Data Quality Inspection and Validation Module for SIH26139."""
from typing import Any

import numpy as np
import pandas as pd

from ml.exceptions import ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Performs rigorous biomedical data quality inspection."""

    def __init__(self, target_column: str = "Outcome"):
        self.target_column = target_column
        self.logger = logger

    def inspect_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Runs quality checks and returns structured validation report dictionary."""
        if df is None or df.empty:
            raise ValidationError("Cannot validate empty or None DataFrame.")

        if self.target_column not in df.columns:
            raise ValidationError(f"Target column '{self.target_column}' missing from DataFrame.")

        total_rows = len(df)
        total_cols = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())

        # Target column inspection
        y = df[self.target_column]
        unique_targets = set(y.dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise ValidationError(
                f"Target column '{self.target_column}' contains non-binary values: {unique_targets}. "
                f"Expected binary {{0, 1}}."
            )

        target_counts = y.value_counts().to_dict()
        if len(unique_targets) < 2:
            raise ValidationError(
                f"Target column '{self.target_column}' lacks class diversity (found only class {unique_targets})."
            )

        # Feature matrix inspection
        features_df = df.drop(columns=[self.target_column])
        feature_names = list(features_df.columns)

        # Confirm target is not in feature matrix
        if self.target_column in feature_names:
            raise ValidationError(
                f"Data Leakage Violation: Target '{self.target_column}' remains inside feature matrix."
            )

        # Check for infinite values
        has_inf = np.isinf(features_df.to_numpy(dtype=float, na_value=0.0)).any()
        if has_inf:
            raise ValidationError("Dataset contains infinite numeric values.")

        missing_per_col = features_df.isna().sum().to_dict()

        report = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows,
            "target_column": self.target_column,
            "target_distribution": target_counts,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "missing_values_per_column": missing_per_col,
            "is_valid": True,
        }

        self.logger.info(
            f"Quality Inspection Passed: {total_rows} samples, {len(feature_names)} features, "
            f"duplicates: {duplicate_rows}, target counts: {target_counts}"
        )
        return report
