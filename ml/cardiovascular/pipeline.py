"""Cardiovascular Preprocessing Pipeline for UCI Cleveland Dataset."""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from ml.exceptions import ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)

CARDIO_FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CARDIO_TARGET_COLUMN = "target"


class CardiovascularPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-free preprocessing pipeline for Cardiovascular clinical features."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = list(CARDIO_FEATURE_NAMES)
        self.pipeline_version = "v1.0"
        self.is_fitted = False

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates feature columns and numeric finite values."""
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValidationError(f"Missing required cardiovascular feature columns: {missing}")

        df_feat = df[self.feature_names].copy()
        arr = df_feat.to_numpy(dtype=float)
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValidationError("Cardiovascular input features contain NaN or Infinite values.")
        return df_feat

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "CardiovascularPreprocessor":
        """Fits StandardScaler strictly on training data."""
        if isinstance(X, np.ndarray):
            df_feat = pd.DataFrame(X, columns=self.feature_names)
        else:
            df_feat = self.validate_input(X)

        self.scaler.fit(df_feat[self.feature_names])
        self.is_fitted = True
        logger.info(f"CardiovascularPreprocessor fitted successfully on {len(df_feat)} training samples.")
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms input features using train-fitted StandardScaler moments."""
        if not self.is_fitted:
            raise ValidationError("CardiovascularPreprocessor is not fitted. Call fit() before transform().")

        if isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValidationError(
                    f"Expected {len(self.feature_names)} features, got {X.shape[1]}."
                )
            return self.scaler.transform(X)

        df_feat = self.validate_input(X)
        return self.scaler.transform(df_feat[self.feature_names])

    def get_feature_names(self) -> list[str]:
        """Returns list of input feature names."""
        return list(self.feature_names)

    def get_metadata(self) -> dict[str, Any]:
        """Returns preprocessor metadata provenance."""
        return {
            "pipeline_version": self.pipeline_version,
            "raw_features_count": len(self.feature_names),
            "engineered_features_count": 0,
            "classical_features_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "scaler": "StandardScaler",
        }
