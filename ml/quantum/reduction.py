"""Leakage-Safe Quantum Feature Reduction and Angle Normalization Module."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from ml.exceptions import PreprocessingError, ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)


class QuantumFeatureReducer:
    """Projects classical features into a low-dimensional representation for quantum encoding.

    Fits SelectKBest and angle scaling STRICTLY on training data to prevent data leakage.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        target_range: tuple[float, float] = (0.0, np.pi),
        random_seed: int = 42,
    ):
        if n_qubits < 1:
            raise ValidationError(f"n_qubits must be >= 1, got {n_qubits}.")

        self.n_qubits = n_qubits
        self.target_range = target_range
        self.random_seed = random_seed
        self.selector_: SelectKBest | None = None
        self.scaler_: MinMaxScaler | None = None
        self.input_feature_names_: list[str] = []
        self.selected_indices_: list[int] = []
        self.selected_feature_names_: list[str] = []
        self.is_fitted_: bool = False
        self.logger = logger

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "QuantumFeatureReducer":
        """Fits SelectKBest and angle scaler strictly on training features."""
        if X is None:
            raise PreprocessingError("Cannot fit QuantumFeatureReducer on None.")

        if isinstance(X, pd.DataFrame):
            in_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if feature_names:
                in_names = list(feature_names)
            elif X_arr.ndim == 2 and X_arr.shape[1] == 11:
                from ml.preprocessing.pipeline import CLASSICAL_FEATURE_NAMES
                in_names = list(CLASSICAL_FEATURE_NAMES)
            else:
                in_names = [f"feature_{i}" for i in range(X_arr.shape[1] if X_arr.ndim == 2 else 0)]

        if X_arr.size == 0 or X_arr.ndim != 2:
            raise PreprocessingError("Input X must be a non-empty 2D array.")

        n_features = X_arr.shape[1]
        if n_features < self.n_qubits:
            raise PreprocessingError(
                f"Cannot reduce {n_features} features to {self.n_qubits} qubits. "
                f"n_features must be >= n_qubits."
            )

        self.input_feature_names_ = in_names

        # Fit SelectKBest only if feature count exceeds qubit count
        if n_features > self.n_qubits:
            if y is None:
                raise PreprocessingError("SelectKBest(mutual_info_classif) requires target labels (y).")
            # Set random seed for mutual_info_classif
            np.random.seed(self.random_seed)
            self.selector_ = SelectKBest(score_func=mutual_info_classif, k=self.n_qubits)
            X_reduced = self.selector_.fit_transform(X_arr, y)
            self.selected_indices_ = [int(i) for i in self.selector_.get_support(indices=True)]
        else:
            self.selector_ = None
            X_reduced = X_arr.copy()
            self.selected_indices_ = list(range(n_features))

        self.selected_feature_names_ = [
            self.input_feature_names_[i] if i < len(self.input_feature_names_) else f"feature_{i}"
            for i in self.selected_indices_
        ]

        # Fit angle scaler into quantum rotation range [-pi, pi]
        self.scaler_ = MinMaxScaler(feature_range=self.target_range)
        self.scaler_.fit(X_reduced)

        self.is_fitted_ = True
        logger.info(
            f"QuantumFeatureReducer fitted: {n_features} features -> {self.n_qubits} qubits "
            f"({self.selected_feature_names_}), range={self.target_range}"
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms input features using parameters learned strictly on training data."""
        if not self.is_fitted_:
            raise PreprocessingError(
                "QuantumFeatureReducer must be fitted before calling transform()."
            )

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if self.selector_ is not None:
            X_reduced = self.selector_.transform(X_arr)
        else:
            X_reduced = X_arr

        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_reduced)
        else:
            X_scaled = X_reduced

        return X_scaled

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Fits on training data and returns transformed quantum-ready matrix."""
        return self.fit(X, y, feature_names=feature_names).transform(X)

    def get_selection_stats(self) -> dict[str, Any]:
        """Returns quantum feature selection statistics and selected feature provenance."""
        if not self.is_fitted_:
            raise PreprocessingError("Reducer must be fitted before retrieving stats.")

        scores = []
        if self.selector_ is not None:
            scores = [float(s) for s in self.selector_.scores_] if hasattr(self.selector_, "scores_") else []

        return {
            "n_components": self.n_qubits,
            "n_qubits": self.n_qubits,
            "method": "SelectKBest(mutual_info_classif)",
            "selected_indices": list(self.selected_indices_),
            "selected_feature_names": list(self.selected_feature_names_),
            "scores": scores,
        }

    def get_variance_stats(self) -> dict[str, Any]:
        """Backward-compatible alias for get_selection_stats()."""
        return self.get_selection_stats()

