from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml.exceptions import PreprocessingError, ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)

# Canonical Raw Feature Schema (8 input features)
RAW_FEATURE_NAMES: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN: str = "Outcome"

# Medically implausible zero values in Pima Indian Diabetes dataset
FEATURES_WITH_INVALID_ZEROS: list[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Canonical Engineered Feature Schema (3 deterministic features)
ENGINEERED_FEATURE_NAMES: list[str] = [
    "Glucose_BMI",
    "Insulin_Glucose",
    "AgeGroup",
]

# Canonical Classical Feature Schema (8 raw + 3 engineered = 11 features)
CLASSICAL_FEATURE_NAMES: list[str] = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES

PIPELINE_VERSION: str = "v2.0"


def split_data(
    df: pd.DataFrame,
    target_column: str = "Outcome",
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' missing from DataFrame.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify_target,
    )
    return X_train, X_test, y_train, y_test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 3 canonical deterministic engineered clinical features.

    Formulas:
    1. Glucose_BMI = Glucose * BMI
       - Input columns: 'Glucose', 'BMI'
       - Denominator: N/A (Multiplicative interaction)
       - Output column: 'Glucose_BMI' (float64)
       - Interpretation: Interaction index between circulating glucose and body mass index.

    2. Insulin_Glucose = Insulin / (Glucose + 1e-9)
       - Input columns: 'Insulin', 'Glucose'
       - Zero/NaN handling: Epsilon regularization (+ 1e-9) prevents division-by-zero.
       - Output column: 'Insulin_Glucose' (float64)
       - Interpretation: Surrogate pancreatic beta-cell insulin response ratio.

    3. AgeGroup = Categorical monotonic binning of Age into [0, 1, 2, 3]
       - Input column: 'Age'
       - Frozen Bins:
           (-inf, 30] -> 0.0 (Young Adult, <= 30)
           (30, 45]   -> 1.0 (Early Adult, 31-45)
           (45, 60]   -> 2.0 (Middle Adult, 46-60)
           (60, inf)  -> 3.0 (Senior Adult, > 60)
       - Output column: 'AgeGroup' (float64)
       - Deterministic: Fully monotonic closed-interval mapping. No NaNs for any finite age.

    Args:
        df: DataFrame containing at least ['Glucose', 'BMI', 'Insulin', 'Age'].

    Returns:
        pd.DataFrame: Copy of DataFrame with the 3 engineered features appended.
    """
    out = df.copy()

    if "Glucose" in out.columns and "BMI" in out.columns:
        out["Glucose_BMI"] = (out["Glucose"].astype(float) * out["BMI"].astype(float)).astype(float)

    if "Insulin" in out.columns and "Glucose" in out.columns:
        denom = out["Glucose"].astype(float) + 1e-9
        out["Insulin_Glucose"] = (out["Insulin"].astype(float) / denom).astype(float)

    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(
            out["Age"].astype(float),
            bins=[-np.inf, 30.0, 45.0, 60.0, np.inf],
            labels=[0.0, 1.0, 2.0, 3.0],
        ).astype(float)

    return out


class BiomedicalPreprocessor:
    """Canonical Leakage-Safe Biomedical Preprocessor for Tabular Clinical Data.

    All transformation parameters (invalid-zero medians, IQR clipping bounds,
    and StandardScaler moments) are fitted strictly on the training partition.
    Validation, test, and inference data reuse the frozen parameters without refitting.
    """

    def __init__(
        self,
        invalid_zero_columns: list[str] | None = None,
        scale_features: bool = True,
    ):
        self.invalid_zero_columns = list(invalid_zero_columns or FEATURES_WITH_INVALID_ZEROS)
        self.scale_features = scale_features
        self.impute_medians_: dict[str, float] = {}
        self.iqr_bounds_: dict[str, dict[str, float]] = {}
        self.scaler_: StandardScaler | None = None
        self.raw_feature_names_: list[str] = list(RAW_FEATURE_NAMES)
        self.feature_names_: list[str] = list(RAW_FEATURE_NAMES)
        self.engineered_feature_names_: list[str] = list(CLASSICAL_FEATURE_NAMES)
        self.is_fitted_: bool = False
        self.pipeline_version: str = PIPELINE_VERSION
        self.logger = logger

    def __getattr__(self, name: str) -> Any:
        """Fallback for attributes on unpickled instances from earlier iterations."""
        if name == "raw_feature_names_":
            if "feature_names_" in self.__dict__ and len(self.__dict__["feature_names_"]) == len(RAW_FEATURE_NAMES):
                return list(self.__dict__["feature_names_"])
            return list(RAW_FEATURE_NAMES)
        if name == "pipeline_version":
            return PIPELINE_VERSION
        if name == "engineered_feature_names_":
            return list(CLASSICAL_FEATURE_NAMES)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "BiomedicalPreprocessor":
        """Fits preprocessing statistics strictly on training data."""
        if X_train is None or X_train.empty:
            raise PreprocessingError("Cannot fit preprocessor on empty DataFrame.")

        self.raw_feature_names_ = list(X_train.columns)
        self.feature_names_ = list(X_train.columns)
        X_copy = X_train.copy()

        # 1. Convert medically implausible zeros to NaN
        for col in self.invalid_zero_columns:
            if col in X_copy.columns:
                mask = X_copy[col] == 0
                X_copy.loc[mask, col] = np.nan

        # 2. Compute and store train-fitted medians
        self.impute_medians_ = {}
        for col in X_copy.columns:
            median_val = X_copy[col].median()
            self.impute_medians_[col] = float(median_val) if pd.notna(median_val) else 0.0

        for col, median_val in self.impute_medians_.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(median_val)

        # 3. Compute and store train-fitted IQR bounds
        self.iqr_bounds_ = {}
        for col in X_copy.columns:
            q1 = float(X_copy[col].quantile(0.25))
            q3 = float(X_copy[col].quantile(0.75))
            iqr = q3 - q1
            self.iqr_bounds_[col] = {
                "lower": float(q1 - 1.5 * iqr),
                "upper": float(q3 + 1.5 * iqr),
            }
            # Clip training data using fitted bounds
            X_copy[col] = X_copy[col].clip(
                lower=self.iqr_bounds_[col]["lower"],
                upper=self.iqr_bounds_[col]["upper"],
            )

        # 4. Deterministic Feature Engineering (8 raw -> 11 classical)
        X_copy = engineer_features(X_copy)
        self.engineered_feature_names_ = [c for c in CLASSICAL_FEATURE_NAMES if c in X_copy.columns]
        if len(self.engineered_feature_names_) != len(CLASSICAL_FEATURE_NAMES):
            self.engineered_feature_names_ = list(CLASSICAL_FEATURE_NAMES)

        # 5. Fit StandardScaler on the 11 engineered/clipped features
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_mat = X_copy[self.engineered_feature_names_].to_numpy(dtype=float)
            self.scaler_.fit(X_mat)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms validation, test, or inference data using frozen train-fitted parameters."""
        if not self.is_fitted_:
            raise PreprocessingError("Preprocessor must be fitted on training data before calling transform().")

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.raw_feature_names_)
        else:
            X_df = X.copy()

        # 1. Convert invalid zeros to NaN
        for col in self.invalid_zero_columns:
            if col in X_df.columns:
                mask = X_df[col] == 0
                X_df.loc[mask, col] = np.nan

        # 2. Impute missing values using frozen training medians
        for col, median_val in self.impute_medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(median_val)

        # 3. Clip outliers using frozen training IQR bounds
        for col in self.raw_feature_names_:
            if col in X_df.columns and col in self.iqr_bounds_:
                bounds = self.iqr_bounds_[col]
                X_df[col] = X_df[col].clip(lower=bounds["lower"], upper=bounds["upper"])

        # 4. Deterministic Feature Engineering
        X_df = engineer_features(X_df)

        # 5. Ensure exact canonical feature ordering
        for col in self.engineered_feature_names_:
            if col not in X_df.columns:
                X_df[col] = 0.0

        X_array = X_df[self.engineered_feature_names_].to_numpy(dtype=float)

        # 6. Standardize using frozen training scaler moments
        if self.scale_features and self.scaler_ is not None:
            X_array = self.scaler_.transform(X_array)

        return X_array

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> np.ndarray:
        """Convenience method fitting on training data and returning transformed matrix."""
        return self.fit(X_train, y_train).transform(X_train)

    def get_feature_names(self) -> list[str]:
        """Returns the 11 classical engineered feature names in canonical order."""
        return list(self.engineered_feature_names_)

    def get_raw_feature_names(self) -> list[str]:
        """Returns the 8 raw feature names in canonical order."""
        return list(self.raw_feature_names_)

    def get_metadata(self) -> dict[str, Any]:
        """Returns explicit provenance distinguishing raw vs engineered feature schemas."""
        raw_count = len(self.raw_feature_names_)
        total_count = len(self.engineered_feature_names_)
        return {
            "pipeline_version": self.pipeline_version,
            "raw_features_count": raw_count,
            "raw_feature_names": list(self.raw_feature_names_),
            "engineered_features_count": total_count - raw_count,
            "engineered_feature_names": [c for c in self.engineered_feature_names_ if c not in self.raw_feature_names_],
            "classical_features_count": total_count,
            "classical_feature_names": list(self.engineered_feature_names_),
            "invalid_zero_handling": "NaN conversion for " + ", ".join(self.invalid_zero_columns),
            "missing_value_handling": "Median (Train-only fitted)",
            "outlier_handling": "IQR Bounds (Train-only fitted)",
            "scaling": "StandardScaler (Train-only fitted)" if self.scale_features else "None",
        }


class QuantumFeatureSelector:
    """Wrapper delegating to canonical QuantumFeatureReducer for quantum reduction."""
    def __init__(self, n_qubits: int = 4, random_seed: int = 42):
        from ml.quantum.reduction import QuantumFeatureReducer
        self._reducer = QuantumFeatureReducer(n_qubits=n_qubits, random_seed=random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QuantumFeatureSelector":
        self._reducer.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self._reducer.fit_transform(X, y)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reducer, name)
