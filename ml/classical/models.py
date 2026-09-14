"""Real Python Classical Machine Learning Engine for SIH26139.

Provides production implementations of Logistic Regression, RBF-kernel SVM,
Random Forest, and XGBoost baseline classifiers.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.exceptions import ModelError, ValidationError
from ml.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_CLASSICAL_MODELS = {
    "logistic_regression": ["logistic_regression", "logistic", "lr"],
    "svm": ["svm", "svc", "support_vector_machine", "rbf_svm"],
    "linear_svm": ["linear_svm", "linearsvm"],
    "knn": ["knn", "k_nearest_neighbors", "k_neighbors"],
    "decision_tree": ["decision_tree", "dt", "tree"],
    "random_forest": ["random_forest", "rf", "forest"],
    "extra_trees": ["extra_trees", "et"],
    "gradient_boosting": ["gradient_boosting", "gb", "gbm"],
    "hist_gradient_boosting": ["hist_gradient_boosting", "hist_gb", "hgbm"],
    "xgboost": ["xgboost", "xgb"],
    "soft_voting": ["soft_voting", "soft_voting_ensemble"],
    "hard_voting": ["hard_voting", "hard_voting_ensemble"],
    "stacking": ["stacking", "stacking_ensemble"],
}


class ClassicalModel:
    """Base interface wrapper for Classical Machine Learning models."""

    def __init__(self, model_name: str, model_type: str, params: dict[str, Any], random_seed: int = 42):
        self.model_name = model_name
        self.model_type = model_type
        self.params = params
        self.random_seed = random_seed
        self.estimator_: Any = None
        self.feature_names_: list[str] = []
        self.n_features_in_: int | None = None
        self.is_fitted_: bool = False
        self.logger = logger

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "ClassicalModel":
        """Fits underlying estimator on feature matrix X and target y."""
        X_arr, y_arr = self._validate_inputs(X, y, is_fit=True)
        try:
            self.estimator_.fit(X_arr, y_arr)
            self.is_fitted_ = True
            self.logger.info(
                f"Model '{self.model_name}' trained successfully on {len(X_arr)} samples "
                f"across {X_arr.shape[1]} features."
            )
        except (ValueError, TypeError, RuntimeError, Exception) as e:
            raise ModelError(f"Training failed for model '{self.model_name}': {e!s}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts binary class labels (0 or 1)."""
        if not self.is_fitted_:
            raise ModelError(f"Model '{self.model_name}' must be fitted before calling predict().")
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        return self.estimator_.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class probabilities shape (N, 2)."""
        if not self.is_fitted_:
            raise ModelError(f"Model '{self.model_name}' must be fitted before calling predict_proba().")
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        if not hasattr(self.estimator_, "predict_proba"):
            raise ModelError(f"Model '{self.model_name}' does not support probability predictions.")
        return self.estimator_.predict_proba(X_arr)

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        is_fit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Validates feature dimensions, NaNs, non-empty shapes, and target labels."""
        if X is None:
            raise ValidationError("Feature matrix X cannot be None.")

        if isinstance(X, pd.DataFrame):
            if is_fit:
                self.feature_names_ = list(X.columns)
            elif self.feature_names_ and list(X.columns) != self.feature_names_:
                raise ValidationError(
                    f"Feature mismatch. Expected columns {self.feature_names_}, got {list(X.columns)}"
                )
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.size == 0 or X_arr.ndim != 2:
            raise ValidationError("Feature matrix X must be a non-empty 2D array.")

        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValidationError("Feature matrix X contains NaN or infinite values. Run preprocessing first.")

        if is_fit:
            self.n_features_in_ = X_arr.shape[1]
        elif self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"Feature dimension mismatch. Expected {self.n_features_in_} features, got {X_arr.shape[1]}."
            )

        y_arr = None
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy().ravel()
            else:
                y_arr = np.asarray(y).ravel()

            if y_arr.shape[0] != X_arr.shape[0]:
                raise ValidationError(
                    f"Sample count mismatch: X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]} samples."
                )

            unique_classes = np.unique(y_arr[~np.isnan(y_arr)])
            if not set(unique_classes).issubset({0, 1}):
                raise ValidationError(f"Target vector y contains non-binary values: {unique_classes}")

        return X_arr, y_arr


class LogisticRegressionModel(ClassicalModel):
    """Real scikit-learn LogisticRegression model implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "random_state": random_seed,
        }
        if params:
            cleaned_params = {k: v for k, v in params.items() if k != "penalty"}
            defaults.update(cleaned_params)
        defaults["random_state"] = random_seed
        super().__init__("Logistic Regression", "logistic_regression", defaults, random_seed)
        self.estimator_ = LogisticRegression(**defaults)


class LinearSVMModel(ClassicalModel):
    """Real scikit-learn SVC model with linear kernel."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "C": 1.0,
            "kernel": "linear",
            "probability": True,
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["kernel"] = "linear"
        defaults["probability"] = True
        defaults["random_state"] = random_seed
        super().__init__("Support Vector Machine (Linear)", "linear_svm", defaults, random_seed)
        self.estimator_ = SVC(**defaults)


class SVMModel(ClassicalModel):
    """Real scikit-learn SVC model with RBF kernel."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["kernel"] = "rbf"
        defaults["probability"] = True
        defaults["random_state"] = random_seed
        super().__init__("Support Vector Machine (RBF)", "svm", defaults, random_seed)
        self.estimator_ = SVC(**defaults)


class KNNModel(ClassicalModel):
    """Real scikit-learn KNeighborsClassifier implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
        }
        if params:
            defaults.update(params)
        super().__init__("K-Nearest Neighbors", "knn", defaults, random_seed)
        self.estimator_ = KNeighborsClassifier(**defaults)


class DecisionTreeModel(ClassicalModel):
    """Real scikit-learn DecisionTreeClassifier implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "max_depth": 5,
            "min_samples_split": 4,
            "criterion": "gini",
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("Decision Tree", "decision_tree", defaults, random_seed)
        self.estimator_ = DecisionTreeClassifier(**defaults)


class RandomForestModel(ClassicalModel):
    """Real scikit-learn RandomForestClassifier model implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 4,
            "criterion": "gini",
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("Random Forest", "random_forest", defaults, random_seed)
        self.estimator_ = RandomForestClassifier(**defaults)


class ExtraTreesModel(ClassicalModel):
    """Real scikit-learn ExtraTreesClassifier implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 4,
            "criterion": "gini",
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("Extra Trees", "extra_trees", defaults, random_seed)
        self.estimator_ = ExtraTreesClassifier(**defaults)


class GradientBoostingModel(ClassicalModel):
    """Real scikit-learn GradientBoostingClassifier implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("Gradient Boosting", "gradient_boosting", defaults, random_seed)
        self.estimator_ = GradientBoostingClassifier(**defaults)


class HistGradientBoostingModel(ClassicalModel):
    """Real scikit-learn HistGradientBoostingClassifier implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "max_iter": 100,
            "learning_rate": 0.1,
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("Histogram Gradient Boosting", "hist_gradient_boosting", defaults, random_seed)
        self.estimator_ = HistGradientBoostingClassifier(**defaults)


class XGBoostModel(ClassicalModel):
    """Real xgboost.XGBClassifier model implementation."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        defaults = {
            "n_estimators": 85,
            "learning_rate": 0.08,
            "max_depth": 4,
            "subsample": 0.8,
            "eval_metric": "logloss",
            "random_state": random_seed,
        }
        if params:
            defaults.update(params)
        defaults["random_state"] = random_seed
        super().__init__("XGBoost", "xgboost", defaults, random_seed)
        self.estimator_ = XGBClassifier(**defaults)


class SoftVotingEnsembleModel(ClassicalModel):
    """Soft Voting Ensemble combining probability predictions from diverse estimators."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        estimators = [
            ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=random_seed)),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed)),
            ("xgb", XGBClassifier(n_estimators=85, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=random_seed)),
            ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=random_seed)),
        ]
        defaults = {"voting": "soft"}
        if params:
            defaults.update(params)
        defaults["voting"] = "soft"
        super().__init__("Soft Voting Ensemble", "soft_voting", defaults, random_seed)
        self.estimator_ = VotingClassifier(estimators=estimators, **defaults)


class HardVotingEnsembleModel(ClassicalModel):
    """Hard Voting Ensemble combining majority class votes from tree/instance estimators."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        estimators = [
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=random_seed)),
            ("lsvm", SVC(C=1.0, kernel="linear", random_state=random_seed)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
        defaults = {"voting": "hard"}
        if params:
            defaults.update(params)
        defaults["voting"] = "hard"
        super().__init__("Hard Voting Ensemble", "hard_voting", defaults, random_seed)
        self.estimator_ = VotingClassifier(estimators=estimators, **defaults)


class StackingEnsembleModel(ClassicalModel):
    """Stacking Ensemble combining base estimators with cross-validation meta-features."""

    def __init__(self, params: dict[str, Any] | None = None, random_seed: int = 42):
        estimators = [
            ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=random_seed)),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_seed)),
            ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_seed)),
            ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=random_seed)),
        ]
        final_estimator = LogisticRegression(C=1.0, max_iter=500, random_state=random_seed)
        defaults = {"cv": 5}
        if params:
            defaults.update(params)
        super().__init__("Stacking Ensemble", "stacking", defaults, random_seed)
        self.estimator_ = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=defaults.get("cv", 5),
        )


class ClassicalModelFactory:
    """Factory for instantiating Classical ML models by key/alias."""

    @classmethod
    def create_model(
        cls,
        model_name: str,
        params: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> ClassicalModel:
        name_key = model_name.strip().lower()

        if name_key in SUPPORTED_CLASSICAL_MODELS["logistic_regression"]:
            return LogisticRegressionModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["linear_svm"]:
            return LinearSVMModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["svm"]:
            return SVMModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["knn"]:
            return KNNModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["decision_tree"]:
            return DecisionTreeModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["random_forest"]:
            return RandomForestModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["extra_trees"]:
            return ExtraTreesModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["gradient_boosting"]:
            return GradientBoostingModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["hist_gradient_boosting"]:
            return HistGradientBoostingModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["xgboost"]:
            return XGBoostModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["soft_voting"]:
            return SoftVotingEnsembleModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["hard_voting"]:
            return HardVotingEnsembleModel(params, random_seed)
        elif name_key in SUPPORTED_CLASSICAL_MODELS["stacking"]:
            return StackingEnsembleModel(params, random_seed)
        else:
            raise ValidationError(
                f"Unsupported classical model name: '{model_name}'. "
                f"Supported options: {list(SUPPORTED_CLASSICAL_MODELS.keys())}"
            )

    create = create_model


def save_model(model: ClassicalModel, filepath: str | Path) -> Path:
    """Serializes a ClassicalModel instance using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model artifact '{model.model_name}' to '{path}'")
    return path


def load_model(filepath: str | Path) -> ClassicalModel:
    """Loads a serialized ClassicalModel instance using joblib."""
    path = Path(filepath)
    if not path.exists():
        raise ModelError(f"Model artifact file not found at: '{path}'")
    try:
        model = joblib.load(path)
        logger.info(f"Successfully loaded model artifact '{model.model_name}' from '{path}'")
        return model
    except (OSError, ValueError, RuntimeError, Exception) as e:
        raise ModelError(f"Failed to load model artifact at '{path}': {e!s}")

