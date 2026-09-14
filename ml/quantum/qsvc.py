"""Real Quantum Support Vector Classifier (QSVC) Model Wrapper."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import qiskit_machine_learning
from qiskit_machine_learning.algorithms.classifiers import QSVC

from ml.exceptions import ModelError, ValidationError
from ml.logger import get_logger
from ml.quantum.circuit import create_feature_map
from ml.quantum.kernel import create_quantum_kernel

logger = get_logger(__name__)


class QuantumSVCModel:
    """Real Quantum Support Vector Classifier (QSVC) wrapping Qiskit ML QSVC."""

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map_name: str = "ZZFeatureMap",
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        probability: bool = True,
        class_weight: str | dict | None = "balanced",
        threshold: float | None = None,
        shots: int = 1024,
        random_seed: int = 42,
    ):
        self.model_name = f"Quantum Support Vector Classifier (QSVC-{num_qubits}Q)"
        self.model_type = "qsvc"
        self.num_qubits = num_qubits
        self.feature_map_name = feature_map_name
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.probability = probability
        self.class_weight = class_weight
        self.threshold_ = threshold
        self.shots = shots
        self.random_seed = random_seed

        # Construct quantum circuit & kernel
        self.feature_map_ = create_feature_map(
            name=feature_map_name,
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
        self.kernel_ = create_quantum_kernel(self.feature_map_)

        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        self.estimator_ = SVC(
            C=C,
            kernel="precomputed",
            probability=False,
            class_weight=class_weight,
            random_state=random_seed,
        )
        self.calibrator_ = LogisticRegression(class_weight="balanced", random_state=random_seed)

        self.is_fitted_: bool = False
        self.n_features_in_: int = num_qubits
        self.logger = logger
        self.X_train_: np.ndarray | None = None
        self.Psi_train_: np.ndarray | None = None

    def _compute_statevectors(self, X_arr: np.ndarray) -> np.ndarray:
        """Computes statevectors for feature map evaluated on rows of X."""
        from qiskit.quantum_info import Statevector
        svs = [
            Statevector.from_instruction(self.feature_map_.assign_parameters(row)).data
            for row in X_arr
        ]
        return np.array(svs, dtype=np.complex128)

    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Computes quantum fidelity Gram matrix between X1 and X2."""
        Psi1 = self._compute_statevectors(X1)
        if X2 is None or X1 is X2:
            Psi2 = Psi1
        elif self.Psi_train_ is not None and np.array_equal(X2, self.X_train_):
            Psi2 = self.Psi_train_
        else:
            Psi2 = self._compute_statevectors(X2)

        Gram = np.abs(Psi1 @ Psi2.conj().T) ** 2
        return np.clip(Gram, 0.0, 1.0)

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "QuantumSVCModel":
        """Trains QSVC on quantum-reduced feature matrix X and labels y."""
        X_arr, y_arr = self._validate_inputs(X, y, is_fit=True)
        try:
            self.logger.info(
                f"Starting QSVC quantum kernel matrix calculation & training on {len(X_arr)} samples..."
            )
            self.X_train_ = X_arr
            self.Psi_train_ = self._compute_statevectors(X_arr)
            K_train = np.clip(np.abs(self.Psi_train_ @ self.Psi_train_.conj().T) ** 2, 0.0, 1.0)

            self.estimator_.fit(K_train, y_arr)
            
            # Fit Platt probability calibrator on training decision margins
            df_train = self.estimator_.decision_function(K_train)
            self.calibrator_.fit(df_train.reshape(-1, 1), y_arr)

            self.is_fitted_ = True
            self.logger.info(
                f"QSVC model trained successfully on {len(X_arr)} samples ({self.num_qubits} qubits)."
            )
        except Exception as e:
            raise ModelError(f"QSVC training failed: {e!s}")
        return self

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Evaluates hyperplane decision scores."""
        if not self.is_fitted_:
            raise ModelError("QuantumSVCModel must be fitted before calling decision_function().")
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        if self.X_train_ is not None:
            K_test = self._compute_kernel_matrix(X_arr, self.X_train_)
            return self.estimator_.decision_function(K_test)
        return self.estimator_.decision_function(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts calibrated class probabilities shape (N, 2)."""
        if not self.is_fitted_:
            raise ModelError("QuantumSVCModel must be fitted before calling predict_proba().")
        df = self.decision_function(X)
        if hasattr(self, "calibrator_") and self.calibrator_ is not None:
            return self.calibrator_.predict_proba(df.reshape(-1, 1))
        # Fallback sigmoid
        p1 = 1.0 / (1.0 + np.exp(-df))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class labels using quantum kernel evaluation with threshold awareness."""
        if not self.is_fitted_:
            raise ModelError("QuantumSVCModel must be fitted before calling predict().")
        thresh = getattr(self, "threshold_", None)
        if thresh is not None:
            probs = self.predict_proba(X)
            return (probs[:, 1] >= thresh).astype(int)
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        if self.X_train_ is not None:
            K_test = self._compute_kernel_matrix(X_arr, self.X_train_)
            return self.estimator_.predict(K_test)
        return self.estimator_.predict(X_arr)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Evaluates hyperplane decision scores."""
        if not self.is_fitted_:
            raise ModelError("QuantumSVCModel must be fitted before calling decision_function().")
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        if getattr(self, "X_train_", None) is not None:
            K_test = self._compute_kernel_matrix(X_arr, self.X_train_)
            return self.estimator_.decision_function(K_test)
        return self.estimator_.decision_function(X_arr)

    def get_circuit_metadata(self) -> dict[str, Any]:
        """Returns quantum circuit, backend, and library metadata."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "num_qubits": self.num_qubits,
            "feature_map_name": self.feature_map_name,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "circuit_depth": self.feature_map_.depth(),
            "circuit_gate_count": len(self.feature_map_.data),
            "backend": "aer_simulator (local)",
            "shots": self.shots,
            "random_seed": self.random_seed,
            "qiskit_version": qiskit.__version__,
            "qiskit_aer_version": qiskit_aer.__version__,
            "qiskit_machine_learning_version": qiskit_machine_learning.__version__,
        }

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        is_fit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Validates feature dimensions, NaNs, and target labels."""
        if X is None:
            raise ValidationError("Feature matrix X cannot be None.")

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if X_arr.size == 0 or X_arr.ndim != 2:
            raise ValidationError("Feature matrix X must be a non-empty 2D array.")

        if np.isnan(X_arr).any() or np.isinf(X_arr).any():
            raise ValidationError("Feature matrix contains NaN or infinite values.")

        if X_arr.shape[1] != self.num_qubits:
            raise ValidationError(
                f"Feature dimension mismatch for quantum circuit. "
                f"Expected {self.num_qubits} features (matching {self.num_qubits} qubits), "
                f"got {X_arr.shape[1]} features."
            )

        y_arr = None
        if y is not None:
            y_arr = y.to_numpy().ravel() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y).ravel()
            if y_arr.shape[0] != X_arr.shape[0]:
                raise ValidationError(
                    f"Sample count mismatch: X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]} samples."
                )
            unique_classes = np.unique(y_arr[~np.isnan(y_arr)])
            if not set(unique_classes).issubset({0, 1}):
                raise ValidationError(f"Target vector y contains non-binary values: {unique_classes}")

        return X_arr, y_arr


def save_quantum_model(model: QuantumSVCModel, filepath: str | Path) -> Path:
    """Serializes a QuantumSVCModel instance."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved Quantum model artifact '{model.model_name}' to '{path}'")
    return path


def load_quantum_model(filepath: str | Path) -> QuantumSVCModel:
    """Loads a serialized QuantumSVCModel instance."""
    path = Path(filepath)
    if not path.exists():
        raise ModelError(f"Model artifact file not found at: '{path}'")
    try:
        model = joblib.load(path)
        logger.info(f"Successfully loaded Quantum model '{model.model_name}' from '{path}'")
        return model
    except Exception as e:
        raise ModelError(f"Failed to load Quantum model artifact at '{path}': {e!s}")
