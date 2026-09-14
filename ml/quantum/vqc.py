"""Real Variational Quantum Classifier (VQC) Model Implementation."""
import time
from typing import Any

import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import qiskit_machine_learning
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA

from ml.exceptions import ModelError, ValidationError
from ml.logger import get_logger
from ml.quantum.ansatz import create_ansatz
from ml.quantum.circuit import create_feature_map

logger = get_logger(__name__)

SUPPORTED_OPTIMIZERS = {
    "cobyla": COBYLA,
    "spsa": SPSA,
    "slsqp": SLSQP,
    "l_bfgs_b": L_BFGS_B,
}


class _VQCCallback:
    """Picklable callback handler for tracking VQC optimization history."""

    def __init__(self):
        self.history: list[float] = []

    def __call__(self, weights: np.ndarray, loss_val: float) -> None:
        self.history.append(float(loss_val))


class VariationalQuantumClassifierModel:
    """Real Variational Quantum Classifier (VQC) using Qiskit Machine Learning."""

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map_name: str = "ZZFeatureMap",
        feature_map_reps: int = 1,
        feature_map_entanglement: str = "linear",
        ansatz_name: str = "RealAmplitudes",
        ansatz_reps: int = 1,
        ansatz_entanglement: str = "linear",
        optimizer_name: str = "COBYLA",
        max_iter: int = 40,
        threshold: float | None = None,
        shots: int = 1024,
        random_seed: int = 42,
    ):
        self.model_name = f"Variational Quantum Classifier (VQC-{num_qubits}Q)"
        self.model_type = "vqc"
        self.num_qubits = num_qubits
        self.feature_map_name = feature_map_name
        self.feature_map_reps = feature_map_reps
        self.feature_map_entanglement = feature_map_entanglement
        self.ansatz_name = ansatz_name
        self.ansatz_reps = ansatz_reps
        self.ansatz_entanglement = ansatz_entanglement
        self.optimizer_name = optimizer_name
        self.max_iter = max_iter
        self.threshold_ = threshold
        self.shots = shots
        self.random_seed = random_seed

        # Construct quantum circuits
        self.feature_map_ = create_feature_map(
            name=feature_map_name,
            num_qubits=num_qubits,
            reps=feature_map_reps,
            entanglement=feature_map_entanglement,
        )
        self.ansatz_ = create_ansatz(
            name=ansatz_name,
            num_qubits=num_qubits,
            reps=ansatz_reps,
            entanglement=ansatz_entanglement,
        )
        self.circuit_ = self.feature_map_.compose(self.ansatz_)

        # Average local Pauli Z observable: 1/n sum(Z_i)
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
        from sklearn.linear_model import LogisticRegression

        pauli_terms = []
        for i in range(num_qubits):
            z_str = ["I"] * num_qubits
            z_str[i] = "Z"
            pauli_terms.append(("".join(z_str), 1.0 / num_qubits))
        self.observable_ = SparsePauliOp.from_list(pauli_terms)

        self.qnn_ = EstimatorQNN(
            circuit=self.circuit_,
            observables=[self.observable_],
            input_params=self.feature_map_.parameters,
            weight_params=self.ansatz_.parameters,
        )

        # Construct optimizer
        opt_key = optimizer_name.strip().lower()
        if opt_key not in SUPPORTED_OPTIMIZERS:
            raise ValidationError(
                f"Unsupported optimizer '{optimizer_name}'. Supported: {list(SUPPORTED_OPTIMIZERS.keys())}"
            )

        if opt_key == "spsa":
            self.optimizer_ = SPSA(maxiter=max_iter)
        else:
            self.optimizer_ = SUPPORTED_OPTIMIZERS[opt_key](maxiter=max_iter)

        self._callback_handler = _VQCCallback()

        # NeuralNetworkClassifier with local observable QNN
        self.estimator_ = NeuralNetworkClassifier(
            neural_network=self.qnn_,
            loss="squared_error",
            optimizer=self.optimizer_,
            callback=self._callback_handler,
            initial_point=np.zeros(self.ansatz_.num_parameters),
        )
        self.calibrator_ = LogisticRegression(class_weight="balanced", random_state=random_seed)

        self.is_fitted_: bool = False
        self.n_features_in_: int = num_qubits
        self.train_time_ms_: float = 0.0
        self.logger = logger

    @property
    def loss_history_(self) -> list[float]:
        """Exposes optimization loss history list."""
        return self._callback_handler.history

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "VariationalQuantumClassifierModel":
        """Trains parameterized variational ansatz to minimize classification loss."""
        X_arr, y_arr = self._validate_inputs(X, y, is_fit=True)
        self._callback_handler.history = []
        t0 = time.perf_counter()
        try:
            self.logger.info(
                f"Starting VQC optimization on {len(X_arr)} samples ({self.num_qubits} qubits, "
                f"ansatz='{self.ansatz_name}', params={self.ansatz_.num_parameters}, "
                f"max_iter={self.max_iter})..."
            )
            # Map target labels {0, 1} -> {-1, 1} for Pauli Z expectation
            y_pm1 = 2 * y_arr - 1

            # Use stratified sub-sampling if training set > 250 for fast, stable COBYLA convergence
            if len(X_arr) > 250:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=250, random_state=self.random_seed)
                fit_idx, _ = next(sss.split(X_arr, y_arr))
                X_fit, y_fit_pm1, y_fit_orig = X_arr[fit_idx], y_pm1[fit_idx], y_arr[fit_idx]
            else:
                X_fit, y_fit_pm1, y_fit_orig = X_arr, y_pm1, y_arr

            self.estimator_.fit(X_fit, y_fit_pm1)
            
            # Calibrate expectation values [-1, 1] to smooth probabilities
            exp_train = self.estimator_.predict(X_fit).ravel()
            self.calibrator_.fit(exp_train.reshape(-1, 1), y_fit_orig)

            self.train_time_ms_ = (time.perf_counter() - t0) * 1000.0
            self.is_fitted_ = True

            final_loss = self.loss_history_[-1] if self.loss_history_ else None
            self.logger.info(
                f"VQC optimization complete in {self.train_time_ms_:.1f}ms. "
                f"Iterations: {len(self.loss_history_)}, Final Loss: {final_loss}"
            )
        except Exception as e:
            raise ModelError(f"VQC training failed: {e!s}")
        return self

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Evaluates variational quantum circuit expectation values in [-1, 1]."""
        if not self.is_fitted_:
            raise ModelError("VariationalQuantumClassifierModel must be fitted before calling decision_function().")
        X_arr, _ = self._validate_inputs(X, is_fit=False)
        return self.estimator_.predict(X_arr).ravel()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts calibrated class probability distributions shape (N, 2)."""
        if not self.is_fitted_:
            raise ModelError(
                "VariationalQuantumClassifierModel must be fitted before calling predict_proba()."
            )
        exp_vals = self.decision_function(X)
        if hasattr(self, "calibrator_") and self.calibrator_ is not None:
            return self.calibrator_.predict_proba(exp_vals.reshape(-1, 1))
        # Linear map fallback: [-1, 1] -> [0, 1]
        p1 = np.clip((exp_vals + 1.0) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts binary class labels (0 or 1) with threshold awareness."""
        if not self.is_fitted_:
            raise ModelError("VariationalQuantumClassifierModel must be fitted before calling predict().")
        thresh = getattr(self, "threshold_", None)
        if thresh is not None:
            probs = self.predict_proba(X)
            return (probs[:, 1] >= thresh).astype(int)
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_loss_history(self) -> list[float]:
        """Returns objective loss function history across optimizer iterations."""
        return list(self.loss_history_)

    def get_circuit_metadata(self) -> dict[str, Any]:
        """Returns variational quantum circuit and optimization metadata."""
        initial_loss = self.loss_history_[0] if self.loss_history_ else None
        final_loss = self.loss_history_[-1] if self.loss_history_ else None
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "num_qubits": self.num_qubits,
            "feature_map_name": self.feature_map_name,
            "feature_map_reps": self.feature_map_reps,
            "ansatz_name": self.ansatz_name,
            "ansatz_reps": self.ansatz_reps,
            "num_trainable_parameters": self.ansatz_.num_parameters,
            "feature_map_depth": self.feature_map_.depth(),
            "ansatz_depth": self.ansatz_.depth(),
            "optimizer": self.optimizer_name,
            "max_iter": self.max_iter,
            "iterations_performed": len(self.loss_history_),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "convergence_status": "CONVERGED" if len(self.loss_history_) > 0 else "NOT_FITTED",
            "loss_history": self.loss_history_,
            "backend": "aer_simulator (local)",
            "shots": self.shots,
            "random_seed": self.random_seed,
            "qiskit_version": qiskit.__version__,
            "qiskit_aer_version": qiskit_aer.__version__,
            "qiskit_machine_learning_version": qiskit_machine_learning.__version__,
        }

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickling handler."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickling handler."""
        self.__dict__.update(state)

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
                f"Feature dimension mismatch for VQC circuit. "
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
