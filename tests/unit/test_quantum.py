"""Unit tests for ml.quantum module (Qiskit Circuit, Kernel, Reducer, QSVC, Scaling, VQC)."""
from pathlib import Path

import numpy as np
import pytest
import qiskit

from ml.evaluation import ModelEvaluator
from ml.exceptions import QuantumCircuitError, ValidationError
from ml.quantum import (
    QuantumFeatureReducer,
    QuantumSVCModel,
    VariationalQuantumClassifierModel,
    create_ansatz,
    create_feature_map,
    create_quantum_kernel,
    get_quantum_backend,
    load_quantum_model,
    save_quantum_model,
)


@pytest.fixture
def small_quantum_fixture():
    """Small deterministic dataset for quantum circuit execution tests."""
    np.random.seed(42)
    n = 12
    X = np.random.uniform(-2.0, 2.0, size=(n, 8))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test_quantum_package_import():
    """Verify quantum module package structure imports cleanly."""
    import ml.quantum
    assert hasattr(ml.quantum, "create_feature_map")
    assert hasattr(ml.quantum, "create_ansatz")
    assert hasattr(ml.quantum, "QuantumFeatureReducer")
    assert hasattr(ml.quantum, "QuantumSVCModel")
    assert hasattr(ml.quantum, "VariationalQuantumClassifierModel")


def test_local_aer_backend_availability():
    """Verify local AerSimulator backend initializes cleanly without external credentials."""
    backend = get_quantum_backend(name="aer_simulator", shots=512, random_seed=42)
    assert backend is not None
    assert backend.name == "aer_simulator"


def test_feature_map_construction():
    """Verify real Qiskit feature map creation with correct qubit count and depth."""
    fm = create_feature_map(name="ZZFeatureMap", num_qubits=4, reps=2, entanglement="linear")
    assert isinstance(fm, qiskit.QuantumCircuit)
    assert fm.num_qubits == 4
    assert fm.depth() > 0


def test_ansatz_construction():
    """Verify real Qiskit parameterized ansatz creation and parameter counting."""
    ansatz_ra = create_ansatz(name="RealAmplitudes", num_qubits=4, reps=1, entanglement="full")
    assert isinstance(ansatz_ra, qiskit.QuantumCircuit)
    assert ansatz_ra.num_qubits == 4
    assert ansatz_ra.num_parameters > 0

    ansatz_su2 = create_ansatz(name="EfficientSU2", num_qubits=4, reps=1, entanglement="linear")
    assert ansatz_su2.num_parameters > 0


def test_invalid_ansatz_name_raises_error():
    """Verify unknown ansatz name raises QuantumCircuitError."""
    with pytest.raises(QuantumCircuitError, match="Unsupported ansatz type"):
        create_ansatz(name="InvalidAnsatz", num_qubits=4)


def test_quantum_feature_reducer_leakage_safe(small_quantum_fixture):
    """Verify QuantumFeatureReducer reduces 8 features -> 4 qubits and scales within [-pi, pi]."""
    X, y = small_quantum_fixture
    X_train = X[:8]
    y_train = y[:8]
    X_test = X[8:]

    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_train_reduced = reducer.fit_transform(X_train, y_train)
    X_test_reduced = reducer.transform(X_test)

    assert X_train_reduced.shape == (8, 4)
    assert X_test_reduced.shape == (4, 4)
    assert np.all(X_train_reduced >= -np.pi - 1e-5)
    assert np.all(X_train_reduced <= np.pi + 1e-5)
    assert reducer.scaler_.data_min_.shape == (4,)


def test_quantum_feature_reducer_explained_variance(small_quantum_fixture):
    X, y = small_quantum_fixture
    r4 = QuantumFeatureReducer(n_qubits=4, random_seed=42).fit(X, y)
    v4 = r4.get_variance_stats()
    assert v4["n_components"] == 4
    assert "SelectKBest" in v4["method"]
    assert "scores" in v4


def test_quantum_kernel_creation():
    """Verify FidelityQuantumKernel instantiates with real Qiskit circuit."""
    fm = create_feature_map(name="ZZFeatureMap", num_qubits=4, reps=1)
    kernel = create_quantum_kernel(fm)
    assert kernel is not None


def test_qsvc_training_prediction_and_metadata(small_quantum_fixture):
    """Verify real QSVC trains on quantum features, predicts labels, and reports circuit metadata."""
    X, y = small_quantum_fixture
    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_trans = reducer.fit_transform(X, y)

    X_train, X_test = X_trans[:8], X_trans[8:]
    y_train, _y_test = y[:8], y[8:]

    qsvc = QuantumSVCModel(num_qubits=4, feature_map_name="ZZFeatureMap", reps=1, C=1.0, random_seed=42)
    qsvc.fit(X_train, y_train)

    assert qsvc.is_fitted_ is True

    preds = qsvc.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})

    probs = qsvc.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    metadata = qsvc.get_circuit_metadata()
    assert metadata["num_qubits"] == 4
    assert metadata["circuit_depth"] > 0
    assert metadata["backend"] == "aer_simulator (local)"
    assert "qiskit_version" in metadata


def test_vqc_training_prediction_and_loss_history(small_quantum_fixture):
    """Verify real VQC trains, minimizes loss, records loss history, and predicts probabilities."""
    X, y = small_quantum_fixture
    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_trans = reducer.fit_transform(X, y)

    X_train, X_test = X_trans[:8], X_trans[8:]
    y_train, _y_test = y[:8], y[8:]

    vqc = VariationalQuantumClassifierModel(
        num_qubits=4,
        ansatz_name="RealAmplitudes",
        ansatz_reps=1,
        optimizer_name="COBYLA",
        max_iter=20,
        random_seed=42,
    )
    vqc.fit(X_train, y_train)

    assert vqc.is_fitted_ is True
    loss_hist = vqc.get_loss_history()
    assert len(loss_hist) > 0

    preds = vqc.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    probs = vqc.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    meta = vqc.get_circuit_metadata()
    assert meta["num_trainable_parameters"] > 0
    assert meta["convergence_status"] == "CONVERGED"


def test_vqc_dimension_mismatch_raises_validation_error():
    """Verify VQC input dimension mismatch raises ValidationError."""
    vqc = VariationalQuantumClassifierModel(num_qubits=4)
    bad_X = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValidationError, match="Feature dimension mismatch"):
        vqc.fit(bad_X, y)


def test_vqc_evaluator_integration(small_quantum_fixture):
    """Verify Step 7 ModelEvaluator evaluates VariationalQuantumClassifierModel."""
    X, y = small_quantum_fixture
    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_trans = reducer.fit_transform(X, y)

    X_train, X_test = X_trans[:8], X_trans[8:]
    y_train, y_test = y[:8], y[8:]

    vqc = VariationalQuantumClassifierModel(num_qubits=4, max_iter=8, random_seed=42)
    vqc.fit(X_train, y_train)

    evaluator = ModelEvaluator(random_seed=42)
    report = evaluator.evaluate_model(vqc, X_test, y_test, compute_bootstrap=False)

    assert report["model_type"] == "vqc"
    assert report["metrics"]["accuracy"] >= 0.0
    assert report["metrics"]["roc_auc"] is not None
    assert "confusion_matrix" in report


def test_vqc_model_persistence(tmp_path: Path, small_quantum_fixture):
    """Verify save_quantum_model and load_quantum_model serialize and restore VQC."""
    X, y = small_quantum_fixture
    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_trans = reducer.fit_transform(X, y)

    vqc = VariationalQuantumClassifierModel(num_qubits=4, max_iter=8, random_seed=42)
    vqc.fit(X_trans[:8], y[:8])

    save_path = tmp_path / "vqc_test.joblib"
    save_quantum_model(vqc, save_path)

    loaded = load_quantum_model(save_path)
    assert loaded.num_qubits == 4
    assert loaded.is_fitted_ is True

    preds = loaded.predict(X_trans[8:])
    assert len(preds) == 4
