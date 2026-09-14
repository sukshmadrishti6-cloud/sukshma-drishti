"""Unit tests for ml.artifacts module (Packaging, Provenance, Integrity, Round-trip)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.artifacts import (
    ModelArtifact,
    compute_sha256_hash,
    get_environment_software_versions,
    list_saved_artifacts,
    load_artifact,
    save_artifact,
)
from ml.classical import LogisticRegressionModel, RandomForestModel
from ml.exceptions import ModelError, ValidationError
from ml.preprocessing import BiomedicalPreprocessor
from ml.quantum import QuantumFeatureReducer, QuantumSVCModel, VariationalQuantumClassifierModel


@pytest.fixture
def mock_dataset():
    """Small deterministic biomedical dataset for artifact round-trip tests."""
    np.random.seed(42)
    n = 20
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 5, size=n),
            "Glucose": np.random.uniform(70, 160, size=n),
            "BloodPressure": np.random.uniform(60, 90, size=n),
            "SkinThickness": np.random.uniform(10, 30, size=n),
            "Insulin": np.random.uniform(15, 150, size=n),
            "BMI": np.random.uniform(18, 32, size=n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.2, size=n),
            "Age": np.random.randint(21, 60, size=n),
            "Outcome": np.array([0, 1] * 10),
        }
    )
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def test_software_version_capture():
    """Verify runtime environment software packages are accurately recorded."""
    versions = get_environment_software_versions()
    assert "python" in versions
    assert "scikit-learn" in versions
    assert "qiskit" in versions
    assert "qiskit-aer" in versions
    assert "qiskit-machine-learning" in versions


def test_sha256_hashing(tmp_path: Path):
    """Verify deterministic SHA-256 calculation for strings and files."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("SIH26139 Quantum Platform", encoding="utf-8")

    h1 = compute_sha256_hash(test_file)
    h2 = compute_sha256_hash(b"SIH26139 Quantum Platform")
    assert h1 == h2
    assert len(h1) == 64


def test_classical_model_artifact_round_trip(tmp_path: Path, mock_dataset):
    """Verify Classical ModelArtifact saves, loads, and preserves preprocessing & predictions."""
    X, y = mock_dataset
    X_train, X_test = X.iloc[:16], X.iloc[16:]
    y_train, _y_test = y.iloc[:16], y.iloc[16:]

    prep = BiomedicalPreprocessor()
    X_train_trans = prep.fit_transform(X_train)

    # Test with Random Forest
    rf = RandomForestModel(random_seed=42)
    rf.fit(X_train_trans, y_train)

    orig_artifact = ModelArtifact(
        artifact_id="test_rf_artifact",
        model_id="rf_v1",
        model_name="Random Forest",
        model_family="classical",
        model_type="random_forest",
        model=rf,
        preprocessor=prep,
        dataset_provenance={"name": "mock_pima", "hash": "abc12345"},
        configuration_provenance={"n_estimators": 100},
        random_seed=42,
    )

    # Predict directly on RAW unscaled test data
    orig_preds = orig_artifact.predict(X_test)
    orig_probs = orig_artifact.predict_proba(X_test)

    # Save artifact
    art_path = save_artifact(orig_artifact, base_dir=tmp_path)
    assert art_path.exists()
    assert (tmp_path / "classical" / "test_rf_artifact_manifest.json").exists()

    # Load artifact
    loaded_artifact = load_artifact(art_path)
    assert loaded_artifact.artifact_id == "test_rf_artifact"
    assert loaded_artifact.integrity_hash is not None

    # Verify identical prediction on RAW unscaled test data
    loaded_preds = loaded_artifact.predict(X_test)
    loaded_probs = loaded_artifact.predict_proba(X_test)

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs)


def test_quantum_qsvc_artifact_round_trip(tmp_path: Path, mock_dataset):
    """Verify Quantum SVC ModelArtifact saves, loads, and preserves reducer & predictions."""
    X, y = mock_dataset
    X_train, X_test = X.iloc[:16], X.iloc[16:]
    y_train, _y_test = y.iloc[:16], y.iloc[16:]

    prep = BiomedicalPreprocessor()
    X_tr_c = prep.fit_transform(X_train)

    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_tr_q = reducer.fit_transform(X_tr_c, y_train)

    qsvc = QuantumSVCModel(num_qubits=4, reps=1, shots=256, random_seed=42)
    qsvc.fit(X_tr_q, y_train)

    orig_art = ModelArtifact(
        artifact_id="test_qsvc_artifact",
        model_id="qsvc_v1",
        model_name=qsvc.model_name,
        model_family="quantum",
        model_type="qsvc",
        model=qsvc,
        preprocessor=prep,
        quantum_reducer=reducer,
        random_seed=42,
    )

    orig_preds = orig_art.predict(X_test)

    art_path = save_artifact(orig_art, base_dir=tmp_path)
    loaded_art = load_artifact(art_path)

    loaded_preds = loaded_art.predict(X_test)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_quantum_vqc_artifact_round_trip(tmp_path: Path, mock_dataset):
    """Verify VQC ModelArtifact saves, loads, and restores predictions cleanly."""
    X, y = mock_dataset
    X_train, X_test = X.iloc[:16], X.iloc[16:]
    y_train, _y_test = y.iloc[:16], y.iloc[16:]

    prep = BiomedicalPreprocessor()
    X_tr_c = prep.fit_transform(X_train)

    reducer = QuantumFeatureReducer(n_qubits=4, random_seed=42)
    X_tr_q = reducer.fit_transform(X_tr_c, y_train)

    vqc = VariationalQuantumClassifierModel(num_qubits=4, max_iter=5, shots=256, random_seed=42)
    vqc.fit(X_tr_q, y_train)

    orig_art = ModelArtifact(
        artifact_id="test_vqc_artifact",
        model_id="vqc_v1",
        model_name=vqc.model_name,
        model_family="quantum",
        model_type="vqc",
        model=vqc,
        preprocessor=prep,
        quantum_reducer=reducer,
        random_seed=42,
    )

    orig_preds = orig_art.predict(X_test)

    art_path = save_artifact(orig_art, base_dir=tmp_path)
    loaded_art = load_artifact(art_path)

    loaded_preds = loaded_art.predict(X_test)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_path_traversal_protection(tmp_path: Path, mock_dataset):
    """Verify security check blocks path traversal in artifact IDs."""
    X, y = mock_dataset
    prep = BiomedicalPreprocessor().fit(X)
    lr = LogisticRegressionModel(random_seed=42).fit(prep.transform(X), y)

    bad_art = ModelArtifact(
        artifact_id="../../bad_path",
        model_id="lr_v1",
        model_name="LR",
        model_family="classical",
        model_type="logistic_regression",
        model=lr,
        preprocessor=prep,
    )

    with pytest.raises(ValidationError, match="Invalid artifact_id"):
        save_artifact(bad_art, base_dir=tmp_path)

    with pytest.raises(ValidationError, match="Path traversal sequence"):
        load_artifact("../evil_id", base_dir=tmp_path)


def test_missing_artifact_raises_model_error(tmp_path: Path):
    """Verify requesting non-existent artifact raises ModelError without silent fallback."""
    with pytest.raises(ModelError, match="not found"):
        load_artifact("non_existent_artifact_id_12345", base_dir=tmp_path)


def test_corrupted_artifact_raises_model_error(tmp_path: Path):
    """Verify corrupted artifact binary fails safely."""
    corrupt_file = tmp_path / "classical" / "corrupt_art.joblib"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_bytes(b"CORRUPTED_NOT_A_VALID_JOBLIB_FILE")

    with pytest.raises(ModelError, match="Failed to deserialize"):
        load_artifact(corrupt_file)


def test_no_secret_leakage_in_metadata_summary(mock_dataset):
    """Verify summary dict contains only public metadata and zero secrets."""
    X, y = mock_dataset
    prep = BiomedicalPreprocessor().fit(X)
    lr = LogisticRegressionModel().fit(prep.transform(X), y)

    art = ModelArtifact(
        artifact_id="safe_artifact",
        model_id="lr_v1",
        model_name="Logistic Regression",
        model_family="classical",
        model_type="logistic_regression",
        model=lr,
        preprocessor=prep,
    )

    summary = art.get_summary()
    summary_str = str(summary)
    assert "SUPABASE_SERVICE_ROLE_KEY" not in summary_str
    assert "password" not in summary_str.lower()
    assert "secret" not in summary_str.lower()


def test_list_saved_artifacts(tmp_path: Path, mock_dataset):
    """Verify artifact discovery lists all saved manifests."""
    X, y = mock_dataset
    prep = BiomedicalPreprocessor().fit(X)
    lr = LogisticRegressionModel().fit(prep.transform(X), y)

    art = ModelArtifact(
        artifact_id="listed_art_1",
        model_id="lr_v1",
        model_name="Logistic Regression",
        model_family="classical",
        model_type="logistic_regression",
        model=lr,
        preprocessor=prep,
    )
    save_artifact(art, base_dir=tmp_path)

    manifests = list_saved_artifacts(base_dir=tmp_path)
    assert len(manifests) == 1
    assert manifests[0]["artifact_id"] == "listed_art_1"
