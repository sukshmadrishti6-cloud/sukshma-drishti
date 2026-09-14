"""Unified Model Artifact Packaging, Provenance, and Reproducibility Module."""
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import qiskit_machine_learning
import sklearn
import xgboost

from ml.exceptions import ModelError, ValidationError
from ml.logger import get_logger
from ml.preprocessing.pipeline import BiomedicalPreprocessor
from ml.quantum.reduction import QuantumFeatureReducer

logger = get_logger(__name__)

VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


def get_environment_software_versions() -> dict[str, str]:
    """Captures exact runtime software package versions."""
    return {
        "python": sys.version.split()[0],
        "scikit-learn": getattr(sklearn, "__version__", "unknown"),
        "xgboost": getattr(xgboost, "__version__", "unknown"),
        "qiskit": getattr(qiskit, "__version__", "unknown"),
        "qiskit-aer": getattr(qiskit_aer, "__version__", "unknown"),
        "qiskit-machine-learning": getattr(qiskit_machine_learning, "__version__", "unknown"),
    }


def compute_sha256_hash(data: bytes | str | Path) -> str:
    """Computes deterministic SHA-256 hash for bytes or files."""
    hasher = hashlib.sha256()
    if isinstance(data, (str, Path)) and Path(data).exists():
        with open(data, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    else:
        raise ValidationError(f"Cannot compute SHA-256 for data of type: {type(data)}")
    return hasher.hexdigest()


@dataclass
class ModelArtifact:
    """Unified container encapsulating model, fitted preprocessing, configuration, and provenance."""

    artifact_id: str
    model_id: str
    model_name: str
    model_family: str  # 'classical' | 'quantum'
    model_type: str  # 'logistic_regression' | 'svm' | 'random_forest' | 'xgboost' | 'qsvc' | 'vqc'
    model: Any
    preprocessor: BiomedicalPreprocessor
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    quantum_reducer: QuantumFeatureReducer | None = None
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    configuration_provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=get_environment_software_versions)
    random_seed: int = 42
    integrity_hash: str | None = None
    artifact_version: str = "v3.0"
    evaluation_provenance: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Executes full inference pipeline: Raw Features -> Preprocessor -> [Reducer] -> Model.

        Includes guard against double preprocessing if transformed data is already supplied.
        """
        # Guard against double preprocessing
        if isinstance(X, np.ndarray) and X.ndim == 2:
            if self.model_family == "classical" and hasattr(self.preprocessor, "get_feature_names") and X.shape[1] == len(self.preprocessor.get_feature_names()):
                preds = self.model.predict(X)
                return np.asarray(preds).ravel()
            elif self.model_family == "quantum" and self.quantum_reducer is not None and X.shape[1] == self.quantum_reducer.n_qubits:
                preds = self.model.predict(X)
                return np.asarray(preds).ravel()

        X_clean = self.preprocessor.transform(X)
        if self.model_family == "quantum" and self.quantum_reducer is not None:
            X_input = self.quantum_reducer.transform(X_clean)
        else:
            X_input = X_clean
        preds = self.model.predict(X_input)
        return np.asarray(preds).ravel()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Executes full probability inference pipeline with double-preprocessing guard."""
        # Guard against double preprocessing
        if isinstance(X, np.ndarray) and X.ndim == 2:
            if self.model_family == "classical" and hasattr(self.preprocessor, "get_feature_names") and X.shape[1] == len(self.preprocessor.get_feature_names()):
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    probs_arr = np.asarray(probs)
                    return np.column_stack([1.0 - probs_arr, probs_arr]) if probs_arr.ndim == 1 else probs_arr
            elif self.model_family == "quantum" and self.quantum_reducer is not None and X.shape[1] == self.quantum_reducer.n_qubits:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    probs_arr = np.asarray(probs)
                    return np.column_stack([1.0 - probs_arr, probs_arr]) if probs_arr.ndim == 1 else probs_arr

        X_clean = self.preprocessor.transform(X)
        if self.model_family == "quantum" and self.quantum_reducer is not None:
            X_input = self.quantum_reducer.transform(X_clean)
        else:
            X_input = X_clean
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)
            probs_arr = np.asarray(probs)
            if probs_arr.ndim == 1:
                return np.column_stack([1.0 - probs_arr, probs_arr])
            return probs_arr
        raise ModelError(f"Model '{self.model_name}' does not support probability predictions.")

    def get_summary(self) -> dict[str, Any]:
        """Returns structured metadata summary distinguishing raw, engineered, and classical schemas."""
        schema_prov = {}
        if hasattr(self.preprocessor, "get_metadata"):
            schema_prov = self.preprocessor.get_metadata()
        elif hasattr(self.preprocessor, "get_feature_names"):
            feat_names = self.preprocessor.get_feature_names()
            schema_prov = {
                "raw_features_count": 8,
                "engineered_features_count": len(feat_names) - 8,
                "classical_features_count": len(feat_names),
                "classical_feature_names": feat_names,
            }

        q_prov = {}
        if self.quantum_reducer is not None and hasattr(self.quantum_reducer, "get_selection_stats"):
            q_prov = self.quantum_reducer.get_selection_stats()

        return {
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_type": self.model_type,
            "artifact_version": getattr(self, "artifact_version", "v3.0"),
            "pipeline_version": getattr(self.preprocessor, "pipeline_version", "v2.0"),
            "created_at": self.created_at,
            "random_seed": self.random_seed,
            "dataset_provenance": self.dataset_provenance,
            "schema_provenance": schema_prov,
            "quantum_provenance": q_prov,
            "configuration_provenance": self.configuration_provenance,
            "evaluation_provenance": getattr(self, "evaluation_provenance", {}),
            "software_versions": self.software_versions,
            "metadata": self.metadata,
            "integrity_hash": self.integrity_hash,
        }



def save_artifact(
    artifact: ModelArtifact,
    base_dir: str | Path = "models",
) -> Path:
    """Persists a ModelArtifact bundle and its companion JSON metadata manifest.

    Args:
        artifact: ModelArtifact container.
        base_dir: Root storage directory.

    Returns:
        Path: Path to saved artifact binary file.
    """
    # Security: Validate identifier against path traversal attacks
    if not VALID_ID_PATTERN.match(artifact.artifact_id):
        raise ValidationError(
            f"Invalid artifact_id '{artifact.artifact_id}'. Must be alphanumeric with '-' or '_' only."
        )

    subfolder = "quantum" if artifact.model_family == "quantum" else "classical"
    base_p = Path(base_dir)
    if base_p.name == subfolder:
        target_dir = base_p
    else:
        target_dir = base_p / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    artifact_file = target_dir / f"{artifact.artifact_id}.joblib"
    manifest_file = target_dir / f"{artifact.artifact_id}_manifest.json"

    # Serialize binary artifact bundle
    joblib.dump(artifact, artifact_file)

    # Compute exact SHA-256 hash of written file
    file_hash = compute_sha256_hash(artifact_file)
    artifact.integrity_hash = file_hash

    # Write human-readable companion manifest with exact hash
    manifest_payload = artifact.get_summary()
    manifest_payload["integrity_hash"] = file_hash
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    logger.info(
        f"Saved ModelArtifact '{artifact.artifact_id}' ({artifact.model_name}) to '{artifact_file}' "
        f"[SHA-256: {file_hash[:12]}]"
    )
    return artifact_file


def load_artifact(
    artifact_path_or_id: str | Path,
    base_dir: str | Path = "models",
    verify_integrity: bool = True,
) -> ModelArtifact:
    """Loads and verifies a ModelArtifact bundle.

    Args:
        artifact_path_or_id: File path or artifact ID.
        base_dir: Root storage directory to resolve IDs.
        verify_integrity: Whether to verify SHA-256 checksum against manifest.

    Returns:
        ModelArtifact: Restored and validated ModelArtifact instance.
    """
    input_str = str(artifact_path_or_id).strip()

    # Security check: Disallow relative path traversal components like '..'
    if ".." in input_str:
        raise ValidationError(f"Path traversal sequence '..' detected in identifier: '{input_str}'")

    if "/" not in input_str and "\\" not in input_str:
        if not VALID_ID_PATTERN.match(input_str):
            raise ValidationError(f"Invalid artifact identifier: '{input_str}'")
        candidate_paths = [
            Path(base_dir) / "classical" / f"{input_str}.joblib",
            Path(base_dir) / "quantum" / f"{input_str}.joblib",
            Path(base_dir) / "cardiovascular" / "classical" / f"{input_str}.joblib",
            Path(base_dir) / "cardiovascular" / "quantum" / f"{input_str}.joblib",
            Path(base_dir) / "cardiovascular" / "quantum" / "quantum" / f"{input_str}.joblib",
            Path(base_dir) / "cardiovascular" / f"{input_str}.joblib",
            Path(base_dir) / "cancer" / "classical" / f"{input_str}.joblib",
            Path(base_dir) / "cancer" / "quantum" / f"{input_str}.joblib",
            Path(base_dir) / "cancer" / "quantum" / "quantum" / f"{input_str}.joblib",
            Path(base_dir) / "cancer" / f"{input_str}.joblib",
            Path(base_dir) / f"{input_str}.joblib",
        ]

        target_path = None
        for cp in candidate_paths:
            if cp.exists():
                target_path = cp
                break

        if target_path is None:
            # Fallback search in base_dir recursively
            found = list(Path(base_dir).glob(f"**/{input_str}.joblib"))
            if found:
                target_path = found[0]

        if target_path is None:
            raise ModelError(f"Model artifact ID '{input_str}' not found in '{base_dir}'.")
    else:
        target_path = Path(input_str).resolve()
        if not target_path.exists():
            raise ModelError(f"Model artifact file not found at '{target_path}'.")

    try:
        artifact = joblib.load(target_path)
    except Exception as e:
        raise ModelError(f"Failed to deserialize model artifact at '{target_path}': {e!s}")

    if not isinstance(artifact, ModelArtifact):
        raise ModelError(
            f"Loaded file at '{target_path}' is not a valid ModelArtifact instance (got {type(artifact)})."
        )

    # Integrity verification
    current_hash = compute_sha256_hash(target_path)
    artifact.integrity_hash = current_hash

    if verify_integrity:
        manifest_path = target_path.parent / f"{artifact.artifact_id}_manifest.json"
        if not manifest_path.exists():
            raise ModelError(f"Artifact manifest missing for '{artifact.artifact_id}'. Loading aborted.")
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                expected_hash = manifest_data.get("integrity_hash")
                if not expected_hash:
                    raise ModelError(f"Manifest for '{artifact.artifact_id}' is missing integrity_hash.")
                if current_hash != expected_hash:
                    raise ModelError(
                        f"Artifact integrity hash mismatch! Expected {expected_hash}, got {current_hash}. "
                        f"The model artifact file may be corrupted or tampered."
                    )
                # Keep manifest accessible
                artifact.manifest = type('Manifest', (object,), manifest_data)()
                # Basic attrs from dict
                for k, v in manifest_data.items():
                    setattr(artifact.manifest, k, v)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Could not verify manifest integrity for '{artifact.artifact_id}': {e!s}")

    logger.info(f"Successfully loaded ModelArtifact '{artifact.artifact_id}' from '{target_path}'")
    return artifact


def list_saved_artifacts(base_dir: str | Path = "models") -> list[dict[str, Any]]:
    """Discovers all saved model artifact manifests in the models directory."""
    root = Path(base_dir)
    if not root.exists():
        return []

    manifests = []
    for manifest_path in root.glob("**/*_manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["manifest_file"] = str(manifest_path)
                manifests.append(data)
        except Exception as e:
            logger.warning(f"Could not read manifest at '{manifest_path}': {e!s}")

    return sorted(manifests, key=lambda x: x.get("created_at", ""), reverse=True)
