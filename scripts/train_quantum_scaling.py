#!/usr/bin/env python3
"""Train, evaluate, package, and persist 6-Qubit and 8-Qubit Quantum Models (QSVC-6Q, VQC-6Q, QSVC-8Q, VQC-8Q).

SIH26139 Quantum Machine Learning Scaling Engine for SukshmaDrishti.
Operates strictly on the canonical 80/20 train/test split without data fabrication or leakage.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from ml.artifacts import (
    ModelArtifact,
    compute_sha256_hash,
    get_environment_software_versions,
    save_artifact,
)
from ml.data import DataLoader
from ml.evaluation import (
    ModelEvaluator,
    calculate_calibration_data,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
)
from ml.exceptions import QMLError
from ml.preprocessing import (
    CLASSICAL_FEATURE_NAMES,
    PIPELINE_VERSION,
    RAW_FEATURE_NAMES,
    TARGET_COLUMN,
    BiomedicalPreprocessor,
    split_data,
)
from ml.quantum import (
    QuantumFeatureReducer,
    QuantumSVCModel,
    VariationalQuantumClassifierModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_quantum_scaling")


def train_and_evaluate_quantum_model(
    model_type: str,  # 'qsvc' or 'vqc'
    num_qubits: int,  # 6 or 8
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: BiomedicalPreprocessor,
    dataset_prov: dict[str, Any],
    version_suffix: str = "v1",
    random_seed: int = 42,
    shots: int = 1024,
    max_iter: int = 60,
    C: float = 1.0,
) -> tuple[ModelArtifact, dict[str, Any]]:
    """Trains a specific quantum model (QSVC or VQC) on k-qubit reduced features,
    evaluates on holdout test partition, and packages a validated ModelArtifact.
    """
    model_id = f"{model_type}_{num_qubits}q_{version_suffix}"
    logger.info(f"=== Starting Training for {model_id.upper()} ({num_qubits} Qubits, shots={shots}) ===")

    # 1. Transform raw training data to 11 classical features
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # 2. Leakage-safe Quantum Feature Reduction (SelectKBest fitted strictly on X_train_trans, y_train)
    t_red_0 = time.perf_counter()
    reducer = QuantumFeatureReducer(n_qubits=num_qubits, random_seed=random_seed)
    X_train_q = reducer.fit_transform(X_train_trans, y_train)
    X_test_q = reducer.transform(X_test_trans)
    red_time_ms = (time.perf_counter() - t_red_0) * 1000.0

    logger.info(f"[{model_id}] Selected {num_qubits} Features: {reducer.selected_feature_names_}")

    # 3. Instantiate Quantum Model
    fmap_name = "ZZFeatureMap"
    fmap_reps = 2
    fmap_ent = "linear"
    ansatz_name = "RealAmplitudes"
    ansatz_reps = 2
    ansatz_ent = "full"
    opt_name = "COBYLA"

    t_train_0 = time.perf_counter()
    if model_type == "qsvc":
        q_model = QuantumSVCModel(
            num_qubits=num_qubits,
            feature_map_name=fmap_name,
            reps=fmap_reps,
            entanglement=fmap_ent,
            C=C,
            shots=shots,
            random_seed=random_seed,
        )
        q_model.fit(X_train_q, y_train)
    elif model_type == "vqc":
        q_model = VariationalQuantumClassifierModel(
            num_qubits=num_qubits,
            feature_map_name=fmap_name,
            feature_map_reps=fmap_reps,
            feature_map_entanglement=fmap_ent,
            ansatz_name=ansatz_name,
            ansatz_reps=ansatz_reps,
            ansatz_entanglement=ansatz_ent,
            optimizer_name=opt_name,
            max_iter=max_iter,
            shots=shots,
            random_seed=random_seed,
        )
        q_model.fit(X_train_q, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_time_ms = (time.perf_counter() - t_train_0) * 1000.0
    logger.info(f"[{model_id}] Model fitted in {train_time_ms:.1f}ms")

    # 4. Construct ModelArtifact
    artifact_version = f"{version_suffix}.0"
    config_provenance = {
        "backend": {"name": "aer_simulator", "shots": shots, "use_gpu": False},
        "qubits": num_qubits,
        "feature_map": {"name": fmap_name, "reps": fmap_reps, "entanglement": fmap_ent},
        "ansatz": {"name": ansatz_name, "reps": ansatz_reps, "entanglement": ansatz_ent} if model_type == "vqc" else None,
        "optimizer": {"name": opt_name, "maxiter": max_iter} if model_type == "vqc" else None,
        "qsvc": {"C": C} if model_type == "qsvc" else None,
    }

    artifact = ModelArtifact(
        artifact_id=model_id,
        model_id=model_id,
        model_name=q_model.model_name,
        model_family="quantum",
        model_type=model_type,
        artifact_version=artifact_version,
        model=q_model,
        preprocessor=preprocessor,
        quantum_reducer=reducer,
        dataset_provenance=dataset_prov,
        configuration_provenance=config_provenance,
        metadata=q_model.get_circuit_metadata(),
        random_seed=random_seed,
    )
    artifact.metadata["train_time_ms"] = train_time_ms
    artifact.metadata["feature_reduction_time_ms"] = red_time_ms

    # 5. Evaluate on Holdout Test Partition (154 samples)
    evaluator = ModelEvaluator(random_seed=random_seed)
    logger.info(f"[{model_id}] Evaluating on {len(X_test_raw)} holdout test samples...")
    eval_report = evaluator.evaluate_model(artifact, X_test_raw, y_test, compute_bootstrap=False)

    artifact.evaluation_provenance = {
        "protocol": "80/20 Stratified Holdout Test Split",
        "sample_count": len(X_test_raw),
        "metrics": eval_report["metrics"],
        "confusion_matrix": eval_report["confusion_matrix"],
        "timing": eval_report["timing"],
    }

    # 6. Calculate Performance Curves on Holdout
    y_prob_pos = None
    if hasattr(artifact, "predict_proba"):
        try:
            probs = artifact.predict_proba(X_test_raw)
            y_prob_pos = probs[:, 1] if probs.ndim == 2 else probs.ravel()
        except Exception as e:
            logger.warning(f"predict_proba failed during curve generation for {model_id}: {e}")

    roc_curve = calculate_roc_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
    pr_curve = calculate_pr_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
    calibration = calculate_calibration_data(y_test, y_prob_pos) if y_prob_pos is not None else {
        "status": "unavailable",
        "reason": "Probabilities unavailable.",
    }

    # 7. Save binary artifact and manifest
    save_path = save_artifact(artifact, base_dir="models")
    logger.info(f"[{model_id}] Successfully saved artifact to '{save_path}'")

    # Benchmark summary record
    benchmark_entry = {
        "model_id": model_id,
        "name": artifact.model_name,
        "family": "quantum",
        "disease": "diabetes",
        "status": "SUCCESS",
        "features_used": num_qubits,
        "qubits": num_qubits,
        "feature_space": f"SelectKBest-Reduced Quantum Angle Encoding [-pi, pi] ({num_qubits} features: {reducer.selected_feature_names_})",
        "metrics": eval_report["metrics"],
        "confusion_matrix": eval_report["confusion_matrix"],
        "timings": {
            "train_time_ms": train_time_ms,
            "inference_time_ms": eval_report["timing"]["inference_time_ms"],
            "total_evaluation_time_ms": eval_report["timing"]["evaluation_time_ms"],
        },
        "curves": {
            "roc_curve": roc_curve,
            "pr_curve": pr_curve,
        },
        "calibration": calibration,
        "circuit_metadata": artifact.metadata,
    }

    return artifact, benchmark_entry


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate 6Q and 8Q quantum models.")
    parser.add_argument("--shots", type=int, default=1024, help="Aer simulator shots")
    parser.add_argument("--max-iter", type=int, default=60, help="VQC COBYLA max iterations")
    parser.add_argument("--raw-path", type=str, default="data/raw/pima_diabetes.csv")
    parser.add_argument("--version", type=str, default="v1", help="Version suffix (e.g. v1)")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        logger.error(f"Dataset not found at '{raw_path}'")
        sys.exit(1)

    loader = DataLoader(str(raw_path), target_column="Outcome")
    df, file_hash = loader.load()
    logger.info(f"Loaded dataset '{raw_path}' ({len(df)} rows, hash: {file_hash[:12]})")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = split_data(df, target_column="Outcome", test_size=0.2, random_seed=42)
    logger.info(f"Split data: {len(X_train)} train, {len(X_test)} holdout test")

    # Fit biomedical preprocessor on training data
    preprocessor = BiomedicalPreprocessor()
    preprocessor.fit(X_train)

    target_dist = {str(k): int(v) for k, v in df["Outcome"].value_counts().items()}
    dataset_prov = {
        "name": "pima_diabetes.csv (Real Raw)",
        "hash": file_hash,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": "Outcome",
        "target_distribution": target_dist,
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "raw_feature_count": len(RAW_FEATURE_NAMES),
    }

    models_to_train = [
        ("qsvc", 6),
        ("vqc", 6),
        ("qsvc", 8),
        ("vqc", 8),
    ]

    new_benchmark_entries = []
    scaling_records = []

    for m_type, q_count in models_to_train:
        art, bench_entry = train_and_evaluate_quantum_model(
            model_type=m_type,
            num_qubits=q_count,
            X_train_raw=X_train,
            y_train=y_train,
            X_test_raw=X_test,
            y_test=y_test,
            preprocessor=preprocessor,
            dataset_prov=dataset_prov,
            version_suffix=args.version,
            shots=args.shots,
            max_iter=args.max_iter,
        )
        new_benchmark_entries.append(bench_entry)
        m = bench_entry["metrics"]
        t = bench_entry["timings"]
        scaling_records.append({
            "configuration": f"{m_type.upper()}-{q_count}Q",
            "model_id": bench_entry["model_id"],
            "qubits": q_count,
            "features": q_count,
            "status": "EVALUATED",
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1_score": m.get("f1_score"),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "train_time_ms": t.get("train_time_ms"),
            "inference_time_ms": t.get("inference_time_ms"),
        })

    # Update data/processed/benchmark_results.json
    bench_file = Path("data/processed/benchmark_results.json")
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        existing_models = bench_data.get("models", [])
        # Replace or append new quantum models
        model_id_to_idx = {m["model_id"]: i for i, m in enumerate(existing_models)}
        for entry in new_benchmark_entries:
            m_id = entry["model_id"]
            if m_id in model_id_to_idx:
                existing_models[model_id_to_idx[m_id]] = entry
            else:
                existing_models.append(entry)
        bench_data["models"] = existing_models

        # Update quantum_scaling array
        existing_scaling = bench_data.get("quantum_scaling", [])
        scaling_map = {r.get("configuration"): i for i, r in enumerate(existing_scaling)}
        for rec in scaling_records:
            cfg = rec["configuration"]
            if cfg in scaling_map:
                existing_scaling[scaling_map[cfg]] = rec
            else:
                existing_scaling.append(rec)
        bench_data["quantum_scaling"] = existing_scaling

        with open(bench_file, "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=2)
        logger.info(f"Updated benchmark results file '{bench_file}' with 6Q and 8Q models.")

    print("\n========================================================")
    print("6Q & 8Q QUANTUM MODELS TRAINING & EVALUATION COMPLETE")
    print("========================================================")
    for rec in scaling_records:
        print(f"[{rec['configuration']}] Acc: {rec['accuracy']:.4f} | F1: {rec['f1_score']:.4f} | ROC-AUC: {rec['roc_auc']:.4f} | Train: {rec['train_time_ms']:.1f}ms")


if __name__ == "__main__":
    main()
