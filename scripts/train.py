#!/usr/bin/env python3
"""CLI script for training classical and quantum machine learning models with full artifact packaging.

Production Model Rebuild & Benchmark Engine for SukshmaDrishti (SIH26139).

Usage:
    python scripts/train.py --model all --version v3
"""
import argparse
import hashlib
import json
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
    load_artifact,
    save_artifact,
)
from ml.classical import SUPPORTED_CLASSICAL_MODELS, ClassicalModelFactory
from ml.data import DataLoader
from ml.evaluation import (
    ModelEvaluator,
    calculate_calibration_data,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
    evaluate_cross_validation,
)
from ml.exceptions import ModelError, PreprocessingError, QMLError, ValidationError
from ml.preprocessing import (
    CLASSICAL_FEATURE_NAMES,
    ENGINEERED_FEATURE_NAMES,
    FEATURES_WITH_INVALID_ZEROS,
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


def main():
    parser = argparse.ArgumentParser(
        description="Train classical and quantum ML models with full provenance artifacts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model family to train (classical, quantum, all, or specific model name: lr, svm, rf, xgboost, qsvc, vqc)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        help="Artifact version suffix (default: v3 for Phase B canonical rebuild)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--classical-config",
        type=str,
        default="configs/classical.yaml",
        help="Path to classical configuration file",
    )
    parser.add_argument(
        "--quantum-config",
        type=str,
        default="configs/quantum.yaml",
        help="Path to quantum configuration file",
    )
    parser.add_argument(
        "--allow-synthetic-fixture",
        action="store_true",
        help="Allow running on synthetic test fixture only for isolated unit tests",
    )
    args = parser.parse_args()

    print("==================================================")
    print(f"SIH26139 - Production Model Rebuild ({args.version.upper()})")
    print("==================================================")

    config_path = Path(args.config)
    classical_config_path = Path(args.classical_config)
    quantum_config_path = Path(args.quantum_config)

    base_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    classical_cfg = {}
    if classical_config_path.exists():
        with open(classical_config_path, "r", encoding="utf-8") as f:
            classical_cfg = yaml.safe_load(f) or {}

    quantum_cfg = {}
    if quantum_config_path.exists():
        with open(quantum_config_path, "r", encoding="utf-8") as f:
            quantum_cfg = yaml.safe_load(f) or {}

    data_cfg = base_cfg.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/pima_diabetes.csv")
    target_column = data_cfg.get("target_column", TARGET_COLUMN)
    test_size = data_cfg.get("test_size", 0.2)
    random_seed = base_cfg.get("random_seed", 42)

    # 1. Dataset Verification & Ingestion
    if Path(raw_path).exists():
        loader = DataLoader(raw_path, target_column=target_column)
        df, file_hash = loader.load()
        dataset_name = "pima_diabetes.csv (Real Raw)"
        print(f"[OK] Verified canonical dataset '{raw_path}' ({len(df)} rows, hash: {file_hash[:12]}...)")
    else:
        if args.allow_synthetic_fixture or "--allow-synthetic-fixture" in sys.argv:
            print("WARNING: Using synthetic fixture (--allow-synthetic-fixture explicitly specified).")
            np.random.seed(random_seed)
            n = 100
            df = pd.DataFrame(
                {
                    "Pregnancies": np.random.randint(0, 10, size=n),
                    "Glucose": np.random.uniform(70, 180, size=n),
                    "BloodPressure": np.random.uniform(60, 90, size=n),
                    "SkinThickness": np.random.uniform(10, 40, size=n),
                    "Insulin": np.random.uniform(15, 200, size=n),
                    "BMI": np.random.uniform(18, 35, size=n),
                    "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.5, size=n),
                    "Age": np.random.randint(21, 65, size=n),
                    "Outcome": np.random.choice([0, 1], size=n, p=[0.65, 0.35]),
                }
            )
            file_hash = "synthetic_fixture_seed42"
            dataset_name = "Synthetic Test Fixture (100 samples)"
        else:
            print("\nBLOCKED — APPROVED DATASET NOT AVAILABLE")
            print(f"Expected real dataset at '{raw_path}' was not found.")
            print("Production model rebuilding requires the authentic Pima dataset.")
            sys.exit(1)

    # Validate Schema
    expected_cols = list(RAW_FEATURE_NAMES) + [target_column]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing expected dataset columns: {missing_cols}")
        sys.exit(1)

    target_dist = {str(k): int(v) for k, v in df[target_column].value_counts().items()}

    try:
        # 2. Canonical Train / Test Split
        X_train, X_test, y_train, y_test = split_data(
            df, target_column=target_column, test_size=test_size, random_seed=random_seed
        )
        print(f"[OK] Stratified Split: {len(X_train)} train, {len(X_test)} holdout test (seed={random_seed})")

        # 3. Fit Canonical Preprocessor strictly on X_train
        preprocessor = BiomedicalPreprocessor()
        X_train_trans = preprocessor.fit_transform(X_train)
        print(f"[OK] Preprocessor fitted on X_train (11 classical features generated)")

        dataset_prov = {
            "name": dataset_name,
            "hash": file_hash,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "target_column": target_column,
            "target_distribution": target_dist,
            "raw_feature_names": list(RAW_FEATURE_NAMES),
            "raw_feature_count": len(RAW_FEATURE_NAMES),
        }

        train_classical = args.model in [
            "classical", "all", "lr", "svm", "rf", "xgboost", "logistic_regression", "random_forest"
        ]
        train_quantum = args.model in ["quantum", "all", "qsvc", "vqc"]

        rebuilt_artifacts: dict[str, ModelArtifact] = {}
        evaluator = ModelEvaluator(random_seed=random_seed)

        # ----------------------------------------------------
        # 4. Classical Model Rebuild
        # ----------------------------------------------------
        if train_classical:
            classical_models = (
                list(SUPPORTED_CLASSICAL_MODELS.keys())
                if args.model in ["classical", "all"]
                else [args.model]
            )
            for m_key in classical_models:
                params = classical_cfg.get(m_key, {})
                print(f"\n--- Training Classical Model: {m_key.upper()} ---")
                t0 = time.perf_counter()
                model_inst = ClassicalModelFactory.create_model(m_key, params=params, random_seed=random_seed)
                model_inst.fit(X_train_trans, y_train)
                train_time_ms = (time.perf_counter() - t0) * 1000.0

                art_id = f"{m_key}_{args.version}"
                artifact = ModelArtifact(
                    artifact_id=art_id,
                    model_id=art_id,
                    model_name=model_inst.model_name,
                    model_family="classical",
                    model_type=model_inst.model_type,
                    artifact_version=f"{args.version}.0",
                    model=model_inst,
                    preprocessor=preprocessor,
                    dataset_provenance=dataset_prov,
                    configuration_provenance={"hyperparameters": params},
                    metadata={"train_time_ms": train_time_ms},
                    random_seed=random_seed,
                )

                # Evaluate on holdout test partition (80/20) using raw DataFrame
                print(f"  Evaluating '{art_id}' on {len(X_test)} holdout test samples...")
                eval_report = evaluator.evaluate_model(artifact, X_test, y_test, compute_bootstrap=False)
                artifact.evaluation_provenance = {
                    "protocol": "80/20 Stratified Holdout Test Split",
                    "sample_count": len(X_test),
                    "metrics": eval_report["metrics"],
                    "confusion_matrix": eval_report["confusion_matrix"],
                    "timing": eval_report["timing"],
                }

                save_path = save_artifact(artifact, base_dir="models")
                rebuilt_artifacts[art_id] = artifact
                print(f"  [OK] Rebuilt & Saved: '{art_id}' -> '{save_path}' (Accuracy: {eval_report['metrics']['accuracy']:.4f})")

        # ----------------------------------------------------
        # 5. Quantum Model Rebuild (4 Qubits)
        # ----------------------------------------------------
        if train_quantum:
            qubits = quantum_cfg.get("qubits", 4)
            shots = quantum_cfg.get("backend", {}).get("shots", 1024)
            fmap = quantum_cfg.get("feature_map", {}).get("name", "ZZFeatureMap")
            fmap_reps = quantum_cfg.get("feature_map", {}).get("reps", 2)
            fmap_ent = quantum_cfg.get("feature_map", {}).get("entanglement", "linear")
            ansatz_name = quantum_cfg.get("ansatz", {}).get("name", "RealAmplitudes")
            ansatz_reps = quantum_cfg.get("ansatz", {}).get("reps", 2)
            ansatz_ent = quantum_cfg.get("ansatz", {}).get("entanglement", "full")
            opt_name = quantum_cfg.get("optimizer", {}).get("name", "COBYLA")
            opt_maxiter = quantum_cfg.get("optimizer", {}).get("maxiter", 60)

            print(f"\n--- Fitting Quantum Feature Reducer (k={qubits}) ---")
            reducer_4q = QuantumFeatureReducer(n_qubits=qubits, random_seed=random_seed)
            X_train_q4 = reducer_4q.fit_transform(X_train_trans, y_train)
            print(f"  Selected Features: {reducer_4q.selected_feature_names_}")

            if args.model in ["quantum", "all", "qsvc"]:
                C_val = quantum_cfg.get("qsvc", {}).get("C", 1.0)
                print(f"\n--- Training Quantum Model: QSVC-{qubits}Q ---")
                t0_qsvc = time.perf_counter()
                qsvc_model = QuantumSVCModel(
                    num_qubits=qubits,
                    feature_map_name=fmap,
                    reps=fmap_reps,
                    entanglement=fmap_ent,
                    C=C_val,
                    shots=shots,
                    random_seed=random_seed,
                )
                qsvc_model.fit(X_train_q4, y_train)
                qsvc_train_time = (time.perf_counter() - t0_qsvc) * 1000.0

                art_id = f"qsvc_{args.version}"
                artifact_qsvc = ModelArtifact(
                    artifact_id=art_id,
                    model_id=art_id,
                    model_name=qsvc_model.model_name,
                    model_family="quantum",
                    model_type="qsvc",
                    artifact_version=f"{args.version}.0",
                    model=qsvc_model,
                    preprocessor=preprocessor,
                    quantum_reducer=reducer_4q,
                    dataset_provenance=dataset_prov,
                    configuration_provenance=quantum_cfg,
                    metadata=qsvc_model.get_circuit_metadata(),
                    random_seed=random_seed,
                )

                print(f"  Evaluating '{art_id}' on {len(X_test)} holdout test samples...")
                eval_qsvc = evaluator.evaluate_model(artifact_qsvc, X_test, y_test, compute_bootstrap=False)
                artifact_qsvc.evaluation_provenance = {
                    "protocol": "80/20 Stratified Holdout Test Split",
                    "sample_count": len(X_test),
                    "metrics": eval_qsvc["metrics"],
                    "confusion_matrix": eval_qsvc["confusion_matrix"],
                    "timing": eval_qsvc["timing"],
                }

                save_path = save_artifact(artifact_qsvc, base_dir="models")
                rebuilt_artifacts[art_id] = artifact_qsvc
                print(f"  [OK] Rebuilt & Saved: '{art_id}' -> '{save_path}' (Accuracy: {eval_qsvc['metrics']['accuracy']:.4f})")

            if args.model in ["quantum", "all", "vqc"]:
                print(f"\n--- Training Quantum Model: VQC-{qubits}Q ---")
                t0_vqc = time.perf_counter()
                vqc_model = VariationalQuantumClassifierModel(
                    num_qubits=qubits,
                    feature_map_name=fmap,
                    feature_map_reps=fmap_reps,
                    feature_map_entanglement=fmap_ent,
                    ansatz_name=ansatz_name,
                    ansatz_reps=ansatz_reps,
                    ansatz_entanglement=ansatz_ent,
                    optimizer_name=opt_name,
                    max_iter=opt_maxiter,
                    shots=shots,
                    random_seed=random_seed,
                )
                vqc_model.fit(X_train_q4, y_train)
                vqc_train_time = (time.perf_counter() - t0_vqc) * 1000.0

                art_id = f"vqc_{args.version}"
                artifact_vqc = ModelArtifact(
                    artifact_id=art_id,
                    model_id=art_id,
                    model_name=vqc_model.model_name,
                    model_family="quantum",
                    model_type="vqc",
                    artifact_version=f"{args.version}.0",
                    model=vqc_model,
                    preprocessor=preprocessor,
                    quantum_reducer=reducer_4q,
                    dataset_provenance=dataset_prov,
                    configuration_provenance=quantum_cfg,
                    metadata=vqc_model.get_circuit_metadata(),
                    random_seed=random_seed,
                )

                print(f"  Evaluating '{art_id}' on {len(X_test)} holdout test samples...")
                eval_vqc = evaluator.evaluate_model(artifact_vqc, X_test, y_test, compute_bootstrap=False)
                artifact_vqc.evaluation_provenance = {
                    "protocol": "80/20 Stratified Holdout Test Split",
                    "sample_count": len(X_test),
                    "metrics": eval_vqc["metrics"],
                    "confusion_matrix": eval_vqc["confusion_matrix"],
                    "timing": eval_vqc["timing"],
                }

                save_path = save_artifact(artifact_vqc, base_dir="models")
                rebuilt_artifacts[art_id] = artifact_vqc
                print(f"  [OK] Rebuilt & Saved: '{art_id}' -> '{save_path}' (Accuracy: {eval_vqc['metrics']['accuracy']:.4f})")

        # ----------------------------------------------------
        # 6. Cross-Validation Evaluation (Isolated from holdout)
        # ----------------------------------------------------
        cv_summary = {}
        if args.model in ["classical", "all"]:
            print("\n--- Running 5-Fold Stratified Cross-Validation on Training Split ---")
            train_df = X_train.copy()
            train_df[target_column] = y_train.values

            for m_key in ["lr", "svm", "rf", "xgboost"]:
                params = classical_cfg.get(m_key, {})
                cv_res = evaluate_cross_validation(
                    m_key,
                    train_df,
                    target_column=target_column,
                    n_splits=5,
                    random_seed=random_seed,
                    model_params=params,
                )
                cv_summary[m_key] = cv_res["metrics"]
                acc_mean = cv_res["metrics"]["accuracy"]["mean"]
                acc_std = cv_res["metrics"]["accuracy"]["std"]
                print(f"  5-Fold CV '{m_key.upper()}': Accuracy = {acc_mean:.4f} +/- {acc_std:.4f}")

        # ----------------------------------------------------
        # 7. Generate Unified Benchmark Record
        # ----------------------------------------------------
        if args.model == "all":
            print("\n--- Generating Authoritative Benchmark Results ---")
            models_benchmark_list = []

            for art_key, art in rebuilt_artifacts.items():
                eval_prov = art.evaluation_provenance
                metrics = eval_prov.get("metrics", {})

                # Curves computation on holdout test set
                y_prob_pos = None
                if hasattr(art, "predict_proba"):
                    try:
                        probs = art.predict_proba(X_test)
                        y_prob_pos = probs[:, 1] if probs.ndim == 2 else probs.ravel()
                    except Exception:
                        pass

                roc_curve = calculate_roc_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
                pr_curve = calculate_pr_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
                calibration = calculate_calibration_data(y_test, y_prob_pos) if y_prob_pos is not None else {
                    "status": "unavailable",
                    "reason": "Calibration curve unavailable for model.",
                }

                is_q = art.model_family == "quantum"
                qubits = 4 if is_q else None
                features_used = 4 if is_q else 11

                models_benchmark_list.append({
                    "model_id": art.artifact_id,
                    "name": art.model_name,
                    "family": art.model_family,
                    "status": "SUCCESS",
                    "features_used": features_used,
                    "qubits": qubits,
                    "feature_space": (
                        "SelectKBest-Reduced Quantum Angle Encoding [-pi, pi] (4 features)"
                        if is_q
                        else "Cleaned & Engineered Classical (11 features: 8 raw + 3 engineered)"
                    ),
                    "metrics": metrics,
                    "confusion_matrix": eval_prov.get("confusion_matrix"),
                    "timings": {
                        "train_time_ms": art.metadata.get("train_time_ms", 0.0),
                        "inference_time_ms": eval_prov.get("timing", {}).get("inference_time_ms", 0.0),
                        "total_evaluation_time_ms": eval_prov.get("timing", {}).get("evaluation_time_ms", 0.0),
                    },
                    "curves": {
                        "roc_curve": roc_curve,
                        "pr_curve": pr_curve,
                    },
                    "calibration": calibration,
                    "circuit_metadata": art.metadata if is_q else None,
                })

            # Quantum Scaling Section: 4Q real, 6Q & 8Q explicitly marked unavailable
            qsvc_art = rebuilt_artifacts.get(f"qsvc_{args.version}")
            qsvc_metrics = qsvc_art.evaluation_provenance.get("metrics", {}) if qsvc_art else {}

            quantum_scaling_records = [
                {
                    "configuration": "QSVC-4Q",
                    "qubits": 4,
                    "features": 4,
                    "status": "EVALUATED",
                    "accuracy": qsvc_metrics.get("accuracy"),
                    "f1_score": qsvc_metrics.get("f1_score"),
                    "roc_auc": qsvc_metrics.get("roc_auc"),
                    "train_time_ms": qsvc_art.metadata.get("train_time_ms", 0.0) if qsvc_art else 0.0,
                    "inference_time_ms": qsvc_art.evaluation_provenance.get("timing", {}).get("inference_time_ms", 0.0) if qsvc_art else 0.0,
                },
                {
                    "configuration": "QSVC-6Q",
                    "qubits": 6,
                    "features": 6,
                    "status": "CIRCUIT_ONLY",
                    "message": "Evaluation Unavailable: No pre-trained 6Q production artifact.",
                    "accuracy": None,
                    "f1_score": None,
                    "roc_auc": None,
                    "train_time_ms": None,
                    "inference_time_ms": None,
                },
                {
                    "configuration": "QSVC-8Q",
                    "qubits": 8,
                    "features": 8,
                    "status": "CIRCUIT_ONLY",
                    "message": "Evaluation Unavailable: No pre-trained 8Q production artifact.",
                    "accuracy": None,
                    "f1_score": None,
                    "roc_auc": None,
                    "train_time_ms": None,
                    "inference_time_ms": None,
                },
            ]

            benchmark_payload = {
                "benchmark_id": f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.version}",
                "timestamp": datetime.now().isoformat(),
                "artifact_version": f"{args.version}.0",
                "dataset": {
                    "name": dataset_name,
                    "hash": file_hash,
                    "total_samples": len(df),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "target_column": target_column,
                    "stratification": "Stratified on target",
                    "random_seed": random_seed,
                },
                "preprocessing": {
                    "pipeline_version": PIPELINE_VERSION,
                    "imputation": "median (train-fitted)",
                    "outlier_handling": "IQR clipping (train-fitted)",
                    "feature_engineering": "Deterministic (8 raw -> 11 classical)",
                    "scaling": "StandardScaler (train-fitted)",
                    "quantum_reduction": "SelectKBest mutual_info_classif (11 -> 4 components) + angle scaling [-pi, pi]",
                },
                "evaluation_protocol": {
                    "type": "80/20 Stratified Holdout Test Split",
                    "holdout_samples": len(X_test),
                    "isolated_test_partition": True,
                },
                "cross_validation_protocol": {
                    "type": "5-Fold Stratified Cross-Validation (Training Split Only)",
                    "train_samples": len(X_train),
                    "metrics": cv_summary,
                },
                "models": models_benchmark_list,
                "quantum_scaling": quantum_scaling_records,
            }

            canonical_bench_file = Path("data/processed/benchmark_results.json")
            canonical_bench_file.parent.mkdir(parents=True, exist_ok=True)
            with open(canonical_bench_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_payload, f, indent=2)
            print(f"[OK] Wrote canonical benchmark record -> '{canonical_bench_file}'")

            exp_bench_file = Path(f"experiments/results/unified_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.version}.json")
            exp_bench_file.parent.mkdir(parents=True, exist_ok=True)
            with open(exp_bench_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_payload, f, indent=2)
            print(f"[OK] Wrote archive benchmark record -> '{exp_bench_file}'")

        print("\n==================================================")
        print("Phase B Production Model Rebuild Completed Successfully")
        print("==================================================")
        return 0

    except QMLError as e:
        print(f"Error during training: {e!s}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
