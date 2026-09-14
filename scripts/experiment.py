#!/usr/bin/env python3
"""CLI script for running controlled quantum scaling experiments across 4, 6, and 8 qubits.

Usage:
    python scripts/experiment.py --qubits all
    python scripts/experiment.py --qubits 4 --shots 1024
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from datetime import datetime
import json
import yaml
import numpy as np
import pandas as pd

from ml.data import DataLoader
from ml.preprocessing import BiomedicalPreprocessor, split_data
from ml.quantum import QuantumFeatureReducer, QuantumSVCModel
from ml.evaluation import ModelEvaluator
from ml.exceptions import QMLError, ValidationError


def run_single_quantum_experiment(
    df: pd.DataFrame,
    num_qubits: int,
    target_column: str = "Outcome",
    test_size: float = 0.2,
    random_seed: int = 42,
    shots: int = 1024,
    feature_map_name: str = "ZZFeatureMap",
    reps: int = 2,
    C: float = 1.0,
) -> dict:
    """Executes a single controlled quantum scaling experiment for a given qubit count."""
    t_start = time.perf_counter()

    # 1. Stratified Split (same split for all experiments)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df, target_column=target_column, test_size=test_size, random_seed=random_seed
    )

    # 2. Classical Preprocessing (leakage-safe, fit on train only)
    t_prep_0 = time.perf_counter()
    preprocessor = BiomedicalPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_test_clean = preprocessor.transform(X_test_raw)
    t_prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

    # 3. Quantum Feature Reduction (SelectKBest mutual_info_classif + angle scaling into [-pi, pi], fit strictly on train)
    t_red_0 = time.perf_counter()
    reducer = QuantumFeatureReducer(n_qubits=num_qubits, random_seed=random_seed)
    X_train_q = reducer.fit_transform(X_train_clean)
    X_test_q = reducer.transform(X_test_clean)
    t_red_ms = (time.perf_counter() - t_red_0) * 1000.0
    var_stats = reducer.get_variance_stats()

    # 4. Quantum Circuit & Kernel Model Initialization
    t_init_0 = time.perf_counter()
    qsvc = QuantumSVCModel(
        num_qubits=num_qubits,
        feature_map_name=feature_map_name,
        reps=reps,
        C=C,
        shots=shots,
        random_seed=random_seed,
    )
    t_init_ms = (time.perf_counter() - t_init_0) * 1000.0

    # 5. Model Training (Quantum Kernel Matrix computation on train fold)
    t_train_0 = time.perf_counter()
    qsvc.fit(X_train_q, y_train)
    t_train_ms = (time.perf_counter() - t_train_0) * 1000.0

    # 6. Evaluation using Step 7 Evaluator
    evaluator = ModelEvaluator(random_seed=random_seed)
    eval_report = evaluator.evaluate_model(qsvc, X_test_q, y_test, compute_bootstrap=False)

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    circuit_meta = qsvc.get_circuit_metadata()

    return {
        "configuration": f"QML-{num_qubits}Q",
        "num_features": num_qubits,
        "num_qubits": num_qubits,
        "status": "COMPLETED",
        "circuit_metadata": circuit_meta,
        "variance_stats": var_stats,
        "samples": {
            "train_samples": len(X_train_q),
            "test_samples": len(X_test_q),
        },
        "timings": {
            "preprocessing_time_ms": t_prep_ms,
            "feature_reduction_time_ms": t_red_ms,
            "kernel_init_time_ms": t_init_ms,
            "train_time_ms": t_train_ms,
            "inference_time_ms": eval_report["timing"]["inference_time_ms"],
            "total_experiment_time_ms": t_total_ms,
        },
        "metrics": eval_report["metrics"],
        "confusion_matrix": eval_report["confusion_matrix"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run controlled Quantum Scaling Experiments")
    parser.add_argument(
        "--qubits",
        type=str,
        default="all",
        help="Number of qubits to benchmark (4, 6, 8, or 'all')",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of Aer simulator shots",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--quantum-config",
        type=str,
        default="configs/quantum.yaml",
        help="Path to quantum config file",
    )
    args = parser.parse_args()

    print("==================================================")
    print("SIH26139 - Controlled Quantum Scaling Benchmarking")
    print("==================================================")

    # Load configs
    base_cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    q_cfg = {}
    if Path(args.quantum_config).exists():
        with open(args.quantum_config, "r", encoding="utf-8") as f:
            q_cfg = yaml.safe_load(f) or {}

    data_cfg = base_cfg.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/pima_diabetes.csv")
    target_column = data_cfg.get("target_column", "Outcome")
    test_size = data_cfg.get("test_size", 0.2)
    random_seed = base_cfg.get("random_seed", 42)

    shots = args.shots or q_cfg.get("backend", {}).get("shots", 1024)
    feature_map_name = q_cfg.get("feature_map", {}).get("name", "ZZFeatureMap")
    reps = q_cfg.get("feature_map", {}).get("reps", 2)
    C = q_cfg.get("qsvc", {}).get("C", 1.0)

    # Determine dataset
    if Path(raw_path).exists():
        loader = DataLoader(raw_path, target_column=target_column)
        df, file_hash = loader.load()
        dataset_name = "pima_diabetes.csv (Real Raw)"
    else:
        if "--allow-synthetic-fixture" in sys.argv:
            print("Raw dataset file not found; generating deterministic 100-sample fixture for scaling test...")
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
            print("ERROR: Approved real dataset not found at expected path. Experiment aborted.")
            sys.exit(1)

    # Determine qubit counts to benchmark
    if args.qubits == "all":
        qubit_list = [4, 6, 8]
    else:
        qubit_list = [int(args.qubits)]

    results_list = []

    print(f"\nDataset: {dataset_name} (Hash: {file_hash[:12]})")
    print(f"Random Seed: {random_seed} | Shots: {shots} | Feature Map: {feature_map_name} (reps={reps})\n")

    for q in qubit_list:
        print(f"--- Running Quantum Experiment: {q} Features -> {q} Qubits ---")
        try:
            exp_res = run_single_quantum_experiment(
                df=df,
                num_qubits=q,
                target_column=target_column,
                test_size=test_size,
                random_seed=random_seed,
                shots=shots,
                feature_map_name=feature_map_name,
                reps=reps,
                C=C,
            )
            results_list.append(exp_res)
            m = exp_res["metrics"]
            t = exp_res["timings"]
            v = exp_res["variance_stats"]
            roc_val = m.get("roc_auc")
            roc_str = f"{roc_val:.4f}" if isinstance(roc_val, float) else "N/A"
            print(
                f"  [OK] {exp_res['configuration']} Done | "
                f"Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | ROC-AUC: {roc_str} | "
                f"Train: {t['train_time_ms']:.1f}ms | Infer: {t['inference_time_ms']:.1f}ms | "
                f"Exp Var: {v['cumulative_explained_variance']:.4f}"
            )
        except Exception as e:
            print(f"  [FAIL] Failed for {q} qubits: {e!s}")
            results_list.append(
                {
                    "configuration": f"QML-{q}Q",
                    "num_features": q,
                    "num_qubits": q,
                    "status": "FAILED",
                    "error": str(e),
                }
            )

    # Save results to experiments/results/
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(f"experiments/results/quantum_scaling_{timestamp_str}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "experiment_type": "quantum_scaling_4_6_8_qubits",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "dataset_hash": file_hash,
        "random_seed": random_seed,
        "experiments": results_list,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nSaved structured experiment results to '{out_file}'")

    # Display comparison table
    table_rows = []
    for r in results_list:
        if r.get("status") == "COMPLETED":
            m = r["metrics"]
            t = r["timings"]
            v = r["variance_stats"]
            roc_val = m.get("roc_auc")
            roc_str = f"{roc_val:.4f}" if isinstance(roc_val, float) else "N/A"
            table_rows.append(
                {
                    "Configuration": r["configuration"],
                    "Features / Qubits": r["num_qubits"],
                    "Accuracy": f"{m['accuracy']:.4f}",
                    "F1-Score": f"{m['f1_score']:.4f}",
                    "Sensitivity": f"{m['sensitivity']:.4f}",
                    "Specificity": f"{m['specificity']:.4f}",
                    "ROC-AUC": roc_str,
                    "Cum. Var": f"{v['cumulative_explained_variance']:.4f}",
                    "Train Time (ms)": f"{t['train_time_ms']:.1f}",
                    "Infer Time (ms)": f"{t['inference_time_ms']:.1f}",
                }
            )

    if table_rows:
        df_summary = pd.DataFrame(table_rows)
        print("\n=== QUANTUM SCALING COMPARISON TABLE ===")
        print(df_summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
