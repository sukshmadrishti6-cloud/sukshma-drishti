#!/usr/bin/env python3
"""CLI script for evaluating trained model artifacts, running cross-validation, and executing the Unified Classical vs Quantum Benchmark.

Usage:
    python scripts/evaluate.py --benchmark
    python scripts/evaluate.py --model-path models/classical/rf_v1.joblib
    python scripts/evaluate.py --cv --model rf
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import yaml
import numpy as np
import pandas as pd

from ml.data import DataLoader
from ml.preprocessing import BiomedicalPreprocessor, split_data
from ml.classical import load_model, ClassicalModelFactory
from ml.evaluation import (
    ModelEvaluator,
    evaluate_cross_validation,
    compare_models,
    UnifiedBenchmarkRunner,
    generate_benchmark_summary_table,
)
from ml.exceptions import QMLError


def main():
    parser = argparse.ArgumentParser(description="Evaluate models and execute Unified Benchmark")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the complete Unified Classical vs Quantum Benchmark (LR, SVM, RF, XGBoost, QSVC, VQC)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model artifact (.joblib)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run cross-validation on the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        help="Model identifier to evaluate with CV (e.g. lr, svm, rf, xgboost)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of shots for quantum models",
    )
    args = parser.parse_args()

    print("==================================================")
    print("SIH26139 - Evaluation & Unified Benchmark CLI")
    print("==================================================")

    config_path = Path(args.config)
    base_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    data_cfg = base_cfg.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/pima_diabetes.csv")
    target_column = data_cfg.get("target_column", "Outcome")
    test_size = data_cfg.get("test_size", 0.2)
    random_seed = base_cfg.get("random_seed", 42)

    # Determine dataset
    if Path(raw_path).exists():
        loader = DataLoader(raw_path, target_column=target_column)
        df, file_hash = loader.load()
        dataset_name = "pima_diabetes.csv (Real Raw)"
    else:
        if "--allow-synthetic-fixture" in sys.argv:
            print("Raw dataset file not found; generating deterministic 100-sample fixture for evaluation...")
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
            print("ERROR: Approved real dataset not found at expected path. Evaluation aborted.")
            sys.exit(1)

    try:
        if args.benchmark:
            print(f"\nExecuting Unified Benchmark across 6 models on '{dataset_name}'...")
            runner = UnifiedBenchmarkRunner(
                random_seed=random_seed,
                shots=args.shots,
                quantum_qubits=4,
                vqc_max_iter=30,
            )
            bench_results = runner.run_benchmark(
                df,
                target_column=target_column,
                test_size=test_size,
                dataset_name=dataset_name,
                dataset_hash=file_hash,
                include_scaling=False,
                save_results=True,
            )

            df_summary = generate_benchmark_summary_table(bench_results)
            print("\n=== UNIFIED MODEL BENCHMARK RESULTS (TEST SPLIT) ===")
            print(df_summary.to_string(index=False))

            if bench_results.get("quantum_scaling"):
                df_scale = pd.DataFrame(bench_results["quantum_scaling"])
                print("\n=== QUANTUM SCALING RESULTS (SEPARATE SECTION) ===")
                print(df_scale.to_string(index=False))
            return 0

        if args.cv:
            print(f"\nRunning 5-fold Stratified Cross-Validation for '{args.model}'...")
            cv_results = evaluate_cross_validation(
                args.model, df, target_column=target_column, n_splits=5, random_seed=random_seed
            )
            print("\nCross-Validation Results:")
            for m_name, stats in cv_results["metrics"].items():
                if stats:
                    print(f"  {m_name:15s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
            return 0

        if args.model_path:
            model_file = Path(args.model_path)
            if not model_file.exists():
                print(f"Error: Model artifact not found at '{model_file}'.")
                return 1

            from ml.artifacts import load_artifact
            try:
                model = load_artifact(model_file)
            except Exception:
                model = load_model(model_file)

            X_train, X_test, y_train, y_test = split_data(
                df, target_column=target_column, test_size=test_size, random_seed=random_seed
            )

            evaluator = ModelEvaluator(random_seed=random_seed)
            report = evaluator.evaluate_model(model, X_test, y_test)

            print(f"\nEvaluation Results for '{model.model_name}':")
            for k, v in report["metrics"].items():
                if k != "confusion_matrix":
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"  {k:15s}: {val_str}")

            print(f"\nConfusion Matrix: {report['confusion_matrix']}")
            return 0

        print("Please provide --benchmark, --model-path, or --cv to execute.")
        return 0
    except QMLError as e:
        print(f"Evaluation error: {e!s}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
