"""Unified Classical vs Quantum Benchmarking Engine for SIH26139."""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.classical import ClassicalModelFactory
from ml.evaluation.evaluator import ModelEvaluator
from ml.evaluation.metrics import (
    calculate_calibration_data,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
)
from ml.logger import get_logger
from ml.preprocessing import BiomedicalPreprocessor, split_data
from ml.quantum import QuantumFeatureReducer, QuantumSVCModel, VariationalQuantumClassifierModel

logger = get_logger(__name__)


class UnifiedBenchmarkRunner:
    """Orchestrates controlled, fair benchmarking across Classical and Quantum ML models."""

    def __init__(
        self,
        random_seed: int = 42,
        shots: int = 1024,
        quantum_qubits: int = 4,
        vqc_max_iter: int = 60,
    ):
        self.random_seed = random_seed
        self.shots = shots
        self.quantum_qubits = quantum_qubits
        self.vqc_max_iter = vqc_max_iter
        self.logger = logger

    def run_benchmark(
        self,
        df: pd.DataFrame,
        target_column: str = "Outcome",
        test_size: float = 0.2,
        dataset_name: str = "pima_diabetes",
        dataset_hash: str = "unknown",
        include_scaling: bool = True,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """Executes the full comparative benchmark under strictly identical split conditions.

        Args:
            df: Input raw or validated dataset.
            target_column: Binary label column.
            test_size: Fraction of held-out test data (default 0.2).
            dataset_name: Identifier name for reporting.
            dataset_hash: SHA-256 data hash.
            include_scaling: Whether to include 4/6/8-qubit scaling results.
            save_results: Whether to persist JSON benchmark artifact.

        Returns:
            Dict[str, Any]: Complete structured benchmark results.
        """
        self.logger.info(
            f"Starting Unified Benchmark on '{dataset_name}' ({len(df)} samples, seed={self.random_seed})"
        )
        t_bench_start = time.perf_counter()

        # 1. Stratified Data Split (Identical for all models)
        X_train_raw, X_test_raw, y_train, y_test = split_data(
            df, target_column=target_column, test_size=test_size, random_seed=self.random_seed
        )

        # 2. Classical Preprocessing (Fitted strictly on X_train)
        t0_prep = time.perf_counter()
        preprocessor = BiomedicalPreprocessor()
        X_train_classical = preprocessor.fit_transform(X_train_raw)
        X_test_classical = preprocessor.transform(X_test_raw)
        prep_time_ms = (time.perf_counter() - t0_prep) * 1000.0

        # 3. Quantum Feature Reduction (Fitted strictly on X_train_classical)
        t0_red = time.perf_counter()
        reducer_4q = QuantumFeatureReducer(n_qubits=self.quantum_qubits, random_seed=self.random_seed)
        X_train_q4 = reducer_4q.fit_transform(X_train_classical, y_train)
        X_test_q4 = reducer_4q.transform(X_test_classical)
        red_time_ms = (time.perf_counter() - t0_red) * 1000.0
        var_stats_4q = reducer_4q.get_variance_stats()

        # 4. Model Registry for Unified Comparison
        models_to_evaluate = [
            # Classical Models (11 features: 8 raw + 3 engineered)
            {
                "model_id": "lr",
                "name": "Logistic Regression",
                "family": "classical",
                "features_used": X_train_classical.shape[1],
                "qubits": None,
                "feature_space": "Cleaned & Engineered Classical (11 features: 8 raw + 3 engineered)",
                "X_train": X_train_classical,
                "X_test": X_test_classical,
                "instance_factory": lambda: ClassicalModelFactory.create("lr", random_seed=self.random_seed),
            },
            {
                "model_id": "svm_rbf",
                "name": "SVM (RBF Kernel)",
                "family": "classical",
                "features_used": X_train_classical.shape[1],
                "qubits": None,
                "feature_space": "Cleaned & Engineered Classical (11 features: 8 raw + 3 engineered)",
                "X_train": X_train_classical,
                "X_test": X_test_classical,
                "instance_factory": lambda: ClassicalModelFactory.create("svm", random_seed=self.random_seed),
            },
            {
                "model_id": "rf",
                "name": "Random Forest",
                "family": "classical",
                "features_used": X_train_classical.shape[1],
                "qubits": None,
                "feature_space": "Cleaned & Engineered Classical (11 features: 8 raw + 3 engineered)",
                "X_train": X_train_classical,
                "X_test": X_test_classical,
                "instance_factory": lambda: ClassicalModelFactory.create("rf", random_seed=self.random_seed),
            },
            {
                "model_id": "xgboost",
                "name": "XGBoost",
                "family": "classical",
                "features_used": X_train_classical.shape[1],
                "qubits": None,
                "feature_space": "Cleaned & Engineered Classical (11 features: 8 raw + 3 engineered)",
                "X_train": X_train_classical,
                "X_test": X_test_classical,
                "instance_factory": lambda: ClassicalModelFactory.create("xgboost", random_seed=self.random_seed),
            },
            # Quantum Models (4 features -> 4 qubits)
            {
                "model_id": "qsvc_4q",
                "name": "Quantum SVC (QSVC-4Q)",
                "family": "quantum",
                "features_used": 4,
                "qubits": 4,
                "feature_space": "SelectKBest-Reduced Quantum Angle Encoding [-pi, pi] (4 features)",
                "X_train": X_train_q4,
                "X_test": X_test_q4,
                "instance_factory": lambda: QuantumSVCModel(
                    num_qubits=4,
                    shots=self.shots,
                    random_seed=self.random_seed,
                ),
            },
            {
                "model_id": "vqc_4q",
                "name": "Variational Quantum Classifier (VQC-4Q)",
                "family": "quantum",
                "features_used": 4,
                "qubits": 4,
                "feature_space": "SelectKBest-Reduced Quantum Angle Encoding [-pi, pi] (4 features)",
                "X_train": X_train_q4,
                "X_test": X_test_q4,
                "instance_factory": lambda: VariationalQuantumClassifierModel(
                    num_qubits=4,
                    max_iter=self.vqc_max_iter,
                    shots=self.shots,
                    random_seed=self.random_seed,
                ),
            },
        ]

        evaluator = ModelEvaluator(random_seed=self.random_seed)
        model_results: list[dict[str, Any]] = []

        for spec in models_to_evaluate:
            m_id = spec["model_id"]
            m_name = spec["name"]
            self.logger.info(f"Training and evaluating benchmark model '{m_name}'...")
            t_train_start = time.perf_counter()
            try:
                model_obj = spec["instance_factory"]()
                model_obj.fit(spec["X_train"], y_train)
                train_time_ms = (time.perf_counter() - t_train_start) * 1000.0

                # Evaluate using Step 7 Evaluator
                eval_report = evaluator.evaluate_model(
                    model_obj, spec["X_test"], y_test, compute_bootstrap=False
                )

                # Generate curve data for visualization payloads
                y_prob_pos = None
                if hasattr(model_obj, "predict_proba"):
                    try:
                        probs = model_obj.predict_proba(spec["X_test"])
                        y_prob_pos = probs[:, 1] if probs.ndim == 2 else probs.ravel()
                    except Exception:
                        pass

                roc_curve_payload = (
                    calculate_roc_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
                )
                pr_curve_payload = (
                    calculate_pr_curve_data(y_test, y_prob_pos) if y_prob_pos is not None else None
                )
                calibration_payload = (
                    calculate_calibration_data(y_test, y_prob_pos)
                    if y_prob_pos is not None
                    else {
                        "status": "unavailable",
                        "reason": "Calibration unavailable: Model outputs decision scores rather than calibrated probabilities.",
                    }
                )

                circuit_meta = None
                if hasattr(model_obj, "get_circuit_metadata"):
                    circuit_meta = model_obj.get_circuit_metadata()

                res_entry = {
                    "model_id": m_id,
                    "name": m_name,
                    "family": spec["family"],
                    "status": "SUCCESS",
                    "features_used": spec["features_used"],
                    "qubits": spec["qubits"],
                    "feature_space": spec["feature_space"],
                    "metrics": eval_report["metrics"],
                    "confusion_matrix": eval_report["confusion_matrix"],
                    "timings": {
                        "preprocessing_time_ms": prep_time_ms,
                        "feature_reduction_time_ms": red_time_ms if spec["family"] == "quantum" else 0.0,
                        "train_time_ms": train_time_ms,
                        "inference_time_ms": eval_report["timing"]["inference_time_ms"],
                        "total_evaluation_time_ms": eval_report["timing"]["evaluation_time_ms"],
                    },
                    "curves": {
                        "roc_curve": roc_curve_payload,
                        "pr_curve": pr_curve_payload,
                    },
                    "calibration": calibration_payload,
                    "circuit_metadata": circuit_meta,
                }
                model_results.append(res_entry)
            except Exception as e:
                self.logger.error(f"Model benchmark failed for '{m_name}': {e!s}")
                model_results.append(
                    {
                        "model_id": m_id,
                        "name": m_name,
                        "family": spec["family"],
                        "status": "FAILED",
                        "error": str(e),
                    }
                )

        # 5. Quantum Scaling Section (Optional / Attached from Step 9)
        scaling_results = []
        if include_scaling:
            self.logger.info("Executing Quantum Scaling Sub-Benchmark (4Q, 6Q, 8Q)...")
            for q_count in [4, 6, 8]:
                try:
                    red_q = QuantumFeatureReducer(n_qubits=q_count, random_seed=self.random_seed)
                    X_tr_q = red_q.fit_transform(X_train_classical, y_train)
                    X_te_q = red_q.transform(X_test_classical)
                    qsvc_sc = QuantumSVCModel(num_qubits=q_count, shots=self.shots, random_seed=self.random_seed)
                    t0_sc = time.perf_counter()
                    qsvc_sc.fit(X_tr_q, y_train)
                    t_sc_tr = (time.perf_counter() - t0_sc) * 1000.0
                    sc_report = evaluator.evaluate_model(qsvc_sc, X_te_q, y_test, compute_bootstrap=False)
                    scaling_results.append(
                        {
                            "configuration": f"QSVC-{q_count}Q",
                            "qubits": q_count,
                            "features": q_count,
                            "cumulative_explained_variance": red_q.get_variance_stats()["cumulative_explained_variance"],
                            "accuracy": sc_report["metrics"]["accuracy"],
                            "f1_score": sc_report["metrics"]["f1_score"],
                            "roc_auc": sc_report["metrics"]["roc_auc"],
                            "train_time_ms": t_sc_tr,
                            "inference_time_ms": sc_report["timing"]["inference_time_ms"],
                        }
                    )
                except Exception as e:
                    scaling_results.append(
                        {
                            "configuration": f"QSVC-{q_count}Q",
                            "qubits": q_count,
                            "status": "FAILED",
                            "error": str(e),
                        }
                    )

        total_bench_time_ms = (time.perf_counter() - t_bench_start) * 1000.0

        benchmark_payload = {
            "benchmark_id": f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "name": dataset_name,
                "hash": dataset_hash,
                "total_samples": len(df),
                "train_samples": len(X_train_raw),
                "test_samples": len(X_test_raw),
                "target_column": target_column,
                "stratification": "Stratified on target",
                "random_seed": self.random_seed,
            },
            "preprocessing": {
                "imputation": "median",
                "scaling": "standard_scaler (classical) / minmax [-pi, pi] (quantum)",
                "quantum_reduction": f"SelectKBest (8 -> {self.quantum_qubits} components)",
                "quantum_variance_explained": var_stats_4q.get("cumulative_explained_variance", 0.0),
            },
            "timings": {
                "total_benchmark_time_ms": total_bench_time_ms,
            },
            "models": model_results,
            "quantum_scaling": scaling_results,
        }

        # Save to experiments/results/
        if save_results:
            out_dir = Path("experiments/results")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"unified_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_payload, f, indent=2)
            self.logger.info(f"Saved benchmark results artifact to '{out_file}'")

            canonical_file = Path("data/processed/benchmark_results.json")
            canonical_file.parent.mkdir(parents=True, exist_ok=True)

            existing_payload = {}
            if canonical_file.exists():
                try:
                    with open(canonical_file, "r", encoding="utf-8") as f:
                        existing_payload = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load existing canonical benchmark for merging: {e!s}")

            # Merge models additively
            existing_models = existing_payload.get("models", [])
            existing_model_map = {m.get("model_id"): m for m in existing_models if m.get("model_id")}
            for m in model_results:
                m_id = m.get("model_id")
                if m_id:
                    existing_model_map[m_id] = m
            merged_models = list(existing_model_map.values())

            # Merge results additively
            existing_results = existing_payload.get("results", [])
            existing_res_map = {r.get("model_id"): r for r in existing_results if r.get("model_id")}
            dis_tag = "diabetes"
            if "cardio" in dataset_name.lower():
                dis_tag = "cardiovascular"
            elif "cancer" in dataset_name.lower():
                dis_tag = "cancer"

            for m in model_results:
                m_id = m.get("model_id")
                if m_id:
                    existing_res_map[m_id] = {
                        "model_id": m_id,
                        "model_name": m.get("name", m_id),
                        "disease": dis_tag,
                        "model_family": m.get("family", "classical"),
                        "features_used": m.get("features_used"),
                        "qubits": m.get("qubits"),
                        "metrics": m.get("metrics"),
                        "confusion_matrix": m.get("confusion_matrix"),
                        "timings": m.get("timings"),
                        "curves": m.get("curves"),
                        "calibration": m.get("calibration"),
                    }
            merged_results = list(existing_res_map.values())

            # Merge scaling additively
            existing_scaling = existing_payload.get("quantum_scaling", [])
            scaling_map = {s.get("configuration"): s for s in existing_scaling if s.get("configuration")}
            for s in scaling_results:
                c_key = s.get("configuration")
                if c_key:
                    scaling_map[c_key] = s
            merged_scaling = list(scaling_map.values()) if scaling_map else scaling_results

            final_canonical = {
                "benchmark_id": benchmark_payload.get("benchmark_id"),
                "timestamp": benchmark_payload.get("timestamp"),
                "dataset": benchmark_payload.get("dataset"),
                "preprocessing": benchmark_payload.get("preprocessing"),
                "timings": benchmark_payload.get("timings"),
                "models": merged_models,
                "results": merged_results,
                "quantum_scaling": merged_scaling,
            }
            for k, v in existing_payload.items():
                if k not in final_canonical:
                    final_canonical[k] = v

            with open(canonical_file, "w", encoding="utf-8") as f:
                json.dump(final_canonical, f, indent=2)
            self.logger.info(
                f"Additively updated canonical benchmark results to '{canonical_file}' "
                f"({len(merged_models)} models, {len(merged_results)} results, {len(merged_scaling)} scaling)"
            )

        return benchmark_payload


def generate_benchmark_summary_table(benchmark_payload: dict[str, Any]) -> pd.DataFrame:
    """Builds a formatted pandas DataFrame comparison table from benchmark results."""
    rows = []
    for m in benchmark_payload.get("models", []):
        if m.get("status") == "SUCCESS":
            metrics = m.get("metrics", {})
            timings = m.get("timings", {})
            roc_val = metrics.get("roc_auc")
            pr_val = metrics.get("pr_auc")
            rows.append(
                {
                    "Model": m["name"],
                    "Family": m["family"].capitalize(),
                    "Features / Qubits": f"{m['features_used']}F" if m['qubits'] is None else f"{m['features_used']}F / {m['qubits']}Q",
                    "Accuracy": metrics.get("accuracy"),
                    "Sensitivity": metrics.get("sensitivity"),
                    "Specificity": metrics.get("specificity"),
                    "F1-Score": metrics.get("f1_score"),
                    "ROC-AUC": roc_val,
                    "PR-AUC": pr_val,
                    "Train (ms)": timings.get("train_time_ms"),
                    "Infer (ms)": timings.get("inference_time_ms"),
                }
            )
        else:
            rows.append(
                {
                    "Model": m["name"],
                    "Family": m.get("family", "unknown").capitalize(),
                    "Features / Qubits": "N/A",
                    "Accuracy": None,
                    "Sensitivity": None,
                    "Specificity": None,
                    "F1-Score": None,
                    "ROC-AUC": None,
                    "PR-AUC": None,
                    "Train (ms)": None,
                    "Infer (ms)": None,
                }
            )

    df_summary = pd.DataFrame(rows)
    return df_summary
