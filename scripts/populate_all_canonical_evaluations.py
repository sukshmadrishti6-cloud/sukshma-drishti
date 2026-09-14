"""Evaluates all 67 canonical models across all 3 diseases on isolated holdout sets and additively populates benchmark_results.json."""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.artifacts import load_artifact
from ml.cancer.pipeline import CANCER_FEATURE_NAMES, CANCER_TARGET_COLUMN
from ml.cardiovascular.pipeline import CARDIO_FEATURE_NAMES, CARDIO_TARGET_COLUMN
from ml.evaluation.metrics import (
    calculate_calibration_data,
    calculate_classification_metrics,
    calculate_pr_curve_data,
    calculate_roc_curve_data,
)
from ml.preprocessing import (
    RAW_FEATURE_NAMES,
    split_data,
)
from app.services.ml_service import CANONICAL_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("populate_canonical_evaluations")

BENCHMARK_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_results.json"
RANDOM_SEED = 42


def populate_all_evaluations():
    logger.info("==========================================================================")
    logger.info("POPULATING ALL 67 CANONICAL MODEL HOLDOUT EVALUATIONS")
    logger.info(f"Target file: {BENCHMARK_JSON_PATH}")
    logger.info("==========================================================================")

    # 1. Load Datasets & Splits
    # Diabetes
    dib_path = PROJECT_ROOT / "data" / "raw" / "pima_diabetes.csv"
    if not dib_path.exists():
        dib_path = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
    df_dib = pd.read_csv(dib_path)
    X_dib_tr, X_dib_te, y_dib_tr, y_dib_te = split_data(
        df_dib, target_column="Outcome", test_size=0.2, random_seed=RANDOM_SEED
    )

    # Cardiovascular
    cardio_path = PROJECT_ROOT / "data" / "raw" / "cardiovascular.csv"
    df_cardio = pd.read_csv(cardio_path)
    X_c_tr, X_c_te, y_c_tr, y_c_te = split_data(
        df_cardio, target_column=CARDIO_TARGET_COLUMN, test_size=0.2, random_seed=RANDOM_SEED
    )

    # Breast Cancer
    cancer_path = PROJECT_ROOT / "data" / "raw" / "cancer.csv"
    df_cancer = pd.read_csv(cancer_path)
    X_ca_tr, X_ca_te, y_ca_tr, y_ca_te = split_data(
        df_cancer, target_column=CANCER_TARGET_COLUMN, test_size=0.2, random_seed=RANDOM_SEED
    )

    disease_test_sets = {
        "diabetes": (X_dib_te, y_dib_te),
        "cardiovascular": (X_c_te, y_c_te),
        "cancer": (X_ca_te, y_ca_te),
    }

    # Load existing benchmark and optimization data
    if BENCHMARK_JSON_PATH.exists():
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
    else:
        bench_data = {"models": [], "results": []}

    opt_json_path = PROJECT_ROOT / "data" / "processed" / "optimization_results.json"
    opt_data = {}
    if opt_json_path.exists():
        try:
            with open(opt_json_path, "r", encoding="utf-8") as f:
                opt_data = json.load(f)
        except Exception:
            pass

    results_by_id = {r["model_id"]: r for r in bench_data.get("results", []) if "model_id" in r}
    models_by_id = {m["model_id"]: m for m in bench_data.get("models", []) if "model_id" in m}

    success_count = 0
    missing_count = 0

    for model_id, m_def in CANONICAL_MODELS.items():
        disease = m_def["disease"]
        X_test, y_test = disease_test_sets[disease]
        artifact_path = PROJECT_ROOT / m_def["artifact_path"]

        # Check if model artifact exists on disk
        if not artifact_path.exists():
            # Check optimization_results.json first
            opt_rec = next((m for m in opt_data.get("optimized_models", []) if m.get("model_id") == model_id), None)
            if opt_rec and opt_rec.get("metrics", {}).get("accuracy") is not None:
                results_by_id[model_id] = opt_rec
                if disease == "diabetes":
                    models_by_id[model_id] = opt_rec
                logger.info(f"[{model_id}] Artifact not on disk but found in optimization_results.json, retaining.")
                success_count += 1
                continue
            # If already evaluated in results_by_id with valid accuracy, keep existing
            if model_id in results_by_id and results_by_id[model_id].get("metrics", {}).get("accuracy") is not None:
                logger.info(f"[{model_id}] Artifact not on disk but found in benchmark_results.json, retaining.")
                success_count += 1
                continue
            logger.warning(f"[{model_id}] Artifact not found at {artifact_path} and not in benchmark.")
            missing_count += 1
            continue

        try:
            artifact = load_artifact(artifact_path)
            t0 = time.perf_counter()
            y_pred = artifact.predict(X_test)
            inf_time_ms = (time.perf_counter() - t0) * 1000.0 / len(X_test)

            y_prob = None
            if hasattr(artifact, "predict_proba"):
                try:
                    y_prob = artifact.predict_proba(X_test)
                except Exception as ex:
                    logger.debug(f"predict_proba not available for {model_id}: {ex}")
                    y_prob = None

            # Calculate classification metrics
            metrics_dict = calculate_classification_metrics(y_test, y_pred, y_prob)
            cm_data = metrics_dict.pop("confusion_matrix", {})

            # Curves & calibration
            roc_curve = None
            pr_curve = None
            calib_data = {"status": "unsupported"}

            if y_prob is not None:
                y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                try:
                    roc_curve = calculate_roc_curve_data(y_test, y_prob_pos)
                except Exception:
                    pass
                try:
                    pr_curve = calculate_pr_curve_data(y_test, y_prob_pos)
                except Exception:
                    pass
                try:
                    calib_data = calculate_calibration_data(y_test, y_prob_pos)
                except Exception:
                    calib_data = {"status": "available", "brier_score": metrics_dict.get("brier_score")}

            # Create Record
            record = {
                "model_id": model_id,
                "model_name": m_def["display_name"],
                "disease": disease,
                "model_family": m_def["family"],
                "version": m_def.get("version", "v1.0"),
                "features_used": m_def.get("classical_feature_count") or m_def.get("raw_feature_count"),
                "qubits": m_def.get("qubit_count"),
                "feature_space": f"{disease.capitalize()} Cleaned & Standardized ({m_def.get('classical_feature_count', m_def.get('raw_feature_count'))} features)",
                "metrics": metrics_dict,
                "confusion_matrix": cm_data,
                "timings": {
                    "training_time_ms": artifact.metadata.get("train_time_ms", 0.0),
                    "inference_time_ms": inf_time_ms,
                },
                "curves": {
                    "roc_curve": roc_curve,
                    "pr_curve": pr_curve,
                } if roc_curve or pr_curve else {},
                "calibration": calib_data,
                "circuit_specs": artifact.metadata.get("circuit_metadata") if m_def["family"] == "quantum" else None,
                "is_official": True,
            }

            results_by_id[model_id] = record

            # If diabetes, also register in bench_data["models"]
            if disease == "diabetes":
                models_by_id[model_id] = {
                    "model_id": model_id,
                    "name": m_def["display_name"],
                    "family": m_def["family"],
                    "version": m_def.get("version", "v1.0"),
                    "features_used": m_def.get("classical_feature_count") or m_def.get("raw_feature_count"),
                    "qubits": m_def.get("qubit_count"),
                    "feature_space": record["feature_space"],
                    "metrics": metrics_dict,
                    "confusion_matrix": cm_data,
                    "timings": record["timings"],
                    "curves": record["curves"],
                    "calibration": calib_data,
                    "circuit_specs": record.get("circuit_specs"),
                    "status": "SUCCESS",
                    "is_official": True,
                }

            logger.info(
                f"[{model_id}] Evaluated successfully | Acc: {metrics_dict.get('accuracy', 0):.4f}, "
                f"F1: {metrics_dict.get('f1_score', 0):.4f}, ROC-AUC: {metrics_dict.get('roc_auc', 0):.4f}"
            )
            success_count += 1

        except Exception as e:
            logger.error(f"[{model_id}] Evaluation failed: {e!s}", exc_info=True)
            missing_count += 1

    # Persist updated benchmark_results.json
    bench_data["results"] = list(results_by_id.values())
    bench_data["models"] = list(models_by_id.values())
    bench_data["evaluation_protocol"] = {
        "type": "80/20 Stratified Holdout Test Split",
        "isolated_test_partition": True,
        "random_seed": RANDOM_SEED,
        "total_evaluated_models": len(results_by_id),
    }

    BENCHMARK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    logger.info(
        f"Evaluation population completed. Success: {success_count}, Missing/Failed: {missing_count}. "
        f"Total models in results: {len(bench_data['results'])}, in models (diabetes): {len(bench_data['models'])}"
    )


if __name__ == "__main__":
    populate_all_evaluations()
