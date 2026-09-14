"""Training Script for Phase E Cardiovascular Risk ML Baseline Models."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from ml.artifacts import ModelArtifact, save_artifact
from ml.cardiovascular.pipeline import (
    CARDIO_FEATURE_NAMES,
    CARDIO_TARGET_COLUMN,
    CardiovascularPreprocessor,
)
from ml.logger import get_logger

logger = get_logger("train_cardiovascular")

DATASET_PATH = Path("data/raw/cardiovascular.csv")
BENCHMARK_JSON_PATH = Path("data/processed/benchmark_results.json")
RANDOM_SEED = 42


def train_and_evaluate_cardiovascular_models():
    """Trains Logistic Regression and Random Forest on UCI Cleveland dataset and saves artifacts."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Cardiovascular dataset missing at {DATASET_PATH}")

    logger.info(f"Loading cardiovascular dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    X = df[CARDIO_FEATURE_NAMES]
    y = df[CARDIO_TARGET_COLUMN].astype(int)

    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    logger.info(f"Split data into Train ({len(X_train)} samples) and Test ({len(X_test)} samples).")

    # Fit Preprocessor strictly on training data
    preprocessor = CardiovascularPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_SEED)
    lr_model.fit(X_train_trans, y_train)

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)
    rf_model.fit(X_train_trans, y_train)

    models_config = [
        {
            "artifact_id": "cardiovascular_logistic_regression_v1",
            "model_id": "cardiovascular_logistic_regression_v1",
            "model_name": "Cardiovascular Logistic Regression",
            "model_family": "classical",
            "model_type": "logistic_regression",
            "model": lr_model,
        },
        {
            "artifact_id": "cardiovascular_random_forest_v1",
            "model_id": "cardiovascular_random_forest_v1",
            "model_name": "Cardiovascular Random Forest",
            "model_family": "classical",
            "model_type": "random_forest",
            "model": rf_model,
        },
    ]

    benchmark_records = []

    for cfg in models_config:
        model = cfg["model"]
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(int, cm.ravel())
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        fpr, tpr, thresh_roc = roc_curve(y_test, y_prob)
        prec_pts, rec_pts, thresh_pr = precision_recall_curve(y_test, y_prob)
        curves = {
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr.tolist()],
                "tpr": [round(float(x), 4) for x in tpr.tolist()],
                "thresholds": [round(float(x), 4) for x in thresh_roc.tolist()],
                "auc": round(roc_auc, 4),
            },
            "pr_curve": {
                "precision": [round(float(x), 4) for x in prec_pts.tolist()],
                "recall": [round(float(x), 4) for x in rec_pts.tolist()],
                "thresholds": [round(float(x), 4) for x in thresh_pr.tolist()],
                "auc": round(pr_auc, 4),
            },
        }

        eval_provenance = {
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
            },
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "curves": curves,
            "holdout_samples": len(y_test),
        }

        artifact = ModelArtifact(
            artifact_id=cfg["artifact_id"],
            model_id=cfg["model_id"],
            model_name=cfg["model_name"],
            model_family=cfg["model_family"],
            model_type=cfg["model_type"],
            model=model,
            preprocessor=preprocessor,
            dataset_provenance={
                "disease": "cardiovascular",
                "name": "UCI Cleveland Heart Disease Dataset",
                "openml_id": "heart-disease_v1",
                "samples": len(df),
                "features": len(CARDIO_FEATURE_NAMES),
                "hash": "d0eb82e29f70cbbdd73789209674ab609151e96ce5ffd5ef0666ca0113798c03",
            },
            configuration_provenance={
                "random_seed": RANDOM_SEED,
                "test_size": 0.2,
                "stratify": True,
            },
            evaluation_provenance=eval_provenance,
            artifact_version="v1.0",
        )

        output_path = save_artifact(artifact, base_dir="models/cardiovascular")
        logger.info(
            f"Saved artifact '{cfg['artifact_id']}' -> {output_path} | "
            f"Acc: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}"
        )

        benchmark_records.append(
            {
                "disease": "cardiovascular",
                "dataset_version": "v1.0",
                "pipeline_version": "v1.0",
                "model_id": cfg["artifact_id"],
                "model_name": cfg["model_name"],
                "model_family": cfg["model_family"],
                "model_version": "v1.0",
                "metrics": eval_provenance["metrics"],
                "confusion_matrix": eval_provenance["confusion_matrix"],
                "curves": curves,
                "is_official": True,
            }
        )

    # Append to benchmark_results.json if it exists
    if BENCHMARK_JSON_PATH.exists():
        try:
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Additive upsert by model_id to never drop existing models
            existing_results = data.get("results", [])
            existing_by_id = {r.get("model_id"): r for r in existing_results if r.get("model_id")}
            for rec in benchmark_records:
                existing_by_id[rec["model_id"]] = rec
            data["results"] = list(existing_by_id.values())
            
            with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Updated benchmark records in '{BENCHMARK_JSON_PATH}'.")
        except Exception as e:
            logger.warning(f"Could not update benchmark JSON file: {e!s}")


if __name__ == "__main__":
    train_and_evaluate_cardiovascular_models()
