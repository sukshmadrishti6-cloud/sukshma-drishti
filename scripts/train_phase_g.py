#!/usr/bin/env python3
"""Phase G Training and Benchmark Evaluation Script.

Expands classical ML portfolios across Diabetes, Cardiovascular, and Cancer datasets.
Performs 5-fold Stratified Cross-Validation on training data and held-out test evaluation.
Saves ModelArtifact joblib bundles, manifests, and updates data/processed/benchmark_results.json.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Ensure project root & backend are in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from ml.artifacts import ModelArtifact, save_artifact
from ml.cancer.pipeline import CANCER_FEATURE_NAMES, CANCER_TARGET_COLUMN, CancerPreprocessor
from ml.cardiovascular.pipeline import (
    CARDIO_FEATURE_NAMES,
    CARDIO_TARGET_COLUMN,
    CardiovascularPreprocessor,
)
from ml.classical.models import ClassicalModelFactory
from ml.logger import get_logger
from ml.preprocessing import RAW_FEATURE_NAMES, TARGET_COLUMN, BiomedicalPreprocessor, split_data

logger = get_logger("train_phase_g")

BENCHMARK_JSON_PATH = Path("data/processed/benchmark_results.json")
RANDOM_SEED = 42


def evaluate_model_on_data(model, X_train_trans, y_train, X_test_trans, y_test):
    """Conducts 5-fold CV on train data and computes holdout test metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Cross validation on training fold only
    cv_res = cross_validate(model, X_train_trans, y_train, cv=skf, scoring=["accuracy", "f1"], error_score="raise")
    mean_cv_acc = float(np.mean(cv_res["test_accuracy"]))
    mean_cv_f1 = float(np.mean(cv_res["test_f1"]))

    # Fit model on entire training set
    model.fit(X_train_trans, y_train)

    # Test evaluation
    y_pred = model.predict(X_test_trans)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    pr_auc = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test_trans)
            if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
                y_prob = probs[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_prob))
                pr_auc = float(average_precision_score(y_test, y_prob))
        except Exception as e:
            logger.debug(f"Could not compute probabilities for model {type(model)}: {e}")

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "cv_mean_accuracy": round(mean_cv_acc, 4),
        "cv_mean_f1": round(mean_cv_f1, 4),
    }

    conf_matrix = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    return model, metrics, conf_matrix


def train_diabetes_portfolio():
    """Trains Phase G Diabetes classical model portfolio."""
    logger.info("--- Training Diabetes Portfolio ---")
    dataset_path = Path("data/raw/pima_diabetes.csv")
    df = pd.read_csv(dataset_path)
    X = df[RAW_FEATURE_NAMES]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN, test_size=0.2, random_seed=RANDOM_SEED)

    preprocessor = BiomedicalPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    models_def = [
        ("linear_svm_v1", "linear_svm", "Support Vector Machine (Linear)", "linear_svm"),
        ("knn_v1", "knn", "K-Nearest Neighbors", "knn"),
        ("decision_tree_v1", "decision_tree", "Decision Tree", "decision_tree"),
        ("extra_trees_v1", "extra_trees", "Extra Trees", "extra_trees"),
        ("gradient_boosting_v1", "gradient_boosting", "Gradient Boosting", "gradient_boosting"),
        ("hist_gradient_boosting_v1", "hist_gradient_boosting", "Histogram Gradient Boosting", "hist_gradient_boosting"),
        ("soft_voting_ensemble_v1", "soft_voting", "Soft Voting Ensemble", "soft_voting"),
        ("hard_voting_ensemble_v1", "hard_voting", "Hard Voting Ensemble", "hard_voting"),
        ("stacking_ensemble_v1", "stacking", "Stacking Ensemble", "stacking"),
    ]

    records = []
    for art_id, model_key, model_name, model_type in models_def:
        wrapper = ClassicalModelFactory.create_model(model_key, random_seed=RANDOM_SEED)
        raw_model = wrapper.estimator_

        trained_estimator, metrics, conf_matrix = evaluate_model_on_data(
            raw_model, X_train_trans, y_train, X_test_trans, y_test
        )

        wrapper.estimator_ = trained_estimator
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_train_trans.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=model_name,
            model_family="classical",
            model_type=model_type,
            model=wrapper,
            preprocessor=preprocessor,
            dataset_provenance={
                "disease": "diabetes",
                "name": "Pima Indians Diabetes Dataset",
                "samples": len(df),
                "features": len(RAW_FEATURE_NAMES),
            },
            configuration_provenance={
                "random_seed": RANDOM_SEED,
                "test_size": 0.2,
                "stratify": True,
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "holdout_samples": len(y_test),
            },
            artifact_version="v1.0",
        )

        output_path = save_artifact(artifact, base_dir="models")
        logger.info(f"Saved Diabetes model '{art_id}' -> {output_path} | Acc: {metrics['accuracy']}")

        records.append(
            {
                "disease": "diabetes",
                "dataset_version": "v1.0",
                "pipeline_version": "v2.0",
                "model_id": art_id,
                "model_name": model_name,
                "model_family": "classical",
                "model_version": "v1.0",
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "is_official": True,
            }
        )
    return records


def train_cardiovascular_portfolio():
    """Trains Phase G Cardiovascular classical model portfolio."""
    logger.info("--- Training Cardiovascular Portfolio ---")
    dataset_path = Path("data/raw/cardiovascular.csv")
    df = pd.read_csv(dataset_path)
    X = df[CARDIO_FEATURE_NAMES]
    y = df[CARDIO_TARGET_COLUMN].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = CardiovascularPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    models_def = [
        ("cardiovascular_linear_svm_v1", "linear_svm", "Cardiovascular Linear SVM", "linear_svm"),
        ("cardiovascular_rbf_svm_v1", "svm", "Cardiovascular RBF SVM", "svm"),
        ("cardiovascular_knn_v1", "knn", "Cardiovascular KNN", "knn"),
        ("cardiovascular_decision_tree_v1", "decision_tree", "Cardiovascular Decision Tree", "decision_tree"),
        ("cardiovascular_extra_trees_v1", "extra_trees", "Cardiovascular Extra Trees", "extra_trees"),
        ("cardiovascular_gradient_boosting_v1", "gradient_boosting", "Cardiovascular Gradient Boosting", "gradient_boosting"),
        ("cardiovascular_hist_gradient_boosting_v1", "hist_gradient_boosting", "Cardiovascular Hist Gradient Boosting", "hist_gradient_boosting"),
        ("cardiovascular_xgboost_v1", "xgboost", "Cardiovascular XGBoost", "xgboost"),
        ("cardiovascular_soft_voting_ensemble_v1", "soft_voting", "Cardiovascular Soft Voting Ensemble", "soft_voting"),
        ("cardiovascular_hard_voting_ensemble_v1", "hard_voting", "Cardiovascular Hard Voting Ensemble", "hard_voting"),
        ("cardiovascular_stacking_ensemble_v1", "stacking", "Cardiovascular Stacking Ensemble", "stacking"),
    ]

    records = []
    for art_id, model_key, model_name, model_type in models_def:
        wrapper = ClassicalModelFactory.create_model(model_key, random_seed=RANDOM_SEED)
        raw_model = wrapper.estimator_

        trained_estimator, metrics, conf_matrix = evaluate_model_on_data(
            raw_model, X_train_trans, y_train, X_test_trans, y_test
        )

        wrapper.estimator_ = trained_estimator
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_train_trans.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=model_name,
            model_family="classical",
            model_type=model_type,
            model=wrapper,
            preprocessor=preprocessor,
            dataset_provenance={
                "disease": "cardiovascular",
                "name": "UCI Cleveland Heart Disease Dataset",
                "samples": len(df),
                "features": len(CARDIO_FEATURE_NAMES),
            },
            configuration_provenance={
                "random_seed": RANDOM_SEED,
                "test_size": 0.2,
                "stratify": True,
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "holdout_samples": len(y_test),
            },
            artifact_version="v1.0",
        )

        output_path = save_artifact(artifact, base_dir="models/cardiovascular")
        logger.info(f"Saved Cardio model '{art_id}' -> {output_path} | Acc: {metrics['accuracy']}")

        records.append(
            {
                "disease": "cardiovascular",
                "dataset_version": "v1.0",
                "pipeline_version": "v1.0",
                "model_id": art_id,
                "model_name": model_name,
                "model_family": "classical",
                "model_version": "v1.0",
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "is_official": True,
            }
        )
    return records


def train_cancer_portfolio():
    """Trains Phase G Cancer classical model portfolio."""
    logger.info("--- Training Cancer Portfolio ---")
    dataset_path = Path("data/raw/cancer.csv")
    df = pd.read_csv(dataset_path)
    X = df[CANCER_FEATURE_NAMES]
    y = df[CANCER_TARGET_COLUMN].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = CancerPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    models_def = [
        ("cancer_linear_svm_v1", "linear_svm", "Breast Cancer Linear SVM", "linear_svm"),
        ("cancer_rbf_svm_v1", "svm", "Breast Cancer RBF SVM", "svm"),
        ("cancer_knn_v1", "knn", "Breast Cancer KNN", "knn"),
        ("cancer_decision_tree_v1", "decision_tree", "Breast Cancer Decision Tree", "decision_tree"),
        ("cancer_extra_trees_v1", "extra_trees", "Breast Cancer Extra Trees", "extra_trees"),
        ("cancer_gradient_boosting_v1", "gradient_boosting", "Breast Cancer Gradient Boosting", "gradient_boosting"),
        ("cancer_hist_gradient_boosting_v1", "hist_gradient_boosting", "Breast Cancer Hist Gradient Boosting", "hist_gradient_boosting"),
        ("cancer_xgboost_v1", "xgboost", "Breast Cancer XGBoost", "xgboost"),
        ("cancer_soft_voting_ensemble_v1", "soft_voting", "Breast Cancer Soft Voting Ensemble", "soft_voting"),
        ("cancer_hard_voting_ensemble_v1", "hard_voting", "Breast Cancer Hard Voting Ensemble", "hard_voting"),
        ("cancer_stacking_ensemble_v1", "stacking", "Breast Cancer Stacking Ensemble", "stacking"),
    ]

    records = []
    for art_id, model_key, model_name, model_type in models_def:
        wrapper = ClassicalModelFactory.create_model(model_key, random_seed=RANDOM_SEED)
        raw_model = wrapper.estimator_

        trained_estimator, metrics, conf_matrix = evaluate_model_on_data(
            raw_model, X_train_trans, y_train, X_test_trans, y_test
        )

        wrapper.estimator_ = trained_estimator
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_train_trans.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=model_name,
            model_family="classical",
            model_type=model_type,
            model=wrapper,
            preprocessor=preprocessor,
            dataset_provenance={
                "disease": "cancer",
                "name": "UCI Breast Cancer Wisconsin (Diagnostic) Dataset",
                "samples": len(df),
                "features": len(CANCER_FEATURE_NAMES),
            },
            configuration_provenance={
                "random_seed": RANDOM_SEED,
                "test_size": 0.2,
                "stratify": True,
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "holdout_samples": len(y_test),
            },
            artifact_version="v1.0",
        )

        output_path = save_artifact(artifact, base_dir="models/cancer")
        logger.info(f"Saved Cancer model '{art_id}' -> {output_path} | Acc: {metrics['accuracy']}")

        records.append(
            {
                "disease": "cancer",
                "dataset_version": "v1.0",
                "pipeline_version": "v1.0",
                "model_id": art_id,
                "model_name": model_name,
                "model_family": "classical",
                "model_version": "v1.0",
                "metrics": metrics,
                "confusion_matrix": conf_matrix,
                "is_official": True,
            }
        )
    return records


def main():
    logger.info("Starting Phase G Model Portfolio Training & Benchmark Run...")

    dib_records = train_diabetes_portfolio()
    cardio_records = train_cardiovascular_portfolio()
    cancer_records = train_cancer_portfolio()

    all_new_records = dib_records + cardio_records + cancer_records

    # Update benchmark_results.json while preserving existing baseline records
    if BENCHMARK_JSON_PATH.exists():
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        existing_results = data.get("results", [])
        new_ids = {r["model_id"] for r in all_new_records}
        filtered = [r for r in existing_results if r["model_id"] not in new_ids]
        filtered.extend(all_new_records)
        data["results"] = filtered

        with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully updated benchmark results in '{BENCHMARK_JSON_PATH}'. Total models: {len(filtered)}")

    logger.info("Phase G Portfolio Training Complete!")


if __name__ == "__main__":
    main()
