#!/usr/bin/env python3
"""Phase H Hyperparameter Optimization, Calibration, and Threshold Tuning Script.

Performs systematic 5-fold Stratified Cross-Validation hyperparameter search (RandomizedSearchCV)
on training data (X_train) optimizing for PR-AUC (average_precision).
Applies leakage-free probability calibration (CalibratedClassifierCV) and threshold optimization (cross_val_predict).
Evaluates final candidates on locked holdout test set (X_test), computes Brier score, ROC-AUC, PR-AUC, and Confusion Matrices.
Persists ModelArtifact bundles, manifests, data/processed/benchmark_results.json, and data/processed/optimization_results.json.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
from ml.classical.models import ClassicalModel, ClassicalModelFactory
from ml.logger import get_logger
from ml.preprocessing import RAW_FEATURE_NAMES, TARGET_COLUMN, BiomedicalPreprocessor, split_data

logger = get_logger("optimize_phase_h")

BENCHMARK_JSON_PATH = Path("data/processed/benchmark_results.json")
OPTIMIZATION_JSON_PATH = Path("data/processed/optimization_results.json")
RANDOM_SEED = 42


def find_optimal_threshold(y_true, y_probs_train):
    """Finds decision threshold in [0.1, 0.9] maximizing F1 score on training CV predictions."""
    best_thresh = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.10, 0.90, 81)
    for t in thresholds:
        preds = (y_probs_train >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return round(best_thresh, 4), round(best_f1, 4)


def optimize_and_evaluate(
    model_key: str,
    base_estimator: Any,
    param_distributions: dict[str, Any],
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 10,
    calibrate: bool = True,
):
    """Runs RandomizedSearchCV, probability calibration, and leakage-free threshold optimization."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 1. Hyperparameter Search on X_train only using PR-AUC (average_precision)
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=skf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train_trans, y_train)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_cv_pr_auc = float(search.best_score_)

    # Get std of CV score across folds for best params
    best_index = search.best_index_
    cv_std_pr_auc = float(search.cv_results_["std_test_score"][best_index])

    # 2. Probability Calibration (if requested & estimator supports probabilities)
    calibrated_estimator = best_estimator
    is_calibrated = False
    if calibrate and hasattr(best_estimator, "predict_proba"):
        try:
            calib = CalibratedClassifierCV(estimator=best_estimator, cv=5, method="sigmoid")
            calib.fit(X_train_trans, y_train)
            calibrated_estimator = calib
            is_calibrated = True
        except Exception as e:
            logger.debug(f"Calibration skipped for {model_key}: {e}")

    # 3. Leakage-Free Threshold Optimization on Training CV Predictions
    best_threshold = 0.5
    train_opt_f1 = 0.0
    if hasattr(calibrated_estimator, "predict_proba"):
        try:
            cv_train_probs = cross_val_predict(
                calibrated_estimator, X_train_trans, y_train, cv=skf, method="predict_proba"
            )[:, 1]
            best_threshold, train_opt_f1 = find_optimal_threshold(y_train, cv_train_probs)
        except Exception as e:
            logger.debug(f"Threshold tuning skipped for {model_key}: {e}")

    # 4. Final Evaluation on Held-Out Test Set (X_test) using Frozen Threshold
    if hasattr(calibrated_estimator, "predict_proba"):
        y_prob = calibrated_estimator.predict_proba(X_test_trans)[:, 1]
        y_pred = (y_prob >= best_threshold).astype(int)
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))
    else:
        y_pred = calibrated_estimator.predict(X_test_trans)
        y_prob = None
        roc_auc = None
        pr_auc = None
        brier = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = map(int, cm.ravel())
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "cv_mean_pr_auc": round(best_cv_pr_auc, 4),
        "cv_std_pr_auc": round(cv_std_pr_auc, 4),
        "train_optimal_f1": train_opt_f1,
        "frozen_threshold": best_threshold,
        "is_calibrated": is_calibrated,
    }

    conf_matrix = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    search_meta = {
        "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v)) for k, v in best_params.items()},
        "cv_folds": 5,
        "n_iter": n_iter,
        "scoring_objective": "average_precision (PR-AUC)",
    }

    return calibrated_estimator, metrics, conf_matrix, search_meta


def optimize_diabetes_portfolio():
    """Runs Phase H optimization across Diabetes models."""
    logger.info("=== Phase H: Optimizing Diabetes Model Portfolio ===")
    df = pd.read_csv("data/raw/pima_diabetes.csv")
    X = df[RAW_FEATURE_NAMES]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN, test_size=0.2, random_seed=RANDOM_SEED)

    preprocessor = BiomedicalPreprocessor()
    preprocessor.fit(X_train)

    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)
    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()

    configs = [
        (
            "diabetes_rf_opt_v1",
            "Random Forest (Optimized)",
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 150, 200],
                "max_depth": [4, 6, 8, 10],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "diabetes_gb_opt_v1",
            "Gradient Boosting (Optimized)",
            "gradient_boosting",
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [80, 120, 160],
                "learning_rate": [0.03, 0.08, 0.15],
                "max_depth": [3, 4, 5],
                "subsample": [0.8, 0.9, 1.0],
            },
        ),
        (
            "diabetes_xgb_opt_v1",
            "XGBoost (Optimized)",
            "xgboost",
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
            {
                "n_estimators": [80, 120, 160],
                "learning_rate": [0.03, 0.08, 0.15],
                "max_depth": [3, 4, 5],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        ),
        (
            "diabetes_soft_voting_opt_v1",
            "Soft Voting Ensemble (Optimized)",
            "soft_voting",
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)),
                    ("rf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=RANDOM_SEED)),
                    ("xgb", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=RANDOM_SEED)),
                    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ],
                voting="soft",
            ),
            {},  # Presets trained directly
        ),
    ]

    benchmark_records = []
    opt_records = []

    for art_id, name, mtype, base_est, p_grid in configs:
        n_iter = 10 if p_grid else 1
        if not p_grid:
            p_grid = {"cv": [5]} if hasattr(base_est, "cv") else {}

        calib_est, metrics, cm, search_meta = optimize_and_evaluate(
            art_id, base_est, p_grid, X_tr, y_tr, X_te, y_te, n_iter=n_iter, calibrate=True
        )

        wrapper = ClassicalModel(name, mtype, search_meta["best_params"], RANDOM_SEED)
        wrapper.estimator_ = calib_est
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_tr.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=name,
            model_family="classical",
            model_type=mtype,
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
                "optimization": search_meta,
                "frozen_threshold": metrics["frozen_threshold"],
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": cm,
                "holdout_samples": len(y_te),
            },
            artifact_version="v2.0_optimized",
        )

        out_path = save_artifact(artifact, base_dir="models")
        logger.info(
            f"Saved Diabetes optimized model '{art_id}' -> {out_path} | "
            f"PR-AUC: {metrics['pr_auc']}, Brier: {metrics['brier_score']}, Thresh: {metrics['frozen_threshold']}"
        )

        rec = {
            "disease": "diabetes",
            "dataset_version": "v1.0",
            "pipeline_version": "v2.0",
            "model_id": art_id,
            "model_name": name,
            "model_family": "classical",
            "model_version": "v2.0_optimized",
            "metrics": metrics,
            "confusion_matrix": cm,
            "is_official": True,
        }
        benchmark_records.append(rec)
        opt_records.append({**rec, "optimization_provenance": search_meta})

    return benchmark_records, opt_records


def optimize_cardiovascular_portfolio():
    """Runs Phase H optimization across Cardiovascular models."""
    logger.info("=== Phase H: Optimizing Cardiovascular Model Portfolio ===")
    df = pd.read_csv("data/raw/cardiovascular.csv")
    X = df[CARDIO_FEATURE_NAMES]
    y = df[CARDIO_TARGET_COLUMN].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = CardiovascularPreprocessor()
    preprocessor.fit(X_train)

    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)
    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()

    configs = [
        (
            "cardiovascular_extra_trees_opt_v1",
            "Cardiovascular Extra Trees (Optimized)",
            "extra_trees",
            ExtraTreesClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 150, 200],
                "max_depth": [4, 6, 8, None],
                "min_samples_split": [2, 4, 6],
                "criterion": ["gini", "entropy"],
            },
        ),
        (
            "cardiovascular_rbf_svm_opt_v1",
            "Cardiovascular RBF SVM (Optimized)",
            "svm",
            SVC(probability=True, random_state=RANDOM_SEED),
            {
                "C": [0.5, 1.0, 2.0, 5.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "cardiovascular_stacking_opt_v1",
            "Cardiovascular Stacking Ensemble (Optimized)",
            "stacking",
            StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)),
                    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
                    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED)),
                    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ],
                final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED),
                cv=5,
            ),
            {},
        ),
    ]

    benchmark_records = []
    opt_records = []

    for art_id, name, mtype, base_est, p_grid in configs:
        n_iter = 10 if p_grid else 1
        if not p_grid:
            p_grid = {"cv": [5]}

        calib_est, metrics, cm, search_meta = optimize_and_evaluate(
            art_id, base_est, p_grid, X_tr, y_tr, X_te, y_te, n_iter=n_iter, calibrate=True
        )

        wrapper = ClassicalModel(name, mtype, search_meta["best_params"], RANDOM_SEED)
        wrapper.estimator_ = calib_est
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_tr.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=name,
            model_family="classical",
            model_type=mtype,
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
                "optimization": search_meta,
                "frozen_threshold": metrics["frozen_threshold"],
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": cm,
                "holdout_samples": len(y_te),
            },
            artifact_version="v2.0_optimized",
        )

        out_path = save_artifact(artifact, base_dir="models/cardiovascular")
        logger.info(
            f"Saved Cardio optimized model '{art_id}' -> {out_path} | "
            f"PR-AUC: {metrics['pr_auc']}, Brier: {metrics['brier_score']}, Thresh: {metrics['frozen_threshold']}"
        )

        rec = {
            "disease": "cardiovascular",
            "dataset_version": "v1.0",
            "pipeline_version": "v1.0",
            "model_id": art_id,
            "model_name": name,
            "model_family": "classical",
            "model_version": "v2.0_optimized",
            "metrics": metrics,
            "confusion_matrix": cm,
            "is_official": True,
        }
        benchmark_records.append(rec)
        opt_records.append({**rec, "optimization_provenance": search_meta})

    return benchmark_records, opt_records


def optimize_cancer_portfolio():
    """Runs Phase H optimization across Cancer models."""
    logger.info("=== Phase H: Optimizing Cancer Model Portfolio ===")
    df = pd.read_csv("data/raw/cancer.csv")
    X = df[CANCER_FEATURE_NAMES]
    y = df[CANCER_TARGET_COLUMN].astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = CancerPreprocessor()
    preprocessor.fit(X_train)

    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)
    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()

    configs = [
        (
            "cancer_stacking_opt_v1",
            "Breast Cancer Stacking Ensemble (Optimized)",
            "stacking",
            StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)),
                    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
                    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED)),
                    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ],
                final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED),
                cv=5,
            ),
            {},
        ),
        (
            "cancer_rf_opt_v1",
            "Breast Cancer Random Forest (Optimized)",
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": [100, 150, 200],
                "max_depth": [4, 6, 8, None],
                "min_samples_split": [2, 4, 6],
                "criterion": ["gini", "entropy"],
            },
        ),
        (
            "cancer_soft_voting_opt_v1",
            "Breast Cancer Soft Voting Ensemble (Optimized)",
            "soft_voting",
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)),
                    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED)),
                    ("xgb", XGBClassifier(n_estimators=85, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=RANDOM_SEED)),
                    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_SEED)),
                ],
                voting="soft",
            ),
            {},
        ),
    ]

    benchmark_records = []
    opt_records = []

    for art_id, name, mtype, base_est, p_grid in configs:
        n_iter = 10 if p_grid else 1
        if not p_grid:
            p_grid = {"cv": [5]} if hasattr(base_est, "cv") else {"voting": ["soft"]}

        calib_est, metrics, cm, search_meta = optimize_and_evaluate(
            art_id, base_est, p_grid, X_tr, y_tr, X_te, y_te, n_iter=n_iter, calibrate=True
        )

        wrapper = ClassicalModel(name, mtype, search_meta["best_params"], RANDOM_SEED)
        wrapper.estimator_ = calib_est
        wrapper.is_fitted_ = True
        wrapper.n_features_in_ = X_tr.shape[1]

        artifact = ModelArtifact(
            artifact_id=art_id,
            model_id=art_id,
            model_name=name,
            model_family="classical",
            model_type=mtype,
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
                "optimization": search_meta,
                "frozen_threshold": metrics["frozen_threshold"],
            },
            evaluation_provenance={
                "metrics": metrics,
                "confusion_matrix": cm,
                "holdout_samples": len(y_te),
            },
            artifact_version="v2.0_optimized",
        )

        out_path = save_artifact(artifact, base_dir="models/cancer")
        logger.info(
            f"Saved Cancer optimized model '{art_id}' -> {out_path} | "
            f"PR-AUC: {metrics['pr_auc']}, Brier: {metrics['brier_score']}, Thresh: {metrics['frozen_threshold']}"
        )

        rec = {
            "disease": "cancer",
            "dataset_version": "v1.0",
            "pipeline_version": "v1.0",
            "model_id": art_id,
            "model_name": name,
            "model_family": "classical",
            "model_version": "v2.0_optimized",
            "metrics": metrics,
            "confusion_matrix": cm,
            "is_official": True,
        }
        benchmark_records.append(rec)
        opt_records.append({**rec, "optimization_provenance": search_meta})

    return benchmark_records, opt_records


def main():
    logger.info("Starting Phase H Hyperparameter Optimization, Calibration & Threshold Run...")

    dib_b, dib_opt = optimize_diabetes_portfolio()
    cardio_b, cardio_opt = optimize_cardiovascular_portfolio()
    cancer_b, cancer_opt = optimize_cancer_portfolio()

    all_b = dib_b + cardio_b + cancer_b
    all_opt = dib_opt + cardio_opt + cancer_opt

    # Update benchmark_results.json while preserving existing baseline records
    if BENCHMARK_JSON_PATH.exists():
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        existing_results = data.get("results", [])
        new_ids = {r["model_id"] for r in all_b}
        filtered = [r for r in existing_results if r["model_id"] not in new_ids]
        filtered.extend(all_b)
        data["results"] = filtered

        with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Updated benchmark results in '{BENCHMARK_JSON_PATH}'. Total models: {len(filtered)}")

    # Write dedicated optimization_results.json
    with open(OPTIMIZATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "Phase H",
                "timestamp": pd.Timestamp.now().isoformat(),
                "optimization_objective": "PR-AUC (average_precision)",
                "cross_validation": "5-Fold StratifiedKFold (X_train only)",
                "test_set_status": "Locked (X_test used for final holdout evaluation only)",
                "optimized_models": all_opt,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved optimization run metadata to '{OPTIMIZATION_JSON_PATH}'.")

    logger.info("Phase H Optimization, Calibration & Threshold Tuning Complete!")


if __name__ == "__main__":
    main()
