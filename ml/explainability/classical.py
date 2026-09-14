"""Classical Model Explainers (Logistic Regression, Tree Ensembles, RBF SVM, and Ensembles)."""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml.explainability.base import (
    BaseExplainer,
    ExplanationResult,
    FeatureContribution,
    GlobalExplanationResult,
    GlobalFeatureImportance,
)

logger = logging.getLogger(__name__)


def _extract_coef(model: Any) -> tuple[np.ndarray | None, float | None]:
    """Safely extracts weights and intercept from estimator or CalibratedClassifierCV."""
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        return model.coef_[0], float(model.intercept_[0])
    if hasattr(model, "calibrated_classifiers_"):
        coefs = [
            cc.estimator.coef_[0]
            for cc in model.calibrated_classifiers_
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "coef_")
        ]
        intercepts = [
            float(cc.estimator.intercept_[0])
            for cc in model.calibrated_classifiers_
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "intercept_")
        ]
        if coefs:
            return np.mean(coefs, axis=0), float(np.mean(intercepts))
    if hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
        return model.estimator.coef_[0], float(model.estimator.intercept_[0])
    return None, None


def _extract_feature_importances(model: Any) -> np.ndarray | None:
    """Safely extracts feature_importances_ from tree estimator or CalibratedClassifierCV."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "calibrated_classifiers_"):
        imps = [
            cc.estimator.feature_importances_
            for cc in model.calibrated_classifiers_
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "feature_importances_")
        ]
        if imps:
            return np.mean(imps, axis=0)
    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        return model.estimator.feature_importances_
    return None


def _get_baseline_features(preprocessor: Any, feature_names: list[str]) -> dict[str, float]:
    """Extracts reference baseline feature values from any disease preprocessor."""
    if hasattr(preprocessor, "impute_medians_") and preprocessor.impute_medians_:
        return {col: float(preprocessor.impute_medians_.get(col, 0.0)) for col in feature_names}
    if hasattr(preprocessor, "scaler") and hasattr(preprocessor.scaler, "mean_") and preprocessor.scaler.mean_ is not None:
        return {col: float(preprocessor.scaler.mean_[i]) for i, col in enumerate(feature_names)}
    if hasattr(preprocessor, "scaler_") and hasattr(preprocessor.scaler_, "mean_") and preprocessor.scaler_.mean_ is not None:
        return {col: float(preprocessor.scaler_.mean_[i]) for i, col in enumerate(feature_names)}
    return {col: 0.0 for col in feature_names}


def _load_reference_dataset(artifact: Any) -> tuple[pd.DataFrame | None, pd.Series | None, str]:
    """Loads appropriate 80/20 stratified holdout test split for the artifact's disease domain."""
    disease = getattr(artifact, "disease", None)
    art_id = getattr(artifact, "artifact_id", getattr(artifact, "model_id", ""))

    if not disease:
        if "cancer" in art_id:
            disease = "cancer"
        elif "cardio" in art_id:
            disease = "cardiovascular"
        else:
            disease = "diabetes"

    seed = getattr(artifact, "random_seed", 42) or 42

    try:
        from sklearn.model_selection import train_test_split

        if disease == "cancer":
            p = Path("data/raw/cancer.csv")
            if p.exists():
                df = pd.read_csv(p)
                X = df.drop(columns=["target"])
                y = df["target"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                return X_test, y_test, disease
        elif disease == "cardiovascular":
            p = Path("data/raw/cardiovascular.csv")
            if p.exists():
                df = pd.read_csv(p)
                X = df.drop(columns=["target"])
                y = df["target"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                return X_test, y_test, disease
        else:
            p = Path("data/raw/diabetes.csv")
            if not p.exists():
                p = Path("data/raw/pima_diabetes.csv")
            if p.exists():
                df = pd.read_csv(p)
                X = df.drop(columns=["Outcome"])
                y = df["Outcome"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
                return X_test, y_test, disease
    except Exception as e:
        logger.warning(f"Failed to load reference dataset for {disease}: {e}")
    return None, None, disease


class LogisticRegressionExplainer(BaseExplainer):
    """Additive linear coefficient attribution explainer for Logistic Regression and Linear Models."""

    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        model = getattr(artifact.model, "estimator_", artifact.model)
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        # Transform raw features
        X_trans = preprocessor.transform(raw_df)
        x_vec = X_trans[0]

        # Extract weights (w_i) and intercept (w_0)
        weights, intercept = _extract_coef(model)
        if weights is None or intercept is None:
            return ExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                explanation_type="local",
                method="linear_additive_attribution",
                status="unsupported",
                baseline_value=0.0,
                features=[],
                limitations=["Linear model coefficients could not be extracted."],
            )

        # Additive log-odds contributions: c_i = w_i * x_i
        contributions = weights * x_vec
        total_abs = float(np.sum(np.abs(contributions))) + 1e-12

        feature_items: list[FeatureContribution] = []
        for i, col in enumerate(feature_names):
            c_val = float(contributions[i])
            raw_val = float(raw_df[col].iloc[0]) if col in raw_df.columns else 0.0
            norm_wt = (abs(c_val) / total_abs) * 100.0
            direction = "positive" if c_val > 1e-5 else ("negative" if c_val < -1e-5 else "neutral")
            feature_items.append(
                FeatureContribution(
                    name=col,
                    value=raw_val,
                    contribution=c_val,
                    direction=direction,
                    normalized_weight=norm_wt,
                )
            )

        # Sort by absolute contribution magnitude descending
        feature_items.sort(key=lambda x: abs(x.contribution), reverse=True)
        if top_k is not None and top_k > 0:
            feature_items = feature_items[:top_k]

        return ExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            explanation_type="local",
            method="linear_additive_attribution",
            status="available",
            baseline_value=intercept,
            features=feature_items,
            limitations=[
                "Linear additive decomposition assumes features act independently on the log-odds scale.",
                "Feature contributions are evaluated in standardized feature space mapped back to named variables.",
            ],
        )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
    ) -> GlobalExplanationResult:
        model = getattr(artifact.model, "estimator_", artifact.model)
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        weights, _ = _extract_coef(model)
        if weights is None:
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method="standardized_coefficient_magnitude",
                status="unsupported",
                features=[],
                limitations=["Linear model coefficients could not be extracted."],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )

        abs_weights = np.abs(weights)
        total_abs = float(np.sum(abs_weights)) + 1e-12

        items: list[GlobalFeatureImportance] = []
        for i, col in enumerate(feature_names):
            imp = float(abs_weights[i])
            norm_wt = (imp / total_abs) * 100.0
            items.append(
                GlobalFeatureImportance(
                    name=col,
                    importance=imp,
                    normalized_weight=norm_wt,
                )
            )

        items.sort(key=lambda x: x.importance, reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        return GlobalExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            method="standardized_coefficient_magnitude",
            status="available",
            features=items,
            limitations=[
                "Global importance represents absolute coefficient magnitudes in standardized z-score feature space.",
                "Assumes monotonic linear relationships across normalized features without non-linear interaction terms.",
            ],
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
        )


class TreeEnsembleExplainer(BaseExplainer):
    """Marginal feature perturbation explainer for Decision Trees, Random Forest, and Gradient Boosted trees."""

    def __init__(self) -> None:
        self._cached_global: dict[str, GlobalExplanationResult] = {}

    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        baseline_dict = _get_baseline_features(preprocessor, feature_names)
        baseline_df = pd.DataFrame([baseline_dict])
        p_base = float(artifact.predict_proba(baseline_df)[0, 1])

        # Current sample prediction
        p_full = float(artifact.predict_proba(raw_df)[0, 1])

        # Marginal contribution of each feature: replacing with baseline reference
        contributions = []
        for col in feature_names:
            perturbed_df = raw_df.copy()
            perturbed_df[col] = baseline_dict.get(col, 0.0)
            p_without_i = float(artifact.predict_proba(perturbed_df)[0, 1])
            delta_p = p_full - p_without_i
            contributions.append(delta_p)

        contributions_arr = np.array(contributions)
        total_abs = float(np.sum(np.abs(contributions_arr))) + 1e-12

        feature_items: list[FeatureContribution] = []
        for i, col in enumerate(feature_names):
            c_val = float(contributions_arr[i])
            raw_val = float(raw_df[col].iloc[0]) if col in raw_df.columns else 0.0
            norm_wt = (abs(c_val) / total_abs) * 100.0
            direction = "positive" if c_val > 1e-5 else ("negative" if c_val < -1e-5 else "neutral")
            feature_items.append(
                FeatureContribution(
                    name=col,
                    value=raw_val,
                    contribution=c_val,
                    direction=direction,
                    normalized_weight=norm_wt,
                )
            )

        feature_items.sort(key=lambda x: abs(x.contribution), reverse=True)
        if top_k is not None and top_k > 0:
            feature_items = feature_items[:top_k]

        return ExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            explanation_type="local",
            method="marginal_feature_perturbation",
            status="available",
            baseline_value=p_base,
            features=feature_items,
            limitations=[
                "Marginal perturbation measures probability shift relative to median population reference values.",
                "Non-linear feature interactions are evaluated conditioned on surrounding fixed features.",
            ],
        )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
    ) -> GlobalExplanationResult:
        art_id = getattr(artifact, "artifact_id", getattr(artifact, "model_id", "tree_model"))
        if art_id in self._cached_global:
            cached_res = self._cached_global[art_id]
            if top_k is not None and top_k > 0:
                return GlobalExplanationResult(
                    model_name=cached_res.model_name,
                    model_family=cached_res.model_family,
                    method=cached_res.method,
                    status=cached_res.status,
                    features=cached_res.features[:top_k],
                    limitations=cached_res.limitations,
                    pipeline_version=cached_res.pipeline_version,
                    metadata=cached_res.metadata,
                )
            return cached_res

        model = getattr(artifact.model, "estimator_", artifact.model)
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        importances = _extract_feature_importances(model)
        if importances is not None:
            total_imp = float(np.sum(importances)) + 1e-12
            items: list[GlobalFeatureImportance] = []
            for i, col in enumerate(feature_names):
                imp = float(importances[i])
                norm_wt = (imp / total_imp) * 100.0
                items.append(
                    GlobalFeatureImportance(
                        name=col,
                        importance=imp,
                        normalized_weight=norm_wt,
                    )
                )

            items.sort(key=lambda x: x.importance, reverse=True)
            is_xgb = "xgb" in getattr(artifact, "model_type", "").lower()
            method_name = "gain_feature_importance" if is_xgb else "mean_decrease_impurity"

            res = GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method=method_name,
                status="available",
                features=items,
                limitations=[
                    "Measures total reduction in split criterion (Gini/Gain) brought by each feature across all trees.",
                    "Impurity-based importances can slightly overestimate the importance of high-cardinality continuous features.",
                ],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )
            self._cached_global[art_id] = res
            if top_k is not None and top_k > 0:
                return GlobalExplanationResult(
                    model_name=res.model_name,
                    model_family=res.model_family,
                    method=res.method,
                    status=res.status,
                    features=items[:top_k],
                    limitations=res.limitations,
                    pipeline_version=res.pipeline_version,
                    metadata=res.metadata,
                )
            return res

        # Fallback for models like HistGradientBoosting that do not expose feature_importances_: Permutation Importance
        X_test_raw, y_test, dis_name = _load_reference_dataset(artifact)
        if X_test_raw is None or y_test is None:
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method="mean_decrease_impurity",
                status="unsupported",
                features=[],
                limitations=["Fitted tree model does not expose feature_importances_."],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )

        X_test_trans = preprocessor.transform(X_test_raw)
        perm_res = permutation_importance(
            model,
            X_test_trans,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=getattr(artifact, "random_seed", 42) or 42,
        )
        importances_mean = perm_res.importances_mean
        non_neg_imp = np.maximum(0.0, importances_mean)
        total_imp = float(np.sum(non_neg_imp)) + 1e-12

        items = []
        for i, col in enumerate(feature_names):
            raw_imp = float(importances_mean[i])
            norm_wt = float((non_neg_imp[i] / total_imp) * 100.0)
            items.append(
                GlobalFeatureImportance(
                    name=col,
                    importance=raw_imp,
                    normalized_weight=norm_wt,
                )
            )
        items.sort(key=lambda x: x.importance, reverse=True)

        res = GlobalExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            method="permutation_importance",
            status="available",
            features=items,
            limitations=[
                "Calculated via permutation feature importance on holdout test partition.",
            ],
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
        )
        self._cached_global[art_id] = res
        if top_k is not None and top_k > 0:
            return GlobalExplanationResult(
                model_name=res.model_name,
                model_family=res.model_family,
                method=res.method,
                status=res.status,
                features=items[:top_k],
                limitations=res.limitations,
                pipeline_version=res.pipeline_version,
                metadata=res.metadata,
            )
        return res


class SVMRBFExplainer(BaseExplainer):
    """Local numerical sensitivity gradient and global permutation importance explainer for RBF-kernel SVMs."""

    def __init__(self) -> None:
        self._cached_global: dict[str, GlobalExplanationResult] = {}

    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        # Base probability
        p_base = float(artifact.predict_proba(raw_df)[0, 1])

        # Finite difference step (5% of scale or 0.1)
        sensitivities = []
        for col in feature_names:
            scale_val = 1.0
            if hasattr(preprocessor, "scaler_") and preprocessor.scaler_ is not None and hasattr(preprocessor.scaler_, "scale_"):
                col_idx = feature_names.index(col)
                scale_val = float(preprocessor.scaler_.scale_[col_idx])
            elif hasattr(preprocessor, "scaler") and preprocessor.scaler is not None and hasattr(preprocessor.scaler, "scale_"):
                col_idx = feature_names.index(col)
                scale_val = float(preprocessor.scaler.scale_[col_idx])

            delta = 0.05 * scale_val
            raw_val = float(raw_df[col].iloc[0]) if col in raw_df.columns else 0.0

            df_plus = raw_df.copy()
            df_plus[col] = raw_val + delta
            p_plus = float(artifact.predict_proba(df_plus)[0, 1])

            df_minus = raw_df.copy()
            df_minus[col] = max(0.0, raw_val - delta)
            p_minus = float(artifact.predict_proba(df_minus)[0, 1])

            grad = (p_plus - p_minus) / (2.0 * delta + 1e-12)
            sensitivities.append(grad)

        sens_arr = np.array(sensitivities)
        total_abs = float(np.sum(np.abs(sens_arr))) + 1e-12

        feature_items: list[FeatureContribution] = []
        for i, col in enumerate(feature_names):
            s_val = float(sens_arr[i])
            raw_val = float(raw_df[col].iloc[0]) if col in raw_df.columns else 0.0
            norm_wt = (abs(s_val) / total_abs) * 100.0
            direction = "positive" if s_val > 1e-5 else ("negative" if s_val < -1e-5 else "neutral")
            feature_items.append(
                FeatureContribution(
                    name=col,
                    value=raw_val,
                    contribution=s_val,
                    direction=direction,
                    normalized_weight=norm_wt,
                )
            )

        feature_items.sort(key=lambda x: abs(x.contribution), reverse=True)
        if top_k is not None and top_k > 0:
            feature_items = feature_items[:top_k]

        return ExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            explanation_type="local",
            method="local_sensitivity_gradient",
            status="available",
            baseline_value=p_base,
            features=feature_items,
            limitations=[
                "RBF SVM is non-linear; gradient reflects local rate of probability change around input point.",
                "Feature sensitivities do not assume linearity over broad clinical intervals.",
            ],
        )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
        reference_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> GlobalExplanationResult:
        art_id = getattr(artifact, "artifact_id", getattr(artifact, "model_id", "svm_rbf"))
        if art_id in self._cached_global and reference_data is None:
            cached_res = self._cached_global[art_id]
            if top_k is not None and top_k > 0:
                return GlobalExplanationResult(
                    model_name=cached_res.model_name,
                    model_family=cached_res.model_family,
                    method=cached_res.method,
                    status=cached_res.status,
                    features=cached_res.features[:top_k],
                    limitations=cached_res.limitations,
                    pipeline_version=cached_res.pipeline_version,
                    metadata=cached_res.metadata,
                )
            return cached_res

        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()
        seed = getattr(artifact, "random_seed", 42) or 42

        # 1. Obtain held-out reference evaluation data for the specific disease
        X_test_raw = None
        y_test = None
        dis_name = "diabetes"
        if reference_data is not None:
            X_test_raw, y_test = reference_data
        else:
            X_test_raw, y_test, dis_name = _load_reference_dataset(artifact)

        if X_test_raw is None or y_test is None:
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method="permutation_importance",
                status="unavailable",
                features=[],
                limitations=[
                    "Held-out evaluation dataset is required to compute empirical permutation importance.",
                ],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
                metadata={"reason": f"Reference dataset not found for disease '{dis_name}'"},
            )

        # 2. Transform holdout data
        X_test_trans = preprocessor.transform(X_test_raw)

        # 3. Compute permutation importance
        estimator = getattr(artifact.model, "estimator_", artifact.model)
        perm_res = permutation_importance(
            estimator,
            X_test_trans,
            y_test,
            scoring="roc_auc",
            n_repeats=5,
            random_state=seed,
        )

        importances_mean = perm_res.importances_mean
        non_neg_imp = np.maximum(0.0, importances_mean)
        total_imp = float(np.sum(non_neg_imp)) + 1e-12

        items: list[GlobalFeatureImportance] = []
        for i, col in enumerate(feature_names):
            raw_imp = float(importances_mean[i])
            norm_wt = float((non_neg_imp[i] / total_imp) * 100.0)
            items.append(
                GlobalFeatureImportance(
                    name=col,
                    importance=raw_imp,
                    normalized_weight=norm_wt,
                )
            )

        items.sort(key=lambda x: x.importance, reverse=True)

        meta = {
            "model_id": getattr(artifact, "model_id", art_id),
            "model_version": getattr(artifact, "version", "v2.0"),
            "disease": dis_name,
            "pipeline_version": getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            "evaluation_context": f"{len(y_test)}-sample held-out test split (80/20 stratified, seed {seed})",
            "random_seed": seed,
            "scoring_metric": "roc_auc",
            "sample_count": len(y_test),
            "repeats": 5,
        }

        limits = [
            f"Permutation importance measures empirical decrease in ROC-AUC when a feature is shuffled across the {len(y_test)}-sample holdout test partition.",
            "RBF kernel maps inputs into a non-linear Hilbert space; permutation importance provides a model-agnostic global ranking.",
        ]

        result = GlobalExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            method="permutation_importance",
            status="available",
            features=items,
            limitations=limits,
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            metadata=meta,
        )

        self._cached_global[art_id] = result

        if top_k is not None and top_k > 0:
            return GlobalExplanationResult(
                model_name=result.model_name,
                model_family=result.model_family,
                method=result.method,
                status=result.status,
                features=items[:top_k],
                limitations=result.limitations,
                pipeline_version=result.pipeline_version,
                metadata=result.metadata,
            )

        return result


class EnsembleExplainer(BaseExplainer):
    """Model-agnostic explainer for Voting Ensembles, Stacking Classifiers, and KNN models."""

    def __init__(self) -> None:
        self._cached_global: dict[str, GlobalExplanationResult] = {}

    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()

        baseline_dict = _get_baseline_features(preprocessor, feature_names)
        baseline_df = pd.DataFrame([baseline_dict])

        # Check if probability predictions are supported
        has_proba = hasattr(artifact, "predict_proba") or hasattr(artifact.model, "predict_proba")
        if not has_proba:
            # For Hard Voting without predict_proba: return graceful status
            return ExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                explanation_type="local",
                method="marginal_feature_perturbation",
                status="unsupported",
                baseline_value=0.0,
                features=[],
                limitations=["Hard voting classifier does not output continuous probabilities for marginal delta attributions."],
            )

        p_base = float(artifact.predict_proba(baseline_df)[0, 1])
        p_full = float(artifact.predict_proba(raw_df)[0, 1])

        contributions = []
        for col in feature_names:
            perturbed_df = raw_df.copy()
            perturbed_df[col] = baseline_dict.get(col, 0.0)
            p_without_i = float(artifact.predict_proba(perturbed_df)[0, 1])
            delta_p = p_full - p_without_i
            contributions.append(delta_p)

        contributions_arr = np.array(contributions)
        total_abs = float(np.sum(np.abs(contributions_arr))) + 1e-12

        feature_items: list[FeatureContribution] = []
        for i, col in enumerate(feature_names):
            c_val = float(contributions_arr[i])
            raw_val = float(raw_df[col].iloc[0]) if col in raw_df.columns else 0.0
            norm_wt = (abs(c_val) / total_abs) * 100.0
            direction = "positive" if c_val > 1e-5 else ("negative" if c_val < -1e-5 else "neutral")
            feature_items.append(
                FeatureContribution(
                    name=col,
                    value=raw_val,
                    contribution=c_val,
                    direction=direction,
                    normalized_weight=norm_wt,
                )
            )

        feature_items.sort(key=lambda x: abs(x.contribution), reverse=True)
        if top_k is not None and top_k > 0:
            feature_items = feature_items[:top_k]

        return ExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            explanation_type="local",
            method="marginal_feature_perturbation",
            status="available",
            baseline_value=p_base,
            features=feature_items,
            limitations=[
                "Model-agnostic marginal perturbation evaluates ensemble output sensitivity relative to population baseline.",
                "Interactions among heterogeneous base estimators are captured conditioned on surrounding features.",
            ],
        )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
        reference_data: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> GlobalExplanationResult:
        art_id = getattr(artifact, "artifact_id", getattr(artifact, "model_id", "ensemble_model"))
        if art_id in self._cached_global and reference_data is None:
            cached_res = self._cached_global[art_id]
            if top_k is not None and top_k > 0:
                return GlobalExplanationResult(
                    model_name=cached_res.model_name,
                    model_family=cached_res.model_family,
                    method=cached_res.method,
                    status=cached_res.status,
                    features=cached_res.features[:top_k],
                    limitations=cached_res.limitations,
                    pipeline_version=cached_res.pipeline_version,
                    metadata=cached_res.metadata,
                )
            return cached_res

        preprocessor = artifact.preprocessor
        feature_names = preprocessor.get_feature_names()
        seed = getattr(artifact, "random_seed", 42) or 42

        X_test_raw = None
        y_test = None
        dis_name = "diabetes"
        if reference_data is not None:
            X_test_raw, y_test = reference_data
        else:
            X_test_raw, y_test, dis_name = _load_reference_dataset(artifact)

        if X_test_raw is None or y_test is None:
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method="permutation_importance",
                status="unavailable",
                features=[],
                limitations=["Held-out reference dataset is required for ensemble permutation importance."],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )

        X_test_trans = preprocessor.transform(X_test_raw)
        estimator = getattr(artifact.model, "estimator_", artifact.model)

        scoring_metric = "roc_auc"
        if "hard_voting" in getattr(artifact, "model_type", "").lower() or "hard_voting" in art_id:
            scoring_metric = "accuracy"

        try:
            perm_res = permutation_importance(
                estimator,
                X_test_trans,
                y_test,
                scoring=scoring_metric,
                n_repeats=5,
                random_state=seed,
            )
            importances_mean = perm_res.importances_mean
        except Exception as e:
            logger.warning(f"Permutation importance failed for ensemble {art_id}: {e}")
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="classical",
                method="permutation_importance",
                status="unsupported",
                features=[],
                limitations=[f"Permutation importance calculation failed: {e!s}"],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )

        non_neg_imp = np.maximum(0.0, importances_mean)
        total_imp = float(np.sum(non_neg_imp)) + 1e-12

        items: list[GlobalFeatureImportance] = []
        for i, col in enumerate(feature_names):
            raw_imp = float(importances_mean[i])
            norm_wt = float((non_neg_imp[i] / total_imp) * 100.0)
            items.append(
                GlobalFeatureImportance(
                    name=col,
                    importance=raw_imp,
                    normalized_weight=norm_wt,
                )
            )

        items.sort(key=lambda x: x.importance, reverse=True)

        meta = {
            "model_id": getattr(artifact, "model_id", art_id),
            "disease": dis_name,
            "scoring_metric": scoring_metric,
            "sample_count": len(y_test),
        }

        result = GlobalExplanationResult(
            model_name=artifact.model_name,
            model_family="classical",
            method="permutation_importance",
            status="available",
            features=items,
            limitations=[
                f"Global permutation importance calculated on {len(y_test)}-sample holdout test partition.",
                "Measures empirical evaluation metric degradation when individual features are randomly permuted.",
            ],
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            metadata=meta,
        )

        self._cached_global[art_id] = result

        if top_k is not None and top_k > 0:
            return GlobalExplanationResult(
                model_name=result.model_name,
                model_family=result.model_family,
                method=result.method,
                status=result.status,
                features=items[:top_k],
                limitations=result.limitations,
                pipeline_version=result.pipeline_version,
                metadata=result.metadata,
            )

        return result
