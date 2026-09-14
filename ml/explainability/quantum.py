"""Quantum Model Interpretability and Sensitivity Analysis (QSVC & VQC)."""
from typing import Any

import numpy as np
import pandas as pd

from ml.explainability.base import (
    BaseExplainer,
    ExplanationResult,
    FeatureContribution,
    GlobalExplanationResult,
    GlobalFeatureImportance,
)


class QuantumModelExplainer(BaseExplainer):
    """Quantum model explainer providing circuit metadata and quantum feature sensitivity analysis."""

    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        preprocessor = artifact.preprocessor
        reducer = artifact.quantum_reducer
        feature_names = preprocessor.get_feature_names()

        # Extract quantum circuit metadata from artifact
        circuit_meta: dict[str, Any] = dict(artifact.metadata or {})
        circuit_meta.update(
            {
                "qubit_count": getattr(artifact.model, "num_qubits", 4),
                "model_family": "quantum",
                "quantum_backend": "AerSimulator",
                "feature_map": circuit_meta.get("feature_map_name", getattr(artifact.model, "feature_map_name", "ZZFeatureMap")),
                "ansatz": circuit_meta.get("ansatz_name", getattr(artifact.model, "ansatz_name", None)),
                "circuit_depth": circuit_meta.get("circuit_depth", None),
                "parameter_count": circuit_meta.get("parameter_count", None),
            }
        )

        # 1. Base prediction probability
        p_base = 0.5
        try:
            p_base = float(artifact.predict_proba(raw_df)[0, 1])
        except Exception:
            pass

        # 2. Local sensitivity on quantum angle-encoded features
        # Clean & reduce raw sample
        X_clean = preprocessor.transform(raw_df)
        X_reduced = reducer.transform(X_clean) if reducer is not None else X_clean
        n_components = X_reduced.shape[1]

        comp_sensitivities = []
        delta = 0.1  # small angle perturbation in [-pi, pi] space

        for j in range(n_components):
            X_plus = X_reduced.copy()
            X_plus[0, j] += delta

            X_minus = X_reduced.copy()
            X_minus[0, j] -= delta

            try:
                p_p = float(artifact.model.predict_proba(X_plus)[0, 1])
                p_m = float(artifact.model.predict_proba(X_minus)[0, 1])
                grad_j = (p_p - p_m) / (2.0 * delta)
            except Exception:
                # If predict_proba is not directly available, evaluate discrete output shift
                y_p = float(artifact.model.predict(X_plus)[0])
                y_m = float(artifact.model.predict(X_minus)[0])
                grad_j = (y_p - y_m) / (2.0 * delta)
            comp_sensitivities.append(grad_j)

        comp_sens_arr = np.array(comp_sensitivities)

        # 3. Project component sensitivities back to classical clinical variables
        if reducer is not None and hasattr(reducer, "selector_") and reducer.selector_ is not None:
            comp_scores = getattr(reducer.selector_, "scores_", None)
            if comp_scores is not None:
                # Use SelectKBest scores for pseudo-projection
                scores_arr = np.array(comp_scores)
                scores_arr = scores_arr / (np.sum(scores_arr) + 1e-12)
                # Map back from qubits to features
                feature_sens = np.zeros(len(feature_names))
                q_count = getattr(reducer, "n_qubits", len(comp_sens_arr))
                for q in range(min(len(comp_sens_arr), q_count)):
                    for i in range(min(len(feature_names), len(scores_arr))):
                        feature_sens[i] += (comp_sens_arr[q] * scores_arr[i])
            else:
                feature_sens = np.zeros(len(feature_names))
        else:
            feature_sens = np.zeros(len(feature_names))

        total_abs = float(np.sum(np.abs(feature_sens))) + 1e-12

        feature_items: list[FeatureContribution] = []
        for i, col in enumerate(feature_names):
            s_val = float(feature_sens[i])
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
            model_family="quantum",
            explanation_type="local",
            method="quantum_feature_sensitivity_projection",
            status="partial",
            baseline_value=p_base,
            features=feature_items,
            quantum_metadata=circuit_meta,
            limitations=[
                "Quantum models evaluate data in non-linear entangled Hilbert space; classical feature attribution is an approximation.",
                "Feature sensitivities reflect back-projection from selected quantum angle-encoded features to classical variables using mutual information weights.",
                "Attributions represent model sensitivity to input perturbation, NOT medical causality.",
            ],
        )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
    ) -> GlobalExplanationResult:
        preprocessor = artifact.preprocessor
        reducer = artifact.quantum_reducer
        feature_names = preprocessor.get_feature_names()

        scores = None
        if reducer is not None and hasattr(reducer, "selector_") and reducer.selector_ is not None:
            scores = getattr(reducer.selector_, "scores_", None)

        if scores is None:
            return GlobalExplanationResult(
                model_name=artifact.model_name,
                model_family="quantum",
                method="quantum_feature_reduction_selection",
                status="unsupported",
                features=[],
                limitations=[
                    "Quantum kernel and variational models do not directly provide linear global feature importances.",
                    "Global feature reduction selection scores are not present on this artifact.",
                ],
                pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
            )

        scores_arr = np.nan_to_num(np.array(scores), nan=0.0)
        total_score = float(np.sum(scores_arr)) + 1e-12

        items: list[GlobalFeatureImportance] = []
        for i, col in enumerate(feature_names):
            s_val = float(scores_arr[i]) if i < len(scores_arr) else 0.0
            norm_wt = (s_val / total_score) * 100.0
            items.append(
                GlobalFeatureImportance(
                    name=col,
                    importance=s_val,
                    normalized_weight=norm_wt,
                )
            )

        items.sort(key=lambda x: x.importance, reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        return GlobalExplanationResult(
            model_name=artifact.model_name,
            model_family="quantum",
            method="quantum_mutual_info_ranking",
            status="partial",
            features=items,
            limitations=[
                "Global ranking reflects mutual information scores used to select the top features for quantum angle encoding.",
                "Entanglement and kernel state dynamics vary per instance and are captured through local sensitivity analysis.",
            ],
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v2.0") if hasattr(artifact, "manifest") else "v2.0",
        )

