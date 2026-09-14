"""Explanation Service orchestrating model-specific explainer dispatching."""
from typing import Any

import pandas as pd

from ml.explainability.base import BaseExplainer, ExplanationResult, GlobalExplanationResult
from ml.explainability.classical import (
    EnsembleExplainer,
    LogisticRegressionExplainer,
    SVMRBFExplainer,
    TreeEnsembleExplainer,
)
from ml.explainability.quantum import QuantumModelExplainer
from ml.logger import get_logger

logger = get_logger(__name__)


class UnsupportedExplainer(BaseExplainer):
    """Fallback explainer for models that do not support feature decomposition."""

    def explain_instance(self, artifact: Any, raw_df: pd.DataFrame, top_k: int | None = None) -> ExplanationResult:
        m_type = getattr(artifact, "model_type", "unknown")
        return ExplanationResult(
            model_name=getattr(artifact, "model_name", "Unknown"),
            model_family=getattr(artifact, "model_family", "classical"),
            explanation_type="local",
            method="unsupported",
            status="unsupported",
            features=[],
            limitations=[f"Local instance attribution is not mathematically supported for model architecture type '{m_type}'."],
        )

    def explain_global(self, artifact: Any, top_k: int | None = None) -> GlobalExplanationResult:
        m_type = getattr(artifact, "model_type", "unknown")
        return GlobalExplanationResult(
            model_name=getattr(artifact, "model_name", "Unknown"),
            model_family=getattr(artifact, "model_family", "classical"),
            method="unsupported",
            status="unsupported",
            features=[],
            limitations=[f"Global feature importance is mathematically unsupported for model architecture type '{m_type}'."],
            pipeline_version=getattr(artifact.manifest, "pipeline_version", "v1.0") if hasattr(artifact, "manifest") else "v1.0",
        )


class ExplanationService:
    """Orchestrates model-specific explainers across classical and quantum architectures."""

    def __init__(self):
        self._lr_explainer = LogisticRegressionExplainer()
        self._tree_explainer = TreeEnsembleExplainer()
        self._svm_explainer = SVMRBFExplainer()
        self._ensemble_explainer = EnsembleExplainer()
        self._quantum_explainer = QuantumModelExplainer()
        self._unsupported_explainer = UnsupportedExplainer()

    def get_explainer_for_artifact(self, artifact: Any) -> BaseExplainer:
        """Selects the mathematically appropriate explainer for the artifact model type."""
        m_type = getattr(artifact, "model_type", "").lower()
        m_family = getattr(artifact, "model_family", "").lower()

        if m_family == "quantum" or m_type in ["qsvc", "vqc"]:
            return self._quantum_explainer
        elif m_type in ["logistic_regression", "lr", "logistic", "linear_svm"]:
            return self._lr_explainer
        elif m_type in [
            "random_forest", "rf", "xgboost", "xgb",
            "extra_trees", "decision_tree", "gradient_boosting",
            "hist_gradient_boosting"
        ]:
            return self._tree_explainer
        elif m_type in ["svm", "svc", "rbf_svm", "support_vector_machine"]:
            return self._svm_explainer
        elif m_type in ["knn", "soft_voting", "hard_voting", "stacking"]:
            return self._ensemble_explainer
        else:
            return self._unsupported_explainer

    def explain(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = 8,
    ) -> ExplanationResult:
        """Generates local explanation for the given model artifact and raw input features."""
        explainer = self.get_explainer_for_artifact(artifact)
        try:
            return explainer.explain_instance(artifact, raw_df, top_k=top_k)
        except Exception as e:
            logger.error(f"Explanation generation failed for '{artifact.model_name}': {e!s}", exc_info=True)
            return ExplanationResult(
                model_name=getattr(artifact, "model_name", "Unknown"),
                model_family=getattr(artifact, "model_family", "unknown"),
                explanation_type="local",
                method="unavailable",
                status="unsupported",
                features=[],
                limitations=[f"Explanation generation encountered an internal error: {e!s}"],
            )

    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
    ) -> GlobalExplanationResult:
        """Generates population-level global feature importance for the model."""
        explainer = self.get_explainer_for_artifact(artifact)
        try:
            return explainer.explain_global(artifact, top_k=top_k)
        except Exception as e:
            logger.error(f"Global explanation generation failed for '{artifact.model_name}': {e!s}", exc_info=True)
            return GlobalExplanationResult(
                model_name=getattr(artifact, "model_name", "Unknown"),
                model_family=getattr(artifact, "model_family", "unknown"),
                method="unavailable",
                status="unsupported",
                features=[],
                limitations=[f"Global explanation generation encountered an internal error: {e!s}"],
            )


# Singleton explanation service instance
explanation_service = ExplanationService()
