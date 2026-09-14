"""Base dataclasses and interfaces for Model Explainability and Responsible AI."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class FeatureContribution:
    """Individual feature contribution attribution."""

    name: str
    value: float
    contribution: float
    direction: str  # 'positive' (pushes toward higher disease risk) | 'negative' (pushes toward lower risk) | 'neutral'
    normalized_weight: float  # Absolute percentage importance (0.0 to 100.0)


@dataclass
class ExplanationResult:
    """Standardized explanation payload for classical and quantum models."""

    model_name: str
    model_family: str  # 'classical' | 'quantum'
    explanation_type: str  # 'local' | 'global'
    method: str  # e.g. 'linear_additive_attribution', 'marginal_leave_one_feature_out', 'quantum_component_sensitivity'
    status: str  # 'available' | 'partial' | 'unsupported'
    features: list[FeatureContribution] = field(default_factory=list)
    baseline_value: float | None = None
    quantum_metadata: dict[str, Any] | None = None
    limitations: list[str] = field(default_factory=list)
    responsible_ai_disclaimer: str = (
        "Responsible AI Notice: Feature contributions describe model decision behavior and feature sensitivity. "
        "They represent mathematical associations within the trained model and do NOT establish medical causality, "
        "clinical diagnosis, or treatment recommendations. Always consult a qualified healthcare professional."
    )

    def to_dict(self) -> dict[str, Any]:
        """Converts explanation result into clean serializable dictionary."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "explanation_type": self.explanation_type,
            "method": self.method,
            "status": self.status,
            "baseline_value": self.baseline_value,
            "features": [
                {
                    "name": f.name,
                    "value": round(f.value, 4),
                    "contribution": round(f.contribution, 4),
                    "direction": f.direction,
                    "normalized_weight": round(f.normalized_weight, 2),
                }
                for f in self.features
            ],
            "quantum_metadata": self.quantum_metadata,
            "limitations": self.limitations,
            "responsible_ai_disclaimer": self.responsible_ai_disclaimer,
        }


@dataclass
class GlobalFeatureImportance:
    """Global feature importance metric."""

    name: str
    importance: float
    normalized_weight: float  # Percentage (0.0 to 100.0)


@dataclass
class GlobalExplanationResult:
    """Standardized global explanation payload for model architectures."""

    model_name: str
    model_family: str
    method: str
    status: str  # 'available' | 'partial' | 'unsupported'
    features: list[GlobalFeatureImportance] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    pipeline_version: str = "v2.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "method": self.method,
            "status": self.status,
            "features": [
                {
                    "name": f.name,
                    "importance": round(f.importance, 4),
                    "normalized_weight": round(f.normalized_weight, 2),
                }
                for f in self.features
            ],
            "limitations": self.limitations,
            "pipeline_version": self.pipeline_version,
            "metadata": self.metadata,
        }


class BaseExplainer(ABC):
    """Abstract base class for model explainers."""

    @abstractmethod
    def explain_instance(
        self,
        artifact: Any,
        raw_df: pd.DataFrame,
        top_k: int | None = None,
    ) -> ExplanationResult:
        """Generates local feature attribution for a single input record."""

    @abstractmethod
    def explain_global(
        self,
        artifact: Any,
        top_k: int | None = None,
    ) -> GlobalExplanationResult:
        """Generates population-level global feature importance for the model architecture."""

