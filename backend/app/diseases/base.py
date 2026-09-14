"""Abstract base class interface for disease-specific inference adapters."""
from abc import ABC, abstractmethod
from typing import Any
from app.schemas.api import InferenceRequest, InferenceResponse, DiseaseDetail


class BaseDiseaseAdapter(ABC):
    """Abstract interface defining required contract for disease adapters."""

    @property
    @abstractmethod
    def disease_id(self) -> str:
        """Canonical disease identifier (e.g. 'diabetes', 'cardiovascular', 'cancer')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human readable disease title."""
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        """Availability status: 'available' or 'coming_soon'."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """User-facing description of clinical scope."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Disease category (e.g. 'metabolic', 'cardiovascular', 'oncology')."""
        pass

    @property
    def version(self) -> str:
        """Adapter schema version."""
        return "v1.0"

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Executes disease-specific inference pipeline."""
        pass

    @abstractmethod
    def get_metadata(self) -> DiseaseDetail:
        """Returns detailed metadata including supported models and feature names."""
        pass
