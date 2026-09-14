"""Concrete Disease Adapter for Diabetes Risk Inference."""
from app.diseases.base import BaseDiseaseAdapter
from app.schemas.api import DiseaseDetail, InferenceRequest, InferenceResponse
from app.services.ml_service import ml_service
from ml.preprocessing import RAW_FEATURE_NAMES


class DiabetesAdapter(BaseDiseaseAdapter):
    """Adapter wrapping authoritative Diabetes ML and Quantum ML inference pipeline."""

    @property
    def disease_id(self) -> str:
        return "diabetes"

    @property
    def display_name(self) -> str:
        return "Diabetes Mellitus Risk"

    @property
    def status(self) -> str:
        return "available"

    @property
    def description(self) -> str:
        return "Early detection of Type-2 Diabetes Mellitus risk using 8 biomedical clinical indicators and hybrid classical-quantum models."

    @property
    def category(self) -> str:
        return "metabolic"

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Routes inference request to MLService with strict disease boundary validation."""
        request.disease = "diabetes"
        if not request.model_id or request.model_id.lower() in ["default", "production", "optimized"]:
            request.model_id = "diabetes_rf_opt_v1"

        response = ml_service.predict(request)
        response.disease = "diabetes"
        return response

    def get_metadata(self) -> DiseaseDetail:
        """Returns metadata summary for Diabetes prediction models."""
        models_summary = ml_service.get_registered_models_summary(disease="diabetes")
        return DiseaseDetail(
            id=self.disease_id,
            name=self.display_name,
            status=self.status,
            description=self.description,
            category=self.category,
            version=self.version,
            supported_models=models_summary,
            input_features_count=8,
            feature_names=list(RAW_FEATURE_NAMES),
        )
