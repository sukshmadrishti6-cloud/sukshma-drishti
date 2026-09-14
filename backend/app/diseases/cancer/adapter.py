"""Concrete Disease Adapter for Cancer Risk Inference (UCI Breast Cancer Wisconsin Dataset)."""
from fastapi import HTTPException, status
from app.diseases.base import BaseDiseaseAdapter
from app.schemas.api import CancerInput, DiseaseDetail, InferenceRequest, InferenceResponse
from app.services.ml_service import ml_service
from ml.cancer.pipeline import CANCER_FEATURE_NAMES


class CancerAdapter(BaseDiseaseAdapter):
    """Adapter wrapping Cancer Risk ML inference pipeline (UCI Breast Cancer Wisconsin Dataset)."""

    @property
    def disease_id(self) -> str:
        return "cancer"

    @property
    def display_name(self) -> str:
        return "Breast Cancer Risk"

    @property
    def status(self) -> str:
        return "available"

    @property
    def description(self) -> str:
        return "Assessment of breast tumor malignancy risk using 30 digitized cell nucleus morphometric indicators."

    @property
    def category(self) -> str:
        return "oncology"

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Routes cancer inference request to MLService with strict schema and disease boundary validation."""
        request.disease = "cancer"

        # Validate input features against CancerInput schema
        if isinstance(request.features, dict):
            try:
                CancerInput(**request.features)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cancer feature validation failed: {e!s}",
                )
        elif isinstance(request.features, list):
            if len(request.features) != len(CANCER_FEATURE_NAMES):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Feature list must contain exactly {len(CANCER_FEATURE_NAMES)} values (got {len(request.features)}).",
                )
            feat_dict = dict(zip(CANCER_FEATURE_NAMES, request.features))
            try:
                CancerInput(**feat_dict)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cancer feature validation failed: {e!s}",
                )

        if not request.model_id or request.model_id.lower() in ["default", "production", "optimized"]:
            request.model_id = "cancer_stacking_opt_v1"

        response = ml_service.predict(request)
        response.disease = "cancer"
        return response

    def get_metadata(self) -> DiseaseDetail:
        """Returns discovery metadata and model summary for Cancer disease status."""
        models_summary = ml_service.get_registered_models_summary(disease="cancer")
        return DiseaseDetail(
            id=self.disease_id,
            name=self.display_name,
            status=self.status,
            description=self.description,
            category=self.category,
            version=self.version,
            supported_models=models_summary,
            input_features_count=30,
            feature_names=list(CANCER_FEATURE_NAMES),
        )
