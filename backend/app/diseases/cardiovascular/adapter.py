"""Concrete Disease Adapter for Cardiovascular Risk Inference."""
from fastapi import HTTPException, status
from app.diseases.base import BaseDiseaseAdapter
from app.schemas.api import CardiovascularInput, DiseaseDetail, InferenceRequest, InferenceResponse
from app.services.ml_service import ml_service
from ml.cardiovascular.pipeline import CARDIO_FEATURE_NAMES


class CardiovascularAdapter(BaseDiseaseAdapter):
    """Adapter wrapping Cardiovascular Risk ML inference pipeline (UCI Cleveland Dataset)."""

    @property
    def disease_id(self) -> str:
        return "cardiovascular"

    @property
    def display_name(self) -> str:
        return "Cardiovascular Risk"

    @property
    def status(self) -> str:
        return "available"

    @property
    def description(self) -> str:
        return "Assessment of 10-year ischemic heart disease and major adverse cardiovascular event (MACE) risk using 13 clinical indicators."

    @property
    def category(self) -> str:
        return "cardiovascular"

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Routes cardiovascular inference request to MLService with strict schema and disease boundary validation."""
        request.disease = "cardiovascular"

        # Validate input features against CardiovascularInput schema
        if isinstance(request.features, dict):
            try:
                CardiovascularInput(**request.features)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cardiovascular feature validation failed: {e!s}",
                )
        elif isinstance(request.features, list):
            if len(request.features) != len(CARDIO_FEATURE_NAMES):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Feature list must contain exactly {len(CARDIO_FEATURE_NAMES)} values (got {len(request.features)}).",
                )
            feat_dict = dict(zip(CARDIO_FEATURE_NAMES, request.features))
            try:
                CardiovascularInput(**feat_dict)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cardiovascular feature validation failed: {e!s}",
                )

        if not request.model_id or request.model_id.lower() in ["default", "production", "optimized"]:
            request.model_id = "cardiovascular_extra_trees_opt_v1"

        response = ml_service.predict(request)
        response.disease = "cardiovascular"
        return response

    def get_metadata(self) -> DiseaseDetail:
        """Returns discovery metadata and model summary for Cardiovascular disease status."""
        models_summary = ml_service.get_registered_models_summary(disease="cardiovascular")
        return DiseaseDetail(
            id=self.disease_id,
            name=self.display_name,
            status=self.status,
            description=self.description,
            category=self.category,
            version=self.version,
            supported_models=models_summary,
            input_features_count=13,
            feature_names=list(CARDIO_FEATURE_NAMES),
        )
