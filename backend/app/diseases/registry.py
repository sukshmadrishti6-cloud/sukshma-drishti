"""Central Disease Registry and Resolution Service for Multi-Disease Architecture."""
import logging
from typing import Any
from fastapi import HTTPException, status

from app.diseases.base import BaseDiseaseAdapter
from app.diseases.diabetes.adapter import DiabetesAdapter
from app.diseases.cardiovascular.adapter import CardiovascularAdapter
from app.diseases.cancer.adapter import CancerAdapter
from app.schemas.api import DiseaseDetail, DiseaseSummary

logger = logging.getLogger(__name__)

# Controlled disease alias resolution mapping
DISEASE_ALIAS_MAP: dict[str, str] = {
    "diabetes": "diabetes",
    "diabetes_mellitus": "diabetes",
    "dm": "diabetes",
    "cardiovascular": "cardiovascular",
    "cardio": "cardiovascular",
    "heart": "cardiovascular",
    "heart_disease": "cardiovascular",
    "cancer": "cancer",
    "oncology": "cancer",
}


class DiseaseRegistry:
    """Authoritative registry managing registered disease adapters, aliases, and availability."""

    def __init__(self):
        self._adapters: dict[str, BaseDiseaseAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Registers default platform disease adapters."""
        self.register_adapter(DiabetesAdapter())
        self.register_adapter(CardiovascularAdapter())
        self.register_adapter(CancerAdapter())

    def register_adapter(self, adapter: BaseDiseaseAdapter) -> None:
        """Registers a disease adapter instance."""
        disease_id = adapter.disease_id.lower()
        self._adapters[disease_id] = adapter
        logger.info(f"Registered disease adapter '{disease_id}' (status: {adapter.status})")

    def resolve_disease_id(self, raw_identifier: str) -> str:
        """Resolves raw disease string or alias to canonical disease ID.

        Security: Rejects path traversal sequences, invalid characters, or unregistered diseases.
        """
        clean_id = raw_identifier.strip()
        key = clean_id.lower()

        if ".." in clean_id or "/" in clean_id or "\\" in clean_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security violation: Malicious path traversal character detected in disease identifier '{raw_identifier}'.",
            )

        canonical_id = DISEASE_ALIAS_MAP.get(key)
        if not canonical_id or canonical_id not in self._adapters:
            supported = sorted(list(self._adapters.keys()))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Disease '{raw_identifier}' not found in registry. Supported diseases: {supported}",
            )
        return canonical_id

    def get_adapter(self, disease_identifier: str) -> BaseDiseaseAdapter:
        """Resolves disease identifier and returns registered adapter instance."""
        canonical_id = self.resolve_disease_id(disease_identifier)
        return self._adapters[canonical_id]

    def is_available(self, disease_identifier: str) -> bool:
        """Checks if disease model inference is currently available."""
        try:
            adapter = self.get_adapter(disease_identifier)
            return adapter.status == "available"
        except HTTPException:
            return False

    def list_diseases(self) -> list[DiseaseSummary]:
        """Returns summary list of all registered diseases and their deployment status."""
        summaries = []
        for adapter in self._adapters.values():
            summaries.append(
                DiseaseSummary(
                    id=adapter.disease_id,
                    name=adapter.display_name,
                    status=adapter.status,
                    description=adapter.description,
                    category=adapter.category,
                    version=adapter.version,
                )
            )
        return summaries

    def get_disease_detail(self, disease_identifier: str) -> DiseaseDetail:
        """Returns complete metadata details for a registered disease."""
        adapter = self.get_adapter(disease_identifier)
        return adapter.get_metadata()


# Singleton instance
disease_registry = DiseaseRegistry()
