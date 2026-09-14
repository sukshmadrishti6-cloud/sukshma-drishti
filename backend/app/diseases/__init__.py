"""Disease domain package for SukshmaDrishti Multi-Disease Architecture."""
from app.diseases.base import BaseDiseaseAdapter
from app.diseases.registry import disease_registry, DiseaseRegistry

__all__ = ["BaseDiseaseAdapter", "disease_registry", "DiseaseRegistry"]
