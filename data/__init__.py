"""Data sub-package: schemas, adapter, and registry."""

from .adapter import ClinicalFormatAdapter
from .registry import DatasetRegistry
from .schemas import EntitySpan, RelationTriple, StandardizedDocument

__all__ = [
    "ClinicalFormatAdapter",
    "DatasetRegistry",
    "EntitySpan",
    "RelationTriple",
    "StandardizedDocument",
]
