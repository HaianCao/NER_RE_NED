# -*- coding: utf-8 -*-
"""Shared data schemas for the NER + RE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EntitySpan:
    """A named entity extracted from a document."""

    label: str
    """Entity type label, e.g. ``"Problem"``, ``"Treatment"``, ``"Test"``."""

    term: str
    """Surface form of the entity as it appears in the text."""

    start_token_idx: int
    """0-based index of the first whitespace-split token (inclusive)."""

    end_token_idx: int
    """0-based index of the last whitespace-split token (inclusive)."""

    entity_id: str
    """Unique identifier within the document, e.g. ``"T0"``."""


@dataclass
class RelationTriple:
    """A directed relation between two entities."""

    relation_type: str
    """Relation label, e.g. ``"TrAP"``, ``"PIP"``."""

    subject_id: str
    """``entity_id`` of the subject (head) entity."""

    object_id: str
    """``entity_id`` of the object (tail) entity."""


@dataclass
class StandardizedDocument:
    """A normalised document ready for prompt construction."""

    id: str
    """Unique document identifier."""

    text: str
    """Original sentence string."""

    words: List[str]
    """Whitespace-split tokens of ``text``."""

    entities: List[EntitySpan]
    """All labelled entities in the document."""

    relations: List[RelationTriple]
    """All annotated relations in the document."""

    language: str = "en"
    """ISO language code: ``"en"`` or ``"vi"``."""

    def get_entity_by_id(self, entity_id: str) -> EntitySpan | None:
        """Look up an entity by its unique ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
