# -*- coding: utf-8 -*-
"""Shared helpers for building prompt text content.

These functions are purely presentational and have no side effects,
making them easy to unit-test and reuse across Causal and Instruction
builders.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from config import LabelConfig, REMode, TaskMode
from data.schemas import EntitySpan, RelationTriple, StandardizedDocument


# ---------------------------------------------------------------------------
# Entity / relation block formatters
# ---------------------------------------------------------------------------


def format_entity_definitions(entity_labels: Dict[str, str]) -> str:
    """Render entity label definitions as a bullet list."""
    lines = [
        f"- {label}: {desc}" for label, desc in entity_labels.items()
    ]
    return "\n".join(lines)


def format_relation_definitions(relation_labels: Dict[str, str]) -> str:
    """Render relation label definitions as a bullet list."""
    lines = []
    for label, desc in relation_labels.items():
        lines.append(f"- {label}: {desc}")
    lines.append("- NONE: No valid clinical relation exists between the entities")
    return "\n".join(lines)


def format_entity_list(entities: List[EntitySpan]) -> str:
    """Render a compact entity list for GLOBAL RE input prompts."""
    return "\n".join(
        f"- {e.entity_id} ({e.label}): \"{e.term}\"" for e in entities
    )


# ---------------------------------------------------------------------------
# JSON output builders
# ---------------------------------------------------------------------------


def build_ner_output(entities: List[EntitySpan]) -> str:
    """Build the ground-truth JSON string for NER output."""
    return json.dumps(
        {
            "entities": [
                {"id": e.entity_id, "label": e.label, "term": e.term}
                for e in entities
            ]
        },
        ensure_ascii=False,
    )


def build_re_pairwise_output(relation_type: str) -> str:
    """Build the ground-truth JSON string for a single pairwise relation."""
    return json.dumps({"relation": relation_type}, ensure_ascii=False)


def build_re_global_output(relations: List[RelationTriple]) -> str:
    """Build the ground-truth JSON string for global relation extraction."""
    return json.dumps(
        {
            "relations": [
                {
                    "label": r.relation_type,
                    "subject_id": r.subject_id,
                    "object_id": r.object_id,
                }
                for r in relations
            ]
        },
        ensure_ascii=False,
    )


def build_joint_output(
    entities: List[EntitySpan],
    relations: List[RelationTriple],
) -> str:
    """Build the ground-truth JSON for joint NER + RE output."""
    return json.dumps(
        {
            "entities": [
                {"id": e.entity_id, "label": e.label, "term": e.term}
                for e in entities
            ],
            "relations": [
                {
                    "label": r.relation_type,
                    "subject_id": r.subject_id,
                    "object_id": r.object_id,
                }
                for r in relations
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Entity marker insertion (for PAIRWISE RE)
# ---------------------------------------------------------------------------


def insert_entity_markers(
    words: List[str],
    subj: EntitySpan,
    obj: EntitySpan,
) -> str:
    """Wrap subject with ``<e1>…</e1>`` and object with ``<e2>…</e2>``.

    Processes spans from right to left to avoid index drift after
    each insertion.
    """
    new_words: List[str] = list(words)
    spans: List[Tuple[int, int, str, str]] = [
        (subj.start_token_idx, subj.end_token_idx, "<e1>", "</e1>"),
        (obj.start_token_idx, obj.end_token_idx, "<e2>", "</e2>"),
    ]
    # Right-to-left order prevents earlier insertions shifting later indices
    spans.sort(key=lambda s: s[0], reverse=True)
    for start, end, open_tag, close_tag in spans:
        if start < 0 or end < 0:
            continue
        new_words.insert(end + 1, close_tag)
        new_words.insert(start, open_tag)
    return " ".join(new_words)


# ---------------------------------------------------------------------------
# Relation lookup
# ---------------------------------------------------------------------------


def get_pairwise_relation(
    doc: StandardizedDocument,
    subj: EntitySpan,
    obj: EntitySpan,
) -> str:
    """Return the relation label for (subj, obj) or ``"NONE"``."""
    for rel in doc.relations:
        if rel.subject_id == subj.entity_id and rel.object_id == obj.entity_id:
            return rel.relation_type
    return "NONE"


def filter_valid_pairs(
    entities: List[EntitySpan],
    valid_pair_set,
) -> List[Tuple[EntitySpan, EntitySpan]]:
    """Return all (subject, object) entity pairs allowed to have a relation."""
    pairs: List[Tuple[EntitySpan, EntitySpan]] = []
    for i, subj in enumerate(entities):
        for j, obj in enumerate(entities):
            if i == j:
                continue
            if (subj.label, obj.label) in valid_pair_set:
                pairs.append((subj, obj))
    return pairs
