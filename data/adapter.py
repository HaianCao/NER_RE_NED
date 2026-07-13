# -*- coding: utf-8 -*-
"""Adapter that parses clinical JSONL records into StandardizedDocuments.

Follows the Single Responsibility Principle: this class only handles
parsing, never prompt construction or training logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .schemas import EntitySpan, RelationTriple, StandardizedDocument

logger = logging.getLogger(__name__)

# Mapping from language code → (sentence_key, term_key, start_key, end_key)
_LANG_KEY_MAP: Dict[str, Tuple[str, str, str, str]] = {
    "en": (
        "en_sentence_str",
        "en_term",
        "en_start_token_idx",
        "en_end_token_idx",
    ),
    "vi": (
        "vi_sentence_str",
        "vi_term",
        "vi_start_token_idx",
        "vi_end_token_idx",
    ),
}


class ClinicalFormatAdapter:
    """Converts a raw JSONL record into a :class:`StandardizedDocument`.

    Supports English (``en``) and Vietnamese (``vi``) records produced by
    the n2c2-2010 preprocessing pipeline.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(record: dict) -> str:
        """Infer the language of a JSONL record from its keys.

        Returns ``"en"`` when only English keys are present, ``"vi"`` when
        only Vietnamese keys are present, and ``"en"`` as a safe fallback
        when both are present.
        """
        has_en = "en_sentence_str" in record
        has_vi = "vi_sentence_str" in record
        if has_en and not has_vi:
            return "en"
        if has_vi and not has_en:
            return "vi"
        return "en"  # both keys present → default to English

    def parse_record(
        self,
        record: dict,
        language: str,
    ) -> Optional[StandardizedDocument]:
        """Parse one JSONL record for the specified language.

        Returns ``None`` when the sentence text is missing or empty.
        """
        if language not in _LANG_KEY_MAP:
            raise ValueError(
                f"Unsupported language {language!r}. "
                f"Expected one of {list(_LANG_KEY_MAP)}."
            )

        sent_key, term_key, start_key, end_key = _LANG_KEY_MAP[language]
        text: str = record.get(sent_key, "").strip()
        if not text:
            logger.debug("Empty sentence for language %r; skipping record.", language)
            return None

        words: List[str] = text.split()
        entities, id_to_idx = self._parse_entities(
            record, term_key, start_key, end_key
        )
        relations = self._parse_relations(record, id_to_idx)

        doc_id: str = str(record.get("id", ""))
        
        return StandardizedDocument(
            id=doc_id,
            text=text,
            words=words,
            entities=entities,
            relations=relations,
            language=language,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entities(
        record: dict,
        term_key: str,
        start_key: str,
        end_key: str,
    ) -> Tuple[List[EntitySpan], Dict[str, int]]:
        """Parse the ``entities`` list from a raw record."""
        entities: List[EntitySpan] = []
        id_to_idx: Dict[str, int] = {}

        for idx, raw_ent in enumerate(record.get("entities", [])):
            ent_id: str = raw_ent.get("id", f"E{idx}")
            entities.append(
                EntitySpan(
                    label=raw_ent.get("label", "UNKNOWN"),
                    term=raw_ent.get(term_key, ""),
                    start_token_idx=raw_ent.get(start_key, -1),
                    end_token_idx=raw_ent.get(end_key, -1),
                    entity_id=ent_id,
                )
            )
            id_to_idx[ent_id] = idx

        return entities, id_to_idx

    @staticmethod
    def _parse_relations(
        record: dict,
        id_to_idx: Dict[str, int],
    ) -> List[RelationTriple]:
        """Parse the ``relations`` list from a raw record."""
        relations: List[RelationTriple] = []

        for raw_rel in record.get("relations", []):
            subj_id: Optional[str] = raw_rel.get("subject_id") or raw_rel.get(
                "head_idx"
            )
            obj_id: Optional[str] = raw_rel.get("object_id") or raw_rel.get(
                "tail_idx"
            )
            if subj_id in id_to_idx and obj_id in id_to_idx:
                relations.append(
                    RelationTriple(
                        relation_type=raw_rel.get("label", "UNKNOWN"),
                        subject_id=subj_id,
                        object_id=obj_id,
                    )
                )

        return relations
