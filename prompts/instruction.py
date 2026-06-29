# -*- coding: utf-8 -*-
"""Instruction (Format B) prompt builder.

Produces samples with three columns — ``"system"``, ``"user"``, and
``"assistant"`` — that are later assembled into a ChatML sequence via
``tokenizer.apply_chat_template()``.

The system prompt contains the full label ontology so the model learns
the semantic meaning of every label, not just surface patterns.
"""

from __future__ import annotations

from typing import Dict, List

from datasets import Dataset as HFDataset

from config import LabelConfig, REMode, TaskMode
from data.schemas import StandardizedDocument
from ._helpers import (
    build_joint_output,
    build_ner_output,
    build_re_global_output,
    build_re_pairwise_output,
    filter_valid_pairs,
    format_entity_definitions,
    format_entity_list,
    format_relation_definitions,
    get_pairwise_relation,
    insert_entity_markers,
)
from .builder import BasePromptBuilder


class InstructionPromptBuilder(BasePromptBuilder):
    """Builds ChatML-style training samples for instruction fine-tuning.

    The output ``HFDataset`` has three columns:

    * ``"system"``    — label ontology and output schema (constant per config)
    * ``"user"``      — the input sentence with optional entity hints
    * ``"assistant"`` — the gold-standard JSON output

    Loss masking (training only on the ``assistant`` column) is handled by
    :class:`~ner_re_pipeline.training.collator.DataCollatorWithLossMask`.
    """

    def __init__(self, task_mode: TaskMode, re_mode: REMode, labels: LabelConfig) -> None:
        self._task_mode = task_mode
        self._re_mode = re_mode
        self._labels = labels
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # BasePromptBuilder interface
    # ------------------------------------------------------------------

    def generate_training_data(
        self,
        documents: List[StandardizedDocument],
    ) -> HFDataset:
        records: List[Dict[str, str]] = []
        for doc in documents:
            records.extend(self._build_samples(doc))
        return HFDataset.from_list(records)

    # ------------------------------------------------------------------
    # System prompt (built once per config)
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        entity_defs = format_entity_definitions(self._labels.entities)
        relation_defs = format_relation_definitions(self._labels.relations)

        if self._task_mode == TaskMode.NER_ONLY:
            return (
                "You are a clinical Named Entity Recognition expert.\n"
                "Given a medical sentence, identify all named entities.\n\n"
                f"Entity types:\n{entity_defs}\n\n"
                "Output ONLY a valid JSON object. "
                'Schema: {"entities": [{"id": "T0", "label": "...", "term": "..."}]}'
            )

        if self._task_mode == TaskMode.RE_ONLY:
            if self._re_mode == REMode.PAIRWISE:
                return (
                    "You are a clinical Relation Extraction expert.\n"
                    "Given a sentence with two marked entities, classify their relationship.\n\n"
                    f"Relation types:\n{relation_defs}\n\n"
                    "Output ONLY a valid JSON object. "
                    'Schema: {"relation": "<LABEL>"}'
                )
            # GLOBAL
            return (
                "You are a clinical Relation Extraction expert.\n"
                "Given a sentence and a list of named entities, "
                "identify all clinical relations between them.\n\n"
                f"Relation types:\n{relation_defs}\n\n"
                "Omit entity pairs with no valid relation. "
                "Output ONLY a valid JSON object. "
                'Schema: {"relations": [{"label": "...", "subject_id": "T0", "object_id": "T1"}]}'
            )

        # JOINT
        return (
            "You are a clinical information extraction expert.\n"
            "Given a medical sentence, perform both tasks:\n"
            "  1. Named Entity Recognition (NER)\n"
            "  2. Relation Extraction (RE) between identified entities\n\n"
            f"Entity types:\n{entity_defs}\n\n"
            f"Relation types:\n{relation_defs}\n\n"
            "Output ONLY a valid JSON object. Schema:\n"
            '{"entities": [{"id": "T0", "label": "...", "term": "..."}], '
            '"relations": [{"label": "...", "subject_id": "T0", "object_id": "T1"}]}'
        )

    # ------------------------------------------------------------------
    # Sample routing
    # ------------------------------------------------------------------

    def _build_samples(
        self, doc: StandardizedDocument
    ) -> List[Dict[str, str]]:
        if self._task_mode == TaskMode.NER_ONLY:
            return [self._ner_sample(doc)]
        if self._task_mode == TaskMode.RE_ONLY:
            if self._re_mode == REMode.PAIRWISE:
                return self._re_pairwise_samples(doc)
            return [self._re_global_sample(doc)]
        # JOINT
        return [self._joint_sample(doc)]

    def _make_record(self, user: str, assistant: str) -> Dict[str, str]:
        return {"system": self._system_prompt, "user": user, "assistant": assistant}

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def _ner_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        user = f'Sentence: "{doc.text}"'
        return self._make_record(user, build_ner_output(doc.entities))

    # ------------------------------------------------------------------
    # RE – pairwise
    # ------------------------------------------------------------------

    def _re_pairwise_samples(
        self, doc: StandardizedDocument
    ) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        pairs = filter_valid_pairs(doc.entities, self._labels.valid_pair_set)
        for subj, obj in pairs:
            marked = insert_entity_markers(doc.words, subj, obj)
            relation = get_pairwise_relation(doc, subj, obj)
            user = (
                f'Sentence: "{marked}"\n'
                f"Subject <e1> type: {subj.label}\n"
                f"Object  <e2> type: {obj.label}"
            )
            samples.append(self._make_record(user, build_re_pairwise_output(relation)))
        return samples

    # ------------------------------------------------------------------
    # RE – global
    # ------------------------------------------------------------------

    def _re_global_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        entity_list = format_entity_list(doc.entities)
        user = (
            f"Entities:\n{entity_list}\n\n"
            f'Sentence: "{doc.text}"'
        )
        return self._make_record(user, build_re_global_output(doc.relations))

    # ------------------------------------------------------------------
    # JOINT
    # ------------------------------------------------------------------

    def _joint_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        user = f'Sentence: "{doc.text}"'
        return self._make_record(
            user, build_joint_output(doc.entities, doc.relations)
        )
