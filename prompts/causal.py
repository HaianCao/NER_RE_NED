# -*- coding: utf-8 -*-
"""Causal (Format A) prompt builder.

Produces samples with a single ``"text"`` column containing the full
``input_prefix + ### Output:\\n{json}`` string.  No system/user/assistant
separation — the model learns the mapping as a causal language model.
"""

from __future__ import annotations

from typing import Dict, List

from datasets import Dataset as HFDataset

from ..config import LabelConfig, REMode, TaskMode
from ..data.schemas import StandardizedDocument
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


class CausalPromptBuilder(BasePromptBuilder):
    """Builds plain-text training samples for causal LM fine-tuning.

    The output ``HFDataset`` has a single column ``"text"`` which is the
    concatenation of the instruction prefix and the gold-standard JSON output.

    Separating the instruction from the output with ``\\n\\n### Output:\\n``
    gives the model a clear boundary to learn where to start generating.
    """

    _OUTPUT_MARKER = "\n\n### Output:\n"

    def __init__(self, task_mode: TaskMode, re_mode: REMode, labels: LabelConfig) -> None:
        self._task_mode = task_mode
        self._re_mode = re_mode
        self._labels = labels
        self._entity_defs = format_entity_definitions(labels.entities)
        self._relation_defs = format_relation_definitions(labels.relations)

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

    # ------------------------------------------------------------------
    # NER
    # ------------------------------------------------------------------

    def _ner_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        prefix = (
            "Extract all named entities from the following clinical sentence.\n\n"
            f"Entity types:\n{self._entity_defs}\n\n"
            "Output ONLY a valid JSON object. "
            'Schema: {"entities": [{"id": "T0", "label": "...", "term": "..."}]}\n\n'
            f'Sentence: "{doc.text}"'
        )
        return {"text": prefix + self._OUTPUT_MARKER + build_ner_output(doc.entities)}

    # ------------------------------------------------------------------
    # RE – pairwise
    # ------------------------------------------------------------------

    def _re_pairwise_samples(
        self, doc: StandardizedDocument
    ) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        pairs = filter_valid_pairs(doc.entities, self._labels.valid_pair_set)
        for subj, obj in pairs:
            marked_sentence = insert_entity_markers(doc.words, subj, obj)
            relation = get_pairwise_relation(doc, subj, obj)
            prefix = (
                "Classify the clinical relationship between the two marked entities.\n\n"
                f"Relation types:\n{self._relation_defs}\n\n"
                f'Sentence: "{marked_sentence}"\n'
                f"Subject <e1> type: {subj.label}\n"
                f"Object  <e2> type: {obj.label}"
            )
            output = build_re_pairwise_output(relation)
            samples.append({"text": prefix + self._OUTPUT_MARKER + output})
        return samples

    # ------------------------------------------------------------------
    # RE – global
    # ------------------------------------------------------------------

    def _re_global_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        entity_list = format_entity_list(doc.entities)
        prefix = (
            "Extract all clinical relations between the listed entities.\n\n"
            f"Entities:\n{entity_list}\n\n"
            f"Relation types:\n{self._relation_defs}\n\n"
            "Omit entity pairs with no valid relation. "
            "Output ONLY a valid JSON object. "
            'Schema: {"relations": [{"label": "...", "subject_id": "T0", "object_id": "T1"}]}\n\n'
            f'Sentence: "{doc.text}"'
        )
        return {
            "text": prefix + self._OUTPUT_MARKER + build_re_global_output(doc.relations)
        }

    # ------------------------------------------------------------------
    # JOINT
    # ------------------------------------------------------------------

    def _joint_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        prefix = (
            "Extract all named entities and their clinical relations "
            "from the following sentence.\n\n"
            f"Entity types:\n{self._entity_defs}\n\n"
            f"Relation types:\n{self._relation_defs}\n\n"
            "Output ONLY a valid JSON object. Schema:\n"
            '{"entities": [{"id": "T0", "label": "...", "term": "..."}], '
            '"relations": [{"label": "...", "subject_id": "T0", "object_id": "T1"}]}\n\n'
            f'Sentence: "{doc.text}"'
        )
        return {
            "text": (
                prefix
                + self._OUTPUT_MARKER
                + build_joint_output(doc.entities, doc.relations)
            )
        }
