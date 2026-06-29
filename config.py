# -*- coding: utf-8 -*-
"""Pipeline configuration loaded from a YAML file.

Provides strongly-typed dataclasses for every config section and enums
for every categorical option, enabling IDE auto-complete and early
validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple

import yaml


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingFormat(str, Enum):
    """Controls how training samples are formatted."""

    CAUSAL = "causal"
    """Plain text: ``sentence\\n\\n### Output:\\n{json}``."""

    INSTRUCTION = "instruction"
    """ChatML: System / User / Assistant roles."""


class TaskMode(str, Enum):
    """Controls which sub-tasks are trained."""

    NER_ONLY = "ner_only"
    """Named Entity Recognition only."""

    RE_ONLY = "re_only"
    """Relation Extraction only (entities provided as input)."""

    JOINT = "joint"
    """NER and RE in a single forward pass."""


class REMode(str, Enum):
    """Controls how RE samples are constructed (only when task includes RE)."""

    PAIRWISE = "pairwise"
    """One sample per entity pair; N*(N-1) samples per sentence."""

    GLOBAL = "global"
    """One sample per sentence; model predicts all relations at once."""


# ---------------------------------------------------------------------------
# Config sub-sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Backbone model settings."""

    name: str = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    max_seq_length: int = 1024
    seed: int = 42


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation hyper-parameters."""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""

    format: TrainingFormat = TrainingFormat.INSTRUCTION
    task_mode: TaskMode = TaskMode.JOINT
    re_mode: REMode = REMode.GLOBAL
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    output_dir: str = "./outputs"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50

    def __post_init__(self) -> None:
        if isinstance(self.format, str):
            self.format = TrainingFormat(self.format)
        if isinstance(self.task_mode, str):
            self.task_mode = TaskMode(self.task_mode)
        if isinstance(self.re_mode, str):
            self.re_mode = REMode(self.re_mode)


@dataclass
class DataFileEntry:
    """A single JSONL file with an optional language hint."""

    path: str
    language: str = "auto"
    """``"auto"`` | ``"en"`` | ``"vi"``."""


@dataclass
class DataConfig:
    """Dataset file lists for training and evaluation."""

    train_files: List[DataFileEntry] = field(default_factory=list)
    eval_files: List[DataFileEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.train_files = [
            DataFileEntry(**f) if isinstance(f, dict) else f
            for f in self.train_files
        ]
        self.eval_files = [
            DataFileEntry(**f) if isinstance(f, dict) else f
            for f in self.eval_files
        ]


@dataclass
class LabelConfig:
    """Entity and relation label definitions embedded into prompts."""

    entities: Dict[str, str] = field(default_factory=dict)
    """``{label: description}`` for each entity type."""

    relations: Dict[str, str] = field(default_factory=dict)
    """``{label: description}`` for each relation type."""

    valid_pairs: List[List[str]] = field(default_factory=list)
    """List of ``[subject_label, object_label]`` pairs allowed to have a relation."""

    @property
    def valid_pair_set(self) -> Set[Tuple[str, str]]:
        """Return valid pairs as a set of tuples for O(1) lookup."""
        return {(pair[0], pair[1]) for pair in self.valid_pairs}


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Root configuration object assembled from YAML."""

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    labels: LabelConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load and validate configuration from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            labels=LabelConfig(**raw.get("labels", {})),
        )
