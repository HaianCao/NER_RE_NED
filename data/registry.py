# -*- coding: utf-8 -*-
"""Dataset registry that aggregates multiple JSONL files.

Follows the Open/Closed Principle: new file sources can be added by
extending ``DatasetRegistry`` without modifying existing parsing logic.
"""

from __future__ import annotations

import json
import logging
from typing import List

from ..config import DataFileEntry
from .adapter import ClinicalFormatAdapter
from .schemas import StandardizedDocument

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Load and merge an arbitrary number of JSONL files.

    Each file can specify a language hint (``"en"``, ``"vi"``, or
    ``"auto"``).  When ``"auto"`` is used, the language is inferred
    record-by-record from the keys present in each JSON object.

    Example::

        from ner_re_pipeline.config import DataFileEntry
        from ner_re_pipeline.data.registry import DatasetRegistry

        registry = DatasetRegistry([
            DataFileEntry(path="train_en.jsonl"),
            DataFileEntry(path="train_vi.jsonl"),
        ])
        docs = registry.aggregate()
    """

    def __init__(self, entries: List[DataFileEntry]) -> None:
        self._entries = entries
        self._adapter = ClinicalFormatAdapter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def aggregate(self) -> List[StandardizedDocument]:
        """Parse all registered files and return a flat document list."""
        all_docs: List[StandardizedDocument] = []
        for entry in self._entries:
            docs = self._load_file(entry)
            logger.info(
                "Loaded %d documents from '%s' (language hint: %s)",
                len(docs),
                entry.path,
                entry.language,
            )
            all_docs.extend(docs)
        logger.info("Total documents loaded: %d", len(all_docs))
        return all_docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, entry: DataFileEntry) -> List[StandardizedDocument]:
        """Read a single JSONL file and parse each line."""
        docs: List[StandardizedDocument] = []
        try:
            with open(entry.path, encoding="utf-8") as fh:
                for line_num, raw_line in enumerate(fh, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    doc = self._parse_line(raw_line, entry, line_num)
                    if doc is not None:
                        docs.append(doc)
        except FileNotFoundError:
            logger.error("File not found: '%s'", entry.path)
        return docs

    def _parse_line(
        self,
        raw_line: str,
        entry: DataFileEntry,
        line_num: int,
    ) -> StandardizedDocument | None:
        """Decode one JSON line and convert it to a StandardizedDocument."""
        try:
            record: dict = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Skipping malformed JSON at line %d in '%s': %s",
                line_num,
                entry.path,
                exc,
            )
            return None

        lang = (
            self._adapter.detect_language(record)
            if entry.language == "auto"
            else entry.language
        )
        return self._adapter.parse_record(record, lang)
