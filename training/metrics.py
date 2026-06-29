# -*- coding: utf-8 -*-
"""Post-training evaluation metrics for NER and RE.

During training only the cross-entropy loss is tracked (memory-efficient).
This module provides F1 computation utilities intended for a *separate*
evaluation run after training (see ``evaluate.py``).

The functions here are called with *decoded text* (not raw logits) so
they work regardless of model vocabulary size.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set, Tuple

from ..utils.span_utils import normalize_term

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def safe_parse_json(text: str) -> dict:
    """Attempt to parse *text* as JSON; return an empty dict on failure."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# NER metrics
# ---------------------------------------------------------------------------


def extract_ner_gold(parsed: dict) -> Set[Tuple[str, str]]:
    """Extract (label, normalised_term) pairs from a gold or predicted dict."""
    result: Set[Tuple[str, str]] = set()
    for ent in parsed.get("entities", []):
        label = ent.get("label", "")
        term = normalize_term(ent.get("term", ""))
        if label and term:
            result.add((label, term))
    return result


def compute_ner_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute entity-level precision, recall, and F1.

    Matching is done on ``(label, normalised_term)`` pairs (soft span match).

    Args:
        predictions: List of model-generated JSON strings.
        references:  List of gold-standard JSON strings.

    Returns:
        Dict with keys ``ner_precision``, ``ner_recall``, ``ner_f1``.
    """
    tp = fp = fn = 0
    for pred_str, ref_str in zip(predictions, references):
        pred_set = extract_ner_gold(safe_parse_json(pred_str))
        ref_set = extract_ner_gold(safe_parse_json(ref_str))
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    return _compute_prf(tp, fp, fn, prefix="ner")


# ---------------------------------------------------------------------------
# RE metrics
# ---------------------------------------------------------------------------


def extract_re_gold(parsed: dict) -> Set[Tuple[str, str, str]]:
    """Extract (label, subject_id, object_id) triples from a dict."""
    result: Set[Tuple[str, str, str]] = set()
    # JOINT and GLOBAL format
    for rel in parsed.get("relations", []):
        label = rel.get("label", "")
        subj = rel.get("subject_id", "")
        obj = rel.get("object_id", "")
        if label and label != "NONE" and subj and obj:
            result.add((label, subj, obj))
    return result


def extract_re_pairwise(parsed: dict) -> str:
    """Extract the single relation label from a pairwise prediction dict."""
    return parsed.get("relation", "NONE")


def compute_re_metrics(
    predictions: List[str],
    references: List[str],
    pairwise: bool = False,
) -> Dict[str, float]:
    """Compute relation-level precision, recall, and F1.

    Args:
        predictions: Model-generated JSON strings.
        references:  Gold-standard JSON strings.
        pairwise:    ``True`` for PAIRWISE mode (single relation per sample).

    Returns:
        Dict with keys ``re_precision``, ``re_recall``, ``re_f1``.
    """
    tp = fp = fn = 0
    for pred_str, ref_str in zip(predictions, references):
        pred_parsed = safe_parse_json(pred_str)
        ref_parsed = safe_parse_json(ref_str)

        if pairwise:
            pred_label = extract_re_pairwise(pred_parsed)
            ref_label = extract_re_pairwise(ref_parsed)
            if pred_label == ref_label and ref_label != "NONE":
                tp += 1
            elif pred_label != "NONE" and pred_label != ref_label:
                fp += 1
            elif ref_label != "NONE" and pred_label == "NONE":
                fn += 1
        else:
            pred_set = extract_re_gold(pred_parsed)
            ref_set = extract_re_gold(ref_parsed)
            tp += len(pred_set & ref_set)
            fp += len(pred_set - ref_set)
            fn += len(ref_set - pred_set)

    return _compute_prf(tp, fp, fn, prefix="re")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _compute_prf(tp: int, fp: int, fn: int, prefix: str) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        f"{prefix}_precision": round(precision, 4),
        f"{prefix}_recall": round(recall, 4),
        f"{prefix}_f1": round(f1, 4),
    }
