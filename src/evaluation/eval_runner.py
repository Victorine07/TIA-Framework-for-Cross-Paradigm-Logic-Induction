#!/usr/bin/env python3
"""
src/evaluation/eval_runner.py

Shared metric computation, failure-taxonomy classification, and result
aggregation for every evaluation mode (zero-shot, few-shot, fine-tuned).

Shared metric computation for Section 4: Evaluation Framework — results are comparable across zero-shot, few-shot, and fine-tuned evaluation modes.

"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List


def try_load_metrics():
    try:
        from evaluation.metrics_fixed import FixedCryptographicMetrics
        return FixedCryptographicMetrics()
    except Exception:
        return None


def compute_metrics(metrics_obj, prediction: str, reference: str) -> Dict[str, float]:
    if metrics_obj is None:
        return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}

    try:
        result = metrics_obj.evaluate(prediction, reference)
        sv = float(result.get("syntax_validity", 0.0))
        sm = float(result.get("semantic_match", 0.0))
        vc = float(result.get("value_consistency", 0.0))
        overall = (sv + sm + vc) / 3.0
        return {"sv": sv, "sm": sm, "vc": vc, "overall": overall}
    except Exception:
        return {"sv": 0.0, "sm": 0.0, "vc": 0.0, "overall": 0.0}


_WORD_SIZE_PATTERN = re.compile(r"(\d+)\s*word")

FAILURE_CATEGORIES = (
    "correct",
    "syntax_error",
    "type_mismatch",
    "missing_definition",
    "semantic_error",
    "value_error",
)


def classify_failure(generated: str, reference: str, sv: float, sm: float, vc: float) -> str:
    """Best-effort failure-category heuristic for the paper's error analysis.
    Heuristic ordering (first match wins):
    1. empty generation                          -> missing_definition
    2. SV == 0 (fails basic Isabelle structure)   -> syntax_error
    3. word-size annotations disagree             -> type_mismatch
    4. SM < 0.5 (operator/structure mismatch)      -> semantic_error
    5. VC < 0.5 (right structure, wrong constants) -> value_error
    6. all three near 1.0                          -> correct
    7. otherwise (partial credit, no clean bucket) -> semantic_error
    """
    gen = (generated or "").strip()

    if not gen:
        return "missing_definition"

    if sv == 0.0:
        return "syntax_error"

    gen_sizes = set(_WORD_SIZE_PATTERN.findall(gen))
    ref_sizes = set(_WORD_SIZE_PATTERN.findall(reference or ""))
    if gen_sizes and ref_sizes and gen_sizes != ref_sizes:
        return "type_mismatch"

    if sm < 0.5:
        return "semantic_error"

    if vc < 0.5:
        return "value_error"

    if sv >= 0.99 and sm >= 0.99 and vc >= 0.99:
        return "correct"

    return "semantic_error"


def _group_value(row: Dict[str, Any], keys: List[str], default: str = "unknown") -> str:
    metadata = row.get("metadata", {}) or {}
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "avg_sv": 0.0,
            "avg_sm": 0.0,
            "avg_vc": 0.0,
            "avg_overall": 0.0,
            "by_family": {},
            "by_tier": {},
            "by_cipher": {},
            "failure_counts": {},
        }

    def avg(metric_name: str) -> float:
        vals = [r["metrics"][metric_name] for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    def summarize_group(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(group_rows),
            "avg_sv": sum(r["metrics"]["sv"] for r in group_rows) / len(group_rows),
            "avg_sm": sum(r["metrics"]["sm"] for r in group_rows) / len(group_rows),
            "avg_vc": sum(r["metrics"]["vc"] for r in group_rows) / len(group_rows),
            "avg_overall": sum(r["metrics"]["overall"] for r in group_rows) / len(group_rows),
        }

    family_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tier_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cipher_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failure_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        family_groups[_group_value(row, ["family"])].append(row)
        # "tier" is the actual field (T1/T2/T3/T4); component_type kept as legacy fallback
        tier_groups[_group_value(row, ["tier", "component_type", "componenttype"])].append(row)
        cipher_groups[_group_value(row, ["cipher"])].append(row)
        category = row.get("failure_category")
        if category:
            failure_counts[category] += 1

    return {
        "count": len(rows),
        "avg_sv": avg("sv"),
        "avg_sm": avg("sm"),
        "avg_vc": avg("vc"),
        "avg_overall": avg("overall"),
        "by_family": {k: summarize_group(v) for k, v in sorted(family_groups.items())},
        "by_tier": {k: summarize_group(v) for k, v in sorted(tier_groups.items())},
        "by_cipher": {k: summarize_group(v) for k, v in sorted(cipher_groups.items())},
        "failure_counts": dict(sorted(failure_counts.items())),
    }
