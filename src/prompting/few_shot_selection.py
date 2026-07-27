#!/usr/bin/env python3
"""
src/prompting/few_shot_selection.py

Implements the support-example selection evaluated in Section 6: Results and Discussion - Few-Shot Evaluation: Structural Grounding Without Parameter Updates.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

SELECTION_POLICIES = ("same_family", "same_tier", "random")


def _stable_seed(seed: int, key: str) -> int:
    # hash() is per-process-randomized for strings in Python 3 (PYTHONHASHSEED);
    # sha256 is stable across runs/processes, which reproducibility requires.
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _query_key(example: Dict[str, Any]) -> str:
    identifier = example.get("id", example.get("example_id"))
    if identifier is not None:
        return str(identifier)
    return (example.get("instruction", "") + "|" + example.get("input", ""))[:500]


def _is_same_example(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.get("instruction") == b.get("instruction") and a.get("input") == b.get("input")


def select_support_examples(
    query_example: Dict[str, Any],
    pool: List[Dict[str, Any]],
    k: int,
    policy: str = "same_family",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select up to k demonstration examples for `query_example` from `pool`.

    Deterministic: the same (query, pool, k, policy, seed) always returns
    the same support set, but different queries in the same run get
    different (still stable) support sets rather than all sharing one fixed
    block of examples.

    Policies:
    - "same_family": prefer pool examples sharing the query's cipher family
      (metadata.family), padding with other pool examples if too few.
    - "same_tier": same idea, keyed on TIA tier (metadata.tier, T1-T4).
    - "random": ignore family/tier, sample deterministically from the pool.
    """
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"Unknown few-shot selection policy: {policy!r}. Choose from {SELECTION_POLICIES}.")

    if k <= 0:
        return []

    candidates = [ex for ex in pool if not _is_same_example(ex, query_example)]
    if not candidates:
        return []

    rng = random.Random(_stable_seed(seed, _query_key(query_example)))

    if policy == "random":
        ordered = candidates[:]
        rng.shuffle(ordered)
        return ordered[:k]

    meta_key = "family" if policy == "same_family" else "tier"
    query_metadata = query_example.get("metadata", {}) or {}
    target_value = query_metadata.get(meta_key)

    if target_value is not None:
        preferred = [
            ex for ex in candidates
            if (ex.get("metadata", {}) or {}).get(meta_key) == target_value
        ]
    else:
        preferred = []
    preferred_ids = {id(ex) for ex in preferred}
    rest = [ex for ex in candidates if id(ex) not in preferred_ids]

    rng.shuffle(preferred)
    rng.shuffle(rest)
    ordered = preferred + rest
    return ordered[:k]
