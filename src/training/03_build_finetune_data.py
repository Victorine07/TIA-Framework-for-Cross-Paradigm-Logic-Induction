#!/usr/bin/env python3
"""
03_build_finetune_data.py

Stage 3a: build fine-tuning-ready (prompt, completion) pairs from the
canonical evaluation registry's train/val splits

What this script does:
    1. Loads the canonical eval registry (same one 02_build_eval_registry.py
       produces and 04_run_zero_shot_baseline.py consumes).
    2. Reads the requested splits (default: train, val). Never reads "test"
       or any "unseen_*" dataset by default -- those must stay held out from
       training. Reading test/unseen requires explicitly passing them, and
       does so with a loud warning, never silently.
    3. Builds the exact same prompt text used at zero-shot inference time via
       src/prompting/prompt_builder.py, so fine-tuned and zero-shot
       evaluation remain comparable.
    4. Cleans reference Isabelle/HOL output via the shared text_normalization
       helpers (same cleaning applied to predictions during evaluation).
    5. Drops examples with an empty cleaned target, counts them, and fails
       loudly if an entire split ends up empty.
    6. Writes one JSONL file per split plus a manifest JSON describing
       exactly what was built and from what.

Inputs:
    - datasets/processed/eval_registry_*.json (via --registry)

Outputs:
    - datasets/processed/finetune/<split>_sft_<metadata_strategy>.jsonl
    - datasets/processed/finetune/manifest_<tag>.json

Typical usage:
    python src/training/03_build_finetune_data.py --splits train,val

Builds training data for Section 5: Experimental Setup under each metadata strategy from Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.registry_loader import RegistryLoader  # noqa: E402
from evaluation.text_normalization import clean_code  # noqa: E402
from prompting.prompt_builder import build_translation_prompt, example_identifier  # noqa: E402

PROTECTED_SPLITS = {"test"}


def resolve_default_registry(project_root: Path) -> Path:
    candidates = [
        project_root / "datasets" / "processed" / "eval_registry_family_holdout_v1.json",
        project_root / "datasets" / "processed" / "eval_registry.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        meta = row.get("metadata", {}) or {}
        counts[str(meta.get(key, "unknown"))] += 1
    return dict(sorted(counts.items()))


def build_split_records(
    raw_examples: List[Dict[str, Any]],
    metadata_strategy: str,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    dropped_empty_target = 0

    for example in raw_examples:
        prompt_text = build_translation_prompt(example, metadata_strategy)
        target_text = clean_code(example.get("output", ""))

        if not target_text.strip():
            dropped_empty_target += 1
            continue

        records.append(
            {
                "prompt": prompt_text,
                "completion": target_text,
                "full_text": prompt_text + target_text,
                "metadata": example.get("metadata", {}) or {},
                "source_split": split_name,
                "example_id": example_identifier(example),
                "metadata_strategy": metadata_strategy,
                "raw_instruction": example.get("instruction", ""),
                "raw_input": example.get("input", ""),
            }
        )

    return records, dropped_empty_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fine-tuning-ready prompt/completion pairs from the eval registry."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(resolve_default_registry(PROJECT_ROOT)),
        help="Path to eval registry JSON",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated registry dataset names to convert (default: train,val)",
    )
    parser.add_argument(
        "--allow-protected-splits",
        action="store_true",
        help="Allow building from protected splits (e.g. 'test') -- off by default to "
        "prevent accidentally training on held-out evaluation data.",
    )
    parser.add_argument(
        "--metadata-strategy",
        type=str,
        default="none",
        choices=["none", "full", "structured", "algorithmic", "alljson"],
        help="Metadata prompting strategy (must match what 04_run_zero_shot_baseline.py "
        "used for the zero-shot comparison you intend to fine-tune against)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "processed" / "finetune"),
        help="Directory for built SFT JSONL files and manifest",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Manifest tag (default: derived from metadata strategy + timestamp)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on examples per split, for smoke testing",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    blocked = [s for s in requested_splits if s in PROTECTED_SPLITS and not args.allow_protected_splits]
    if blocked:
        raise ValueError(
            f"Refusing to build fine-tuning data from protected split(s) {blocked} "
            f"without --allow-protected-splits. These splits must stay held out for "
            f"evaluation; training on them would invalidate reported test metrics."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.metadata_strategy}_{timestamp}"

    print("=" * 80)
    print("03_build_finetune_data.py")
    print("=" * 80)
    print(f"Registry path:      {registry_path}")
    print(f"Requested splits:   {requested_splits}")
    print(f"Metadata strategy:  {args.metadata_strategy}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Tag:                {tag}")
    print("=" * 80)

    loader = RegistryLoader(registry_path)
    print(f"Loaded registry. Known datasets: {loader.list_dataset_names()}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "tag": tag,
        "timestamp": timestamp,
        "registry_path": str(registry_path),
        "metadata_strategy": args.metadata_strategy,
        "requested_splits": requested_splits,
        "limit": args.limit,
        "splits": {},
    }

    for split_name in requested_splits:
        print("-" * 80)
        print(f"Loading split: {split_name}")
        raw_examples = loader.load_dataset(split_name)
        print(f"  Raw examples loaded: {len(raw_examples)}")

        if args.limit is not None:
            raw_examples = raw_examples[: args.limit]
            print(f"  Truncated to --limit: {len(raw_examples)}")

        if not raw_examples:
            raise ValueError(
                f"Split '{split_name}' resolved to zero examples. Refusing to write an "
                f"empty fine-tuning file -- check the registry and dataset path."
            )

        records, dropped_empty_target = build_split_records(
            raw_examples, args.metadata_strategy, split_name
        )

        if dropped_empty_target:
            print(
                f"  WARNING: dropped {dropped_empty_target} example(s) with an empty "
                f"cleaned target (reference Isabelle/HOL output)."
            )

        if not records:
            raise ValueError(
                f"Split '{split_name}' produced zero usable records after cleaning "
                f"(all {len(raw_examples)} example(s) had empty targets). This indicates "
                f"a normalization or source-data problem, not a normal outcome."
            )

        output_path = output_dir / f"{split_name}_sft_{args.metadata_strategy}.jsonl"
        save_jsonl(output_path, records)
        print(f"  Wrote {len(records)} record(s) to {output_path}")

        manifest["splits"][split_name] = {
            "output_file": output_path.name,
            "raw_examples": len(raw_examples),
            "usable_records": len(records),
            "dropped_empty_target": dropped_empty_target,
            "by_family": count_by(records, "family"),
            "by_cipher": count_by(records, "cipher"),
            "by_tier": count_by(records, "tier"),
        }

    manifest_path = output_dir / f"manifest_{tag}.json"
    save_json(manifest_path, manifest)

    print("=" * 80)
    print("DONE")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
