#!/usr/bin/env python3
"""
Script: 02_build_eval_registry.py

Purpose:
    Build a canonical evaluation registry for experiments from processed
    in-distribution splits and held-out evaluation datasets.

What this script does:
    1. Reads train/val/test splits from datasets/processed/.
    2. Reads split metadata produced by the split creation stage.
    3. Discovers unseen evaluation datasets from datasets/unseen/.
    4. Creates a single registry JSON describing all datasets, counts, paths,
       evaluation groups, and OOD semantics.
    5. Saves a reproducible registry that later training and evaluation scripts
       can load instead of hardcoding dataset paths or cipher names.

Inputs:
    - datasets/processed/train.jsonl
    - datasets/processed/val.jsonl
    - datasets/processed/test.jsonl
    - datasets/processed/split_summary.json
    - datasets/processed/split_indices.json
    - datasets/unseen/*.jsonl
    - optional dataset manifest from 00_prepare_datasets.py

Outputs:
    - datasets/processed/eval_registry_*.json

Notes:
    - Unseen datasets are labeled as cipher-holdout evaluation by default.
      They should not automatically be described as family-level OOD unless the
      holdout protocol was explicitly constructed that way.
    - The registry stores both root-relative and absolute paths. Relative paths
      are better for portability; absolute paths are convenient for local
      debugging.

Corresponds to Section 3: Tiered Isomorphic Alignment - Dataset Construction and Registry (canonical registry that all experiment scripts consume).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set


KNOWN_FAMILY_MAP = {
    "lea": "ARX",
    "speck": "ARX",
    "sparx": "ARX",
    "xtea": "ARX/Feistel-like",
    "cham": "ARX",
    "simon": "Feistel",
    "simeck": "Feistel",
    "hight": "Feistel",
    "rectangle": "SPN",
    "present": "SPN",
    "gift": "SPN",
    "skinny": "SPN",
    "ascon": "Permutation",
    "gift-cofb": "Permutation/AEAD",
    "gift_cofb": "Permutation/AEAD",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    return load_json(path)


def load_jsonl_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_first_jsonl_record(path: Path) -> Optional[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def resolve_manifest_path(root: Path, manifest_arg: str) -> Optional[Path]:
    if not manifest_arg:
        return None
    p = Path(manifest_arg)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def to_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def infer_cipher_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ["_dataset", "-dataset", "_test", "-test"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def infer_family(cipher_name: str, sample_record: Optional[dict] = None) -> str:
    if sample_record:
        metadata = sample_record.get("metadata", {})
        family = metadata.get("family")
        if isinstance(family, str) and family.strip():
            return family.strip()
    return KNOWN_FAMILY_MAP.get(cipher_name.lower(), "unknown")


def collect_seen_ciphers(split_summary: dict) -> Set[str]:
    seen = set()
    for key in ["selected_seen_ciphers", "seen_ciphers", "training_ciphers"]:
        values = split_summary.get(key, [])
        if isinstance(values, list):
            seen.update(str(v).lower() for v in values)
    return seen


def collect_selected_unseen_ciphers(split_summary: dict) -> Set[str]:
    values = split_summary.get("selected_unseen_ciphers", [])
    if isinstance(values, list):
        return set(str(v).lower() for v in values)
    return set()


def build_dataset_entry(
    *,
    root: Path,
    name: str,
    path: Path,
    split_role: str,
    group: str,
    dataset_kind: str,
    family: str,
    cipher: Optional[str],
    description: str,
    ood_level: str = "none",
    family_holdout: bool = False,
) -> dict:
    return {
        "name": name,
        "path_rel": to_relpath(path, root),
        "path_abs": str(path.resolve()),
        "num_examples": load_jsonl_count(path),
        "split_role": split_role,
        "group": group,
        "dataset_kind": dataset_kind,
        "ood_level": ood_level,
        "family_holdout": family_holdout,
        "family": family,
        "cipher": cipher,
        "description": description,
    }


def discover_unseen_files(unseen_dir: Path) -> List[Path]:
    return sorted(
        p for p in unseen_dir.glob("*.jsonl")
        if p.is_file() and not p.name.startswith(".")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical  evaluation registry from processed and unseen datasets"
    )
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="datasets/processed",
        help="Processed split directory",
    )
    parser.add_argument(
        "--unseen_dir",
        type=str,
        default="datasets/unseen",
        help="Directory containing unseen evaluation JSONL files",
    )
    parser.add_argument(
        "--dataset_manifest",
        type=str,
        default="",
        help="Optional dataset manifest from 00_prepare_datasets.py",
    )
    parser.add_argument(
        "--registry_name",
        type=str,
        default="eval_registry_family_holdout_v1.json",
        help="Output registry file name",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="family_holdout_v1",
        help="Experiment tag",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    processed_dir = (root / args.processed_dir).resolve()
    unseen_dir = (root / args.unseen_dir).resolve()
    manifest_path = resolve_manifest_path(root, args.dataset_manifest)

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"
    split_summary_path = processed_dir / "split_summary.json"
    split_indices_path = processed_dir / "split_indices.json"

    required = [train_path, val_path, test_path, split_summary_path, split_indices_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    if not unseen_dir.exists():
        raise FileNotFoundError(f"Unseen dataset directory not found: {unseen_dir}")

    split_summary = load_json(split_summary_path)
    split_indices = load_json(split_indices_path)
    dataset_manifest = maybe_load_json(manifest_path)

    seen_ciphers = collect_seen_ciphers(split_summary)
    selected_unseen_ciphers = collect_selected_unseen_ciphers(split_summary)

    unseen_files = discover_unseen_files(unseen_dir)
    unseen_entries: List[dict] = []
    unseen_ciphers_found: List[str] = []
    overlap_with_seen: List[str] = []
    missing_from_selected: List[str] = []

    for path in unseen_files:
        cipher_name = infer_cipher_name_from_path(path)
        unseen_ciphers_found.append(cipher_name)
        sample = load_first_jsonl_record(path)
        family = infer_family(cipher_name, sample)

        if cipher_name.lower() in seen_ciphers:
            overlap_with_seen.append(cipher_name)
        if selected_unseen_ciphers and cipher_name.lower() not in selected_unseen_ciphers:
            missing_from_selected.append(cipher_name)

        unseen_entries.append(
            build_dataset_entry(
                root=root,
                name=f"unseen_{cipher_name}",
                path=path,
                split_role="cipher_holdout_test",
                group="unseen_cipher",
                dataset_kind="cipher_holdout",
                family=family,
                cipher=cipher_name,
                description=f"Held-out cipher evaluation dataset for {cipher_name}",
                ood_level="cipher",
                family_holdout=False,
            )
        )

    datasets = {
        "train": build_dataset_entry(
            root=root,
            name="train",
            path=train_path,
            split_role="train",
            group="in_distribution",
            dataset_kind="train_split",
            family="mixed",
            cipher=None,
            description="Seen in-distribution training split",
        ),
        "val": build_dataset_entry(
            root=root,
            name="val",
            path=val_path,
            split_role="id_validation",
            group="in_distribution",
            dataset_kind="id_eval",
            family="mixed",
            cipher=None,
            description="Seen in-distribution validation split",
        ),
        "test": build_dataset_entry(
            root=root,
            name="test",
            path=test_path,
            split_role="id_test",
            group="in_distribution",
            dataset_kind="id_eval",
            family="mixed",
            cipher=None,
            description="Seen in-distribution test split",
        ),
        "unseen": unseen_entries,
    }

    unseen_names = [entry["name"] for entry in unseen_entries]
    evaluation_groups: Dict[str, object] = {
        "primary_id": ["val", "test"],
        "primary_cipher_holdout": unseen_names,
        "all_core_eval": ["val", "test"] + unseen_names,
    }

    if unseen_entries:
        by_family: Dict[str, List[str]] = {}
        for entry in unseen_entries:
            by_family.setdefault(entry["family"], []).append(entry["name"])
        evaluation_groups["cipher_holdout_by_family"] = by_family

    registry = {
        "tag": args.tag,
        "registry_version": "2.0",
        "task": "python_to_isabelle_translation",
        "purpose": "Data evaluation registry for in-distribution and held-out cipher evaluation",
        "project_root_abs": str(root),
        "project_root_rel": ".",
        "dataset_manifest_path": to_relpath(manifest_path, root) if manifest_path else None,
        "dataset_manifest_abs": str(manifest_path.resolve()) if manifest_path else None,
        "split_summary_path": to_relpath(split_summary_path, root),
        "split_summary_abs": str(split_summary_path.resolve()),
        "split_indices_path": to_relpath(split_indices_path, root),
        "split_indices_abs": str(split_indices_path.resolve()),
        "selected_unseen_ciphers": sorted(selected_unseen_ciphers) if selected_unseen_ciphers else split_summary.get("selected_unseen_ciphers", []),
        "seen_ciphers": sorted(seen_ciphers),
        "discovered_unseen_ciphers": unseen_ciphers_found,
        "counts": split_summary.get("counts", {}),
        "datasets": datasets,
        "evaluation_groups": evaluation_groups,
        "split_summary": split_summary,
        "split_indices": split_indices,
        "dataset_manifest": dataset_manifest,
        "validation": {
            "seen_unseen_overlap": sorted(set(overlap_with_seen)),
            "discovered_but_not_in_selected_unseen": sorted(set(missing_from_selected)),
            "num_unseen_files": len(unseen_entries),
        },
        "notes": {
            "id_definition": "train/val/test are formed from the seen pool after excluding selected unseen ciphers",
            "cipher_holdout_definition": "unseen datasets are held-out cipher evaluations",
            "ood_definition": "cipher holdout is weaker than family-level OOD unless the protocol explicitly withholds an entire family",
            "motivation": "Designed to support clearer  evaluation bookkeeping and avoid hardcoded unseen dataset assumptions",
        },
    }

    output_path = processed_dir / args.registry_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(" Evaluation Registry")
    print("=" * 72)
    print(f"Tag: {args.tag}")
    print(f"Registry version: {registry['registry_version']}")
    print(f"Train examples: {datasets['train']['num_examples']}")
    print(f"Val examples:   {datasets['val']['num_examples']}")
    print(f"Test examples:  {datasets['test']['num_examples']}")
    print(f"Unseen datasets discovered: {len(unseen_entries)}")
    for entry in unseen_entries:
        print(f"  - {entry['name']}: {entry['num_examples']} [{entry['family']}] role={entry['split_role']}")
    if overlap_with_seen:
        print("\nWARNING: Seen/unseen overlap detected:")
        for cipher_name in sorted(set(overlap_with_seen)):
            print(f"  - {cipher_name}")
    if missing_from_selected:
        print("\nNOTE: Discovered unseen files not listed in selected_unseen_ciphers:")
        for cipher_name in sorted(set(missing_from_selected)):
            print(f"  - {cipher_name}")
    print(f"\nSaved registry to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
