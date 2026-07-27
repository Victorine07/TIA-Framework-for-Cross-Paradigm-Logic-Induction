
#!/usr/bin/env python3

"""
01_create_splits.py

Pipeline stage 1: construct stratified train / val / test splits from the
combined dataset, plus a held-out unseen-cipher partition. Family-aware
partitioning ensures no cipher family appears in both the seen-cipher and
unseen-cipher evaluation sets.

Corresponds to Section 5: Experimental Setup (split construction, train/val/test boundaries, and unseen-cipher held-out design).
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stable_key(item: dict) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def get_metadata_field(item: dict, key: str, default="unknown"):
    return item.get("metadata", {}).get(key, default)


def filter_seen_unseen(
    data: List[dict], unseen_ciphers: List[str]
) -> Tuple[List[dict], List[dict]]:
    unseen_set = {c.lower() for c in unseen_ciphers}
    seen, heldout = [], []
    for item in data:
        cipher = str(get_metadata_field(item, "cipher", "unknown")).lower()
        if cipher in unseen_set:
            heldout.append(item)
        else:
            seen.append(item)
    return seen, heldout


def stratified_split_by_family(
    data: List[dict],
    test_size: float,
    val_size: float,
    seed: int,
    min_per_family_warn: int = 3,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, dict]]:
    rng = random.Random(seed)
    family_groups = defaultdict(list)

    for item in data:
        family = str(get_metadata_field(item, "family", "unknown"))
        family_groups[family].append(item)

    train_data, val_data, test_data = [], [], []
    family_report = {}

    for family, examples in sorted(family_groups.items()):
        exs = examples[:]
        rng.shuffle(exs)
        n = len(exs)

        if n < min_per_family_warn:
            family_report[family] = {
                "count": n,
                "warning": f"Very small family ({n} examples); split may be unstable."
            }
        else:
            family_report[family] = {"count": n}

        n_test = max(1, int(round(n * test_size))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_size))) if n >= 4 else 0

        if n_test + n_val >= n:
            if n >= 3:
                n_test = 1
                n_val = 1 if n >= 4 else 0
            else:
                n_test = 0
                n_val = 0

        n_train = n - n_test - n_val
        if n_train <= 0:
            if n >= 2:
                n_test = 1
                n_val = 0
                n_train = n - 1
            else:
                n_test = 0
                n_val = 0
                n_train = n

        family_train = exs[:n_train]
        family_val = exs[n_train:n_train + n_val]
        family_test = exs[n_train + n_val:n_train + n_val + n_test]

        train_data.extend(family_train)
        val_data.extend(family_val)
        test_data.extend(family_test)

        family_report[family].update({
            "train": len(family_train),
            "val": len(family_val),
            "test": len(family_test),
        })

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data, family_report


def count_by_field(data: List[dict], field: str) -> Dict[str, int]:
    counts = Counter()
    for item in data:
        value = str(get_metadata_field(item, field, "unknown"))
        counts[value] += 1
    return dict(sorted(counts.items()))


def count_by_tier(data: List[dict]) -> Dict[str, int]:
    candidates = ["component_type", "tier", "semantic_tier"]
    counts = Counter()

    for item in data:
        meta = item.get("metadata", {})
        val = None
        for key in candidates:
            if key in meta:
                val = str(meta[key])
                break
        if val is None:
            val = "unknown"
        counts[val] += 1

    return dict(sorted(counts.items()))


def build_indices_map(full_data: List[dict]) -> Dict[str, int]:
    mapping = {}
    for idx, item in enumerate(full_data):
        mapping[stable_key(item)] = idx
    return mapping


def split_indices(split_data: List[dict], key_to_idx: Dict[str, int]) -> List[int]:
    indices = []
    for item in split_data:
        key = stable_key(item)
        if key not in key_to_idx:
            raise KeyError("Split example not found in original dataset index map.")
        indices.append(key_to_idx[key])
    return sorted(indices)


def write_summary(
    output_dir: Path,
    input_path: Path,
    manifest_path: Path,
    unseen_ciphers: List[str],
    full_count: int,
    seen_count: int,
    heldout_count: int,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
    family_report: Dict[str, dict],
    seed: int,
    test_size: float,
    val_size: float,
):
    summary = {
        "input_dataset": str(input_path),
        "dataset_manifest": str(manifest_path) if manifest_path else None,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "selected_unseen_ciphers": unseen_ciphers,
        "counts": {
            "full_dataset": full_count,
            "seen_pool_after_exclusion": seen_count,
            "heldout_unseen_examples": heldout_count,
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "family_split_report": family_report,
        "train_family_counts": count_by_field(train_data, "family"),
        "val_family_counts": count_by_field(val_data, "family"),
        "test_family_counts": count_by_field(test_data, "family"),
        "train_cipher_counts": count_by_field(train_data, "cipher"),
        "val_cipher_counts": count_by_field(val_data, "cipher"),
        "test_cipher_counts": count_by_field(test_data, "cipher"),
        "train_tier_counts": count_by_tier(train_data),
        "val_tier_counts": count_by_tier(val_data),
        "test_tier_counts": count_by_tier(test_data),
    }

    out = output_dir / "split_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_analysis(
    full_count: int,
    seen_count: int,
    heldout_count: int,
    train_data: List[dict],
    val_data: List[dict],
    test_data: List[dict],
):
    total = len(train_data) + len(val_data) + len(test_data)

    print("\n" + "=" * 60)
    print("AAAI Split Analysis")
    print("=" * 60)
    print(f"Full dataset: {full_count}")
    print(f"Seen pool after unseen exclusion: {seen_count}")
    print(f"Held-out unseen examples: {heldout_count}")
    print(f"Train: {len(train_data)} ({len(train_data)/total:.1%})")
    print(f"Val:   {len(val_data)} ({len(val_data)/total:.1%})")
    print(f"Test:  {len(test_data)} ({len(test_data)/total:.1%})")

    for split_name, split_data in [
        ("Train", train_data),
        ("Val", val_data),
        ("Test", test_data),
    ]:
        fam = count_by_field(split_data, "family")
        cip = count_by_field(split_data, "cipher")
        tier = count_by_tier(split_data)

        print(f"\n{split_name} family distribution:")
        for k, v in fam.items():
            print(f"  {k}: {v}")

        print(f"{split_name} cipher distribution:")
        for k, v in cip.items():
            print(f"  {k}: {v}")

        print(f"{split_name} tier distribution:")
        for k, v in tier.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Create AAAI train/val/test splits with configurable unseen cipher exclusion"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/cryptospec_dataset.jsonl",
        help="Canonical merged dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
        help="Directory for train/val/test split outputs"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional dataset manifest path from 00_prepare_aaai_datasets.py"
    )
    parser.add_argument(
        "--unseen_ciphers",
        nargs="*",
        default=[],
        help="Cipher names to exclude from in-distribution splits"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of seen data used for test split"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.10,
        help="Fraction of seen data used for validation split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if not (0.0 <= args.test_size < 1.0 and 0.0 <= args.val_size < 1.0):
        raise ValueError("test_size and val_size must be in [0,1).")

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    full_data = load_jsonl(input_path)
    print(f"Loaded {len(full_data)} examples")

    unseen_ciphers = [c.lower() for c in args.unseen_ciphers]
    seen_data, heldout_data = filter_seen_unseen(full_data, unseen_ciphers)

    if not seen_data:
        raise ValueError("No seen data remains after excluding unseen ciphers.")

    print(f"Excluded unseen ciphers: {unseen_ciphers if unseen_ciphers else 'none'}")
    print(f"Seen pool size: {len(seen_data)}")
    print(f"Held-out unseen size: {len(heldout_data)}")

    train_data, val_data, test_data, family_report = stratified_split_by_family(
        seen_data,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    save_jsonl(output_dir / "train.jsonl", train_data)
    save_jsonl(output_dir / "val.jsonl", val_data)
    save_jsonl(output_dir / "test.jsonl", test_data)

    key_to_idx = build_indices_map(full_data)
    indices = {
        "train": split_indices(train_data, key_to_idx),
        "val": split_indices(val_data, key_to_idx),
        "test": split_indices(test_data, key_to_idx),
        "heldout_unseen": split_indices(heldout_data, key_to_idx),
    }

    with open(output_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2, ensure_ascii=False)

    write_summary(
        output_dir=output_dir,
        input_path=input_path,
        manifest_path=manifest_path,
        unseen_ciphers=unseen_ciphers,
        full_count=len(full_data),
        seen_count=len(seen_data),
        heldout_count=len(heldout_data),
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        family_report=family_report,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    print_analysis(
        full_count=len(full_data),
        seen_count=len(seen_data),
        heldout_count=len(heldout_data),
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    print(f"\nSaved split files to {output_dir}")
    print(f"Saved indices to {output_dir / 'split_indices.json'}")
    print(f"Saved summary to {output_dir / 'split_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
    
