#!/usr/bin/env python3
"""
00_prepare_datasets.py

Pipeline stage 0: merge per-cipher JSONL extractions into a unified raw
dataset. Reads individual extractor outputs from datasets/raw/, deduplicates
records, validates required schema fields, and writes the combined dataset
consumed by the split-creation stage.

Corresponds to Section 3: Tiered Isomorphic Alignment - Cipher Selection and Dataset Design (dataset assembly before split construction).
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_dirs(datasets_dir: Path) -> Dict[str, Path]:
    paths = {
        "raw": datasets_dir / "raw",
        "processed": datasets_dir / "processed",
        "unseen": datasets_dir / "unseen",
        "manifests": datasets_dir / "manifests",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def collect_train_files(search_dir: Path) -> List[Path]:
    return sorted(p for p in search_dir.glob("*_train.jsonl") if p.is_file())


def parse_cipher_name(filename: str) -> str:
    name = filename.replace(".jsonl", "")
    if not name.endswith("_train"):
        return ""
    parts = name.split("_")
    if len(parts) < 3:
        return ""
    return parts[0].lower()


def collect_cipher_files(files: List[Path]) -> Dict[str, List[Path]]:
    cipher_to_files: Dict[str, List[Path]] = {}
    for f in files:
        cipher = parse_cipher_name(f.name)
        if cipher:
            cipher_to_files.setdefault(cipher, []).append(f)
    return dict(sorted(cipher_to_files.items()))


def copy_or_move_files(files: List[Path], raw_dir: Path, move: bool = False):
    for src in files:
        dst = raw_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)


def ensure_canonical_dataset(
    raw_dir: Path,
    canonical_name: str = "cryptospec_dataset.jsonl"
) -> Tuple[Path, int, str]:
    all_train = raw_dir / "all_train.jsonl"
    canonical = raw_dir / canonical_name

    if canonical.exists():
        count = sum(1 for _ in open(canonical, "r", encoding="utf-8"))
        return canonical, count, "existing canonical file"

    if all_train.exists():
        shutil.copy2(str(all_train), str(canonical))
        count = sum(1 for _ in open(canonical, "r", encoding="utf-8"))
        return canonical, count, "copied from all_train.jsonl"

    raise FileNotFoundError(
        f"Could not create canonical dataset. Neither {canonical.name} nor all_train.jsonl "
        f"exists in {raw_dir}"
    )


def merge_cipher_files(raw_dir: Path, cipher: str) -> Tuple[List[dict], List[str]]:
    matched = sorted(raw_dir.glob(f"{cipher}_*_train.jsonl"))
    merged = []
    for path in matched:
        merged.extend(load_jsonl(path))
    return merged, [p.name for p in matched]


def write_unseen_sets(raw_dir: Path, unseen_dir: Path, unseen_ciphers: List[str]) -> Dict[str, dict]:
    summary = {}

    for cipher in unseen_ciphers:
        data, source_files = merge_cipher_files(raw_dir, cipher)
        if not source_files:
            raise FileNotFoundError(
                f"No raw train files found for unseen cipher '{cipher}' in {raw_dir}"
            )

        output_file = unseen_dir / f"{cipher}_dataset.jsonl"
        save_jsonl(output_file, data)

        summary[cipher] = {
            "output_file": output_file.name,
            "num_examples": len(data),
            "source_files": source_files,
        }

    return summary


def write_manifest(
    manifests_dir: Path,
    raw_dir: Path,
    canonical_path: Path,
    unseen_summary: Dict[str, dict],
    selected_unseen: List[str],
    available_ciphers: List[str],
    tag: str,
    train_file_source: str
) -> Path:
    manifest = {
        "tag": tag,
        "canonical_dataset": canonical_path.name,
        "raw_dir": str(raw_dir),
        "train_file_source": train_file_source,
        "selected_unseen_ciphers": selected_unseen,
        "available_ciphers": available_ciphers,
        "unseen_sets": unseen_summary,
        "expected_processed_outputs": [
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "split_indices.json",
            "split_summary.json",
        ],
    }

    manifest_path = manifests_dir / f"dataset_manifest_{tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def find_train_files(datasets_dir: Path, raw_dir: Path) -> Tuple[List[Path], str]:
    raw_files = collect_train_files(raw_dir)
    if raw_files:
        return raw_files, "datasets/raw"

    root_files = collect_train_files(datasets_dir)
    if root_files:
        return root_files, "datasets root"

    return [], "not found"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AAAI datasets with configurable unseen cipher holdouts"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="datasets",
        help="Directory containing dataset folders/files"
    )
    parser.add_argument(
        "--canonical_name",
        type=str,
        default="cryptospec_dataset.jsonl",
        help="Canonical merged dataset filename in raw/"
    )
    parser.add_argument(
        "--unseen_ciphers",
        nargs="+",
        default=["lea", "rectangle", "xtea"],
        help="Cipher names to export into unseen/ as separate dataset files"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Tag used in manifest filename"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them into raw/"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets_dir = (root / args.datasets_dir).resolve()

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    paths = ensure_dirs(datasets_dir)

    train_files, train_source = find_train_files(datasets_dir, paths["raw"])
    if not train_files:
        raise FileNotFoundError(
            f"No *_train.jsonl files found in either:\n"
            f"  - {datasets_dir}\n"
            f"  - {paths['raw']}\n"
            f"Make sure the per-cipher files exist before running this script."
        )

    cipher_map = collect_cipher_files(train_files)
    available_ciphers = sorted(cipher_map.keys())

    unseen_ciphers = [c.lower() for c in args.unseen_ciphers]
    unknown = [c for c in unseen_ciphers if c not in available_ciphers]
    if unknown:
        raise ValueError(
            f"Unknown unseen cipher(s): {unknown}. "
            f"Available ciphers: {available_ciphers}"
        )

    print(f"Found {len(train_files)} train JSONL files from {train_source}.")
    print(f"Available ciphers: {', '.join(available_ciphers)}")
    print(f"Selected unseen ciphers: {', '.join(unseen_ciphers)}")

    if train_source == "datasets root":
        copy_or_move_files(train_files, paths["raw"], move=args.move)
        print(f"Copied/moved raw files into {paths['raw']}")
    else:
        print(f"Using existing raw files in {paths['raw']}")

    canonical_path, canonical_count, canonical_source = ensure_canonical_dataset(
        paths["raw"],
        canonical_name=args.canonical_name
    )
    print(f"Canonical dataset: {canonical_path.name} ({canonical_count} lines, {canonical_source})")

    unseen_summary = write_unseen_sets(paths["raw"], paths["unseen"], unseen_ciphers)
    print(f"Prepared unseen sets in {paths['unseen']}")

    manifest_path = write_manifest(
        manifests_dir=paths["manifests"],
        raw_dir=paths["raw"],
        canonical_path=canonical_path,
        unseen_summary=unseen_summary,
        selected_unseen=unseen_ciphers,
        available_ciphers=available_ciphers,
        tag=args.tag,
        train_file_source=train_source,
    )
    print(f"Manifest written to {manifest_path}")

    print("\nUnseen datasets created:")
    for cipher, info in unseen_summary.items():
        print(f"  - {cipher}: {info['num_examples']} examples -> {info['output_file']}")

    print("\nProcessed directory is ready for the next split script.")
    print("Done.")


if __name__ == "__main__":
    main()
