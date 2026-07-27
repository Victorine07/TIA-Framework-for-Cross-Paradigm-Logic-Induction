#!/usr/bin/env python3
"""
summarize_datasets.py
Compute ground-truth dataset composition statistics 

Reads: datasets/raw/*.jsonl, datasets/processed/{train,val,test}.jsonl,
       datasets/unseen/*.jsonl
Writes: reports/paper/tables/dataset_composition.tex
        reports/paper/tables/dataset_splits.tex
        reports/dataset_composition.json

Usage:
    python src/analysis/summarize_datasets.py
    python src/analysis/summarize_datasets.py --output-dir reports/paper/tables
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))

CIPHER_FAMILY_MAP: dict[str, str] = {
    # ARX
    "speck": "ARX", "SPECK": "ARX",
    "SPARX": "ARX", "sparx": "ARX",
    "cham": "ARX", "CHAM": "ARX",
    # Feistel
    "SIMON": "Feistel", "simon": "Feistel",
    "SIMECK": "Feistel", "simeck": "Feistel",
    "HIGHT": "Feistel", "hight": "Feistel",
    # SPN
    "PRESENT": "SPN", "present": "SPN",
    "GIFT": "SPN", "gift": "SPN",
    "SKINNY": "SPN", "skinny": "SPN",
    "RECTANGLE": "SPN", "rectangle": "SPN",
    # Permutation / AEAD
    "ASCON": "Permutation/AEAD",
    "GIFT-COFB": "Permutation/AEAD",
    # Unseen
    "lea": "ARX", "LEA": "ARX",
    "XTEA": "Feistel", "xtea": "Feistel",
}


def count_jsonl(path: Path) -> dict:
    """Count tier × cipher × family from one JSONL file."""
    tiers: Counter = Counter()
    families: Counter = Counter()
    ciphers: Counter = Counter()
    tier_x_family: Counter = Counter()
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            m = rec.get("metadata", {})
            tier = m.get("tier", "UNK")
            family = m.get("family", "UNK")
            cipher = m.get("cipher", "UNK")
            tiers[tier] += 1
            families[family] += 1
            ciphers[cipher] += 1
            tier_x_family[(tier, family)] += 1
            n += 1
    return dict(tiers=dict(tiers), families=dict(families),
                ciphers=dict(ciphers), tier_x_family={str(k): v for k, v in tier_x_family.items()},
                total=n)


def load_split(split_path: Path) -> list[dict]:
    records = []
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def cipher_table_rows(records: list[dict]) -> dict:
    """Build {cipher: {T1:n, T2:n, T3:n, T4:n, family:str, variants:set}} mapping."""
    table: dict = defaultdict(lambda: {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "variants": set(), "family": "?"})
    for rec in records:
        m = rec.get("metadata", {})
        cipher = m.get("cipher", "UNK")
        tier = m.get("tier", "UNK")
        family = m.get("family", cipher)
        variant = m.get("cipher_variant") or str(m.get("variant", ""))
        if tier in ("T1", "T2", "T3", "T4"):
            table[cipher][tier] += 1
        table[cipher]["family"] = family
        if variant:
            table[cipher]["variants"].add(variant)
    return dict(table)


def tex_cipher_table(rows: dict, split_name: str, out: Path) -> None:
    """Write a LaTeX table: Cipher | Family | Variants | T1 | T2 | T3 | T4 | Total"""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{TIA Dataset Composition — {split_name} split (components by tier and cipher)}}",
        rf"\label{{tab:dataset_{split_name.lower()}}}",
        r"\small",
        r"\begin{tabular}{llcccccr}",
        r"\toprule",
        r"\textbf{Cipher} & \textbf{Family} & \textbf{Var.} & \textbf{T1} & \textbf{T2} & \textbf{T3} & \textbf{T4} & \textbf{Total} \\",
        r"\midrule",
    ]
    family_order = ["ARX", "Feistel", "SPN", "Permutation", "Permutation/AEAD"]
    by_family: dict[str, list] = defaultdict(list)
    for cipher, info in sorted(rows.items()):
        by_family[info["family"]].append(cipher)

    grand_t = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    first_family = True
    for fam in family_order:
        ciphers = by_family.get(fam, [])
        if not ciphers:
            continue
        if not first_family:
            lines.append(r"\midrule")
        first_family = False
        for cipher in sorted(ciphers):
            info = rows[cipher]
            t1, t2, t3, t4 = info["T1"], info["T2"], info["T3"], info["T4"]
            total = t1 + t2 + t3 + t4
            n_variants = len(info["variants"]) or 1
            grand_t["T1"] += t1; grand_t["T2"] += t2
            grand_t["T3"] += t3; grand_t["T4"] += t4
            lines.append(
                f"{cipher} & {fam} & {n_variants} & {t1} & {t2} & {t3} & {t4} & {total} \\\\"
            )
    gt = sum(grand_t.values())
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & & & \\textbf{{{grand_t['T1']}}} & \\textbf{{{grand_t['T2']}}} & "
        f"\\textbf{{{grand_t['T3']}}} & \\textbf{{{grand_t['T4']}}} & \\textbf{{{gt}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def tex_split_table(stats: dict, out: Path) -> None:
    """Write a LaTeX split summary table: Split | T1 | T2 | T3 | T4 | Total"""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TIA Dataset Split Distribution by Tier}",
        r"\label{tab:dataset_splits}",
        r"\small",
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"\textbf{Split} & \textbf{T1} & \textbf{T2} & \textbf{T3} & \textbf{T4} & \textbf{Total} \\",
        r"\midrule",
    ]
    for split, info in stats.items():
        t = info["tiers"]
        total = info["total"]
        lines.append(
            f"{split} & {t.get('T1',0)} & {t.get('T2',0)} & {t.get('T3',0)} & {t.get('T4',0)} & {total} \\\\"
        )
    # Totals row
    all_t = Counter()
    all_n = 0
    for split, info in stats.items():
        if "unseen" not in split.lower():
            for tier, n in info["tiers"].items():
                all_t[tier] += n
            all_n += info["total"]
    lines += [
        r"\midrule",
        f"\\textbf{{Train+Val+Test}} & \\textbf{{{all_t.get('T1',0)}}} & \\textbf{{{all_t.get('T2',0)}}} & "
        f"\\textbf{{{all_t.get('T3',0)}}} & \\textbf{{{all_t.get('T4',0)}}} & \\textbf{{{all_n}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset composition statistics.")
    parser.add_argument("--datasets-dir", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    args = parser.parse_args()

    ddir = Path(args.datasets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/5] Reading datasets from: {ddir}")

    # ── Processed splits ─────────────────────────────────────────────────────
    split_stats: dict = {}
    for split in ["train", "val", "test"]:
        p = ddir / "processed" / f"{split}.jsonl"
        if not p.exists():
            print(f"  [WARNING] {p} not found, skipping")
            continue
        info = count_jsonl(p)
        split_stats[split] = info
        print(f"  {split}: total={info['total']}, tiers={info['tiers']}")

    # ── Unseen splits ─────────────────────────────────────────────────────────
    for p in sorted((ddir / "unseen").glob("*.jsonl")):
        info = count_jsonl(p)
        split_stats[f"unseen_{p.stem.replace('_dataset','')}"] = info
        print(f"  unseen/{p.name}: total={info['total']}, tiers={info['tiers']}")

    print(f"[CHECKPOINT 2/5] Computing per-cipher tier breakdown from train split")
    train_records = load_split(ddir / "processed" / "train.jsonl")
    train_rows = cipher_table_rows(train_records)
    print(f"  Ciphers in train: {sorted(train_rows.keys())}")

    print(f"[CHECKPOINT 3/5] Computing per-cipher breakdown from all splits combined")
    all_records: list[dict] = []
    for split in ["train", "val", "test"]:
        p = ddir / "processed" / f"{split}.jsonl"
        if p.exists():
            all_records.extend(load_split(p))
    for p in sorted((ddir / "unseen").glob("*.jsonl")):
        all_records.extend(load_split(p))
    all_rows = cipher_table_rows(all_records)

    print(f"[CHECKPOINT 4/5] Writing LaTeX tables")
    tex_cipher_table(train_rows, "Train", out_dir / "dataset_train_cipher.tex")
    print(f"  Wrote dataset_train_cipher.tex")
    tex_cipher_table(all_rows, "All", out_dir / "dataset_all_cipher.tex")
    print(f"  Wrote dataset_all_cipher.tex")
    tex_split_table(split_stats, out_dir / "dataset_splits.tex")
    print(f"  Wrote dataset_splits.tex")

    print(f"[CHECKPOINT 5/5] Writing JSON summary")
    summary = {
        "split_stats": split_stats,
        "train_by_cipher": {k: {kk: list(vv) if isinstance(vv, set) else vv
                                for kk, vv in v.items()} for k, v in train_rows.items()},
        "all_by_cipher": {k: {kk: list(vv) if isinstance(vv, set) else vv
                               for kk, vv in v.items()} for k, v in all_rows.items()},
    }
    json_out = PROJECT_ROOT / "reports" / "dataset_composition.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Tables written to: {out_dir}")


if __name__ == "__main__":
    main()
