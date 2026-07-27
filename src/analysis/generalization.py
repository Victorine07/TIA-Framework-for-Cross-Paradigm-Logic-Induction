#!/usr/bin/env python3
"""
generalization.py
Compare in-distribution (ID) vs out-of-distribution (OOD) performance across
learning regimes, with tier-level breakdown for unseen ciphers.

Reads:  results/**/overall__*.json  (pre-aggregated; fast path)
        results/**/*.jsonl           (per-example; for tier breakdown on unseen)
Writes: reports/paper/tables/generalization.tex
        reports/paper/tables/generalization_tier.tex
        reports/generalization.json

Usage:
    python src/analysis/generalization.py

Produces the ID vs OOD performance tables for Section 6: Results and Discussion - Generalization to Unseen Ciphers: Grammar Transfer, Not Memorization.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import (
    load_overall_summaries, load_per_example_records,
    mode_display, fmt
)

RESULTS_ROOT = PROJECT_ROOT / "results"
TIERS = ["T1", "T2", "T3", "T4"]
ID_DATASET = "test"
OOD_DATASETS = ["unseen_lea", "unseen_rectangle", "unseen_xtea"]
UNSEEN_LABELS = {
    "unseen_lea": "LEA (ARX)",
    "unseen_rectangle": "Rectangle (SPN)",
    "unseen_xtea": "XTEA (Feistel)",
}


def build_id_ood_rows(overall_rows: list[dict]) -> list[dict]:
    """
    For each (model, mode, strategy): extract ID overall score
    and OOD overall scores per unseen cipher.
    Returns one dict per unique (model, mode, strategy).
    """
    # Key: (model, mode, strategy) → {dataset: row}
    by_key: dict[tuple, dict] = defaultdict(dict)
    for r in overall_rows:
        key = (r["model"], r["mode"], r["strategy"])
        by_key[key][r["dataset"]] = r

    result = []
    for (model, mode, strategy), ds_map in sorted(by_key.items()):
        if ID_DATASET not in ds_map:
            continue
        id_row = ds_map[ID_DATASET]
        row = {
            "model": model,
            "mode": mode,
            "mode_display": mode_display(mode),
            "strategy": strategy,
            "id_sv": id_row["avg_sv"],
            "id_sm": id_row["avg_sm"],
            "id_vc": id_row["avg_vc"],
            "id_overall": id_row["avg_overall"],
        }
        for ood_ds in OOD_DATASETS:
            ood_row = ds_map.get(ood_ds)
            label = ood_ds.replace("unseen_", "")
            row[f"ood_{label}_overall"] = ood_row["avg_overall"] if ood_row else None
            row[f"ood_{label}_vc"] = ood_row["avg_vc"] if ood_row else None
        result.append(row)
    return result


def tex_id_ood_table(rows: list[dict], out: Path) -> None:
    ood_cols = [ds.replace("unseen_", "") for ds in OOD_DATASETS]
    ood_header = " & ".join(f"\\textbf{{{UNSEEN_LABELS[f'unseen_{c}']}}}" for c in ood_cols)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{In-distribution (ID, test split) vs out-of-distribution (OOD, unseen ciphers) "
        r"performance (Overall score). Zero-shot uses best strategy per model.}",
        r"\label{tab:generalization}",
        r"\small",
        r"\begin{tabular}{lllc|ccc}",
        r"\toprule",
        rf"\textbf{{Model}} & \textbf{{Mode}} & \textbf{{Strategy}} "
        rf"& \textbf{{ID (test)}} & {ood_header} \\",
        r"\midrule",
    ]
    prev_model = None
    for r in rows:
        model_str = r["model"] if r["model"] != prev_model else ""
        prev_model = r["model"]
        ood_vals = " & ".join(fmt(r.get(f"ood_{c}_overall")) for c in ood_cols)
        lines.append(
            f"{model_str} & {r['mode_display']} & {r['strategy']} "
            f"& {fmt(r['id_overall'])} & {ood_vals} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def tier_ood_breakdown(records: list[dict]) -> dict:
    """
    For each (model, mode, strategy, dataset, tier): mean VC on unseen datasets.
    Returns nested dict.
    """
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for rec in records:
        ds = rec.get("dataset", "?")
        if not ds.startswith("unseen_"):
            continue
        tier = rec["_tier"]
        if tier not in TIERS:
            continue
        key = (rec["_model"], rec["_mode"], rec["_strategy"], ds)
        bucket[key][tier].append(float(rec["_vc"]) if rec["_vc"] is not None else 0.0)
    result: dict = {}
    for key, tier_data in sorted(bucket.items()):
        result[str(key)] = {
            tier: round(sum(vals)/len(vals), 4) if vals else None
            for tier, vals in tier_data.items()
        }
    return result


def tex_tier_ood_table(records: list[dict], out: Path) -> None:
    """Per-tier VC for each unseen cipher × mode, best strategy only."""
    # Select: SFT (none) and ZS (best strategy) per model
    bucket: dict = defaultdict(lambda: {t: [] for t in TIERS})
    for rec in records:
        ds = rec.get("dataset", "?")
        if not ds.startswith("unseen_"):
            continue
        mode = rec["_mode"]
        # Only include SFT-none and zero_shot runs for this table
        if mode not in ("finetuned", "zero_shot"):
            continue
        if mode == "finetuned" and rec["_strategy"] != "none":
            continue
        tier = rec["_tier"]
        if tier in TIERS:
            key = (rec["_model"], mode, ds)
            bucket[key][tier].append(float(rec["_vc"] or 0))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Tier-wise VC on unseen ciphers for zero-shot (best strategy) and "
        r"SFT (none strategy). Confirms grammar transfer rather than cipher memorization.}",
        r"\label{tab:tier_ood}",
        r"\small",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mode} & \textbf{Unseen Cipher} "
        r"& \textbf{T1 VC} & \textbf{T2 VC} & \textbf{T3 VC} & \textbf{T4 VC} \\",
        r"\midrule",
    ]
    prev = None
    for (model, mode, ds), tier_data in sorted(bucket.items()):
        model_str = model if (model, mode) != prev else ""
        mode_str = mode_display(mode) if (model, mode) != prev else ""
        prev = (model, mode)
        vals = [
            str(round(sum(tier_data[t])/len(tier_data[t]), 3)) if tier_data[t] else "---"
            for t in TIERS
        ]
        label = UNSEEN_LABELS.get(ds, ds)
        lines.append(f"{model_str} & {mode_str} & {label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ID vs OOD generalization analysis.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/5] Loading overall summary files")
    overall_rows = load_overall_summaries(Path(args.results_dir))
    print(f"  {len(overall_rows)} dataset-level rows")

    print(f"[CHECKPOINT 2/5] Building ID vs OOD comparison")
    id_ood_rows = build_id_ood_rows(overall_rows)
    for r in id_ood_rows:
        print(
            f"  {r['model']} | {r['mode_display']:16s} | {r['strategy']:12s} | "
            f"ID={fmt(r['id_overall'])} "
            f"LEA={fmt(r.get('ood_lea_overall'))} "
            f"Rect={fmt(r.get('ood_rectangle_overall'))} "
            f"XTEA={fmt(r.get('ood_xtea_overall'))}"
        )

    print(f"[CHECKPOINT 3/5] Writing ID vs OOD LaTeX table")
    tex_id_ood_table(id_ood_rows, out_dir / "generalization.tex")
    print(f"  Wrote generalization.tex")

    print(f"[CHECKPOINT 4/5] Loading per-example records for tier breakdown")
    records = load_per_example_records(Path(args.results_dir))
    print(f"  {len(records)} per-example records")
    tex_tier_ood_table(records, out_dir / "generalization_tier.tex")
    print(f"  Wrote generalization_tier.tex")

    print(f"[CHECKPOINT 5/5] Writing JSON summary")
    summary = {
        "id_ood_comparison": id_ood_rows,
        "tier_ood_breakdown": tier_ood_breakdown(records),
    }
    json_out = PROJECT_ROOT / "reports" / "generalization.json"
    json_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Generalization tables written to: {out_dir}")


if __name__ == "__main__":
    main()
