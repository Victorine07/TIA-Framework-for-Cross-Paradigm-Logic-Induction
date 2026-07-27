#!/usr/bin/env python3
"""
metadata_ablation.py
Analyze the five metadata strategies (none / full / structured / algorithmic / alljson)
for each model, computing SV/SM/VC on the test split and per-tier ΔVC vs baseline.

Reads:  results/**/*.jsonl  (per-example records, for tier-level breakdown)
        results/**/overall__*.json (for aggregate ablation table)
Writes: reports/paper/tables/metadata_ablation.tex
        reports/paper/tables/metadata_ablation_tier.tex
        reports/metadata_ablation.json

Usage:
    python src/analysis/metadata_ablation.py
    python src/analysis/metadata_ablation.py --dataset test --mode zero_shot
    
Produces the ablation tables for Section 3: Tiered Isomorphic Alignment - Metadata Strategy Ablation and the corresponding rows in Section 6: Results and Discussion.
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
    load_overall_summaries, load_per_example_records, fmt
)

RESULTS_ROOT = PROJECT_ROOT / "results"
STRATEGIES = ["none", "algorithmic", "structured", "full", "alljson"]
TIERS = ["T1", "T2", "T3", "T4"]


def ablation_table(rows: list[dict], model: str, dataset: str, mode: str) -> list[dict]:
    """Build per-strategy rows for one (model, dataset, mode) combination."""
    result = []
    base_vc = None
    base_ov = None
    # Extract none baseline first
    for strategy in STRATEGIES:
        match = next(
            (r for r in rows if r["model"] == model and r["dataset"] == dataset
             and r["mode"] == mode and r["strategy"] == strategy), None
        )
        sv = match["avg_sv"] if match else None
        sm = match["avg_sm"] if match else None
        vc = match["avg_vc"] if match else None
        ov = match["avg_overall"] if match else None
        if strategy == "none":
            base_vc = vc
            base_ov = ov
        result.append({
            "model": model,
            "mode": mode,
            "dataset": dataset,
            "strategy": strategy,
            "sv": sv, "sm": sm, "vc": vc, "overall": ov,
            "delta_vc": (vc - base_vc) if (vc is not None and base_vc is not None) else None,
            "delta_overall": (ov - base_ov) if (ov is not None and base_ov is not None) else None,
        })
    return result


def tex_ablation_table(model_rows: dict[str, list[dict]], out: Path) -> None:
    """model_rows: {model: [strategy rows]}"""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-shot metadata strategy ablation (test split). "
        r"$\Delta$ = gain in Overall vs \textit{none} baseline. "
        r"Best per model in \textbf{bold}.}",
        r"\label{tab:metadata_ablation}",
        r"\small",
        r"\begin{tabular}{llccccr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Strategy} & \textbf{SV} & \textbf{SM} & "
        r"\textbf{VC} & \textbf{Overall} & \textbf{$\Delta$Ovrl} \\",
        r"\midrule",
    ]
    for model, strat_rows in sorted(model_rows.items()):
        # Find best overall
        best_ov = max((r["overall"] or 0) for r in strat_rows)
        first = True
        for r in strat_rows:
            model_str = model if first else ""
            first = False
            is_best = r["overall"] is not None and abs(r["overall"] - best_ov) < 1e-6
            ov_str = f"\\textbf{{{fmt(r['overall'])}}}" if is_best else fmt(r["overall"])
            delta_str = "---"
            if r["delta_overall"] is not None:
                delta_str = f"{'+' if r['delta_overall'] >= 0 else ''}{r['delta_overall']:.3f}"
            lines.append(
                f"{model_str} & {r['strategy']} & {fmt(r['sv'])} & {fmt(r['sm'])} & "
                f"{fmt(r['vc'])} & {ov_str} & {delta_str} \\\\"
            )
        lines.append(r"\midrule")
    # Remove trailing \midrule
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def tier_ablation(records: list[dict], model: str, dataset: str, mode: str) -> dict:
    """
    For each (strategy, tier), compute mean VC from per-example records.
    Returns {strategy: {tier: mean_vc}}.
    """
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["_model"] != model:
            continue
        if rec.get("dataset") != dataset:
            continue
        if rec["_mode"] != mode:
            continue
        tier = rec["_tier"]
        strategy = rec["_strategy"]
        vc = rec["_vc"]
        if vc is not None and tier in TIERS:
            bucket[strategy][tier].append(float(vc))
    result = {}
    for strategy in STRATEGIES:
        result[strategy] = {}
        for tier in TIERS:
            vals = bucket[strategy].get(tier, [])
            result[strategy][tier] = (sum(vals) / len(vals)) if vals else None
    return result


def tex_tier_ablation(tier_data: dict, model: str, out: Path) -> None:
    """Write per-tier VC ablation as LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Tier-wise VC by metadata strategy (zero-shot, test split) for {model}. "
        r"$\Delta$ = gain vs \textit{none} baseline.}}",
        rf"\label{{tab:tier_ablation_{model.replace('.','').replace('-','_')}}}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{T1 VC} & $\Delta$ & \textbf{T2 VC} & $\Delta$ "
        r"& \textbf{T3 VC} & $\Delta$ & \textbf{T4 VC} & $\Delta$ \\",
        r"\midrule",
    ]
    base = tier_data.get("none", {})
    for strategy in STRATEGIES:
        s_data = tier_data.get(strategy, {})
        cols = [strategy]
        for tier in TIERS:
            vc = s_data.get(tier)
            bvc = base.get(tier)
            cols.append(fmt(vc))
            if strategy == "none" or bvc is None or vc is None:
                cols.append("---")
            else:
                d = vc - bvc
                cols.append(f"{'+' if d >= 0 else ''}{d:.3f}")
        lines.append(" & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze metadata strategy ablation.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--mode", default="zero_shot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/5] Loading overall summaries from: {args.results_dir}")
    rows = load_overall_summaries(Path(args.results_dir))
    print(f"  {len(rows)} dataset-level rows loaded")

    print(f"[CHECKPOINT 2/5] Building ablation tables (dataset={args.dataset}, mode={args.mode})")
    models = sorted(set(r["model"] for r in rows
                        if r["dataset"] == args.dataset and r["mode"] == args.mode))
    print(f"  Models found: {models}")

    model_rows: dict[str, list[dict]] = {}
    for model in models:
        abl = ablation_table(rows, model, args.dataset, args.mode)
        model_rows[model] = abl
        for r in abl:
            print(f"  {model} | {r['strategy']:12s} | ov={fmt(r['overall'])} Δ={fmt(r['delta_overall'])}")

    print(f"[CHECKPOINT 3/5] Writing aggregate ablation LaTeX table")
    tex_ablation_table(model_rows, out_dir / "metadata_ablation.tex")
    print(f"  Wrote metadata_ablation.tex")

    print(f"[CHECKPOINT 4/5] Loading per-example records for tier-level ablation")
    records = load_per_example_records(Path(args.results_dir))
    print(f"  {len(records)} per-example records loaded")

    for model in models:
        td = tier_ablation(records, model, args.dataset, args.mode)
        safe = model.replace(".", "").replace("-", "_")
        tex_tier_ablation(td, model, out_dir / f"metadata_ablation_tier_{safe}.tex")
        print(f"  Wrote metadata_ablation_tier_{safe}.tex")

    print(f"[CHECKPOINT 5/5] Writing JSON summary")
    summary = {"model_ablation": {m: v for m, v in model_rows.items()}}
    json_out = PROJECT_ROOT / "reports" / "metadata_ablation.json"
    json_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Metadata ablation tables written to: {out_dir}")


if __name__ == "__main__":
    main()
