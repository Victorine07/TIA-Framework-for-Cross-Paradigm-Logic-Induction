#!/usr/bin/env python3
"""
compare_modes.py
Quantify how much each learning regime (zero-shot, few-shot, SFT) improves
over the zero-shot baseline.  Reads overall summary JSONs (fastest path since
they have pre-aggregated metrics).

Writes: reports/paper/tables/mode_comparison.tex
        reports/mode_comparison.json

Usage:
    python src/analysis/compare_modes.py
    python src/analysis/compare_modes.py --dataset test

Produces the learning-regime comparison table for Section 6: Results and Discussion (zero-shot vs. few-shot vs. SFT deltas).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import load_overall_summaries, mode_display, fmt

RESULTS_ROOT = PROJECT_ROOT / "results"

# Which strategy to treat as "best" for each regime when aggregating.
# We dynamically compute this, but need a fallback ordering.
STRATEGY_PREFERENCE_ORDER = ["none", "structured", "full", "algorithmic", "alljson"]


def best_for_key(rows: list[dict], model: str, mode: str, dataset: str) -> dict | None:
    """Return the row with the highest avg_overall for a given (model, mode, dataset)."""
    candidates = [
        r for r in rows
        if r["model"] == model and r["mode"] == mode and r["dataset"] == dataset
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["avg_overall"] or 0))


def compute_comparison(rows: list[dict], dataset: str) -> list[dict]:
    """
    Build comparison rows for each model, picking the best strategy per regime.
    Returns one row per model with fields for each regime.
    """
    rows_ds = [r for r in rows if r["dataset"] == dataset]
    models = sorted(set(r["model"] for r in rows_ds))
    result = []
    for model in models:
        # Zero-shot best
        zs = best_for_key(rows_ds, model, "zero_shot", dataset)
        # Few-shot k=3 best
        fs3 = best_for_key(rows_ds, model, "fewshot_k3", dataset)
        # Few-shot k=5 best
        fs5 = best_for_key(rows_ds, model, "fewshot_k5", dataset)
        # SFT none
        sft_none = next(
            (r for r in rows_ds if r["model"] == model and r["mode"] == "finetuned"
             and r["strategy"] == "none"), None
        )
        # SFT structured
        sft_struct = next(
            (r for r in rows_ds if r["model"] == model and r["mode"] == "finetuned"
             and r["strategy"] == "structured"), None
        )
        # SFT best
        sft = best_for_key(rows_ds, model, "finetuned", dataset)

        def _vc(r): return r["avg_vc"] if r and r["avg_vc"] is not None else None
        def _ov(r): return r["avg_overall"] if r and r["avg_overall"] is not None else None
        def _sv(r): return r["avg_sv"] if r and r["avg_sv"] is not None else None
        def _sm(r): return r["avg_sm"] if r and r["avg_sm"] is not None else None

        def ratio(a, b):
            if a is not None and b and b > 0:
                return round(a / b, 2)
            return None

        zs_vc = _vc(zs)
        zs_ov = _ov(zs)
        def delta(val, base): return (val - base) if (val is not None and base is not None) else None
        result.append({
            "model": model,
            "dataset": dataset,
            # Zero-shot (best strategy)
            "zs_strategy": zs["strategy"] if zs else "---",
            "zs_sv": _sv(zs), "zs_sm": _sm(zs), "zs_vc": zs_vc, "zs_overall": zs_ov,
            # Few-shot k=3
            "fs3_strategy": fs3["strategy"] if fs3 else "---",
            "fs3_sv": _sv(fs3), "fs3_sm": _sm(fs3), "fs3_vc": _vc(fs3), "fs3_overall": _ov(fs3),
            "fs3_delta_sv": delta(_sv(fs3), _sv(zs)),
            "fs3_delta_sm": delta(_sm(fs3), _sm(zs)),
            "fs3_delta_vc": delta(_vc(fs3), zs_vc),
            "fs3_delta_overall": delta(_ov(fs3), zs_ov),
            "fs3_ratio_overall": ratio(_ov(fs3), zs_ov),
            # Few-shot k=5
            "fs5_strategy": fs5["strategy"] if fs5 else "---",
            "fs5_sv": _sv(fs5), "fs5_sm": _sm(fs5), "fs5_vc": _vc(fs5), "fs5_overall": _ov(fs5),
            "fs5_delta_sv": delta(_sv(fs5), _sv(zs)),
            "fs5_delta_sm": delta(_sm(fs5), _sm(zs)),
            "fs5_delta_vc": delta(_vc(fs5), zs_vc),
            "fs5_delta_overall": delta(_ov(fs5), zs_ov),
            "fs5_ratio_overall": ratio(_ov(fs5), zs_ov),
            # SFT best
            "sft_strategy": sft["strategy"] if sft else "---",
            "sft_sv": _sv(sft), "sft_sm": _sm(sft), "sft_vc": _vc(sft), "sft_overall": _ov(sft),
            "sft_delta_sv": delta(_sv(sft), _sv(zs)),
            "sft_delta_sm": delta(_sm(sft), _sm(zs)),
            "sft_delta_vc": delta(_vc(sft), zs_vc),
            "sft_delta_overall": delta(_ov(sft), zs_ov),
            "sft_ratio_overall": ratio(_ov(sft), zs_ov),
        })
    return result


def tex_mode_comparison(comp_rows: list[dict], out: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Learning Regime Comparison (test split). ZS = best zero-shot strategy; "
        r"FS-3 / FS-5 = few-shot $k$=3/$k$=5 (none strategy); SFT = best fine-tuned. "
        r"$\times$ = Overall ratio vs ZS baseline.}",
        r"\label{tab:mode_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{ZS (best)}} "
        r"& \multicolumn{2}{c}{\textbf{FS-3}} & \multicolumn{2}{c}{\textbf{FS-5}} "
        r"& \multicolumn{2}{c}{\textbf{SFT (best)}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r" & Ovrl & --- & Ovrl & $\times$ ZS & Ovrl & $\times$ ZS & Ovrl & $\times$ ZS \\",
        r"\midrule",
    ]
    for r in comp_rows:
        lines.append(
            f"{r['model']} & {fmt(r['zs_overall'])} & {r['zs_strategy']} "
            f"& {fmt(r['fs3_overall'])} & {fmt(r['fs3_ratio_overall'])}$\\times$ "
            f"& {fmt(r['fs5_overall'])} & {fmt(r['fs5_ratio_overall'])}$\\times$ "
            f"& {fmt(r['sft_overall'])} & {fmt(r['sft_ratio_overall'])}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def _sgn(v) -> str:
    if v is None: return "---"
    return f"{'+' if v >= 0 else ''}{v:.3f}"


def tex_metric_delta_table(comp_rows: list[dict], metric: str, out: Path) -> None:
    """Delta table for one metric (sv / sm / vc) across regimes."""
    label = metric.upper()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{label} gains over zero-shot baseline (test split). "
        rf"$\Delta$ = absolute improvement; $\times$ = ratio vs ZS for Overall.}}",
        rf"\label{{tab:{metric}_deltas}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"\textbf{{Model}} & \textbf{{ZS {label}}} & \textbf{{FS-3 $\Delta${label}}} "
        rf"& \textbf{{FS-5 $\Delta${label}}} & \textbf{{SFT $\Delta${label}}} "
        rf"& \textbf{{SFT $\times$Overall}} \\",
        r"\midrule",
    ]
    for r in comp_rows:
        lines.append(
            f"{r['model']} & {fmt(r[f'zs_{metric}'])} "
            f"& {_sgn(r[f'fs3_delta_{metric}'])} "
            f"& {_sgn(r[f'fs5_delta_{metric}'])} "
            f"& {_sgn(r[f'sft_delta_{metric}'])} "
            f"& {fmt(r['sft_ratio_overall'])}$\\times$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def tex_full_delta_table(comp_rows: list[dict], out: Path) -> None:
    """Comprehensive delta table across all three metrics and all regimes."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Absolute gains ($\Delta$) over zero-shot baseline across all metrics and "
        r"regimes (test split). Best strategy per model per regime used for ZS; SFT = "
        r"\textit{none} strategy unless \textit{structured} is better.}",
        r"\label{tab:full_deltas}",
        r"\small",
        r"\begin{tabular}{ll ccc ccc ccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{ZS base}} "
        r"& \multicolumn{3}{c}{\textbf{FS-3 $\Delta$}} "
        r"& \multicolumn{3}{c}{\textbf{FS-5 $\Delta$}} "
        r"& \multicolumn{3}{c}{\textbf{SFT $\Delta$}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r" & Ovrl & SV & SM & VC & SV & SM & VC & SV & SM & VC \\",
        r"\midrule",
    ]
    for r in comp_rows:
        lines.append(
            f"{r['model']} & {fmt(r['zs_overall'])} "
            f"& {_sgn(r['fs3_delta_sv'])} & {_sgn(r['fs3_delta_sm'])} & {_sgn(r['fs3_delta_vc'])} "
            f"& {_sgn(r['fs5_delta_sv'])} & {_sgn(r['fs5_delta_sm'])} & {_sgn(r['fs5_delta_vc'])} "
            f"& {_sgn(r['sft_delta_sv'])} & {_sgn(r['sft_delta_sm'])} & {_sgn(r['sft_delta_vc'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare learning regimes on SV/SM/VC.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    parser.add_argument("--dataset", default="test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/4] Loading overall summary files from: {args.results_dir}")
    rows = load_overall_summaries(Path(args.results_dir))
    print(f"  Loaded {len(rows)} dataset-level rows across all runs")

    print(f"[CHECKPOINT 2/4] Computing regime comparison for dataset='{args.dataset}'")
    comp_rows = compute_comparison(rows, args.dataset)
    for r in comp_rows:
        print(
            f"  {r['model']}: ZS={fmt(r['zs_overall'])} (strat={r['zs_strategy']}) "
            f"FS3={fmt(r['fs3_overall'])} ({fmt(r['fs3_ratio_overall'])}×) "
            f"SFT={fmt(r['sft_overall'])} ({fmt(r['sft_ratio_overall'])}×)"
        )

    print(f"[CHECKPOINT 3/4] Writing LaTeX tables")
    tex_mode_comparison(comp_rows, out_dir / "mode_comparison.tex")
    print(f"  Wrote mode_comparison.tex")
    for metric in ["sv", "sm", "vc"]:
        tex_metric_delta_table(comp_rows, metric, out_dir / f"{metric}_deltas.tex")
        print(f"  Wrote {metric}_deltas.tex")
    tex_full_delta_table(comp_rows, out_dir / "full_deltas.tex")
    print(f"  Wrote full_deltas.tex")

    print(f"[CHECKPOINT 4/4] Writing JSON summary")
    json_out = PROJECT_ROOT / "reports" / "mode_comparison.json"
    json_out.write_text(json.dumps(comp_rows, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Mode comparison written to: {out_dir}")


if __name__ == "__main__":
    main()
