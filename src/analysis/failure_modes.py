#!/usr/bin/env python3
"""
failure_modes.py
Aggregate failure category distributions and extract representative qualitative
failure examples for paper discussion.

Reads:  results/**/*.jsonl
Writes: reports/paper/tables/error_taxonomy.tex
        reports/failure_analysis.json
        reports/failure_examples.jsonl   (representative examples per category)

Failure categories:
    correct / syntax_error / type_mismatch / semantic_error / value_error

Usage:
    python src/analysis/failure_modes.py
    python src/analysis/failure_modes.py --dataset test --mode finetuned

Produces the error taxonomy reported in Section 6: Results and Discussion (Error Taxonomy: Tracing the Failure Trajectory).

"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import load_per_example_records, mode_display, fmt

RESULTS_ROOT = PROJECT_ROOT / "results"
FAILURE_CATS = ["correct", "syntax_error", "type_mismatch", "semantic_error", "value_error"]
TIERS = ["T1", "T2", "T3", "T4"]

# ── Interesting failure patterns to extract qualitatively ─────────────────────
QUALITATIVE_PATTERNS = {
    "high_sv_low_vc":   lambda r: (r["_sv"] or 0) >= 0.8 and (r["_vc"] or 1) <= 0.3,
    "high_vc_low_sm":   lambda r: (r["_vc"] or 0) >= 0.7 and (r["_sm"] or 1) <= 0.3,
    "t4_only_failure":  lambda r: r["_tier"] == "T4" and (r["_overall"] or 1) <= 0.5,
    "correct":          lambda r: r["_failure"] == "correct",
    "syntax_error":     lambda r: r["_failure"] == "syntax_error",
    "semantic_error":   lambda r: r["_failure"] == "semantic_error",
}


def count_failures(records: list[dict]) -> dict:
    """
    Count failure categories per (model, mode, dataset, tier).
    Returns nested dict: {(model,mode,dataset): {tier: Counter}}
    """
    bucket: dict = defaultdict(lambda: defaultdict(Counter))
    for rec in records:
        key = (rec["_model"], rec["_mode"], rec.get("dataset", "?"))
        tier = rec["_tier"]
        cat = rec["_failure"] or "unknown"
        bucket[key][tier][cat] += 1
    return bucket


def tex_error_taxonomy(records: list[dict], dataset: str, out: Path) -> None:
    """
    Write error taxonomy table for test split, grouped by model × mode.
    Columns: correct | syntax | type | semantic | value | total
    """
    target = [r for r in records if r.get("dataset") == dataset]
    # Aggregate by (model, mode)
    counts: dict = defaultdict(Counter)
    for rec in target:
        key = (rec["_model"], rec["_mode"])
        counts[key][rec["_failure"] or "unknown"] += 1

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Failure category distribution on \texttt{{{dataset}}} split ($n$ per condition).}}",
        r"\label{tab:error_taxonomy}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mode} & \textbf{Correct} & \textbf{Syntax} "
        r"& \textbf{Type} & \textbf{Semantic} & \textbf{Value} \\",
        r"\midrule",
    ]
    prev_model = None
    for (model, mode) in sorted(counts.keys()):
        c = counts[(model, mode)]
        total = sum(c.values())
        model_str = model if model != prev_model else ""
        prev_model = model
        lines.append(
            f"{model_str} & {mode_display(mode)} "
            f"& {c.get('correct',0)} & {c.get('syntax_error',0)} "
            f"& {c.get('type_mismatch',0)} & {c.get('semantic_error',0)} "
            f"& {c.get('value_error',0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def tex_tier_failure_table(records: list[dict], model: str, mode: str, dataset: str, out: Path) -> None:
    """Per-tier failure distribution for one (model, mode, dataset)."""
    target = [
        r for r in records
        if r["_model"] == model and r["_mode"] == mode and r.get("dataset") == dataset
    ]
    tier_counts: dict[str, Counter] = defaultdict(Counter)
    for rec in target:
        tier_counts[rec["_tier"]][rec["_failure"] or "unknown"] += 1

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Tier-wise failure distribution: {model} / {mode_display(mode)} / \texttt{{{dataset}}}}}",
        rf"\label{{tab:tier_failure_{model.replace('.','').replace('-','_')}_{mode}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Tier} & \textbf{Correct} & \textbf{Syntax} & \textbf{Type} "
        r"& \textbf{Semantic} & \textbf{Value} \\",
        r"\midrule",
    ]
    for tier in TIERS:
        c = tier_counts.get(tier, Counter())
        lines.append(
            f"{tier} & {c.get('correct',0)} & {c.get('syntax_error',0)} "
            f"& {c.get('type_mismatch',0)} & {c.get('semantic_error',0)} "
            f"& {c.get('value_error',0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def extract_qualitative_examples(records: list[dict]) -> dict[str, list[dict]]:
    """Extract one representative example per qualitative pattern per model."""
    examples: dict[str, list[dict]] = defaultdict(list)
    for pattern_name, pred_fn in QUALITATIVE_PATTERNS.items():
        by_model: dict[str, list] = defaultdict(list)
        for rec in records:
            if pred_fn(rec):
                by_model[rec["_model"]].append(rec)
        for model, model_recs in by_model.items():
            # Pick one representative (e.g. the one where the gap is largest for the pattern)
            if pattern_name == "high_sv_low_vc":
                pick = min(model_recs, key=lambda r: (r["_vc"] or 1))
            elif pattern_name == "t4_only_failure":
                pick = min(model_recs, key=lambda r: (r["_overall"] or 1))
            else:
                pick = model_recs[0]
            examples[pattern_name].append({
                "model": model,
                "mode": pick["_mode"],
                "strategy": pick["_strategy"],
                "tier": pick["_tier"],
                "cipher": pick["_cipher"],
                "family": pick["_family"],
                "failure_category": pick["_failure"],
                "sv": pick["_sv"],
                "sm": pick["_sm"],
                "vc": pick["_vc"],
                "overall": pick["_overall"],
                "reference_snippet": (pick.get("reference_output") or "")[:300],
                "prediction_snippet": (pick.get("prediction") or "")[:300],
                "example_id": pick.get("example_id"),
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure mode analysis and error taxonomy.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    parser.add_argument("--dataset", default="test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/5] Loading per-example records from: {args.results_dir}")
    records = load_per_example_records(Path(args.results_dir))
    print(f"  Loaded {len(records)} records total")

    test_records = [r for r in records if r.get("dataset") == args.dataset]
    print(f"  Records for dataset='{args.dataset}': {len(test_records)}")

    print(f"[CHECKPOINT 2/5] Printing failure distribution summary")
    bucket = count_failures(test_records)
    for (model, mode, dataset), tier_data in sorted(bucket.items()):
        total_c: Counter = Counter()
        for tc in tier_data.values():
            total_c.update(tc)
        print(
            f"  {model} | {mode_display(mode):16s} | correct={total_c.get('correct',0)} "
            f"syntax={total_c.get('syntax_error',0)} type={total_c.get('type_mismatch',0)} "
            f"semantic={total_c.get('semantic_error',0)} value={total_c.get('value_error',0)}"
        )

    print(f"[CHECKPOINT 3/5] Writing error taxonomy LaTeX table")
    tex_error_taxonomy(test_records, args.dataset, out_dir / "error_taxonomy.tex")
    print(f"  Wrote error_taxonomy.tex")

    # Write per-tier failure tables for SFT runs (most interesting for paper)
    for (model, mode, dataset) in sorted(bucket.keys()):
        if mode != "finetuned" or dataset != args.dataset:
            continue
        safe_model = model.replace(".", "").replace("-", "_")
        out_path = out_dir / f"tier_failure_{safe_model}_{mode}.tex"
        tex_tier_failure_table(records, model, mode, dataset, out_path)
        print(f"  Wrote {out_path.name}")

    print(f"[CHECKPOINT 4/5] Extracting qualitative examples")
    examples = extract_qualitative_examples(test_records)
    for pattern, exs in examples.items():
        print(f"  {pattern}: {len(exs)} example(s) extracted")

    print(f"[CHECKPOINT 5/5] Writing JSON and JSONL outputs")
    json_out = PROJECT_ROOT / "reports" / "failure_analysis.json"
    failure_summary = {
        "patterns": {k: v for k, v in examples.items()},
        "counts": {
            str(k): dict(v_total)
            for (model, mode, dataset), tier_data in bucket.items()
            for k in [(model, mode, dataset)]
            for v_total in [sum((tc for tc in tier_data.values()), Counter())]
        },
    }
    json_out.write_text(json.dumps(failure_summary, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")

    ex_jsonl = PROJECT_ROOT / "reports" / "failure_examples.jsonl"
    with ex_jsonl.open("w", encoding="utf-8") as f:
        for pattern, exs in examples.items():
            for ex in exs:
                ex["pattern"] = pattern
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {ex_jsonl}")
    print(f"\n[DONE] Failure analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
