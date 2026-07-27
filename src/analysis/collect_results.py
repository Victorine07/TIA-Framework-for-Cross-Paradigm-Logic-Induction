#!/usr/bin/env python3
"""
collect_results.py

Scans all results/ directories for evaluation summary JSONs, aggregates
SV/SM/VC metrics by model × mode × metadata_strategy × dataset, and
writes paper-ready tables to the reports/ directory.

Usage:
    python src/analysis/collect_results.py
    python src/analysis/collect_results.py --results-dir results/ --output-dir reports/
    python src/analysis/collect_results.py --dataset test          # filter to one dataset
    python src/analysis/collect_results.py --latex                 # also write LaTeX table

Output:
    reports/results_summary.csv     -- flat table of all runs
    reports/results_by_model.json   -- nested by model/mode/strategy
    reports/results_table.txt       -- human-readable console table
    reports/results_table.tex       -- LaTeX tabular (with --latex)

Produces the aggregated SV/SM/VC results tables reported across Section 6: Results and Discussion.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def find_overall_summaries(results_dir: Path) -> list[Path]:
    """Recursively find all overall__*__summary.json files."""
    # Overall summaries follow the naming: overall__{model}__{mode}__{strategy}__{ts}.json
    return sorted(results_dir.rglob("overall__*.json"))


def infer_mode_from_path(path: Path) -> str:
    """Infer experiment mode from the parent directory path."""
    parts = [p.lower() for p in path.parts]
    if "few_shot" in parts or "few-shot" in parts:
        return "few_shot"
    if "finetuned" in parts:
        return "finetuned"
    if "zero_shot" in parts:
        return "zero_shot"
    return "unknown"


def parse_summary(path: Path) -> list[dict]:
    """
    Parse one overall summary JSON into a list of flat result rows,
    one per dataset. Returns empty list on parse error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARNING] Could not parse {path}: {e}", file=sys.stderr)
        return []

    mode_from_path = infer_mode_from_path(path)

    rows = []
    datasets = data.get("datasets", {})
    if not datasets:
        # Might be a per-dataset summary -- skip, only process overall files
        return []

    for dataset_name, ds in datasets.items():
        row = {
            # zero-shot uses "model"; finetuned uses "base_model" — check both
            "model":             (ds.get("model") or ds.get("base_model")
                                  or data.get("model") or data.get("base_model") or "unknown"),
            # scripts write "mode" (fewshot_k3, finetuned, zeroshot); "mode_tag" is an alias
            "mode":              ds.get("mode_tag") or ds.get("mode") or mode_from_path,
            "metadata_strategy": ds.get("metadata_strategy", "unknown"),
            "dataset":           dataset_name,
            # summarize_results() writes "count"; n_examples/total kept as fallbacks
            "n_examples":        ds.get("count", ds.get("n_examples", ds.get("total", 0))),
            "avg_sv":            ds.get("avg_sv", None),
            "avg_sm":            ds.get("avg_sm", None),
            "avg_vc":            ds.get("avg_vc", None),
            "avg_overall":       ds.get("avg_overall", None),
            "failure_syntax":    ds.get("failure_counts", {}).get("syntax_error", 0),
            "failure_type":      ds.get("failure_counts", {}).get("type_mismatch", 0),
            "failure_missing":   ds.get("failure_counts", {}).get("missing_definition", 0),
            "failure_semantic":  ds.get("failure_counts", {}).get("semantic_error", 0),
            "failure_value":     ds.get("failure_counts", {}).get("value_error", 0),
            "correct":           ds.get("failure_counts", {}).get("correct", 0),
            "source_file":       str(path),
        }
        rows.append(row)

    return rows


def model_short_name(model_id: str) -> str:
    """Return a short label for display tables."""
    lm = model_id.lower()
    if "qwen2.5-coder-7b" in lm:
        return "Qwen2.5-7B"
    if "deepseek-coder-v2-lite" in lm:
        return "DS-Coder-V2-Lite"
    return model_id.split("/")[-1][:24]


def mode_display(mode_tag: str) -> str:
    """Normalize mode tags for display."""
    if mode_tag.startswith("fewshot_k"):
        k = mode_tag.replace("fewshot_k", "")
        return f"Few-shot k={k}"
    if mode_tag in ("zeroshot", "zero_shot"):
        return "Zero-shot"
    if mode_tag in ("finetuned", "sft"):
        return "SFT"
    return mode_tag


def fmt(val) -> str:
    if val is None:
        return "  -  "
    return f"{float(val):.3f}"


def render_text_table(rows: list[dict], filter_dataset: str | None) -> str:
    """Render a human-readable table grouped by model × mode × strategy."""
    if filter_dataset:
        rows = [r for r in rows if r["dataset"] == filter_dataset]
    if not rows:
        return "(no results matching filter)"

    lines = []
    header_ds = f"(dataset={filter_dataset})" if filter_dataset else "(all datasets averaged)"
    lines.append(f"  Results Summary {header_ds}")
    lines.append("=" * 90)
    lines.append(
        f"  {'Model':<22} {'Mode':<18} {'Strategy':<14} {'Dataset':<22} "
        f"{'SV':>6} {'SM':>6} {'VC':>6} {'Ovrl':>6}  N"
    )
    lines.append("-" * 90)

    # Group for display
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[model_short_name(r["model"])].append(r)

    for model_name, model_rows in sorted(by_model.items()):
        for r in sorted(model_rows, key=lambda x: (x["mode"], x["metadata_strategy"], x["dataset"])):
            lines.append(
                f"  {model_name:<22} {mode_display(r['mode']):<18} {r['metadata_strategy']:<14} "
                f"{r['dataset']:<22} {fmt(r['avg_sv'])} {fmt(r['avg_sm'])} "
                f"{fmt(r['avg_vc'])} {fmt(r['avg_overall'])}  {r['n_examples']}"
            )
        lines.append("-" * 90)

    return "\n".join(lines)


def render_latex_table(rows: list[dict], filter_dataset: str | None) -> str:
    """Render a LaTeX tabular snippet for paper inclusion."""
    if filter_dataset:
        rows = [r for r in rows if r["dataset"] == filter_dataset]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{TIA evaluation results (SV / SM / VC).}",
        r"\label{tab:results}",
        r"\begin{tabular}{llllccccc}",
        r"\toprule",
        r"Model & Mode & Strategy & Dataset & SV & SM & VC & Overall & N \\",
        r"\midrule",
    ]

    for r in sorted(rows, key=lambda x: (x["model"], x["mode"], x["metadata_strategy"], x["dataset"])):
        lines.append(
            f"{model_short_name(r['model'])} & {mode_display(r['mode'])} & "
            f"{r['metadata_strategy']} & {r['dataset']} & "
            f"{fmt(r['avg_sv'])} & {fmt(r['avg_sm'])} & {fmt(r['avg_vc'])} & "
            f"{fmt(r['avg_overall'])} & {r['n_examples']} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results for paper tables.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Root directory containing zero_shot/, few_shot/, finetuned/ subdirs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="Directory for output tables and CSVs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter to one dataset name (e.g. 'test'). Default: include all.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also write a LaTeX table snippet to reports/results_table.tex",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT] Scanning for summary files under: {results_dir}")

    summaries = find_overall_summaries(results_dir)
    print(f"[CHECKPOINT] Found {len(summaries)} overall summary file(s)")

    if not summaries:
        print("[WARNING] No summary files found. Have any evaluation jobs completed?")
        print(f"          Looked in: {results_dir}")
        sys.exit(0)

    all_rows: list[dict] = []
    for path in summaries:
        rows = parse_summary(path)
        if rows:
            print(f"  Loaded {len(rows)} dataset row(s) from: {path.name}")
        all_rows.extend(rows)

    if not all_rows:
        print("[WARNING] Parsed zero rows from all summary files. Check file format.")
        sys.exit(0)

    print(f"[CHECKPOINT] Total result rows before dedup: {len(all_rows)}")

    # ── Deduplicate: keep only the most recent result per (model, mode, strategy, dataset)
    # Duplicate runs arise when a strategy is run twice (e.g. original single-strategy
    # sbatch AND the sweep sbatch both ran 'none'). The source_file path embeds a
    # timestamp; sorting descending and keeping the first per key retains the newest run.
    seen: dict = {}
    for row in sorted(all_rows, key=lambda r: r["source_file"], reverse=True):
        key = (row["model"], row["mode"], row["metadata_strategy"], row["dataset"])
        if key not in seen:
            seen[key] = row
    all_rows = list(seen.values())
    print(f"[CHECKPOINT] Total result rows after dedup:  {len(all_rows)}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / "results_summary.csv"
    fieldnames = [
        "model", "mode", "metadata_strategy", "dataset", "n_examples",
        "avg_sv", "avg_sm", "avg_vc", "avg_overall",
        "failure_syntax", "failure_type", "failure_missing",
        "failure_semantic", "failure_value", "correct", "source_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[CHECKPOINT] Saved CSV: {csv_path}")

    # ── Write nested JSON ────────────────────────────────────────────────────
    nested: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_rows:
        nested[model_short_name(r["model"])][mode_display(r["mode"])][r["metadata_strategy"]].append(r)
    json_path = output_dir / "results_by_model.json"
    json_path.write_text(
        json.dumps(nested, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[CHECKPOINT] Saved JSON: {json_path}")

    # ── Write text table ─────────────────────────────────────────────────────
    text_table = render_text_table(all_rows, args.dataset)
    txt_path = output_dir / "results_table.txt"
    txt_path.write_text(text_table, encoding="utf-8")
    print(f"[CHECKPOINT] Saved text table: {txt_path}")
    print()
    print(text_table)

    # ── Write LaTeX table (optional) ─────────────────────────────────────────
    if args.latex:
        latex_table = render_latex_table(all_rows, args.dataset)
        tex_path = output_dir / "results_table.tex"
        tex_path.write_text(latex_table, encoding="utf-8")
        print(f"[CHECKPOINT] Saved LaTeX table: {tex_path}")

    # ── Completion notice ────────────────────────────────────────────────────
    print()
    print(f"[CHECKPOINT] Analysis complete. Timestamp: {datetime.now().isoformat()}")
    print(f"  Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
