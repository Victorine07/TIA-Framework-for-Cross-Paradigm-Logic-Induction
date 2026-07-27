#!/usr/bin/env python3
"""
figures.py  (revised: larger fonts, shorter heights, hatch patterns on bar charts)
Generate all paper-quality figures from real experimental data.

Produces:
  1.  tier_metrics_by_mode.pdf       — Line plots: SV/SM/VC vs tier, per model × mode
  2.  mode_comparison_bars.pdf       — Grouped bars: all metrics across regimes
  3.  metadata_ablation_heatmap.pdf  — Heatmap: strategy × tier → metric (per model)
  4.  id_vs_ood_bars.pdf             — Grouped bars: ID vs unseen per model × mode
  5.  error_taxonomy_stacked.pdf     — Stacked bars: failure distribution per mode
  6.  sv_vc_scatter.pdf              — Scatter: SV vs VC coloured by tier/mode
  7.  score_distributions.pdf        — Histograms of SV/SM/VC per regime
  8.  dataset_composition.pdf        — Stacked bars: tier breakdown per cipher
  9.  metric_sensitivity_bars.pdf    — Bar comparison of weight config robustness
  10. sft_generalization_tier.pdf    — Per-tier VC on ID vs unseen (SFT)
  11. improvement_heatmap.pdf        — Delta heatmap: SFT − ZS per metric × tier

Usage:
    python src/analysis/figures.py
    python src/analysis/figures.py --output-dir reports/paper/figures
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import (
    load_overall_summaries, load_per_example_records,
    mode_display, fmt
)

RESULTS_ROOT = PROJECT_ROOT / "results"
TIERS = ["T1", "T2", "T3", "T4"]
METRICS = ["sv", "sm", "vc"]
METRIC_LABELS = {
    "sv": "Syntax Validity (SV)",
    "sm": "Semantic Match (SM)",
    "vc": "Value Consistency (VC)",
}

# ── Shared style ───────────────────────────────────────────────────────────────
# Fonts are set large so that after scaling to  column width 
# labels remain readable.  Base 14 pt → ~7 pt rendered at single-column width.
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.size":             14,
    "axes.titlesize":        15,
    "axes.labelsize":        14,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "legend.fontsize":       12,
    "legend.title_fontsize": 12,
    "lines.linewidth":       2.2,
    "lines.markersize":      7,
    "figure.dpi":            200,
    "hatch.linewidth":       1.3,   # bold hatching, visible when scaled
    "axes.linewidth":        0.9,
})

# ── Color + hatch palette for bar charts ──────────────────────────────────────
# Original category colors retained for on-screen clarity; hatch patterns add
# print-friendly distinguishability for B&W reproduction.
_FALLBACK_BAR = {"facecolor": "white", "hatch": "///", "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85}

MODEL_BAR = {
    "Qwen2.5-7B":       {"facecolor": "#2196F3", "hatch": "///", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    "DS-Coder-V2-Lite": {"facecolor": "#FF5722", "hatch": "xxx", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
}
REGIME_BAR = {
    "zero_shot":  {"facecolor": "#607D8B", "hatch": "///", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    "fewshot_k3": {"facecolor": "#FF9800", "hatch": "...", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    "fewshot_k5": {"facecolor": "#795548", "hatch": "---", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    "finetuned":  {"facecolor": "#4CAF50", "hatch": "xxx", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
}
# Error-taxonomy stacked bars: original category colors + hatch
CAT_BAR = {
    "correct":        {"facecolor": "#4CAF50", "hatch": "///",  "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85},
    "syntax_error":   {"facecolor": "#F44336", "hatch": "\\\\", "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85},
    "type_mismatch":  {"facecolor": "#FF9800", "hatch": "---",  "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85},
    "semantic_error": {"facecolor": "#9C27B0", "hatch": "...",  "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85},
    "value_error":    {"facecolor": "#607D8B", "hatch": "|||",  "edgecolor": "black", "linewidth": 0.6, "alpha": 0.85},
}
CAT_LABELS = {
    "correct":        "Correct",
    "syntax_error":   "Syntax Error",
    "type_mismatch":  "Type Mismatch",
    "semantic_error": "Semantic Error",
    "value_error":    "Value Error",
}

# ── Line-plot palette (color + linestyle + marker) ────────────────────────────
MODEL_COLORS  = {"Qwen2.5-7B": "#2196F3", "DS-Coder-V2-Lite": "#FF5722"}
MODEL_MARKERS = {"Qwen2.5-7B": "o",       "DS-Coder-V2-Lite": "s"}
MODE_STYLES   = {
    "zero_shot":  ("--", 0.70),
    "fewshot_k3": ("-.", 0.85),
    "fewshot_k5": (":",  0.75),
    "finetuned":  ("-",  1.00),
}
DS_LINE = {
    "test":             {"color": "#212121", "ls": "-",  "marker": "o"},
    "unseen_lea":       {"color": "#2196F3", "ls": "--", "marker": "s"},
    "unseen_rectangle": {"color": "#FF5722", "ls": "-.", "marker": "^"},
    "unseen_xtea":      {"color": "#4CAF50", "ls": ":",  "marker": "D"},
}
DS_LABELS = {
    "test":             "ID (test)",
    "unseen_lea":       "LEA",
    "unseen_rectangle": "Rectangle",
    "unseen_xtea":      "XTEA",
}


def _save(fig, out_dir: Path, name: str) -> None:
    p = out_dir / name
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}")


def _legend_patch(style: dict, label: str) -> mpatches.Patch:
    return mpatches.Patch(
        facecolor=style["facecolor"],
        hatch=style["hatch"],
        edgecolor="black",
        label=label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Tier-wise SV/SM/VC line plots per model × mode
# ─────────────────────────────────────────────────────────────────────────────
def fig_tier_metrics(records: list[dict], out_dir: Path) -> None:
    """Line plot per metric: x=tier, y=metric score, curves=model × mode."""
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("dataset") != "test":
            continue
        key = (rec["_model"], rec["_mode"])
        tier = rec["_tier"]
        if tier not in TIERS:
            continue
        for m in METRICS:
            v = rec.get(f"_{m}")
            if v is not None:
                bucket[key][(tier, m)].append(float(v))

    # Shorter height; figure* maps to textwidth ≈ 6.9 in → scale ≈ 0.63
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False,
                              layout="constrained")
    fig.suptitle("Tier-wise Metrics by Learning Regime (test split)", fontsize=18)

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        ax.set_title(METRIC_LABELS[metric], fontsize=15)
        ax.set_xlabel("TIA Tier", fontsize=15)
        ax.set_ylabel(metric.upper() if ax_idx == 0 else "")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks(range(4))
        ax.set_xticklabels(TIERS)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.85, color="gray", lw=0.9, ls="--", alpha=0.6, label="0.85 threshold")
        ax.grid(axis="y", alpha=0.3)

        for (model, mode), tier_data in sorted(bucket.items()):
            vals = [np.mean(tier_data.get((t, metric), [np.nan])) for t in TIERS]
            color  = MODEL_COLORS.get(model, "black")
            ls, alpha = MODE_STYLES.get(mode, ("-", 1.0))
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(range(4), vals, ls=ls, color=color, alpha=alpha,
                    marker=marker, label=f"{model} / {mode_display(mode)}")

    # Shared legend below all three subplots
    legend_handles = [
        mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()
    ] + [
        plt.Line2D([0], [0], color="gray", ls=ls, label=mode_display(md))
        for md, (ls, _) in MODE_STYLES.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.18), framealpha=0.9, fontsize=18)

    _save(fig, out_dir, "tier_metrics_by_mode.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Mode comparison grouped bars (SV / SM / VC per regime per model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_mode_comparison(overall_rows: list[dict], out_dir: Path) -> None:
    modes_order = ["zero_shot", "fewshot_k3", "fewshot_k5", "finetuned"]
    mode_labels  = [mode_display(m) for m in modes_order]
    models       = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]

    def best(model, mode, metric):
        cands = [r for r in overall_rows
                 if r["model"] == model and r["mode"] == mode and r["dataset"] == "test"]
        if not cands:
            return np.nan
        best_r = max(cands, key=lambda r: r["avg_overall"] or 0)
        return best_r.get(f"avg_{metric}") or np.nan

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True, layout="constrained")
    fig.suptitle("SV / SM / VC by Regime — test split (best strategy per condition)",
                 fontsize=14)
    x     = np.arange(len(modes_order))
    width = 0.35

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        ax.set_title(METRIC_LABELS[metric], fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, rotation=22, ha="right", fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.85, color="gray", lw=0.9, ls="--", alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(metric.upper() if ax_idx == 0 else "")

        for m_idx, model in enumerate(models):
            style = MODEL_BAR[model]
            vals  = [best(model, mode, metric) for mode in modes_order]
            bars  = ax.bar(x + (m_idx - 0.5) * width, vals, width,
                           zorder=3, **style)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    handles = [_legend_patch(MODEL_BAR[m], m) for m in models]
    handles.append(plt.Line2D([0], [0], color="gray", ls="--", label="0.85 threshold"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), fontsize=12)
    _save(fig, out_dir, "mode_comparison_bars.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Metadata ablation heatmap (strategy × tier → metric per model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_metadata_ablation_heatmap(records: list[dict], out_dir: Path) -> None:
    strategies = ["none", "algorithmic", "structured", "full", "alljson"]
    models     = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]

    for metric in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
        fig.suptitle(
            f"Metadata Ablation — {METRIC_LABELS[metric]} (ZS, test split)",
            fontsize=14,
        )
        for m_idx, model in enumerate(models):
            ax = axes[m_idx]
            ax.set_title(model, fontsize=13)

            bucket: dict = defaultdict(list)
            for rec in records:
                if rec.get("dataset") != "test":
                    continue
                if rec["_model"] != model or rec["_mode"] != "zero_shot":
                    continue
                tier = rec["_tier"]
                strat = rec["_strategy"]
                v = rec.get(f"_{metric}")
                if v is not None and tier in TIERS:
                    bucket[(strat, tier)].append(float(v))

            mat = np.full((len(strategies), len(TIERS)), np.nan)
            for s_idx, strat in enumerate(strategies):
                for t_idx, tier in enumerate(TIERS):
                    vals = bucket.get((strat, tier), [])
                    if vals:
                        mat[s_idx, t_idx] = np.mean(vals)

            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                           vmin=0.0, vmax=1.0, origin="upper")
            ax.set_xticks(range(len(TIERS)))
            ax.set_xticklabels(TIERS, fontsize=12)
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(strategies, fontsize=12)
            ax.set_xlabel("TIA Tier", fontsize=13)
            if m_idx == 0:
                ax.set_ylabel("Strategy", fontsize=13)

            for s_idx in range(len(strategies)):
                for t_idx in range(len(TIERS)):
                    v = mat[s_idx, t_idx]
                    if not np.isnan(v):
                        ax.text(t_idx, s_idx, f"{v:.2f}",
                                ha="center", va="center", fontsize=9,
                                color="black" if 0.3 < v < 0.8 else "white",
                                fontweight="bold" if v > 0.8 else "normal")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _save(fig, out_dir, f"metadata_ablation_heatmap_{metric}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: ID vs OOD grouped bars (Overall score)
# ─────────────────────────────────────────────────────────────────────────────
def fig_id_vs_ood(overall_rows: list[dict], out_dir: Path) -> None:
    datasets    = ["test", "unseen_lea", "unseen_rectangle", "unseen_xtea"]
    ds_labels   = ["ID\n(test)", "LEA\n(ARX)", "Rect.\n(SPN)", "XTEA\n(Feis.)"]
    modes_spec  = [("finetuned", "none"), ("finetuned", "structured"), ("zero_shot", None)]
    mode_labels = ["SFT-none", "SFT-struct", "ZS-best"]
    mode_bar    = [
        {"facecolor": "#4CAF50", "hatch": "///", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        {"facecolor": "#2196F3", "hatch": "...", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        {"facecolor": "#FF9800", "hatch": "---", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    ]
    models      = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]
    metric      = "overall"  # paper uses Overall score

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True, layout="constrained")
    fig.suptitle("In-Distribution vs Unseen Cipher — Overall Score", fontsize=25)
    x     = np.arange(len(datasets))
    n_b   = len(modes_spec)
    width = 0.15
    bar_gap = 0.11

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        ax.set_title(model, fontsize=27)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontsize=27)
        ax.set_ylim(0, 1.3)
        ax.axhline(0.85, color="gray", lw=0.9, ls="--", alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Overall" if m_idx == 0 else "", fontsize=27)

        ax.tick_params(axis='y', labelsize=20) 

        for mode_i, (mode, strategy) in enumerate(modes_spec):
            vals = []
            for ds in datasets:
                if strategy is not None:
                    match = next(
                        (r for r in overall_rows
                         if r["model"] == model and r["mode"] == mode
                         and r["strategy"] == strategy and r["dataset"] == ds),
                        None,
                    )
                else:
                    cands = [r for r in overall_rows
                             if r["model"] == model and r["mode"] == mode
                             and r["dataset"] == ds]
                    match = max(cands, key=lambda r: r["avg_overall"] or 0) if cands else None
                vals.append(
                    match[f"avg_{metric}"] if match and match[f"avg_{metric}"] else np.nan
                )

            offset = (mode_i - n_b / 2 + 0.5) * (width + bar_gap)
            style  = mode_bar[mode_i]
            bars   = ax.bar(x + offset, vals, width, label=mode_labels[mode_i],
                            zorder=3, **style)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    val_str = f"{v:.2f}".lstrip('0')
                    # Augmentation de la taille (ex: fontsize=19)
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            val_str, ha="center", va="bottom", fontsize=25)
                    
                    # ax.text(bar.get_x() + bar.get_width() / 2,
                    #         bar.get_height() + 0.01,
                    #         f"{v:.2f}", ha="center", va="bottom", fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), fontsize=26)
    _save(fig, out_dir, "id_vs_ood_bars.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Error taxonomy stacked bars per mode (per model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_error_taxonomy(records: list[dict], out_dir: Path) -> None:
    cats        = ["correct", "syntax_error", "type_mismatch", "semantic_error", "value_error"]
    modes_order = ["zero_shot", "fewshot_k3", "fewshot_k5", "finetuned"]
    models      = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
    fig.suptitle("Error Category Distribution by Regime (test split)", fontsize=14)

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        ax.set_title(model, fontsize=13)

        counts: dict = defaultdict(lambda: defaultdict(int))
        for rec in records:
            if rec.get("dataset") != "test" or rec["_model"] != model:
                continue
            counts[rec["_mode"]][rec["_failure"] or "unknown"] += 1

        x       = np.arange(len(modes_order))
        bottoms = np.zeros(len(modes_order))
        for cat in cats:
            style = CAT_BAR.get(cat, _FALLBACK_BAR)
            vals  = []
            for mode in modes_order:
                total = sum(counts[mode].values())
                n     = counts[mode].get(cat, 0)
                vals.append(n / total * 100 if total else 0)
            ax.bar(x, vals, bottom=bottoms, label=CAT_LABELS.get(cat, cat),
                   zorder=3, **style)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [mode_display(m) for m in modes_order], rotation=18, ha="right", fontsize=11
        )
        ax.set_ylabel("% of examples" if m_idx == 0 else "")
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.3)

    cat_handles = [
        _legend_patch(CAT_BAR.get(cat, _FALLBACK_BAR), CAT_LABELS.get(cat, cat))
        for cat in cats
    ]
    fig.legend(handles=cat_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.18), fontsize=12)
    _save(fig, out_dir, "error_taxonomy_stacked.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: SV vs VC scatter, shaped by tier
# ─────────────────────────────────────────────────────────────────────────────
def fig_sv_vc_scatter(records: list[dict], out_dir: Path) -> None:
    tier_style = {
        "T1": {"color": "#4CAF50", "marker": "o"},
        "T2": {"color": "#2196F3", "marker": "s"},
        "T3": {"color": "#FF9800", "marker": "^"},
        "T4": {"color": "#F44336", "marker": "D"},
    }
    modes_to_plot = ["zero_shot", "fewshot_k3", "finetuned"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True, sharey=True,
                              layout="constrained")
    fig.suptitle("SV vs VC by Tier — paradigm-crossing divergence (test split)",
                 fontsize=14)

    for ax_idx, mode in enumerate(modes_to_plot):
        ax = axes[ax_idx]
        ax.set_title(mode_display(mode), fontsize=13)
        ax.set_xlabel("Syntax Validity (SV)", fontsize=12)
        ax.set_ylabel("Value Consistency (VC)" if ax_idx == 0 else "", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.axhline(0.85, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.axvline(0.85, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.grid(alpha=0.2)

        for tier, sty in tier_style.items():
            xs = [float(rec["_sv"]) for rec in records
                  if rec.get("dataset") == "test" and rec["_mode"] == mode
                  and rec.get("_sv") is not None and rec.get("_vc") is not None
                  and rec.get("_tier") == tier]
            ys = [float(rec["_vc"]) for rec in records
                  if rec.get("dataset") == "test" and rec["_mode"] == mode
                  and rec.get("_sv") is not None and rec.get("_vc") is not None
                  and rec.get("_tier") == tier]
            if xs:
                ax.scatter(xs, ys, alpha=0.35, s=14, zorder=3,
                           color=sty["color"], marker=sty["marker"])

    tier_legend_handles = [
        plt.Line2D([0], [0], color=sty["color"], marker=sty["marker"],
                   ls="None", markersize=7, label=tier)
        for tier, sty in tier_style.items()
    ]
    fig.legend(handles=tier_legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.18), fontsize=12)
    _save(fig, out_dir, "sv_vc_scatter.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Score distribution histograms per regime
# ─────────────────────────────────────────────────────────────────────────────
def fig_score_distributions(records: list[dict], out_dir: Path) -> None:
    modes_order = ["zero_shot", "fewshot_k3", "fewshot_k5", "finetuned"]
    hist_styles = [
        {"facecolor": "#2196F3", "hatch": "///", "edgecolor": "black", "alpha": 0.75},
        {"facecolor": "#FF5722", "hatch": "xxx", "edgecolor": "black", "alpha": 0.75},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 4.5), sharex=True,
                              layout="constrained")
    axes = axes.flatten()
    fig.suptitle("Overall Score Distribution by Regime (test, both models)", fontsize=14)

    for ax_idx, mode in enumerate(modes_order):
        ax = axes[ax_idx]
        ax.set_title(mode_display(mode), fontsize=13)
        ax.set_xlabel("Overall Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(0.85, color="black", lw=1.2, ls="--", alpha=0.8, label="0.85 threshold")
        ax.grid(axis="y", alpha=0.3)

        for idx, (model, color) in enumerate(MODEL_COLORS.items()):
            vals = [
                float(rec["_overall"])
                for rec in records
                if rec.get("dataset") == "test" and rec["_mode"] == mode
                and rec["_model"] == model and rec["_overall"] is not None
            ]
            if vals:
                ax.hist(vals, bins=20, range=(0, 1), label=model,
                        zorder=3, **hist_styles[idx % 2])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=12)
    _save(fig, out_dir, "score_distributions.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Dataset composition stacked bars per cipher
# ─────────────────────────────────────────────────────────────────────────────
def fig_dataset_composition(out_dir: Path) -> None:
    comp_path = PROJECT_ROOT / "reports" / "dataset_composition.json"
    if not comp_path.exists():
        print("  [SKIP] dataset_composition.json not found — run summarize_datasets.py first")
        return

    d = json.loads(comp_path.read_text(encoding="utf-8"))
    ciphers_data = d.get("train_by_cipher", {})
    if not ciphers_data:
        return

    ciphers = sorted(ciphers_data.keys())
    tier_style = {
        "T1": {"facecolor": "#4CAF50", "hatch": "///", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        "T2": {"facecolor": "#2196F3", "hatch": "...", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        "T3": {"facecolor": "#FF9800", "hatch": "---", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        "T4": {"facecolor": "#F44336", "hatch": "xxx", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    }

    fig, ax = plt.subplots(figsize=(11, 2.8), layout="constrained")
    fig.suptitle("Training Dataset Composition: Components per Cipher by TIA Tier", fontsize=14)
    x       = np.arange(len(ciphers))
    bottoms = np.zeros(len(ciphers))

    for tier in TIERS:
        vals = [ciphers_data[c].get(tier, 0) for c in ciphers]
        sty  = tier_style[tier]
        ax.bar(x, vals, bottom=bottoms, label=tier, zorder=3, **sty)
        bottoms += np.array(vals, dtype=float)

    # Family letter annotation above bars
    fam_colors = {
        "ARX": "#000000", "Feistel": "#444444", "SPN": "#888888",
        "Permutation": "#555555", "Permutation/AEAD": "#888888",
    }
    families = [ciphers_data[c].get("family", "?") for c in ciphers]
    for i, fam in enumerate(families):
        ax.text(i, bottoms[i] + 1.5, fam[0],
                ha="center", va="bottom", fontsize=10,
                color=fam_colors.get(fam, "gray"), fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ciphers, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Component Count", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    tier_handles = [_legend_patch(tier_style[t], t) for t in TIERS]
    fig.legend(handles=tier_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.18), fontsize=12, title="Tier")
    _save(fig, out_dir, "dataset_composition.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Metric sensitivity bars
# ─────────────────────────────────────────────────────────────────────────────
def fig_metric_sensitivity(out_dir: Path) -> None:
    sens_path = PROJECT_ROOT / "reports" / "metric_sensitivity.json"
    if not sens_path.exists():
        print("  [SKIP] metric_sensitivity.json not found")
        return

    rows         = json.loads(sens_path.read_text(encoding="utf-8"))
    wnames       = ["W1 (SV-heavy)", "W2 (Orig. 1/3)", "W3 (VC-heavy)"]
    weight_bar   = [
        {"facecolor": "#2196F3", "hatch": "///", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        {"facecolor": "#4CAF50", "hatch": "...", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
        {"facecolor": "#FF5722", "hatch": "---", "edgecolor": "black", "linewidth": 0.7, "alpha": 0.82},
    ]
    modes_unique = list(dict.fromkeys(r["mode"] for r in rows))
    models_unique = list(dict.fromkeys(r["model"] for r in rows))

    fig, axes = plt.subplots(1, len(models_unique), figsize=(9, 2.8),
                              sharey=True, layout="constrained")
    if len(models_unique) == 1:
        axes = [axes]
    fig.suptitle("Metric Sensitivity: Overall Score Under Alternative Weighting", fontsize=14)

    x     = np.arange(len(modes_unique))
    width = 0.25

    for m_idx, model in enumerate(models_unique):
        ax = axes[m_idx]
        ax.set_title(model, fontsize=13)
        model_rows = [r for r in rows if r["model"] == model]
        ax.set_xticks(x)
        ax.set_xticklabels(modes_unique, rotation=15, ha="right", fontsize=11)
        ax.set_ylim(0, 1.07)
        ax.axhline(0.85, color="gray", lw=0.9, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Mean Overall Score" if m_idx == 0 else "")

        for w_idx, (wname, sty) in enumerate(zip(wnames, weight_bar)):
            vals = []
            for mode in modes_unique:
                r = next((rr for rr in model_rows if rr["mode"] == mode), None)
                vals.append(r.get(f"{wname}_mean") or np.nan if r else np.nan)
            offset = (w_idx - 1) * width
            ax.bar(x + offset, vals, width, label=wname, zorder=3, **sty)

    handles = [_legend_patch(weight_bar[i], wnames[i]) for i in range(len(wnames))]
    handles.append(plt.Line2D([0], [0], color="gray", ls="--", label="0.85 threshold"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.18), fontsize=12)
    _save(fig, out_dir, "metric_sensitivity_bars.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: SFT per-tier VC on ID vs unseen (generalization profile)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sft_generalization_tier(records: list[dict], out_dir: Path) -> None:
    ds_keys = ["test", "unseen_lea", "unseen_rectangle", "unseen_xtea"]
    models  = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True, layout="constrained")
    fig.suptitle("SFT (none): Per-Tier VC — ID vs Unseen Cipher Generalization", fontsize=14)

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        ax.set_title(model, fontsize=13)
        ax.set_xlabel("TIA Tier", fontsize=13)
        ax.set_ylabel("Value Consistency (VC)" if m_idx == 0 else "")
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels(TIERS)
        ax.set_ylim(-0.05, 1.12)
        ax.axhline(0.85, color="gray", lw=0.9, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        for ds in ds_keys:
            sty    = DS_LINE[ds]
            bucket: dict = defaultdict(list)
            for rec in records:
                if rec.get("dataset") != ds:
                    continue
                if rec["_model"] != model or rec["_mode"] != "finetuned":
                    continue
                if rec["_strategy"] != "none":
                    continue
                tier = rec["_tier"]
                v    = rec.get("_vc")
                if v is not None and tier in TIERS:
                    bucket[tier].append(float(v))

            vals = [np.mean(bucket.get(t, [np.nan])) for t in TIERS]
            ax.plot(range(len(TIERS)), vals, label=DS_LABELS[ds],
                    color=sty["color"], ls=sty["ls"], marker=sty["marker"], zorder=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.18), fontsize=12)
    _save(fig, out_dir, "sft_generalization_tier.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Improvement heatmap (SFT − ZS per metric × tier)
# ─────────────────────────────────────────────────────────────────────────────
def fig_improvement_heatmap(records: list[dict], out_dir: Path) -> None:
    models  = ["Qwen2.5-7B", "DS-Coder-V2-Lite"]
    metrics = ["sv", "sm", "vc"]
    m_short = ["SV", "SM", "VC"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), layout="constrained")
    fig.suptitle(
        r"SFT vs Zero-shot Improvement ($\Delta$) per Tier and Metric (test split)",
        fontsize=14,
    )

    for m_idx, model in enumerate(models):
        ax = axes[m_idx]
        ax.set_title(model, fontsize=13)

        def _mean_by(mode, strategy=None):
            buck: dict = defaultdict(list)
            for rec in records:
                if rec.get("dataset") != "test":
                    continue
                if rec["_model"] != model or rec["_mode"] != mode:
                    continue
                if strategy and rec["_strategy"] != strategy:
                    continue
                tier = rec["_tier"]
                if tier in TIERS:
                    for met in metrics:
                        v = rec.get(f"_{met}")
                        if v is not None:
                            buck[(met, tier)].append(float(v))
            return buck

        zs_b  = _mean_by("zero_shot")
        sft_b = _mean_by("finetuned", "none")

        mat = np.full((len(metrics), len(TIERS)), np.nan)
        for mi, met in enumerate(metrics):
            for ti, tier in enumerate(TIERS):
                zv = zs_b.get((met, tier), [])
                sv = sft_b.get((met, tier), [])
                if zv and sv:
                    mat[mi, ti] = np.mean(sv) - np.mean(zv)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=-0.1, vmax=1.0, origin="upper")
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels(TIERS, fontsize=12)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(m_short, fontsize=12)
        ax.set_xlabel("TIA Tier", fontsize=13)
        if m_idx == 0:
            ax.set_ylabel("Metric", fontsize=13)

        for mi in range(len(metrics)):
            for ti in range(len(TIERS)):
                v = mat[mi, ti]
                if not np.isnan(v):
                    ax.text(ti, mi, f"{v:+.2f}", ha="center", va="center",
                            fontsize=10,
                            color="black" if -0.2 < v < 0.7 else "white",
                            fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=r"$\Delta$ (SFT − ZS)")

    _save(fig, out_dir, "improvement_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures from real data.")
    parser.add_argument("--results-dir",  default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir",   default=str(PROJECT_ROOT / "reports" / "paper" / "figures"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/4] Loading data from: {args.results_dir}")
    overall_rows = load_overall_summaries(Path(args.results_dir))
    print(f"  {len(overall_rows)} overall summary rows")
    records = load_per_example_records(Path(args.results_dir))
    print(f"  {len(records)} per-example records")

    # print(f"\n[CHECKPOINT 2/4] Generating metric / regime figures")
    # fig_tier_metrics(records, out_dir)
    # fig_mode_comparison(overall_rows, out_dir)
    # fig_sv_vc_scatter(records, out_dir)
    # fig_score_distributions(records, out_dir)
    # fig_improvement_heatmap(records, out_dir)

    # print(f"\n[CHECKPOINT 3/4] Generating ablation and generalisation figures")
    # fig_metadata_ablation_heatmap(records, out_dir)
    fig_id_vs_ood(overall_rows, out_dir)
    # fig_sft_generalization_tier(records, out_dir)
    # fig_error_taxonomy(records, out_dir)

    # print(f"\n[CHECKPOINT 4/4] Generating dataset and sensitivity figures")
    # fig_dataset_composition(out_dir)
    # fig_metric_sensitivity(out_dir)

    all_figs = sorted(out_dir.glob("*.pdf"))
    print(f"\n[DONE] {len(all_figs)} figures written to: {out_dir}")
    for f in all_figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
