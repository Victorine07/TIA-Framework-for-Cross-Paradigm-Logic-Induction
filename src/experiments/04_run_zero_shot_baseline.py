#!/usr/bin/env python3
"""
04_run_zero_shot_baseline.py

Zero-shot and few-shot inference runner for Python -> Isabelle/HOL
translation. This is the canonical baseline against which SFT runs
(03_finetune.py + 04_evaluate_finetuned.py) must be compared.

Summary:
- runs zero-shot OR few-shot inference on one or more registry datasets
- supports metadata-aware prompt variants
- writes per-example JSONL plus summary JSON files, including a per-example
  failure-taxonomy bucket for paper error analysis

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from data.registry_loader import RegistryLoader  # noqa: E402
from evaluation.eval_runner import (  # noqa: E402
    classify_failure,
    compute_metrics,
    summarize_results,
    try_load_metrics,
)
from evaluation.generation import (  # noqa: E402
    batched,
    generate_batch,
    get_cuda_info,
    load_model_and_tokenizer,
)
from evaluation.text_normalization import clean_code  # noqa: E402
from prompting.few_shot_selection import select_support_examples  # noqa: E402
from prompting.prompt_builder import (  # noqa: E402
    build_few_shot_prompt,
    build_translation_prompt,
    example_identifier,
)

PROTECTED_FEW_SHOT_SOURCES = {"test"}


def resolve_default_registry(project_root: Path) -> Path:
    candidates = [
        project_root / "datasets" / "processed" / "eval_registry_family_holdout_v1.json",
        project_root / "datasets" / "processed" / "eval_registry.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_run_metadata(args, registry_path: Path, resolved_datasets: List[str], model) -> Dict[str, Any]:
    import socket

    return {
        "model": args.model,
        "registry": str(registry_path),
        "datasets": resolved_datasets,
        "metadata_strategy": args.metadata_strategy,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "few_shot": {
            "k": args.few_shot_k,
            "policy": args.few_shot_policy if args.few_shot_k > 0 else None,
            "seed": args.few_shot_seed if args.few_shot_k > 0 else None,
            "source_split": args.few_shot_source if args.few_shot_k > 0 else None,
        },
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "cuda": get_cuda_info(),
        "model_class": model.__class__.__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot/few-shot baseline on registry datasets.")
    parser.add_argument(
        "--registry",
        type=str,
        default=str(resolve_default_registry(PROJECT_ROOT)),
        help="Path to eval registry JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, comma-separated names, or 'all'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID or local model path",
    )
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="Optional HF cache/models directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "zero_shot"),
        help="Directory for prediction outputs",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size")
    parser.add_argument("--max-input-length", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum generated tokens")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on examples per dataset")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with 4-bit quantization")
    parser.add_argument(
        "--metadata-strategy",
        type=str,
        default="none",
        choices=["none", "full", "structured", "algorithmic", "alljson"],
        help="Metadata prompting strategy",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Skip SV / SM / VC computation")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional Transformers attention implementation override",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code when loading tokenizer/model "
        "(required for some model repos, e.g. DeepSeek-Coder-V2)",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=0,
        help="Number of support examples to prepend (0 = pure zero-shot, unchanged prompt text)",
    )
    parser.add_argument(
        "--few-shot-policy",
        type=str,
        default="same_family",
        choices=["same_family", "same_tier", "random"],
        help="Support-example selection policy when --few-shot-k > 0",
    )
    parser.add_argument(
        "--few-shot-source",
        type=str,
        default="train",
        help="Registry dataset name to draw few-shot support examples from (default: train)",
    )
    parser.add_argument("--few-shot-seed", type=int, default=42, help="Seed for few-shot support selection")
    parser.add_argument(
        "--allow-protected-few-shot-source",
        action="store_true",
        help="Allow drawing few-shot support examples from a protected split (e.g. 'test') -- "
        "off by default to prevent accidental leakage into a paper-facing eval run.",
    )
    args = parser.parse_args()

    if args.few_shot_k > 0 and args.few_shot_source in PROTECTED_FEW_SHOT_SOURCES and not args.allow_protected_few_shot_source:
        raise ValueError(
            f"Refusing to draw few-shot support examples from protected source "
            f"'{args.few_shot_source}' without --allow-protected-few-shot-source. "
            f"This split must stay held out for evaluation."
        )

    registry_path = Path(args.registry).expanduser().resolve()
    loader = RegistryLoader(registry_path)
    project_root = loader.project_root

    if args.dataset.lower() == "all":
        dataset_names = loader.list_dataset_names()
    else:
        dataset_names = [x.strip() for x in args.dataset.split(",") if x.strip()]

    mode_tag = "zeroshot" if args.few_shot_k == 0 else f"fewshot_k{args.few_shot_k}"

    print("=" * 80)
    print(f"{'ZERO-SHOT' if args.few_shot_k == 0 else 'FEW-SHOT'} BASELINE")
    print("=" * 80)
    print(f"Project root:       {project_root}")
    print(f"Model:              {args.model}")
    print(f"Datasets:           {dataset_names}")
    print(f"Metadata strategy:  {args.metadata_strategy}")
    print(f"Few-shot k:         {args.few_shot_k}")
    if args.few_shot_k > 0:
        print(f"Few-shot policy:    {args.few_shot_policy}")
        print(f"Few-shot source:    {args.few_shot_source}")
        print(f"Few-shot seed:      {args.few_shot_seed}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Max input len:      {args.max_input_length}")
    print(f"Max new tokens:     {args.max_new_tokens}")
    print(f"Load in 4-bit:      {args.load_in_4bit}")
    print(f"Trust remote code:  {args.trust_remote_code}")
    print(f"Attention impl:     {args.attn_implementation}")
    print("=" * 80)

    few_shot_pool: List[Dict[str, Any]] = []
    if args.few_shot_k > 0:
        print(f"Loading few-shot support pool from '{args.few_shot_source}'...")
        few_shot_pool = loader.load_dataset(args.few_shot_source)
        print(f"  Support pool size: {len(few_shot_pool)}")

    print("Loading model and tokenizer...")
    start_load = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        hf_cache_dir=args.hf_cache_dir,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    print(f"Model loaded in {time.time() - start_load:.1f}s")
    print(f"CUDA available: {torch.cuda.is_available()}")

    metrics_obj = None if args.skip_metrics else try_load_metrics()
    if metrics_obj is None and not args.skip_metrics:
        print("Metrics module not available; continuing without SV/SM/VC.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(":", "_")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary: Dict[str, Any] = {
        "model": args.model,
        "timestamp": timestamp,
        "metadata_strategy": args.metadata_strategy,
        "mode": mode_tag,
        "run_config": build_run_metadata(args, registry_path, dataset_names, model),
        "datasets": {},
    }

    for dataset_name in dataset_names:
        data = loader.load_dataset(dataset_name)
        dataset_path = loader.resolve_dataset_path(dataset_name)

        if args.limit is not None:
            data = data[: args.limit]

        if not data:
            print(f"WARNING: No examples found in dataset '{dataset_name}'")
            continue

        print("-" * 80)
        print(f"Dataset:      {dataset_name}")
        print(f"Path:         {dataset_path}")
        print(f"Examples:     {len(data)}")

        prompts: List[str] = []
        support_id_lists: List[List[Any]] = []
        for ex in data:
            if args.few_shot_k > 0:
                support = select_support_examples(
                    ex, few_shot_pool, k=args.few_shot_k, policy=args.few_shot_policy, seed=args.few_shot_seed
                )
                prompts.append(build_few_shot_prompt(ex, support, args.metadata_strategy))
                support_id_lists.append([example_identifier(s) for s in support])
            else:
                prompts.append(build_translation_prompt(ex, args.metadata_strategy))
                support_id_lists.append([])

        results: List[Dict[str, Any]] = []
        total_start = time.time()
        processed = 0

        for batch_examples, batch_prompts, batch_support_ids in zip(
            batched(data, args.batch_size),
            batched(prompts, args.batch_size),
            batched(support_id_lists, args.batch_size),
        ):
            batch_outputs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )

            for ex, prompt, pred, support_ids in zip(batch_examples, batch_prompts, batch_outputs, batch_support_ids):
                reference = clean_code(ex.get("output", ""))
                metric_scores = compute_metrics(metrics_obj, pred, reference)
                failure_category = classify_failure(
                    pred, reference, metric_scores["sv"], metric_scores["sm"], metric_scores["vc"]
                )

                row = {
                    "dataset": dataset_name,
                    "model": args.model,
                    "timestamp": timestamp,
                    "mode": mode_tag,
                    "decoding": {
                        "do_sample": False,
                        "temperature": 0.0,
                        "num_beams": 1,
                        "repetition_penalty": 1.0,
                        "max_new_tokens": args.max_new_tokens,
                        "max_input_length": args.max_input_length,
                        "load_in_4bit": args.load_in_4bit,
                    },
                    "prompting": {
                        "metadata_strategy": args.metadata_strategy,
                        "few_shot_k": args.few_shot_k,
                        "few_shot_policy": args.few_shot_policy if args.few_shot_k > 0 else None,
                        "few_shot_source": args.few_shot_source if args.few_shot_k > 0 else None,
                        "few_shot_support_example_ids": support_ids,
                    },
                    "source_path": str(dataset_path),
                    "example_id": example_identifier(ex),
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                    "reference_output": reference,
                    "prediction": pred,
                    "metrics": metric_scores,
                    "failure_category": failure_category,
                    "metadata": ex.get("metadata", {}),
                    "prompt": prompt,
                }
                results.append(row)

            processed += len(batch_examples)
            print(f"Processed {processed}/{len(data)}")

        total_time = time.time() - total_start

        output_jsonl = (
            output_dir
            / f"{dataset_name}__{model_slug}__{mode_tag}__{args.metadata_strategy}__{timestamp}.jsonl"
        )
        output_summary = (
            output_dir
            / f"{dataset_name}__{model_slug}__{mode_tag}__{args.metadata_strategy}__{timestamp}__summary.json"
        )

        save_jsonl(output_jsonl, results)

        dataset_summary = summarize_results(results)
        dataset_summary.update(
            {
                "dataset": dataset_name,
                "dataset_path": str(dataset_path),
                "model": args.model,
                "metadata_strategy": args.metadata_strategy,
                "mode": mode_tag,
                "timestamp": timestamp,
                "prediction_file": str(output_jsonl),
                "total_time_sec": total_time,
                "run_config": build_run_metadata(args, registry_path, [dataset_name], model),
            }
        )
        save_json(output_summary, dataset_summary)
        overall_summary["datasets"][dataset_name] = dataset_summary

        print(f"Saved predictions to: {output_jsonl}")
        print(f"Saved summary to:     {output_summary}")
        print(
            "Averages: "
            f"SV={dataset_summary['avg_sv']:.3f}, "
            f"SM={dataset_summary['avg_sm']:.3f}, "
            f"VC={dataset_summary['avg_vc']:.3f}, "
            f"Overall={dataset_summary['avg_overall']:.3f}"
        )
        print(f"Failure categories: {dataset_summary['failure_counts']}")

    overall_path = (
        output_dir
        / f"overall__{model_slug}__{mode_tag}__{args.metadata_strategy}__{timestamp}.json"
    )
    save_json(overall_path, overall_summary)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved overall summary to: {overall_path}")


if __name__ == "__main__":
    main()
