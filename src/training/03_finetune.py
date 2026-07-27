#!/usr/bin/env python3
"""
03_finetune.py

Stage 3b: LoRA/PEFT fine-tuning on the SFT data produced by
03_build_finetune_data.py.

Inputs:
    - <split>_sft_<metadata_strategy>.jsonl from 03_build_finetune_data.py

Outputs:
    - checkpoints/<run_name>/run_config.json
    - checkpoints/<run_name>/adapter/  (LoRA adapter + tokenizer)
    - checkpoints/<run_name>/train_summary.json

Typical usage:
    python src/training/03_finetune.py \\
        --train-file datasets/processed/finetune/train_sft_none.jsonl \\
        --val-file datasets/processed/finetune/val_sft_none.jsonl \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --load-in-4bit
"""

from __future__ import annotations

import argparse
import inspect
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class SFTDataset(torch.utils.data.Dataset):
    # Implements the instructional label masking described in Section 5: Experimental Setup - prompt tokens
    # receive label −100 (zero gradient contribution); loss is computed exclusively over
    # the Isabelle/HOL completion tokens. See the labels assignment below.
    """Pre-tokenizes (prompt, completion) pairs with loss masked over the prompt span."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_seq_length: int) -> None:
        self.examples: List[Dict[str, List[int]]] = []
        self.dropped_too_long = 0

        eos_id = tokenizer.eos_token_id
        for rec in records:
            prompt_ids = tokenizer(rec["prompt"], add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(rec["completion"], add_special_tokens=False)["input_ids"]
            completion_ids = completion_ids + [eos_id]

            if len(completion_ids) >= max_seq_length:
                self.dropped_too_long += 1
                continue

            budget_for_prompt = max_seq_length - len(completion_ids)
            if len(prompt_ids) > budget_for_prompt:
                prompt_ids = prompt_ids[-budget_for_prompt:]

            input_ids = prompt_ids + completion_ids
            # Corresponds to Section 5: Experimental Setup - label −100 masks the prompt span
            # so cross-entropy loss is computed only over the Isabelle/HOL completion.
            labels = [-100] * len(prompt_ids) + completion_ids

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[i, :length] = torch.tensor(item["labels"], dtype=torch.long)
            attention_mask[i, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def resolve_dtype_and_precision() -> Dict[str, Any]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return {"torch_dtype": torch.bfloat16, "bf16": True, "fp16": False}
        return {"torch_dtype": torch.float16, "bf16": False, "fp16": True}
    print("WARNING: CUDA not available. Falling back to fp32 on CPU -- training will be very slow. "
          "This path exists for local smoke tests only, not real cluster runs.")
    return {"torch_dtype": torch.float32, "bf16": False, "fp16": False}


def build_training_arguments(args, run_dir: Path, has_val: bool, precision: Dict[str, Any]):
    from transformers import TrainingArguments

    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "eval_strategy" if "eval_strategy" in sig_params else "evaluation_strategy"

    kwargs: Dict[str, Any] = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "report_to": [] if args.report_to.lower() == "none" else [args.report_to],
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    kwargs[eval_strategy_key] = args.save_strategy if has_val else "no"
    if has_val:
        kwargs["per_device_eval_batch_size"] = args.per_device_train_batch_size

    return TrainingArguments(**kwargs)


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}

    if args.hf_cache_dir:
        cache_dir = str(Path(args.hf_cache_dir).expanduser().resolve())
        tokenizer_kwargs["cache_dir"] = cache_dir
        model_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, **tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    precision = resolve_dtype_and_precision()

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "BitsAndBytesConfig not available. Install compatible transformers/bitsandbytes."
            ) from e

        # Corresponds to Section 5: Experimental Setup - 4-bit NF4 quantization with bfloat16 compute dtype,
        # enabling 7B-class models to train on a single 16 GB V100.
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=precision["torch_dtype"],
        )
    else:
        model_kwargs["torch_dtype"] = precision["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    return model, tokenizer, precision


def attach_lora(model, args):
    # Corresponds to Section 5: Experimental Setup - LoRA adapter with rank r=16, α=32,
    # dropout=0.05, targeting all seven projection modules (q,k,v,o,gate,up,down_proj).
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA fine-tuning but is not installed. "
            "Install it in the active environment (pip install peft) before running this script."
        ) from e

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_run_metadata(args, run_dir: Path, model, precision: Dict[str, Any]) -> Dict[str, Any]:
    cuda_info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        cuda_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "script": "03_finetune.py",
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "model": args.model,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "limit": args.limit,
        "precision": {k: str(v) for k, v in precision.items()},
        "cuda": cuda_info,
        "model_class": model.__class__.__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LM on TIA SFT data.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train_sft_*.jsonl")
    parser.add_argument("--val-file", type=str, default=None, help="Optional path to val_sft_*.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Hugging Face model ID or local model path")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--run-name", type=str, default=None,
                        help="Defaults to '<model_slug>_<timestamp>'")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str,
                        default=",".join(DEFAULT_LORA_TARGET_MODULES),
                        help="Comma-separated module names LoRA adapters attach to")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--report-to", type=str, default="none",
                        help="'none' disables external logging integrations (default; "
                        "cluster nodes may lack internet access)")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per split, for smoke tests")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    args = parser.parse_args()

    import transformers

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(":", "_")
    run_name = args.run_name or f"{model_slug}_{timestamp}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("03_finetune.py")
    print("=" * 80)
    print(f"Hostname:           {socket.gethostname()}")
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Model:              {args.model}")
    print(f"Train file:         {args.train_file}")
    print(f"Val file:           {args.val_file}")
    print(f"Run dir:            {run_dir}")
    print(f"Max seq length:     {args.max_seq_length}")
    print(f"Epochs:             {args.num_train_epochs}")
    print(f"LoRA r/alpha/drop:  {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Load in 4-bit:      {args.load_in_4bit}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    print("=" * 80)

    train_path = Path(args.train_file).expanduser().resolve()
    train_records = load_jsonl(train_path)
    print(f"Loaded {len(train_records)} train record(s) from {train_path}")

    val_records: Optional[List[Dict[str, Any]]] = None
    if args.val_file:
        val_path = Path(args.val_file).expanduser().resolve()
        val_records = load_jsonl(val_path)
        print(f"Loaded {len(val_records)} val record(s) from {val_path}")

    if args.limit is not None:
        train_records = train_records[: args.limit]
        if val_records is not None:
            val_records = val_records[: args.limit]
        print(f"--limit applied: train={len(train_records)}, val={len(val_records) if val_records else 0}")

    print("Loading model and tokenizer...")
    start_load = time.time()
    model, tokenizer, precision = load_model_and_tokenizer(args)
    print(f"Model loaded in {time.time() - start_load:.1f}s")

    print("Tokenizing train split...")
    train_dataset = SFTDataset(train_records, tokenizer, args.max_seq_length)
    print(f"  Usable: {len(train_dataset)}, dropped (too long): {train_dataset.dropped_too_long}")
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after tokenization/truncation. Aborting.")

    val_dataset = None
    if val_records:
        print("Tokenizing val split...")
        val_dataset = SFTDataset(val_records, tokenizer, args.max_seq_length)
        print(f"  Usable: {len(val_dataset)}, dropped (too long): {val_dataset.dropped_too_long}")
        if len(val_dataset) == 0:
            print("WARNING: val dataset is empty after tokenization; disabling evaluation.")
            val_dataset = None

    print("Attaching LoRA adapters...")
    model = attach_lora(model, args)

    run_metadata = build_run_metadata(args, run_dir, model, precision)
    run_metadata["train_examples_usable"] = len(train_dataset)
    run_metadata["train_examples_dropped_too_long"] = train_dataset.dropped_too_long
    if val_dataset is not None:
        run_metadata["val_examples_usable"] = len(val_dataset)
        run_metadata["val_examples_dropped_too_long"] = val_dataset.dropped_too_long
    save_json(run_dir / "run_config.json", run_metadata)
    print(f"Saved effective run config to {run_dir / 'run_config.json'} (before training starts)")

    training_args = build_training_arguments(args, run_dir, has_val=val_dataset is not None, precision=precision)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Starting training...")
    start_train = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    total_train_time = time.time() - start_train
    print(f"Training finished in {total_train_time:.1f}s")

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter + tokenizer to {adapter_dir}")

    eval_metrics = None
    if val_dataset is not None:
        print("Running final evaluation on val split...")
        eval_metrics = trainer.evaluate()
        print(f"Final eval metrics: {eval_metrics}")

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter_dir),
        "total_train_time_sec": total_train_time,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }
    save_json(run_dir / "train_summary.json", summary)

    print("=" * 80)
    print("DONE")
    print(f"Run dir:      {run_dir}")
    print(f"Adapter:      {adapter_dir}")
    print(f"Summary:      {run_dir / 'train_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
