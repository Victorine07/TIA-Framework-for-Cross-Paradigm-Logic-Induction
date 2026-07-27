# Tiered Isomorphic Alignment (TIA) for Cross-Paradigm Logic Translation

This repository contains the dataset, extraction pipeline, evaluation framework, and training code:

> **Tiered Isomorphic Alignment for Cross-Paradigm Logic Induction: Translating Imperative Cryptographic Code into Formal Isabelle/HOL Specifications**

---

## Overview

Auto-formalization fails when the structural and semantic gap between imperative code and formal logic is too large for language models to bridge directly. TIA addresses this by decomposing each cryptographic algorithm into four alignment tiers before training:

| Tier | Role | Examples |
|------|------|---------|
| **T1** | Lexical — constants and static parameters | word sizes, rotation constants, S-box entries |
| **T2** | Functional — primitive transforms | XOR round step, modular addition, S-box lookup |
| **T3** | Structural — round functions and key schedules | `primrec` round definitions, key expansion |
| **T4** | Orchestration — top-level encrypt / decrypt | full cipher interface, composition |

The dataset spans **1,084 aligned Python ↔ Isabelle/HOL component pairs** across 14 cipher variants from 5 cryptographic families (ARX, Feistel, SPN, Permutation, AEAD), with a clean 766 / 96 / 96 / 126 train / val / test / unseen split.

---

## Key Results

| Model | Regime | Overall | SV | SM | VC |
|-------|--------|---------|-----|-----|-----|
| Qwen2.5-Coder-7B | Zero-Shot | 0.203 | 0.075 | 0.325 | 0.209 |
| Qwen2.5-Coder-7B | Few-Shot k=3 | 0.602 | 0.742 | 0.618 | 0.445 |
| Qwen2.5-Coder-7B | **SFT (TIA)** | **0.969** | 0.959 | 0.978 | 0.968 |
| DeepSeek-Coder-V2-Lite | Zero-Shot | 0.267 | 0.273 | 0.297 | 0.231 |
| DeepSeek-Coder-V2-Lite | Few-Shot k=3 | 0.327 | 0.370 | 0.382 | 0.227 |
| DeepSeek-Coder-V2-Lite | **SFT (TIA)** | **0.954** | 0.949 | 0.974 | 0.939 |

TIA fine-tuned models generalize to unseen cipher families with Overall scores of **0.76–0.92**, confirming grammar transfer rather than memorization.

---

## Requirements

- Python 3.10+
- Isabelle 2025 (for formal verification; not required for dataset use or model training)
- PyTorch 2.1+ with CUDA 11.8
- See `requirements.txt` for the full dependency list

Install:
```bash
# GPU training (CUDA 11.8 wheel)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Quick Start

**1. Extract the TIA dataset from cipher sources:**
```bash
python extractors/run_all.py --output-dir datasets/raw
```

**2. Build train / val / test splits:**
```bash
python scripts/00_prepare_aaai_datasets.py
python scripts/01_create_aaai_splits.py
python scripts/02_build_eval_registry.py
```

**3. Run zero-shot baseline:**
```bash
python src/experiments/04_run_zero_shot_baseline.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset datasets/processed/test.jsonl \
  --strategy none --output-dir results/zero_shot
```

**4. Build SFT training data and fine-tune:**
```bash
python src/training/03_build_finetune_data.py
python src/training/03_finetune.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --train-file datasets/processed/finetune/train_sft_none.jsonl \
  --output-dir checkpoints
```

**5. Evaluate the fine-tuned model:**
```bash
python src/experiments/04_evaluate_finetuned.py \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path checkpoints/<run_dir> \
  --dataset test \
  --output-dir results/finetuned
```

---

## Cluster / HPC Execution

`cluster_jobs.sh` contains SLURM job templates for every pipeline stage (data prep, zero-shot, few-shot, SFT training, fine-tuned evaluation). Replace the three placeholders at the top of the file before submitting:

| Placeholder | Replace with |
|---|---|
| `SLURM_ACCOUNT` | your cluster account / project allocation |
| `CONDA_ENV_PATH` | absolute path to the conda environment |
| `PROJECT_DIR` | absolute path to the repository root |

Resource requirements per stage are documented inside each job block.

---

## Pre-trained Adapters

LoRA adapter weights (both models, `none` and `structured` metadata strategies) are available on Hugging Face.
The base models are `Qwen/Qwen2.5-Coder-7B-Instruct` and `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, available directly from Hugging Face Hub.

---

## Metrics

Three domain-specific metrics evaluate formal translation quality:

- **SV (Syntax Validity):** detects missing or malformed Isabelle/HOL entry points via delimiter-balance verification and foreign-syntax penalties.
- **SM (Semantic Match):** measures cryptographic operator recall (ROUGE-style) and operator-sequence similarity (Gestalt pattern matching).
- **VC (Value Consistency):** performs static analysis of security-critical constants — rotation amounts, bit-widths, S-box entries — with context-adaptive weighting.

Scores ≥ 0.85 are classified as **verification-ready**.

