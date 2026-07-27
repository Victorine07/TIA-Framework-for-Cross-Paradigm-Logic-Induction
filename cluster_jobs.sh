#!/bin/bash
# cluster_jobs.sh
#
# SLURM job templates for all TIA experiment stages. (Replace the three placeholders before submitting)
#
#   SLURM_ACCOUNT   your cluster account / project allocation
#   CONDA_ENV_PATH  absolute path to the conda environment
#   PROJECT_DIR     absolute path to the cloned repository root
#
# Tested on CUDA 11.8.  Adjust --partition and --gres to match your
# cluster's scheduler configuration.
#
# Submit individual stages with:
#   sbatch <(sed -n '/^### STAGE/,/^### END/p' cluster_jobs.sh | grep -v '^###')
# or copy the relevant block into its own .sbatch file.
#
# ─────────────────────────────────────────────────────────────────────────────
# Common header used in every job (shown once for reference)
# ─────────────────────────────────────────────────────────────────────────────
#   #SBATCH --account=SLURM_ACCOUNT
#   module purge
#   module load CUDA/11.8
#   source /path/to/miniconda3/etc/profile.d/conda.sh
#   conda activate CONDA_ENV_PATH
#   cd PROJECT_DIR
#   mkdir -p logs
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# STAGE 0 — Dataset preparation (CPU only, ~15 min)
# =============================================================================
# Merges per-cipher JSONL extractions into a unified raw dataset.
# Resources: 2 CPUs, 8 GB RAM, 30 min wall time.
# No GPU required.
# =============================================================================
cat << 'EOF_STAGE0'
#!/bin/bash
#SBATCH --job-name=tia-prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=SLURM_ACCOUNT

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

python scripts/00_prepare_datasets.py
python scripts/01_create_splits.py --seed 42
python scripts/02_build_eval_registry.py --root . --tag aaai_v1
EOF_STAGE0


# =============================================================================
# STAGE 1 — Build SFT training data (CPU only, ~20 min)
# =============================================================================
# Builds prompt/completion pairs under a given metadata strategy.
# Run once per metadata strategy (none / full / structured / algorithmic / alljson).
# Resources: 2 CPUs, 8 GB RAM, 30 min wall time.
# No GPU required.
# =============================================================================
cat << 'EOF_STAGE1'
#!/bin/bash
#SBATCH --job-name=tia-build-sft
#SBATCH --output=logs/build_sft_%j.out
#SBATCH --error=logs/build_sft_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=SLURM_ACCOUNT

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

# Change --metadata-strategy to run other ablation conditions
python src/training/03_build_finetune_data.py \
  --splits train,val \
  --metadata-strategy none \
  --tag aaai_v1
EOF_STAGE1


# =============================================================================
# STAGE 2a — Zero-shot / few-shot baseline: Qwen2.5-Coder-7B
# =============================================================================
# Resources: 1 × GPU (≥16 GB VRAM), 4 CPUs, 32 GB RAM, 8 h wall time.
# 4-bit NF4 quantization keeps peak VRAM under 10 GB on this model.
# =============================================================================
cat << 'EOF_ZS_QWEN'
#!/bin/bash
#SBATCH --job-name=tia-zs-qwen
#SBATCH --output=logs/zs_qwen_%j.out
#SBATCH --error=logs/zs_qwen_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

# Zero-shot (remove --few-shot-k to disable few-shot)
python src/experiments/04_run_zero_shot_baseline.py \
  --dataset test,unseen_lea,unseen_rectangle,unseen_xtea \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --hf-cache-dir ./models \
  --batch-size 1 \
  --max-input-length 2048 \
  --max-new-tokens 512 \
  --metadata-strategy none \
  --output-dir results/zero_shot/qwen \
  --load-in-4bit

# Few-shot sweep over k=1,2,4 (add to a separate job or loop here)
# for K in 1 2 4; do
#   python src/experiments/04_run_zero_shot_baseline.py \
#     --dataset test,unseen_lea,unseen_rectangle,unseen_xtea \
#     --model Qwen/Qwen2.5-Coder-7B-Instruct \
#     --hf-cache-dir ./models \
#     --batch-size 1 \
#     --max-input-length 2048 \
#     --max-new-tokens 512 \
#     --metadata-strategy none \
#     --output-dir results/few_shot/qwen \
#     --load-in-4bit \
#     --few-shot-k "${K}" \
#     --few-shot-policy same_family \
#     --few-shot-source train \
#     --few-shot-seed 42
# done
EOF_ZS_QWEN


# =============================================================================
# STAGE 2b — Zero-shot / few-shot baseline: DeepSeek-Coder-V2-Lite
# =============================================================================
# Resources: 2 × GPU (≥16 GB VRAM each), 4 CPUs, 48 GB RAM, 16 h wall time.
# DeepSeek requires --trust-remote-code and benefits from 2 GPUs for
# pipeline parallelism at 4-bit quantization.
# =============================================================================
cat << 'EOF_ZS_DS'
#!/bin/bash
#SBATCH --job-name=tia-zs-deepseek
#SBATCH --output=logs/zs_deepseek_%j.out
#SBATCH --error=logs/zs_deepseek_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

python src/experiments/04_run_zero_shot_baseline.py \
  --dataset test,unseen_lea,unseen_rectangle,unseen_xtea \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --hf-cache-dir ./models \
  --batch-size 1 \
  --max-input-length 2048 \
  --max-new-tokens 512 \
  --metadata-strategy none \
  --output-dir results/zero_shot/deepseek \
  --load-in-4bit \
  --trust-remote-code
EOF_ZS_DS


# =============================================================================
# STAGE 3a — LoRA fine-tuning: Qwen2.5-Coder-7B
# =============================================================================
# Resources: 1 × GPU (≥16 GB VRAM), 4 CPUs, 48 GB RAM, 12 h wall time.
# LoRA config: r=16, alpha=32, dropout=0.05, 4-bit NF4 base model.
# =============================================================================
cat << 'EOF_FT_QWEN'
#!/bin/bash
#SBATCH --job-name=tia-ft-qwen
#SBATCH --output=logs/ft_qwen_%j.out
#SBATCH --error=logs/ft_qwen_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

python src/training/03_finetune.py \
  --train-file datasets/processed/finetune/train_sft_none.jsonl \
  --val-file   datasets/processed/finetune/val_sft_none.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --hf-cache-dir ./models \
  --load-in-4bit \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --max-seq-length 2048 \
  --output-dir checkpoints
EOF_FT_QWEN


# =============================================================================
# STAGE 3b — LoRA fine-tuning: DeepSeek-Coder-V2-Lite
# =============================================================================
# Resources: 2 × GPU (≥16 GB VRAM each), 4 CPUs, 64 GB RAM, 16 h wall time.
# Larger gradient-accumulation-steps compensates for the single-example
# per-device batch imposed by sequence length and GPU memory constraints.
# =============================================================================
cat << 'EOF_FT_DS'
#!/bin/bash
#SBATCH --job-name=tia-ft-deepseek
#SBATCH --output=logs/ft_deepseek_%j.out
#SBATCH --error=logs/ft_deepseek_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

python src/training/03_finetune.py \
  --train-file datasets/processed/finetune/train_sft_none.jsonl \
  --val-file   datasets/processed/finetune/val_sft_none.jsonl \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --hf-cache-dir ./models \
  --load-in-4bit \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --max-seq-length 1024 \
  --output-dir checkpoints \
  --trust-remote-code
EOF_FT_DS


# =============================================================================
# STAGE 4a — Fine-tuned evaluation: Qwen2.5-Coder-7B
# =============================================================================
# Resources: 1 × GPU (≥16 GB VRAM), 4 CPUs, 32 GB RAM, 4 h wall time.
# Uses the same prompt construction, normalization, and SV/SM/VC pipeline
# as the zero-shot baseline for a fair comparison.
# =============================================================================
cat << 'EOF_EVAL_QWEN'
#!/bin/bash
#SBATCH --job-name=tia-eval-ft-qwen
#SBATCH --output=logs/eval_ft_qwen_%j.out
#SBATCH --error=logs/eval_ft_qwen_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

# Set ADAPTER_PATH to the checkpoint directory produced by STAGE 3a
ADAPTER_PATH=checkpoints/qwen_sft_none

python src/experiments/04_evaluate_finetuned.py \
  --dataset val,test,unseen_lea,unseen_rectangle,unseen_xtea \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path "${ADAPTER_PATH}" \
  --hf-cache-dir ./models \
  --load-in-4bit \
  --metadata-strategy none \
  --output-dir results/finetuned/qwen
EOF_EVAL_QWEN


# =============================================================================
# STAGE 4b — Fine-tuned evaluation: DeepSeek-Coder-V2-Lite
# =============================================================================
# Resources: 2 × GPU (≥16 GB VRAM each), 4 CPUs, 48 GB RAM, 4 h wall time.
# =============================================================================
cat << 'EOF_EVAL_DS'
#!/bin/bash
#SBATCH --job-name=tia-eval-ft-deepseek
#SBATCH --output=logs/eval_ft_deepseek_%j.out
#SBATCH --error=logs/eval_ft_deepseek_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=SLURM_ACCOUNT

module purge && module load CUDA/11.8
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate CONDA_ENV_PATH
cd PROJECT_DIR && mkdir -p logs

ADAPTER_PATH=checkpoints/deepseek_sft_none

python src/experiments/04_evaluate_finetuned.py \
  --dataset val,test,unseen_lea,unseen_rectangle,unseen_xtea \
  --base-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --adapter-path "${ADAPTER_PATH}" \
  --hf-cache-dir ./models \
  --load-in-4bit \
  --trust-remote-code \
  --metadata-strategy none \
  --output-dir results/finetuned/deepseek
EOF_EVAL_DS
