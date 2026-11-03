#!/bin/bash
#SBATCH --job-name=a6-all
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Every run the report needs (Parts 1, 2 and 3), in one job. A failed
# step is logged and the rest still run.
#   sbatch slurm/run_all.sbatch
# ---------------------------------------------------------------------
set -uo pipefail
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step() { echo; echo "=== $(date +%T) $*"; srun "$@" || echo "!!! FAILED: $*"; }

step python check_env.py
step python -m pytest tests/ -q

# Part 1: the KV cache
step python sample.py --model gpt2-xl --tokens 200
step python sample.py --model gpt2-xl --tokens 800
step python sample.py --prompt "The capital of France is" "Once upon a time" --tokens 50

# Part 2: PagedAttention
step python bench_kv.py --model gpt2-xl --batch 32 --tokens 128 --block-size 8 16 32 64 128 256

# Part 3: continuous batching
step python bench_serve.py --kv-gb 4
step python bench_serve.py --kv-gb 1
step python bench_serve.py --kv-gb 1 --engine continuous --optimistic
echo; echo "=== $(date +%T) done"
