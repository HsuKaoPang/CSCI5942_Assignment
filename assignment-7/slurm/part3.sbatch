#!/bin/bash
#SBATCH --job-name=a7-part3
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
# Part 3 in one job, the README's five commands in order:
#   1. GRPO with the leash (beta 0.04), Gemini judge, 100 steps
#   2. eval.py on out/sft          -> results/sft.json
#   3. eval.py on out/judge-leash  -> results/judge-leash.json
#   4. curves.png for the run
#
#   cd ~/assignment-7 && sbatch slurm/part3.sbatch
# Every judge call is cached in out/judge-cache.json, so a rerun after a
# crash costs nothing for the calls already made.
# ---------------------------------------------------------------------
set -euo pipefail
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set in the shell that ran sbatch}"
mkdir -p logs results
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "===== 1. GRPO, judge-leash (beta 0.04) ====="
python grpo.py --config configs/judge-leash.json --judge gemini

echo; echo "===== 2. eval out/sft ====="
python eval.py --run out/sft --judge gemini

echo; echo "===== 3. eval out/judge-leash ====="
python eval.py --run out/judge-leash --judge gemini

echo; echo "===== 4. curves ====="
python plot_run.py out/judge-leash

echo; echo "===== done ====="
