#!/bin/bash
#SBATCH --job-name=a5-part1
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 1: the four 50-step memory runs, one process each so every run
# reports its own peak.
#   sbatch slurm/part1_mem.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python train.py --config configs/gpt2-small-full.json  --steps 50 --out out/mem-small-full
srun python train.py --config configs/gpt2-medium-full.json --steps 50 --out out/mem-medium-full
srun python train.py --config configs/lora-r4.json          --steps 50 --out out/mem-lora-r4
srun python train.py --config configs/lora-r64.json         --steps 50 --out out/mem-lora-r64
