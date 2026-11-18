#!/bin/bash
#SBATCH --job-name=a7-part2
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 2 on a GPU, in one job:
#   1. the four tests
#   2. eight samples of one MR from out/sft (Part 2 deliverable)
#   3. the plumbing: grpo.py with the mock judge for 2 steps
#
#   cd ~/assignment-7 && sbatch slurm/part2.sbatch
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs results
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "===== 1. tests ====="
python -m pytest tests/ -q

echo; echo "===== 2. eight samples from out/sft ====="
python sample.py --run out/sft --n 8 | tee results/part2-samples.txt

echo; echo "===== 3. plumbing: mock judge, 2 steps ====="
python grpo.py --config configs/judge-leash.json --judge mock --steps 2

echo; echo "===== done ====="
