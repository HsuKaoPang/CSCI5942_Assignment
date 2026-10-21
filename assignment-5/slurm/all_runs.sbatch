#!/bin/bash
#SBATCH --job-name=a5-all
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Parts 3, 4 and 5: every graded training run, then every eval, in one
# job. A failed step is logged and the rest still run.
#   sbatch slurm/all_runs.sbatch
# ---------------------------------------------------------------------
set -uo pipefail
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step() { echo "=== $(date +%T) $*"; srun "$@" || echo "!!! FAILED: $*"; }

RUNS="gpt2-medium-full lora-r1 lora-r4 lora-r16 lora-r64 budget-attn-r16 budget-mlp-r12"
for name in $RUNS; do
    step python train.py --config configs/$name.json
done
step python eval.py --base gpt2-medium
for name in $RUNS; do
    step python eval.py --run out/$name
done
step python plot_knee.py
echo "=== $(date +%T) done"
