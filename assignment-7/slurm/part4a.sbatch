#!/bin/bash
#SBATCH --job-name=a7-part4a
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 4, first half, plus the Part 3 before/after pair:
#   1. greedy description of the default MR from out/sft and out/judge-leash
#      (Part 3: "two descriptions of the same MR before and after")
#   2. GRPO without the leash (beta 0), Gemini judge, 100 steps
#   3. eval.py on out/judge-noleash -> results/judge-noleash.json
#   4. greedy description of the same MR from out/judge-noleash
#
#   cd ~/assignment-7 && sbatch slurm/part4a.sbatch
# plot_run.py is left out: Delta's matplotlib fails to render text
# (FT_Render_Glyph raster overflow), so the curves are drawn locally.
# ---------------------------------------------------------------------
set -euo pipefail
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set in the shell that ran sbatch}"
mkdir -p logs results
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "===== 1. before / after (greedy, default MR) ====="
{
  echo "--- out/sft ---";         python sample.py --run out/sft
  echo "--- out/judge-leash ---"; python sample.py --run out/judge-leash
} | tee results/part3-before-after.txt

echo; echo "===== 2. GRPO, judge-noleash (beta 0) ====="
python grpo.py --config configs/judge-noleash.json --judge gemini

echo; echo "===== 3. eval out/judge-noleash ====="
python eval.py --run out/judge-noleash --judge gemini

echo; echo "===== 4. greedy from out/judge-noleash, same MR ====="
{
  echo "--- out/judge-noleash ---"; python sample.py --run out/judge-noleash
} | tee -a results/part3-before-after.txt

echo; echo "===== done ====="
