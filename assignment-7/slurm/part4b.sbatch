#!/bin/bash
#SBATCH --job-name=a7-part4b
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 4, second half: train on RUBRICS["yours"] and evaluate.
#   1. Part 1's check on the new rubric (humans vs intruder, order swap)
#   2. GRPO on the 'yours' rubric (beta 0.04), Gemini judge, 100 steps
#   3. eval.py on the 'given' rubric -> results/judge-yours.json (the table row)
#   4. eval.py on the 'yours' rubric -> results/judge-yours-on-yours.json
#   5. greedy description of the default MR, appended to the before/after file
#
#   cd ~/assignment-7 && sbatch slurm/part4b.sbatch
# eval.py names its output after the run directory, so step 4 would
# overwrite step 3's file; step 4's result is renamed and step 3's restored.
# ---------------------------------------------------------------------
set -euo pipefail
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set in the shell that ran sbatch}"
mkdir -p logs results
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import judge; print('rubric yours:', judge.RUBRICS['yours'])"

echo; echo "===== 1. Part 1 check on the 'yours' rubric ====="
python judge.py --part1 --judge gemini --rubric yours

echo; echo "===== 2. GRPO, judge-yours ====="
python grpo.py --config configs/judge-yours.json --judge gemini

echo; echo "===== 3. eval out/judge-yours on the 'given' rubric ====="
python eval.py --run out/judge-yours --judge gemini
cp results/judge-yours.json results/judge-yours-given.tmp

echo; echo "===== 4. eval out/judge-yours on the 'yours' rubric ====="
python eval.py --run out/judge-yours --judge gemini --rubric yours
mv results/judge-yours.json results/judge-yours-on-yours.json
mv results/judge-yours-given.tmp results/judge-yours.json

echo; echo "===== 5. greedy from out/judge-yours, same MR ====="
{
  echo "--- out/judge-yours ---"; python sample.py --run out/judge-yours
} | tee -a results/part3-before-after.txt

echo; echo "===== done ====="
