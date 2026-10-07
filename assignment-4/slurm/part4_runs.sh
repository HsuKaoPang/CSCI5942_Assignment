#!/bin/bash
# ---------------------------------------------------------------------
# Part 4: every `huge` mapping submitted, two gpuA40x4 nodes (8 A40s).
# Model, data, seed, global batch 128 and optimizer are unchanged;
# only the mapping varies. Results land in out/huge-<tag>/summary.json.
#
#   bash slurm/part4_runs.sh        # submits all of them
#
# Best configuration: dp=4, tp=2, pp=1, overlapped DDP (tag dp4tp2),
# 12.8 s/step, 10,231 tok/s, 38.2 GiB peak.
# ---------------------------------------------------------------------
set -euo pipefail
C="--steps 6 --warmup-steps 1"

# Pure data parallel: runs out of memory, as expected (60.1 GiB of state per GPU).
sbatch --time=00:10:00 slurm/part4_a40.sbatch $C --dp 8 --tp 1 --pp 1 --ddp naive --tag oom

# Best: shard only as much as needed to fit (tp*pp = 2), give the rest to DP.
sbatch slurm/part4_a40.sbatch $C --dp 4 --tp 2 --pp 1 --ddp overlapped --tag dp4tp2

sbatch slurm/part4_a40.sbatch $C --dp 2 --tp 1 --pp 4 --ddp overlapped --schedule 1f1b --tag dp2pp4
sbatch slurm/part4_a40.sbatch $C --dp 2 --tp 2 --pp 2 --ddp overlapped --schedule 1f1b --tag dp2tp2pp2

# Follow-ups: TP vs. PP at the same memory split, more TP, no DP, and naive DDP on the best mapping.
sbatch --time=00:15:00 slurm/part4_a40.sbatch $C --dp 4 --tp 1 --pp 2 --ddp overlapped --schedule 1f1b --tag dp4pp2
sbatch --time=00:15:00 slurm/part4_a40.sbatch $C --dp 2 --tp 4 --pp 1 --ddp overlapped --tag dp2tp4
sbatch --time=00:15:00 slurm/part4_a40.sbatch $C --dp 1 --tp 4 --pp 2 --ddp none --schedule 1f1b --tag dp1tp4pp2
sbatch --time=00:15:00 slurm/part4_a40.sbatch $C --dp 4 --tp 2 --pp 1 --ddp naive --tag dp4tp2-naive
