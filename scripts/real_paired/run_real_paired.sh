#!/bin/bash
#SBATCH --job-name=real_paired
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
source ~/envs/env_fosta/bin/activate

seeds=(690349 8925503 41184989 14532986 29501205) # 5 seeds

#seeds=(14532986 29501205) # 5 seeds
for seed in "${seeds[@]}"; do
    echo "Running real paired with seed=$seed"
    python3 scripts/real_paired.py -t=2 --seed=$seed

done