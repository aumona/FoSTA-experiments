#!/bin/bash
#SBATCH --job-name=sim_batches
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate


python3 scripts/simulated_batches.py \
    --noise 0.5 \
    --dropout 0.5 \
    --components 2 \
    --seed 3008874 \
    --savename "simulated_batches_special_request"
