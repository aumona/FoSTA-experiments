#!/bin/bash
#SBATCH --job-name=sim_paired_batches
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-35

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate

noise_stds=(0 0.2 0.4 0.6 0.8 1.0)
dropout_probs=(0 0.2 0.4 0.6 0.8 0.95)


noise_index=$((SLURM_ARRAY_TASK_ID / 6))
dropout_index=$((SLURM_ARRAY_TASK_ID % 6))


noise=${noise_stds[$noise_index]}
dropout=${dropout_probs[$dropout_index]}

echo "Running noise=$noise dropout=$dropout"

python3 scripts/simulated_paired_batches.py \
    --noise "$noise" \
    --dropout "$dropout" \
    --components 2