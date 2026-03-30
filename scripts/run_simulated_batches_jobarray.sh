#!/bin/bash
#SBATCH --job-name=sim_batches
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-109
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myriam.lizotte@mila.quebec

cd /home/mila/m/myriam.lizotte/RF-MALI
source ~/envs/env_fosta/bin/activate

# -------------------------
# Define hyperparameters
# -------------------------

noise_stds=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
dropout_probs=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 
n_components=(2) # 3 10 20)

# -------------------------
# Get lengths automatically
# -------------------------

n_noise=${#noise_stds[@]}
n_dropout=${#dropout_probs[@]}
n_comp=${#n_components[@]}

# Total jobs
total_jobs=$((n_noise * n_dropout * n_comp))

echo "Total jobs should be: $total_jobs"
echo "Current task ID: $SLURM_ARRAY_TASK_ID"

# -------------------------
# Index mapping (automatic)
# -------------------------

noise_index=$((SLURM_ARRAY_TASK_ID / (n_dropout * n_comp)))
dropout_index=$(((SLURM_ARRAY_TASK_ID / n_comp) % n_dropout))
component_index=$((SLURM_ARRAY_TASK_ID % n_comp))

# -------------------------
# Extract values
# -------------------------

noise=${noise_stds[$noise_index]}
dropout=${dropout_probs[$dropout_index]}
component=${n_components[$component_index]}

echo "noise=$noise"
echo "dropout=$dropout"
echo "component=$component"

python3 scripts/simulated_batches_benchmark_one.py \
    --noise "$noise" \
    --dropout "$dropout" \
    --components "$component" \
    --savename "simulated_batches/${SLURM_ARRAY_JOB_ID}"
