#!/bin/bash
#SBATCH --job-name=sim_batches_benchmark_only
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-119
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myriam.lizotte@mila.quebec

cd /home/mila/m/myriam.lizotte/RF-MALI
source ~/envs/env_fosta/bin/activate

noise_stds=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0)
dropout_probs=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
n_components=(2)

noise_index=$((SLURM_ARRAY_TASK_ID / 10))
dropout_index=$((SLURM_ARRAY_TASK_ID % 10))
component_index=0

noise=${noise_stds[$noise_index]}
dropout=${dropout_probs[$dropout_index]}
component=${n_components[$component_index]}



echo "Running noise=$noise dropout=$dropout component=$component"

python3 scripts/simulated_batches_benchmark_only.py \
    --noise "$noise" \
    --dropout "$dropout" \
    --components "$component"