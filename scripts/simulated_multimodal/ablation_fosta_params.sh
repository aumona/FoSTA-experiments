#!/bin/bash
#SBATCH --job-name=ablation_fosta_params
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=00:30:00
##SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-449%15

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
source ~/envs/env_fosta/bin/activate

datasets=(
  balance_scale breast_cancer crx diabetes ecoli_5 flare1 glass
  heart_disease heart_failure hepatitis ionosphere parkinsons
  seeds iris tic-tac-toe
) # 15 datasets

splits=(random importance alternating_importance add_noise_features distort rotate) # 6 splits

seeds=(690349 8925503 41184989 14532986 29501205) # 5 seeds

task_id=$SLURM_ARRAY_TASK_ID

n_datasets=${#datasets[@]}   # 15
n_splits=${#splits[@]}       # 6
n_seeds=${#seeds[@]}         # 5

dataset_index=$(( task_id / (n_splits * n_seeds) ))
rem=$(( task_id % (n_splits * n_seeds) ))

split_index=$(( rem / n_seeds ))
seed_index=$(( rem % n_seeds ))

dataset=${datasets[$dataset_index]}
split=${splits[$split_index]}
seed=${seeds[$seed_index]}

echo "Running dataset=$dataset split=$split"


python3 scripts/ablation_fosta_params_on_simulated_multimodal.py \
    --dataset "$dataset" \
    --split "$split" \
    --seed "$seed" \
    -t 2 \
    --savename "ablation_fosta_params/${SLURM_ARRAY_JOB_ID}"
