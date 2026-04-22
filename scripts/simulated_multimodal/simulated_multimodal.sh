#!/bin/bash
#SBATCH --job-name=sim_multimodal
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=00:30:00
##SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-450
#SBATCH --mail-type=END,FAIL

cd /home/mila/m/myriam.lizotte/RF-MALI
source ~/envs/env_fosta/bin/activate

datasets=(
  balance_scale breast_cancer crx diabetes ecoli flare1 glass
  heart_disease heart_failure hepatitis ionosphere parkinsons
  seeds iris tic-tac-toe
) # 15 datasets

splits=(random importance alternating_importance add_noise_features distort rotate) # 6 splits

seeds=(690349 8925503 41184989 14532986 29501205) # 5 seeds


# Map array index → dataset/split
dataset_index=$((SLURM_ARRAY_TASK_ID / ${#splits[@]}))
split_index=$((SLURM_ARRAY_TASK_ID % ${#splits[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))


dataset=${datasets[$dataset_index]}
split=${splits[$split_index]}
seed=${seeds[$seed_index]}

echo "Running dataset=$dataset split=$split seed=$seed"

folder_name=$(date +%Y-%m-%d_%H-%M-%S)
python3 scripts/simulated_multimodal.py \
    --dataset "$dataset" \
    --split "$split" \
    --seed "$seed" \
    -t 2 \
    --savename "simulated_multimodal/${SLURM_ARRAY_JOB_ID}"
