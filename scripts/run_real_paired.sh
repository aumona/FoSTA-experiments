#!/bin/bash
#SBATCH --job-name=real_paired
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/real_paired/%j.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/real_paired/%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --mail-type=END,FAIL

cd /home/mila/m/myriam.lizotte/RF-MALI
source ~/envs/env_fosta/bin/activate

seeds=(690349 8925503 41184989 14532986 29501205) # 5 seeds

#seeds=(14532986 29501205) # 5 seeds
for seed in "${seeds[@]}"; do
    echo "Running real paired with seed=$seed"
    python3 scripts/real_paired.py -t=2 --seed=$seed

done