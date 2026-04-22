#!/bin/bash
#SBATCH --job-name=sim_batches
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=myriam.lizotte@mila.quebec

cd /home/mila/m/myriam.lizotte/RF-MALI
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate


python3 scripts/simulated_batches.py \
    --noise 0.5 \
    --dropout 0.5 \
    --components 2 \
    --seed 3008874 \
    --savename "simulated_batches_special_request"
