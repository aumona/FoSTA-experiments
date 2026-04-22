#!/bin/bash
#SBATCH --job-name=real_batches_investigation
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=20Gb
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-15 

cd /home/mila/m/myriam.lizotte/RF-MALI
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate


batches=("1" "2" "3" "4" "5" "6" "A1" "A2" "A3" "A4" "A5" "A6" "B1" "B2" "B3" "B4") # 16 batches

# Select pair for this job
batch=${batches[$SLURM_ARRAY_TASK_ID]}

python3 scripts/real_batches_investigation_plots_metrics.py \
  --batch "$batch" 