#!/bin/bash
#SBATCH --job-name=real_batches_20pct_masking
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
## SBATCH --partition=short-unkillable
#SBATCH --mem=16Gb
#SBATCH --array=0-119%4  

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
source ~/envs/env_fosta_neurips/bin/activate
# source ~/envs/env_rfmali/bin/activate


batches=("1" "2" "3" "4" "5" "6" "A1" "A2" "A3" "A4" "A5" "A6" "B1" "B2" "B3" "B4")

# Build list of unique pairs
pairs=()
for ((i=0; i<${#batches[@]}; i++)); do
  for ((j=i+1; j<${#batches[@]}; j++)); do
    pairs+=("${batches[i]} ${batches[j]}")
  done
done

# Select pair for this job
pair=${pairs[$SLURM_ARRAY_TASK_ID]}
batch1=$(echo $pair | awk '{print $1}')
batch2=$(echo $pair | awk '{print $2}')

echo "Running pair: $batch1 vs $batch2"

python3 scripts/real_batches_lung/real_batches_lung.py \
  --batch1 "$batch1" \
  --batch2 "$batch2" \
  --components 2 \
  --globalmasking 0.2 \
  --savename "real_batches_20pct_masking"