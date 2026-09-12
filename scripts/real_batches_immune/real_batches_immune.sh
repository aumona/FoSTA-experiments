#!/bin/bash
#SBATCH --job-name=real_batches_immune
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --array=0-44

PROJECT_ROOT="${RF_MALI_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT" || exit 1
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate


n_batches=10

# Build list of unique pairs
pairs=()
for ((i=0; i<${n_batches}; i++)); do
  for ((j=i+1; j<${n_batches}; j++)); do
    pairs+=("$i $j")
  done
done

# Select pair for this job
pair=${pairs[$SLURM_ARRAY_TASK_ID]}
batch1=$(echo $pair | awk '{print $1}')
batch2=$(echo $pair | awk '{print $2}')

echo "Running pair: $batch1 vs $batch2"

python3 scripts/real_batches_immune.py \
  --batch1idx $batch1 \
  --batch2idx $batch2 \
  --components 2 \
  --savename "real_batches_immune/${SLURM_ARRAY_JOB_ID}" \
  --globalmasking 0