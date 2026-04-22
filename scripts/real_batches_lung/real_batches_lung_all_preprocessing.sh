#!/bin/bash
#SBATCH --job-name=real_batches_pca_hvg
#SBATCH --output=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.out
#SBATCH --error=/home/mila/m/myriam.lizotte/scratch/RF-MALI/logs/%x/%A/%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20Gb
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-119  

cd /home/mila/m/myriam.lizotte/RF-MALI
# source ~/envs/env_fosta/bin/activate
source ~/envs/env_rfmali/bin/activate


batches=("1" "2" "3" "4" "5" "6" "A1" "A2" "A3" "A4" "A5" "A6" "B1" "B2" "B3" "B4") # 16 batches, 120 unique pairs

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

python3 scripts/real_batches_benchmark.py \
  --batch1 "$batch1" \
  --batch2 "$batch2" \
  --hvg False \
  --pca False \
  --components 2 \
  --savename "real_batches_no_pca_no_hvg" 


python3 scripts/real_batches_benchmark.py \
  --batch1 "$batch1" \
  --batch2 "$batch2" \
  --hvg False \
  --pca True \
  --components 2 \
  --savename "real_batches_with_pca_no_hvg" 

python3 scripts/real_batches_benchmark.py \
  --batch1 "$batch1" \
  --batch2 "$batch2" \
  --hvg True \
  --pca False \
  --components 2 \
  --savename "real_batches_no_pca_with_hvg" 