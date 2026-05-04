#!/bin/bash

cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA

timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}

log_dir=/NOBACKUP/lizottem/fosta/logs/real_batches_lung/masking/$timestamp
# create log directory
mkdir -p "$log_dir"
# batches=("1" "2" "3" "4" "5" "6" "A1" "A2" "A3" "A4" "A5" "A6" "B1" "B2" "B3" "B4")
batches=("A1" "A2" "A3" "A4" "A5" "A6")
seeds=(11784 39041 56089 79121 4386721)

# Build list of unique pairs
pairs=()
for ((i=0; i<${#batches[@]}; i++)); do
  for ((j=i+1; j<${#batches[@]}; j++)); do
    pairs+=("${batches[i]} ${batches[j]}")
  done
done


for seed in "${seeds[@]}"; do
  for i in "${!pairs[@]}"; do
  
    # Select pair for this job
    pair=${pairs[$i]}
    batch1=$(echo $pair | awk '{print $1}')
    batch2=$(echo $pair | awk '{print $2}')

    echo "Running pair: $batch1 vs $batch2"
    savename="real_batches_lung/masking/${timestamp}"
    
    python3 scripts/real_batches_lung/real_batches_lung.py \
      --batch1 "$batch1" \
      --batch2 "$batch2" \
      --seed "$seed" \
      --components 2 \
      --globalmasking 0.2 \
      --savename "$savename" \
      > "$log_dir/"$batch1"_"$batch2"_"$seed".out" \
      2> "$log_dir/"$batch1"_"$batch2"_"$seed".err"

  done
done