#!/bin/bash

cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA

timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
log_dir=/home/lizottem/RF-MALI/logs/real_batches_immune/$timestamp
# create log directory
mkdir -p "$log_dir"

n_batches=10

# Build list of unique pairs
pairs=()
for ((i=0; i<${n_batches}; i++)); do
  for ((j=i+1; j<${n_batches}; j++)); do
    pairs+=("$i $j")
  done
done

for seed in "${seeds[@]}"; do
  for i in "${!pairs[@]}"; do
    # Select pair for this job
    pair=${pairs[$i]}
    batch1=$(echo $pair | awk '{print $1}')
    batch2=$(echo $pair | awk '{print $2}')

    echo "Running pair: $batch1 vs $batch2"

    python3 scripts/real_batches_immune/real_batches_immune.py \
      --batch1idx $batch1 \
      --batch2idx $batch2 \
      --components 2 \
      --globalmasking 0\
      --savename "real_batches_immune/${timestamp}" \
      --seed "$seed" \
        > "$log_dir/"$batch1"_"$batch2"_"$seed".out" \
        2> "$log_dir/"$batch1"_"$batch2"_"$seed".err"

  done
done