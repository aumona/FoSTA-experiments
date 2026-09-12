#!/bin/bash

# 1. Define the project root
PROJECT_ROOT="${RF_MALI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT" || exit # Exit if the directory doesn't exist

# 2. Activate Virtual Environment
# Using the absolute path is good practice
source "$PROJECT_ROOT/.venv/bin/activate"

# 3. Handle Timestamp
timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
log_dir="${RF_MALI_LOG_DIR:-$PROJECT_ROOT/logs}/real_batches_lung/$timestamp"
mkdir -p "$log_dir"

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
  for pair in "${pairs[@]}"; do
  
    # Select pair
    batch1=$(echo "$pair" | awk '{print $1}')
    batch2=$(echo "$pair" | awk '{print $2}')

    echo "🚀 Running Seed $seed | Pair: $batch1 vs $batch2"
    
    # We pass the timestamp as part of the savename 
    # to keep everything grouped under one folder in /results/
    savename="real_batches_lung/$timestamp"
    
    # Execute (Note: we use the path relative to PROJECT_ROOT)
    python3 scripts/real_batches_lung/real_batches_lung.py \
      --batch1 "$batch1" \
      --batch2 "$batch2" \
      --seed "$seed" \
      --components 2 \
      --globalmasking 0 \
      --savename "$savename" \
      > "$log_dir/${batch1}_${batch2}_${seed}.out" \
      2> "$log_dir/${batch1}_${batch2}_${seed}.err"

  done
done

echo "✅ All jobs finished. Logs are in $log_dir"