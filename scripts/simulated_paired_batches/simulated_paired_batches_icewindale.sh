#!/bin/bash

cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA

timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
log_dir=/NOBACKUP/lizottem/fosta/logs/simulated_paired_batches/$timestamp
# create log directory
mkdir -p "$log_dir"


noise_stds=(0 0.2 0.4 0.6 0.8 1.0)
dropout_probs=(0 0.2 0.4 0.6 0.8 0.95) 
# noise_stds=(0.5)
# dropout_probs=(0.5) 
# noise_stds=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# dropout_probs=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 
# n_components=(2) # 3 10 20)
seeds=(11784 39041 56089 79121 4386721)



# first run 0.5 0.5 with all seeds to get a baseline, then run the rest of the combinations
for seed in "${seeds[@]}"; do
  python3 scripts/simulated_paired_batches/simulated_paired_batches.py \
        --globalmasking 0.2  \
        --savename "simulated_paired_batches/${timestamp}" \
        --noise 0.5 \
        --dropout 0.5 \
        --seed "$seed"  \
       > "$log_dir/noise${noise}_dropout${dropout}_seed${seed}.log" 2>&1
done



for seed in "${seeds[@]}"; do
  for noise in "${noise_stds[@]}"; do
    for dropout in "${dropout_probs[@]}"; do
      python3 scripts/simulated_paired_batches/simulated_paired_batches.py \
        --globalmasking 0.2  \
        --savename "simulated_paired_batches/${timestamp}" \
        --noise "$noise" \
        --dropout "$dropout" \
        --seed "$seed"  \
       > "$log_dir/noise${noise}_dropout${dropout}_seed${seed}.log" 2>&1
    done
  done
done
