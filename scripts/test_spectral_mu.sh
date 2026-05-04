#!/bin/bash
cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA

timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
log_dir=/NOBACKUP/lizottem/fosta/logs/test_spectral_mu/$timestamp
# create log directory
mkdir -p "$log_dir"

# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5  --seed 95 --savename "simulated_paired_batches/try1" --noise 0.5 --dropout 0.5 --test
# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try2" --noise 0.5 --dropout 0.5 --test
# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try3" --noise 0.5 --dropout 0.5 --test


python3 scripts/real_batches_lung/real_batches_lung.py --test --npca 30 --batch1 "A1" --batch2 "A2" --components 2  --globalmasking 0 --savename "real_batches_lung/spectral_mu"  > "$log_dir/"$batch1"_"$batch2"_1.out" 