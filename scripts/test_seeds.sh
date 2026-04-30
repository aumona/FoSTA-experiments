#!/bin/bash
cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA

timestamp=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
log_dir=/NOBACKUP/lizottem/fosta/logs/test_seeds/$timestamp
# create log directory
mkdir -p "$log_dir"

# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5  --seed 95 --savename "simulated_paired_batches/try1" --noise 0.5 --dropout 0.5 --test
# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try2" --noise 0.5 --dropout 0.5 --test
# python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try3" --noise 0.5 --dropout 0.5 --test


python3 scripts/real_batches_lung/real_batches_lung.py --batch1 "A1" --batch2 "A2" --components 2  --globalmasking 0.2 --savename "real_batches_lung/test_seed1"  > "$log_dir/"$batch1"_"$batch2"_1.out" 
python3 scripts/real_batches_lung/real_batches_lung.py --batch1 "A1" --batch2 "A2" --components 2  --globalmasking 0.2 --savename "real_batches_lung/test_seed2"  > "$log_dir/"$batch1"_"$batch2"_2.out" 
