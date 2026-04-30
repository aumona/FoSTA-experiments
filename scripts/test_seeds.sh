#!/bin/bash
cd /home/lizottem/RF-MALI
source ../.profile
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_FOSTA



python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5  --seed 95 --savename "simulated_paired_batches/try1" --noise 0.5 --dropout 0.5 --test
python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try2" --noise 0.5 --dropout 0.5 --test
python3 scripts/simulated_paired_batches/simulated_paired_batches.py --globalmasking 0.5 --seed 95  --savename "simulated_paired_batches/try3" --noise 0.5 --dropout 0.5 --test