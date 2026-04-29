# this code was inspired by from https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

#%%
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import visualization, run_models_from_adata, benchmark_from_adata
from utils.simulation_utils import add_noise, dropout, global_label_masking, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, ensure_label_intersection
from utils.script_utils import main_argparser, set_seeds, prepare_adata, run_methods, evaluate_and_save_results, save_embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, IMMUNE_BATCHES_DATA_PATH, RESULTS_PATH

from methods_configs import methods_params_dict
# methods_params_dict = {"Pamona": {}
#                     } # for testing purposes, only run Pamona.
parser = main_argparser(default_savename="real_batches_immune")
parser.add_argument('--batch1idx', default = 0, type=int) 
parser.add_argument('--batch2idx', default = 1, type=int) 
args = parser.parse_args()

batches_idxs = [args.batch1idx, args.batch2idx]


data_path = IMMUNE_BATCHES_DATA_PATH
label_key = "final_annotation"
batch_key = "batch"


# set save location
save_path_parent = f"{RESULTS_PATH}/{args.savename}"
save_path = f"{save_path_parent}/{batches_idxs[0]}_{batches_idxs[1]}" 


# LOAD DATA 
adata_full = sc.read(data_path)

# subset to the current batches
batch1 = adata_full.obs['batch'].unique()[args.batch1idx]
batch2 = adata_full.obs['batch'].unique()[args.batch2idx]
batches = [batch1, batch2]
print(f"Selected batches: {batches}")
adata = adata_full[(adata_full.obs["batch"].isin(batches))]
# adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)


set_seeds(args.seed)
adata, original_label_key, masked_label_key = prepare_adata(adata, save_path, label_key, batch_key, args=args)
adata, times_dict, memory_dict = run_methods(adata, save_path, masked_label_key, batch_key, methods_params_dict=methods_params_dict, args=args)
evaluate_and_save_results(adata, save_path, save_path_parent, original_label_key, batch_key, times_dict, memory_dict, args, save_name = "real_batches_immune")
save_embeddings(adata, save_path, label_key, batch_key)


    