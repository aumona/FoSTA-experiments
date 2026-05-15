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
from utils.benchmark_utils import set_seeds, visualization, run_models_from_adata, benchmark_from_adata
from utils.simulation_utils import add_noise, dropout, global_label_masking, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, ensure_label_intersection
from utils.script_utils import load_existing_adata, main_argparser, prepare_adata, run_methods, evaluate_and_save_results, save_embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, RESULTS_PATH

from methods_configs import methods_params_dict


data_path = LUNG_BATCHES_DATA_PATH
label_key = "cell_type"
batch_key = "batch"

# # Run this once to fix the warnings
# adata_full = sc.read(data_path)
# adata_full.write(data_path.replace(".h5ad", "_updated.h5ad"))
# sys.exit()

parser = main_argparser(default_savename="real_batches_lung")
parser.add_argument('--batch1', default = "4")
parser.add_argument('--batch2', default = "5")
args = parser.parse_args()

set_seeds(args.seed)

# ensure args.keep_unshared_labels is True (do not mask the labels that are not in both batches)
args.keep_unshared_labels = True

if args.test:
    from methods_configs_test import methods_params_dict

batches = [args.batch1, args.batch2]

# set save location 
save_path_parent = f"{RESULTS_PATH}/{args.savename}"
save_path = f"{save_path_parent}/{batches[0]}_{batches[1]}/seed_{args.seed}" 



# LOAD DATA 
if os.path.exists(f"{save_path}/adata_intermediate.h5ad"):
    # load previously computed results
    adata, methods_params_dict = load_existing_adata(save_path, methods_params_dict) # this will update the adata and methods_params_dict by removing the methods that have already been run (if any)
    
    original_label_key = label_key # the original label key is the same as the input label key, since we are not modifying the labels in this script (we are keeping the unshared labels)
    masked_label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking
    masked_encoded_label_key_with_unshared = f"{label_key}_encoded_with_unshared"
    
    if masked_encoded_label_key_with_unshared not in adata.obs.columns:
        adata = clean_and_encode_labels(adata, label_key=label_key, batch_key= batch_key, encoded_label_key= masked_encoded_label_key_with_unshared, min_cells=0, keep_unshared_labels=args.keep_unshared_labels) 
    
    
else:
    adata_full = sc.read(data_path)

    # subset to the current batches
    adata = adata_full[(adata_full.obs["batch"].isin(batches))].copy()
    # adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)
    
    adata, original_label_key, masked_label_key, masked_encoded_label_key_with_unshared = prepare_adata(adata, save_path, label_key, batch_key, args=args, masked_encoded_label_key=f"{label_key}_encoded_with_unshared")

adata = run_methods(adata, save_path, masked_label_key, encoded_label_key= masked_encoded_label_key_with_unshared, batch_key = batch_key, methods_params_dict=methods_params_dict, args=args)
evaluate_and_save_results(adata, save_path_subfolder=save_path, save_path_parent=save_path_parent, original_label_key=original_label_key, batch_key=batch_key, args=args, save_name = "real_batches_lung")
save_embeddings(adata, save_path, label_key, batch_key)
