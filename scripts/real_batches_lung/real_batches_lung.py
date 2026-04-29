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
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, RESULTS_PATH

from methods_configs import methods_params_dict

data_path = LUNG_BATCHES_DATA_PATH
label_key = "cell_type"
batch_key = "batch"

parser = main_argparser(default_savename="real_batches_lung")
parser.add_argument('--batch1', default = "4")
parser.add_argument('--batch2', default = "5")
args = parser.parse_args()

batches = [args.batch1, args.batch2]

# set save location 
save_path_parent = f"{RESULTS_PATH}/{args.savename}"
save_path = f"{save_path_parent}/{batches[0]}_{batches[1]}" 

# LOAD DATA 
adata_full = sc.read(data_path)

# subset to the current batches
adata = adata_full[(adata_full.obs["batch"].isin(batches))].copy()
# adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)




## only run methods that have not been run yet
## load previously computed intermediate adata if it exists
# if os.path.exists(f"{save_path}/adata_intermediate.h5ad"):
#     print("Loading previously computed intermediate adata...")
#     adata = sc.read_h5ad(f"{save_path}/adata_intermediate.h5ad")
#     for method_ran in adata.obsm.keys():
#         print(f"Method {method_ran} already computed, skipping...")
#         methods_params_dict.pop(method_ran, None)
            
#     print(f"Methods left to run: {list(methods_params_dict.keys())}")

# else:
set_seeds(args.seed)
adata, original_label_key, masked_label_key = prepare_adata(adata, save_path, label_key, batch_key, args=args)
adata = run_methods(adata, save_path, masked_label_key, batch_key, methods_params_dict=methods_params_dict, args=args)
evaluate_and_save_results(adata, save_path_subfolder=save_path, save_path_parent=save_path_parent, original_label_key=original_label_key, batch_key=batch_key, args=args, save_name = "real_batches_lung")
save_embeddings(adata, save_path, label_key, batch_key)
