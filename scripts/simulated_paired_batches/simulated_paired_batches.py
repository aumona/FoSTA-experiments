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
from utils.benchmark_utils import set_seeds, visualization, run_our_models, run_models_from_adata, benchmark_from_adata, run_metrics, run_metrics_from_adata
from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, mask_labels, global_label_masking
from utils.script_utils import main_argparser, prepare_adata, run_methods, paired_evaluate_and_save_results, save_embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, IMMUNE_BATCHES_DATA_PATH, RESULTS_PATH
from methods_configs import methods_params_dict

parser = main_argparser(default_globalmasking=0.5, default_savename="simulated_paired_batches")
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('-d', '--dropout', default = 0) 
parser.add_argument('-n', '--noise', default = 0) 
args = parser.parse_args()
seed = args.seed
batch = args.batch
noise_std = float(args.noise)
dropout_prob = float(args.dropout)
n_components = int(args.components)

if args.test: # only consider a few methods for testing
    from methods_configs_test import methods_params_dict
    
    
# set save location (will create subfolders per dataset and split type)
save_name = f"simulated_paired_batches" #dataset}".format(dataset = dataset_name)

# set save location 
save_path_parent = f"{RESULTS_PATH}/{args.savename}"
save_path = f"{save_path_parent}/{batch}/{n_components}_components/seed_{args.seed}/noise_{noise_std}_dropout_{dropout_prob}"


# LOAD DATA
adata_full = sc.read(LUNG_BATCHES_DATA_PATH)
adata_full

# subset to the current batch
adata = adata_full[(adata_full.obs["batch"] == batch)]

adata.obsm["X"] = adata.X # hack so we can access the original data in obsm["X"]


label_key = "cell_type"
batch_key = "simulated_batch"
encoded_label_key = "cell_type_cleaned_encoded" # never instantiated
masked_encoded_label_key = f"{encoded_label_key}_masked"
masked_encoded_label_key = f"{label_key}_cleaned_encoded_masked"
# embedding_basis = "X_pca"

# adata = clean_and_encode_labels(adata, label_key=label_key, new_label_key=encoded_label_key, min_cells=0)   

# # mask some labels
# labels_masked = mask_labels(adata.obs[encoded_label_key], mask_fraction=0.5, random_state=seed)
# adata.obs[masked_encoded_label_key] = labels_masked


# creates a new column encoded_label_key with cleaned and encoded labels for our methods (that need numbers)
adata = clean_and_encode_labels(adata, label_key=label_key, batch_key= None, encoded_label_key= encoded_label_key, min_cells=0)


dummy_batch_key = "batch" # for the global label masking stratified over batches. we only have 1 batch currently.
adata = global_label_masking(adata, masking_frac=args.globalmasking, label_key=label_key, batch_key=dummy_batch_key, random_state=args.seed) 
masked_label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking)

# now also mask the encoded labels 
adata.obs[masked_encoded_label_key] = adata.obs[encoded_label_key].copy()
adata.obs.loc[adata.obs["mask_indices"] ==1, masked_encoded_label_key] = -1 # set the masked labels to -1


# ----------------- similar to the prepare adata function in script_utils, but in a different order. we want to mask on the single batch, but then preprocess on the combined adata with simulated batch
adata.obs["cell_id"] = adata.obs.index.astype(str) # add the cell identifier column that aligns pairs from each batch

# apply transformations to create a new batch  
adata1 = adata.copy()
embedding_basis = "X"
adata2 = adata.copy()

adata2.obsm[embedding_basis] = add_noise(adata2.obsm[embedding_basis], noise_std, random_state=seed)
adata2.obsm[embedding_basis] = dropout(adata2.obsm[embedding_basis], dropout_prob, random_state=seed)

# concatenate the two adatas to get one adata with simulated batches columns in obs
adata = adata1.concatenate(adata2, batch_key=batch_key, batch_categories=["batch1", "batch2"])

# -----------------
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(f"{save_path}/config.json", "w") as f:
    json.dump(vars(args), f, indent=4)

adata = preprocess_adata(adata, batch_key=batch_key, n_top_genes=args.nhvg, n_pcs=args.npca)

adata = run_methods(adata, save_path, label_key=masked_label_key, encoded_label_key=masked_encoded_label_key, batch_key=batch_key, methods_params_dict=methods_params_dict, args=args)
paired_evaluate_and_save_results(adata, save_path_subfolder=save_path, save_path_parent=save_path_parent, encoded_label_key= label_key, masked_encoded_label_key=masked_encoded_label_key, batch_key=batch_key, args=args, save_name = "simulated_batches")
save_embeddings(adata, save_path, label_key, batch_key)


