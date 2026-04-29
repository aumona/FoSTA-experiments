#%%
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import pickle

import os
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import visualization, run_our_models, run_models_from_adata, benchmark_from_adata, run_metrics, run_metrics_from_adata
from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, mask_labels

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, IMMUNE_BATCHES_DATA_PATH, RESULTS_PATH
base_path = BASE_PATH
scratch_path = RESULTS_PATH

original_methods = ["MALI", "RFMALI", "FoSTA", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"]
# original_methods = ["MALI", "RFMALI", "FoSTA", "Pamona", "KEMArbf", "KEMAlin"]
# noise_stds = [0, 0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10]
# dropout_probs = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
# noise_stds = [0.42]
# dropout_probs = [0, 0.88]
methods = original_methods.copy() # methods might be modified based on what is already run. but we still want to benchmark everything


parser = main_argparser(default_globalmasking=0, default_savename="simulated_paired_batches")
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('-d', '--dropout', default = 0) 
parser.add_argument('-n', '--noise', default = 0) 
args = parser.parse_args()
seed = args.seed
batch = args.batch
noise_std = float(args.noise)
dropout_prob = float(args.dropout)
n_components = int(args.components)

data_path = f"{LUNG_BATCHES_DATA_PATH}"

# set save location (will create subfolders per dataset and split type)
save_name = f"simulated_paired_batches" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"

# LOAD DATA
adata_full = sc.read(data_path)
adata_full

# subset to the current batch
adata = adata_full[(adata_full.obs["batch"] == batch)]

adata.obsm["X"] = adata.X # hack so we can access the original data in obsm["X"]


label_key = "cell_type"
batch_key = "simulated_batch"
encoded_label_key = "cell_type_cleaned_encoded"
masked_encoded_label_key = f"{encoded_label_key}_masked"
# embedding_basis = "X_pca"

adata = clean_and_encode_labels(adata, label_key=label_key, new_label_key=encoded_label_key, min_cells=0)   

# mask some labels
labels_masked = mask_labels(adata.obs[encoded_label_key], mask_fraction=0.5, random_state=seed)
adata.obs[masked_encoded_label_key] = labels_masked

adata.obs["cell_id"] = adata.obs.index.astype(str) # add the cell identifier column that aligns pairs from each batch


# apply transformations to create a new batch  
adata1 = adata.copy()
embedding_basis = "X"
adata2 = adata.copy()

adata2.obsm[embedding_basis] = add_noise(adata2.obsm[embedding_basis], noise_std, random_state=seed)
adata2.obsm[embedding_basis] = dropout(adata2.obsm[embedding_basis], dropout_prob, random_state=seed)

# concatenate the two adatas to get one adata with simulated batches columns in obs
adata = adata1.concatenate(adata2, batch_key=batch_key, batch_categories=["batch1", "batch2"])

adata = preprocess_adata(adata) # to get highly variable genes and PCA


save_path_subfolder = f"{save_path}/noise_{noise_std}_dropout_{dropout_prob}/{n_components}_components"





