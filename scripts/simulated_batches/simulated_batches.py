# this code was inspired by from https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

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
from utils.benchmark_utils import set_seeds, visualization, run_models_from_adata, benchmark_from_adata
from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, split_and_transform_batch_stratified, global_label_masking, ensure_label_intersection
from utils.script_utils import main_argparser, prepare_adata, run_methods, evaluate_and_save_results, save_embeddings



sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, RESULTS_PATH

from methods_configs import methods_params_dict

parser = main_argparser(default_savename="simulated_batches")
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('-d', '--dropout', default = 0, type=float) 
parser.add_argument('-n', '--noise', default = 0, type=float)  
args = parser.parse_args()

data_path = LUNG_BATCHES_DATA_PATH
label_key = "cell_type"
batch_key = "simulated_batch"


# embedding_basis = "X_pca"
batch = args.batch
noise_std = float(args.noise)
dropout_prob = float(args.dropout)
n_components = int(args.components)
seed = args.seed
# t = args.t

# set save location 
save_path_parent = f"{RESULTS_PATH}/{args.savename}"
save_path = f"{save_path_parent}/{batch}" 

# LOAD DATA 
adata_full = sc.read(data_path)
adata_full

# subset to the current batch
adata = adata_full[(adata_full.obs["batch"] == batch)]

# adata = adata[0:200] # for testing purposes




set_seeds(args.seed)


# TRANSFORM the second half with noise and dropout
print(f"Running benchmark for noise std: {noise_std}, dropout prob: {dropout_prob}")
adata = split_and_transform_batch_stratified(adata, noise_std=noise_std, dropout_prob = dropout_prob, new_batch_key= batch_key, embedding_basis="X", seed = seed)



save_path_subfolder = f"{save_path}/noise_{noise_std}_dropout_{dropout_prob}/{n_components}_components"

adata, original_label_key, masked_label_key = prepare_adata(adata, save_path_subfolder, label_key, batch_key, args=args)
adata = run_methods(adata, save_path, masked_label_key, batch_key, methods_params_dict=methods_params_dict, args=args)
evaluate_and_save_results(adata, save_path_subfolder=save_path, save_path_parent=save_path_parent, original_label_key=original_label_key, batch_key=batch_key, args=args, save_name = "simulated_batches")
save_embeddings(adata, save_path, label_key, batch_key)