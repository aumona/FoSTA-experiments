# this code was inspired by from https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

#%%
import argparse
import numpy as np
import phate
import rfphate
import scanpy as sc
import pandas as pd
import pickle

import os
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import visualization, run_models_from_adata, benchmark_from_adata, run_our_models_from_adatas
from utils.simulation_utils import add_noise, dropout, global_label_masking, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, ensure_label_intersection

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, BATCHES_DATA_PATH, RESULTS_PATH
base_path = BASE_PATH
scratch_path = RESULTS_PATH

original_methods = ["FoSTA", "RFMALI", "MALI", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"]
original_methods = ["FoSTA", "RFMALI"] # for testing. to delete
methods = original_methods.copy() # methods might be modified based on what is already run. but we still want to benchmark everything


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "lung_atlas") 
parser.add_argument('--batch', default = "1") 
parser.add_argument('-s', '--save', default = True, type=bool) 
# parser.add_argument('--globalmasking', default = 0.2, type=float) 
parser.add_argument('--savename', default = "real_batches_investigation_metrics", type=str) 
# parser.add_argument('-t', default = "auto") 

args = parser.parse_args()
dataset_name = args.dataset
batch = args.batch
# t = args.t
data_path = f"{BATCHES_DATA_PATH}/{dataset_name}.h5ad"
# set save location (won't be used if args.save is False)
save_name = f"{args.savename}/{batch}" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"


# LOAD DATA 

adata_full = sc.read(data_path)
adata_full
print(adata_full.uns)

label_key = "cell_type"
batch_key = "batch"


# subset to the current batches
adata = adata_full[(adata_full.obs["batch"].isin([batch]))]
# adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)


save_path_subfolder = save_path
if args.save and not os.path.exists(save_path_subfolder):
    os.makedirs(save_path_subfolder)
    
adata = preprocess_adata(adata, batch_key=batch_key)


# run PHATE and RFPHATE
adata.obsm["X_phate_PCA"] = phate.PHATE(n_components=2, random_state=42).fit_transform(adata.obsm["X_pca"])
adata.obsm["X_phate"] = phate.PHATE(n_components=2, random_state=42).fit_transform(adata.X)
adata.obsm["X_rfphate_PCA"] = rfphate.RFPHATE(n_components=2, random_state=42).fit_transform(adata.obsm["X_pca"], adata.obs[label_key])
adata.obsm["X_rfphate"] = rfphate.RFPHATE(n_components=2, random_state=42).fit_transform(adata.X, adata.obs[label_key])


embs= ["X_phate_PCA", "X_phate", "X_rfphate_PCA", "X_rfphate"]

for emb in embs:
    emb_name = '_'.join(emb.split('_')[1:3])
    sc.pl.embedding(adata, basis=emb, color=label_key, title=f"{emb_name} embeddings of batch {batch} colored by {label_key}")
    if args.save:
        plt.savefig(f"{save_path_subfolder}/{emb_name}_embedding.png", bbox_inches='tight')
        plt.show()

# mask some labels (adds a new column "cell_type_masked" with some values "Unknown")
# adata = global_label_masking(adata, masking_frac=args.globalmasking, label_key=label_key, batch_key=batch_key, random_state=42) 
# original_label_key = label_key
# label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking)

# creates a new column "cell_type_cleaned_encoded" with cleaned and encoded labels for our methods (that need numbers)
adata = clean_and_encode_labels(adata, label_key=label_key, batch_key= batch_key, encoded_label_key="cell_type_cleaned_encoded", min_cells=0)

# add batch copy column
adata.obs["batch_copy"] = 1
adata_copy = adata.copy()
adata_copy.obs["batch_copy"] = 2
adata_combined = adata.concatenate(adata_copy)

for method in methods:
    # try: 
    if method in ["RFMALI_WIP", "RFMALI", "FoSTA"]: #(label_key="cell_type", model_name=None, **kwargs):
        adata_combined, time_taken, peak_mem_delta_mb = run_models_from_adata(adata_combined, batch_key = "batch_copy", model_name=method, label_key="cell_type_cleaned_encoded", embedding_basis="X_pca", embedder="PHATE", t = 2, seed=42, n_components = 2)
        
        sc.pl.embedding(adata_combined, basis=method, color=["cell_type", 'batch_copy'], title=f"{method} PHATE embedding colored by cell type")
        plt.savefig(f"{save_path_subfolder}/{method}_PHATE_embedding.png", bbox_inches='tight')
                
            
    
df = benchmark_from_adata(adata_combined, methods, batch_key = 'batch_copy', label_key = label_key, save_path= save_path_subfolder)
# metric_type = df.loc["Metric Type"]
# df = df.drop("Metric Type")
df.insert(0, "method", df.index)

df.to_csv(f"{save_path_subfolder}/real_batches_results.csv", index=False)
    
        
        
        
        


    #%%     



