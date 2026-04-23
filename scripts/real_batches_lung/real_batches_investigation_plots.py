# this script takes one batch of the lung atlas dataset, runs PHATE and RFPHATE on it, and visualizes the embeddings colored by cell type and batch. 
# and runs RFMALI and MALI. the goal is to investigate if our method struggles, and if the geometry of the individual batches can be properly captured by PHATE and RFPHATE (which are the basis of our method). if we can't capture the geometry of the individual batches well, that might explain why our method struggles. if we can capture the geometry well, but our method still struggles, then it might be an issue with how we leverage the labels or how we do the integration.

# # this code was inspired by from https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

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
data_path = BATCHES_DATA_PATH
scratch_path = RESULTS_PATH

original_methods = ["FoSTA", "RFMALI", "MALI", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"]
original_methods = ["FoSTA"]
methods = original_methods.copy() # methods might be modified based on what is already run. but we still want to benchmark everything




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "lung_atlas") 
parser.add_argument('--batch', default = "1") 
parser.add_argument('-s', '--save', default = True, type=bool) 
# parser.add_argument('--globalmasking', default = 0.2, type=float) 
parser.add_argument('--savename', default = "real_batches_investigation", type=str) 
# parser.add_argument('-t', default = "auto") 

args = parser.parse_args()
dataset_name = args.dataset
batch = args.batch
# t = args.t

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


for method in methods:
    # try: 
    if method in ["RFMALI_WIP", "RFMALI", "FoSTA"]: #(label_key="cell_type", model_name=None, **kwargs):
        emb, y0, y1, idxs_d = run_our_models_from_adatas(adata, adata, model_name=method, label_key="cell_type_cleaned_encoded", embedding_basis="X_pca", embedder="PHATE", t = 2, seed=42, n_components = 2)
        original_labels = adata[idxs_d].obs[label_key] #[0:adata.n_obs] # get the original labels (take the first half (same as second half)) 
        
        if args.save:
            fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(24, 7))
            
            # scatter1 = ax1.scatter(emb[:,0], emb[:,1], c=np.concatenate([y0,y1]), cmap="tab20", s=5)
            # ax1.set_title("Colored by encoded cell type")
            # ax1.legend()        
            
            
            # 1. Convert your labels to a pandas Series (if they aren't already)
            labels_series = pd.Series(original_labels)

            # 2. Map labels to integers
            # .astype('category').cat.codes gives you 0, 1, 2... based on the label
            color_indices = labels_series.astype('category').cat.codes
            unique_labels = labels_series.astype('category').cat.categories

            scatter2 = ax2.scatter(emb[:, 0], emb[:, 1], c=color_indices, cmap="tab20", s=5)
            ax2.set_title("Colored by original cell type")

            ax2.legend(
                handles=scatter2.legend_elements()[0], 
                labels=list(unique_labels),
                title="Cell Types",
                loc='upper left',
                bbox_to_anchor=(1.02, 1), # Places legend to the right of ax1
                borderaxespad=0
            )
            
            
            # --- Plot 2: Batch ---
            batch_labels = np.concatenate([np.zeros(len(y0)), np.ones(len(y1))]) # create batch labels based on the order of y0 and y1 (first half is batch 0, second half is batch 1)
            scatter3 = ax3.scatter(emb[:, 0], emb[:, 1], c=batch_labels, cmap="Set1", s=5)
            ax3.set_title("Colored by Batch copy")
            ax3.legend()
            
            # 2. Adjust layout to make room for the legends
            # rect=[0, 0, 0.9, 1] keeps the plots within the left 90% of the figure
            plt.tight_layout(rect=[0, 0, 0.85, 1]) 
            
            
            plt.savefig(f"{save_path_subfolder}/{method}_PHATE_t2.png", bbox_inches='tight')



            
            
            
            
            
            


        #%%     



