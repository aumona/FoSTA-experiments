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


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('-s', '--save', default = True, type=bool) 
parser.add_argument('-d', '--dropout', default = 0) 
parser.add_argument('-n', '--noise', default = 0) 
parser.add_argument('-c', '--components', default = 2) 


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
if not os.path.exists(save_path_subfolder):
    os.makedirs(save_path_subfolder)


results_df = pd.DataFrame()
# ts = ["auto", 2, 10]
ts = [2]
embedders = ["PHATE"] #, "UMAP", "spectral"]

times_dict = {}

rows_list = []

for method in methods:
    try: 
        if method in ["RFMALI_WIP", "RFMALI", "FoSTA"]: # for these methods, try different t values
            for embedder in embedders:
                if embedder == "PHATE":
                    for t in ts:
                        method_emb_t = f"{method}_{embedder}_t{t}"
                        print(f"Running method {method} with embedder={embedder}..., t={t}")
                        adata, time_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = masked_encoded_label_key, label_key = masked_encoded_label_key, embedding_basis="X_pca", embedder=embedder, seed=seed, t=t, n_components = n_components)
                        
                        
                        adata.obsm[method_emb_t] = adata.obsm[method]
                        times_dict[method_emb_t] = time_taken
                        adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
                        
                        
                        res, sil_dom, foscttm, alignment_score = run_metrics_from_adata(adata, method_emb_t, batch_key = batch_key, label_key = encoded_label_key, masked_label_key = masked_encoded_label_key)
                        result_row = {"model": method_emb_t, 
                                    "FOSCTTM": foscttm, 
                                    "Silhouette_domain": sil_dom,
                                    "Accuracy_missing": res['acc_missing'], 
                                    "Accuracy_visible": res['acc_visible'],
                                    "Alignment_score": alignment_score}
                else:
                    method_emb = f"{method}_{embedder}"
                    print(f"Running method {method} with embedder={embedder}...")
                    adata, time_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = masked_encoded_label_key, label_key = masked_encoded_label_key, embedding_basis="X_pca", embedder=embedder, seed=seed, n_components = n_components)
                    adata.obsm[method_emb] = adata.obsm[method]
                    times_dict[method_emb] = time_taken
                    adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
                    
                    res, sil_dom, foscttm, alignment_score = run_metrics_from_adata(adata, method_emb, batch_key = batch_key, label_key = encoded_label_key, masked_label_key = masked_encoded_label_key)
                    result_row = {"model": method_emb, 
                        "FOSCTTM": foscttm, 
                        "Silhouette_domain": sil_dom,
                        "Accuracy_missing": res['acc_missing'], 
                        "Accuracy_visible": res['acc_visible'],
                        "Alignment_score": alignment_score}
        else:
            adata, time_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = masked_encoded_label_key, label_key = masked_encoded_label_key, embedding_basis="X_pca", embedder="PHATE", seed=seed, n_components = n_components)
            times_dict[f"{method}"] = time_taken
            
            res, sil_dom, foscttm, alignment_score = run_metrics_from_adata(adata, method, batch_key = batch_key, label_key = encoded_label_key, masked_label_key = masked_encoded_label_key)
            result_row = {"model": method, 
                  "FOSCTTM": foscttm, 
                  "Silhouette_domain": sil_dom,
                  "Accuracy_missing": res['acc_missing'], 
                  "Accuracy_visible": res['acc_visible'],
                  "Alignment_score": alignment_score}
        
            
        # save adata so that if it crashes later, we don't have to recompute everything
        if args.save:
            adata.write(f"{save_path_subfolder}/adata_intermediate.h5ad")
           
        print(f"\nMethod {method} completed in {time_taken:.2f} seconds\n")
    except Exception as e:
        print(f"\nMethod {method} failed with error: {e}\n")
        import traceback
        print(traceback.format_exc())
        
        result_row = {"model": None, 
                  "FOSCTTM": None, 
                  "Silhouette_domain": None,
                  "Accuracy_missing": None, 
                  "Accuracy_visible": None,
                  "Alignment_score": None}
        continue


    rows_list.append(result_row)
            

results_df = pd.DataFrame(rows_list) 

print("Saving results...")
# update results file (add new results to existing file if exists)
results_file = f"{save_path}/real_paired_results.csv"
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    results_df = pd.concat([existing_results, results_df], ignore_index=True)

results_df.to_csv(results_file, index=False)


# SAVE THE EMBEDDINGS
# for method in adata.obsm.keys():
#     sc.pl.embedding(adata, basis=method, color=[label_key, batch_key], title=method)
#     if args.save:
#         plt.savefig(f"{save_path_subfolder}/{method}.png")

        #%%     



