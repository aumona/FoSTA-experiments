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
from utils.benchmark_utils import visualization, run_models_from_adata, benchmark_from_adata
from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, split_and_transform_batch_stratified, global_label_masking, ensure_label_intersection

original_methods = ["MALI", "RFMALI", "FoSTA", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"]
#original_methods = ["LIGER"]
# noise_stds = [0, 0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10]
# dropout_probs = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9]
# noise_stds = [0.42]
# dropout_probs = [0, 0.88]
methods = original_methods.copy() # methods might be modified based on what is already run. but we still want to benchmark everything

base_path = "."
mali_path = "./../MALI"
scratch_path = "."


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "lung_atlas") 
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('-s', '--save', default = True, type=bool) 
parser.add_argument('-d', '--dropout', default = "0") 
parser.add_argument('-n', '--noise', default = "0") 
parser.add_argument('-c', '--components', default = "30") 
parser.add_argument('--savename', default = "simulated_batches") 
# parser.add_argument('-t', default = "auto") 

args = parser.parse_args()
dataset_name = args.dataset
batch = args.batch
noise_std = float(args.noise)
dropout_prob = float(args.dropout)
n_components = int(args.components)
seed = args.seed
# t = args.t

# set save location (won't be used if args.save is False)
save_name = f"{args.savename}/{batch}" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"


# LOAD DATA 

adata_full = sc.read(data_path)
adata_full

# subset to the current batch
adata = adata_full[(adata_full.obs["batch"] == batch)]

# adata = adata[0:200] # for testing purposes

label_key = "cell_type"
batch_key = "simulated_batch"
# embedding_basis = "X_pca"


# TRANSFORM the second half with noise and dropout
print(f"Running benchmark for noise std: {noise_std}, dropout prob: {dropout_prob}")

save_path_subfolder = f"{save_path}/noise_{noise_std}_dropout_{dropout_prob}/{n_components}_components"
if args.save and not os.path.exists(save_path_subfolder):
    os.makedirs(save_path_subfolder)
    
    

adata = split_and_transform_batch_stratified(adata, noise_std=noise_std, dropout_prob = dropout_prob, new_batch_key= batch_key, embedding_basis="X", seed = seed)
adata = preprocess_adata(adata)

# mask some labels (adds a new column "cell_type_masked" with some values "Unknown")
adata, mask_indices = global_label_masking(adata, masking_frac=0.5, label_key=label_key, batch_key=batch_key, random_state=seed) 
original_label_key = label_key
label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking)

# creates a new column "cell_type_cleaned_encoded" with cleaned and encoded labels for our methods (that need numbers)
adata = clean_and_encode_labels(adata, batch_key = batch_key, label_key=label_key, encoded_label_key="cell_type_cleaned_encoded", min_cells=0)

## ensure our encoded labels are present in both batches. if not, set them as unlabeled
#adata, labels_missing_in_batch1, labels_missing_in_batch2 = ensure_label_intersection(adata, label_key="cell_type_cleaned_encoded", batch_key=batch_key)


# only run methods that have not been run yet
# load previously computed intermediate adata if it exists
if os.path.exists(f"{save_path_subfolder}/adata_intermediate.h5ad"):
    print("Loading previously computed intermediate adata...")
    adata = sc.read_h5ad(f"{save_path_subfolder}/adata_intermediate.h5ad")
    # methods = [method for method in methods if method not in adata.obsm.keys()]
    # print(f"Methods left to run: {methods}")
    
results_df = pd.DataFrame()
# ts = ["auto", 2, 10]
ts = [2]
embedders = ["PHATE"]#, "UMAP", "spectral"]



times_dict = {}
memory_dict= {}

for method in methods:
    try: 
        if method in ["RFMALI_WIP", "RFMALI", "FoSTA"]: # for these methods, try different t values
            for embedder in embedders:
                if embedder == "PHATE":
                    for t in ts:
                        print(f"Running method {method} with embedder={embedder}..., t={t}")
                        adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=42, t=t, n_components = n_components)
                        adata.obsm[f"{method}_{embedder}_t{t}"] = adata.obsm[method]
                        times_dict[f"{method}_{embedder}_t{t}"] = time_taken
                        memory_dict[f"{method}_{embedder}_t{t}"] = memory_taken
                        adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
                else:
                    print(f"Running method {method} with embedder={embedder}...")
                    adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=42, n_components = n_components)
                    adata.obsm[f"{method}_{embedder}"] = adata.obsm[method]
                    times_dict[f"{method}_{embedder}"] = time_taken
                    memory_dict[f"{method}_{embedder}"] = memory_taken

                    adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
        else:
            adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder="PHATE", seed=42, n_components = n_components)
            times_dict[f"{method}"] = time_taken
            memory_dict[f"{method}"] = memory_taken
            
        # save adata so that if it crashes later, we don't have to recompute everything
        if args.save:
            adata.write(f"{save_path_subfolder}/adata_intermediate.h5ad")
           
        print(f"\nMethod {method} completed in {time_taken:.2f} seconds\n")
    except Exception as e:
        print(f"\nMethod {method} failed with error: {e}\n")
        import traceback
        print(traceback.format_exc())
        continue


# # save the times and mem of each method
# times_df = pd.DataFrame(list(times_dict.items()), columns=["method", "time_seconds"])
# methods_memory_df = pd.DataFrame(list(memory_dict.items()), columns=["method", "peak_memory_MB"])
# times_mem_df = times_df.merge(methods_memory_df, on="method")
# if args.save:
#     times_mem_df.to_csv(f"{save_path_subfolder}/method_times_and_memory.csv", index=False)

methods = list(adata.obsm.keys())
methods.remove("X_pca")
methods.remove("Unintegrated")


benchmark_adata = adata[mask_indices].copy()

df = benchmark_from_adata(benchmark_adata, methods, batch_key = batch_key, label_key = original_label_key, save_path= save_path_subfolder)
# metric_type = df.loc["Metric Type"]
# df = df.drop("Metric Type")

df.insert(0, "method", df.index)
df.insert(0, "dropout_prob", dropout_prob)
df.insert(0, "noise_std", noise_std)
df.insert(0, "n_components", n_components)

df["time"] = df["method"].map(times_dict)
df["memory"] = df["method"].map(memory_dict)

# results_df = pd.concat([results_df, df], ignore_index=True)
results_df = df
print("Saving results for this noise std and dropout prob combination...")
# results_df.to_csv(f"{save_path}/simulated_batches_benchmark_results.csv", index=False)
            
        
# pd.concat([metric_type, results_df]) # adding back the metric type row for display
print(results_df)
if args.save:
    results_df.to_csv(f"{save_path_subfolder}/simulated_batches_results.csv", index=False)
                  
# add results to a common file in the parent
if os.path.exists(f"{save_path}/simulated_batches_results.csv"):
    print("Adding to previously computed results...")
    all_results_df = pd.read_csv(f"{save_path}/simulated_batches_results.csv")
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
else:
    all_results_df = results_df       
    print("Saving results as a new result file in the parent folder...")   


if args.save:
    all_results_df.to_csv(f"{save_path}/simulated_batches_results.csv", index=False)
                  
                  
                  
# SAVE THE EMBEDDINGS

for method in methods+["Unintegrated"]:
    sc.pl.embedding(adata, basis=method, color=[label_key, batch_key], title=method)
    if args.save:
        plt.savefig(f"{save_path_subfolder}/{method}.png")

        #%%     



