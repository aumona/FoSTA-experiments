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

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, BATCHES_DATA_PATH, RESULTS_PATH
base_path = BASE_PATH
scratch_path = RESULTS_PATH

# original_methods = ["FoSTA", "RFMALI", "MALI", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"] 
original_methods = ["FoSTA", "RFMALI", "MALI", "Scanorama", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf", "KEMAlin"] 
# original_methods = ["FoSTA"] # for testing. to delete
methods = original_methods.copy() # methods might be modified based on what is already run. but we still want to benchmark everything

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "immune") 
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('--batch1idx', default = 0, type=int) 
parser.add_argument('--batch2idx', default = 1, type=int) 
parser.add_argument('-s', '--save', default = True, type=bool) 
parser.add_argument('--hvg', default = True, type=bool) 
parser.add_argument('--pca', default = True, type=bool) 
parser.add_argument('-c', '--components', default = "30", type=int) 
parser.add_argument('--globalmasking', default = 0.2, type=float) 
parser.add_argument('--savename', default = "real_batches_immune", type=str) 
# parser.add_argument('-t', default = "auto") 

args = parser.parse_args()
dataset_name = args.dataset
batches_idxs = [args.batch1idx, args.batch2idx]
n_components = int(args.components)
# t = args.t
data_path = f"{BATCHES_DATA_PATH}/{dataset_name}.h5ad"
# set save location (won't be used if args.save is False)
save_name = f"{args.savename}/{batches_idxs[0]}_{batches_idxs[1]}" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"

save_path_subfolder = save_path
if args.save and not os.path.exists(save_path_subfolder):
    os.makedirs(save_path_subfolder)
    

args = parser.parse_args()
with open(f"{save_path}/config.json", "w") as f:
    json.dump(vars(args), f, indent=4)

# LOAD DATA 
data_path = "/home/mila/m/myriam.lizotte/scratch/RF-MALI/data/Immune_ALL_human.h5ad"
adata_full = sc.read(data_path)
adata_full

label_key = "final_annotation"
batch_key = "batch"


batch1 = adata_full.obs['batch'].unique()[args.batch1idx]
batch2 = adata_full.obs['batch'].unique()[args.batch2idx]
batches = [batch1, batch2]
print(f"Selected batches: {batches}")

# subset to the current batches
adata = adata_full[(adata_full.obs["batch"].isin(batches))]
# adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)



if args.hvg:
    n_top_genes=2000
else:
    n_top_genes=0
    
if args.pca:
    n_pcs=30
else:
    n_pcs=0
    
adata = preprocess_adata(adata, batch_key=batch_key, n_top_genes=n_top_genes, n_pcs=n_pcs)

# mask some labels (adds a new column "cell_type_masked" with some values "Unknown")
adata = global_label_masking(adata, masking_frac=args.globalmasking, label_key=label_key, batch_key=batch_key, random_state=args.seed) 
original_label_key = label_key
label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking)

# creates a new column "cell_type_cleaned_encoded" with cleaned and encoded labels for our methods (that need numbers)
adata = clean_and_encode_labels(adata, label_key=label_key, batch_key= batch_key, encoded_label_key="cell_type_cleaned_encoded", min_cells=0)


# # ensure our encoded labels are present in both batches. if not, set them as unlabeled -- not needed, already done in clean_and_encode_labels with replace_by=np.nan
# adata, labels_missing_in_batch1, labels_missing_in_batch2 = ensure_label_intersection(adata, label_key="cell_type_cleaned_encoded", batch_key=batch_key)


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
memory_dict = {}


base_fosta_params = {
    "kernel_method": "original",
    "ot_solver": "hiref"
}

oob_fosta_params = {
    "kernel_method": "oob",
    "ot_solver": "hiref"
}

dense_fosta_params = {
    "kernel_method": "original",
    "ot_solver": "dense"
}
    
    
    
for method in methods:
    # try: 
    if method in ["RFMALI_WIP", "RFMALI", "FoSTA"]: # for these methods, try different t values
        for embedder in embedders:
            if embedder == "PHATE":
                for t in ts:
                    print(f"Running method {method} with embedder={embedder}, t={t}")
                    adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=args.seed, t=t, n_components = n_components, **base_fosta_params)
                    adata.obsm[f"{method}_{embedder}_t{t}"] = adata.obsm[method]
                    times_dict[f"{method}_{embedder}_t{t}"] = time_taken
                    memory_dict[f"{method}_{embedder}_t{t}"] = memory_taken
                    adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
                
            if method == "FoSTA": # only for FoSTA, also run the OOB and dense kernel and OT solver variants with the default t (2)
                print(f"Running method {method} with embedder={embedder}, OOB kernel and HiRef OT solver with t=2...")
                adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=args.seed, t=2, n_components = n_components, **oob_fosta_params)
                adata.obsm[f"{method}_{embedder}_OOB"] = adata.obsm[method]
                times_dict[f"{method}_{embedder}_OOB"] = time_taken
                memory_dict[f"{method}_{embedder}_OOB"] = memory_taken
                adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion

                print(f"Running method {method} with embedder={embedder}, original kernel and dense OT solver with t=2...")
                adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=args.seed, t=2, n_components = n_components, **dense_fosta_params)
                adata.obsm[f"{method}_{embedder}_denseOT"] = adata.obsm[method]
                times_dict[f"{method}_{embedder}_denseOT"] = time_taken
                memory_dict[f"{method}_{embedder}_denseOT"] = memory_taken
                adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
                
            else:
                print(f"Running method {method} with embedder={embedder}...")
                adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder=embedder, seed=args.seed, n_components = n_components)
                adata.obsm[f"{method}_{embedder}"] = adata.obsm[method]
                times_dict[f"{method}_{embedder}"] = time_taken
                memory_dict[f"{method}_{embedder}"] = memory_taken
                adata.obsm.pop(method)  # remove adata.obsm[method] to avoid confusion
    else:
        adata, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", embedder="PHATE", seed=args.seed, n_components = n_components)
        # adata_no_pca, time_taken, memory_taken = run_models_from_adata(adata, method, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X", embedder="PHATE", seed=args.seed, n_components = n_components)
        
        times_dict[f"{method}"] = time_taken
        memory_dict[f"{method}"] = memory_taken

    # save adata so that if it crashes later, we don't have to recompute everything
    if args.save:
        adata.write(f"{save_path_subfolder}/adata_intermediate.h5ad")
        
    print(f"\nMethod {method} completed in {time_taken:.2f} seconds\n")
    # except Exception as e:
    #     print(f"\nMethod {method} failed with error: {e}\n")
    #     continue


# save the times of each method
# times_df = pd.DataFrame(list(times_dict.items()), columns=["method", "time_seconds"])
# if args.save:
#     times_df.to_csv(f"{save_path_subfolder}/method_times.csv", index=False)

# save the times and mem of each method
# times_df = pd.DataFrame(list(times_dict.items()), columns=["method", "time_seconds"])
# methods_memory_df = pd.DataFrame(list(memory_dict.items()), columns=["method", "peak_memory_MB"])
# times_mem_df = times_df.merge(methods_memory_df, on="method")
# if args.save:
#     times_mem_df.to_csv(f"{save_path_subfolder}/method_times_and_memory.csv", index=False)

# for key in adata_no_pca.obsm.keys():
#     adata.obsm[f"{key}_no_pca"] = adata_no_pca.obsm[key]

methods_to_benchmark = list(adata.obsm.keys())
try:
    methods_to_benchmark.remove("X_pca")
except ValueError:
    pass
try:
    methods_to_benchmark.remove("Unintegrated")
except ValueError:
    pass

if args.globalmasking>0:
   benchmark_adata = adata[adata.obs['mask_indices'] == 1].copy()
else:
    benchmark_adata = adata.copy() # if we masked nothing, evaluate on everything

df = benchmark_from_adata(benchmark_adata, methods_to_benchmark, batch_key = batch_key, label_key = original_label_key, save_path= save_path_subfolder)
# metric_type = df.loc["Metric Type"]
# df = df.drop("Metric Type")

df.insert(0, "method", df.index)
df.insert(0, "n_components", n_components)

df["time"] = df["method"].map(times_dict)
df["memory"] = df["method"].map(memory_dict)


# results_df = pd.concat([results_df, df], ignore_index=True)
results_df = df
# results_df.to_csv(f"{save_path}/simulated_batches_benchmark_results.csv", index=False)
            
        
# pd.concat([metric_type, results_df]) # adding back the metric type row for display
print(results_df)
if args.save:
    results_df.to_csv(f"{save_path_subfolder}/real_batches_results.csv", index=False)
                  
# add results to a common file in the parent
if os.path.exists(f"{save_path}/real_batches_results.csv"):
    print("Adding to previously computed results...")
    all_results_df = pd.read_csv(f"{save_path}/real_batches_results.csv")
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
else:
    all_results_df = results_df       
    print("Saving results as a new result file in the parent folder...")   


if args.save:
    all_results_df.to_csv(f"{save_path}/real_batches_results.csv", index=False)
                  
                  
                  
# SAVE THE EMBEDDINGS

for method in methods_to_benchmark:
    sc.pl.embedding(adata, basis=method, color=[label_key, batch_key], title=method)
    if args.save:
        plt.savefig(f"{save_path_subfolder}/{method}.png")

        #%%     



