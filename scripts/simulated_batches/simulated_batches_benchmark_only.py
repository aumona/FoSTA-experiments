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
from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata


base_path = "/home/mila/m/myriam.lizotte/RF-MALI"
mali_path = "/home/mila/m/myriam.lizotte/MALI"
scratch_path = "/home/mila/m/myriam.lizotte/scratch/RF-MALI"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "lung_atlas") 
parser.add_argument('-b', '--batch', default = "4") 
parser.add_argument('-s', '--save', default = True, type=bool) 
parser.add_argument('-d', '--dropout', default = "0") 
parser.add_argument('-n', '--noise', default = "0") 
parser.add_argument('-c', '--components', default = "30") 
# parser.add_argument('-t', default = "auto") 

args = parser.parse_args()
dataset_name = args.dataset
batch = args.batch
noise_std = float(args.noise)
dropout_prob = float(args.dropout)
n_components = int(args.components)
# t = args.t


label_key = "cell_type"
batch_key = "simulated_batch"

# set save location (won't be used if args.save is False)
save_name = f"simulated_batches/{batch}" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"

save_path_subfolder = f"{save_path}/noise_{noise_std}_dropout_{dropout_prob}/{n_components}_components"

# load previously computed intermediate adata if it exists
if os.path.exists(f"{save_path_subfolder}/adata_intermediate.h5ad"):
    print("Loading previously computed intermediate adata...")
    adata = sc.read_h5ad(f"{save_path_subfolder}/adata_intermediate.h5ad")
    # methods = [method for method in methods if method not in adata.obsm.keys()]
    # print(f"Methods left to run: {methods}")
    
results_df = pd.DataFrame()
# ts = ["auto", 2, 10]


methods = list(adata.obsm.keys())
methods.remove("X_pca")
methods.remove("Unintegrated")
df = benchmark_from_adata(adata, methods, batch_key = batch_key, label_key = label_key, save_path= save_path_subfolder)
# metric_type = df.loc["Metric Type"]
# df = df.drop("Metric Type")

df.insert(0, "method", df.index)
df.insert(0, "dropout_prob", dropout_prob)
df.insert(0, "noise_std", noise_std)
df.insert(0, "n_components", n_components)

# results_df = pd.concat([results_df, df], ignore_index=True)
results_df = df
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



