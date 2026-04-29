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
import random
import torch

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import visualization, run_models_from_adata, benchmark_from_adata, run_metrics_from_adata
from utils.simulation_utils import add_noise, dropout, global_label_masking, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, ensure_label_intersection


def main_argparser(default_savename="experiment", default_globalmasking=0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 3008874, type=int) 
    parser.add_argument('-c', '--components', default = 2, type=int) 
    parser.add_argument('--nhvg', default = 2000, type=int) 
    parser.add_argument('--npca', default = 30, type=int) 
    parser.add_argument('--globalmasking', default = default_globalmasking, type=float) 
    parser.add_argument('--savename', default = default_savename, type=str) 
    # parser.add_argument('-t', default = "auto") 

    return parser



def prepare_adata(adata, save_path, label_key, batch_key, args):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

        
    adata = preprocess_adata(adata, batch_key=batch_key, n_top_genes=args.nhvg, n_pcs=args.npca)

    if args.globalmasking>0:
        # mask some labels (adds a new column "cell_type_masked" with some values "Unknown")
        adata = global_label_masking(adata, masking_frac=args.globalmasking, label_key=label_key, batch_key=batch_key, random_state=args.seed) 
    
    original_label_key = label_key
    label_key = f"{label_key}_masked" # update label key to the masked version for benchmarking (so that methods that can leverage labels will be affected by the masking)

    # creates a new column "cell_type_cleaned_encoded" with cleaned and encoded labels for our methods (that need numbers)
    adata = clean_and_encode_labels(adata, label_key=label_key, batch_key= batch_key, encoded_label_key="cell_type_cleaned_encoded", min_cells=0)


    # # ensure our encoded labels are present in both batches. if not, set them as unlabeled -- not needed, already done in clean_and_encode_labels with replace_by=np.nan
    # adata, labels_missing_in_batch1, labels_missing_in_batch2 = ensure_label_intersection(adata, label_key="cell_type_cleaned_encoded", batch_key=batch_key)

    return adata, original_label_key, label_key

def run_methods(adata, save_path, label_key, batch_key, methods_params_dict, args):
    if os.path.exists(f"{save_path}/adata_intermediate.h5ad"):
        print("Loading previously computed intermediate adata...")
        adata = sc.read_h5ad(f"{save_path}/adata_intermediate.h5ad")
        for method_ran in adata.obsm.keys():
            print(f"Method {method_ran} already computed, skipping...")
            methods_params_dict.pop(method_ran, None)
        print(f"Methods left to run: {list(methods_params_dict.keys())}")
        times_dict = adata.uns.get("times_dict", {})
        memory_dict = adata.uns.get("memory_dict", {})
    else:
        times_dict = {}
        memory_dict = {}
    
    for method_name, method_params in methods_params_dict.items():
        
        # if there is a parameter called "method_type" in method_params, then we assume "method_name" is the save name, and "method_type" is the actual method to run (ex. FoSTA with different t's will be called "FostA_t2" and "FoSTA_t10" but the method type is "FoSTA" for both)
        # if "method_type" is not present, add it and set it to method_name for simplicity
        if "method_type" not in method_params:
            method_params["method_type"] = method_name
        
        method_type = method_params.pop("method_type", None)
        
        print(f"Running method {method_name}...")
        adata, time_taken, memory_taken = run_models_from_adata(adata, method_type, batch_key = batch_key, label_key_ours = "cell_type_cleaned_encoded", label_key = label_key, embedding_basis="X_pca", seed=args.seed, n_components = args.components, **method_params)
        adata.obsm[f"{method_name}"] = adata.obsm.pop(method_type) # move the embedding to the correct key in obsm. bc the above function saves it in the method_type key, but we want it to be saved in the method_name key (ex. "FoSTA_t2" instead of "FoSTA")
        
        times_dict[f"{method_name}"] = time_taken
        memory_dict[f"{method_name}"] = memory_taken

        adata.uns["times_dict"] = times_dict
        adata.uns["memory_dict"] = memory_dict
        # save adata so that if it crashes later, we don't have to recompute everything
        adata.write(f"{save_path}/adata_intermediate.h5ad")
            
        print(f"\nMethod {method_name} completed in {time_taken:.2f} seconds\n")
    
    return adata, times_dict, memory_dict

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
   
    
    
def evaluate_and_save_results(adata, save_path_subfolder, save_path_parent, original_label_key, batch_key, args, save_name=""):
    times_dict = adata.uns.get("times_dict", {})
    memory_dict = adata.uns.get("memory_dict", {})
    
    methods_to_benchmark = list(adata.obsm.keys())
    try:
        methods_to_benchmark.remove("X_pca")
    except ValueError:
        pass
    try:
        methods_to_benchmark.remove("Unintegrated")
    except ValueError:
        pass
    
    if not "global_masking_fraction" in adata.uns.keys():
        adata.uns["global_masking_fraction"] = 0
        
    if adata.uns["global_masking_fraction"]>0:
        benchmark_adata = adata[adata.obs['mask_indices'] == 1].copy()
    else:
        benchmark_adata = adata.copy() # if we masked nothing, evaluate on everything

    results_df = benchmark_from_adata(benchmark_adata, methods_to_benchmark, batch_key = batch_key, label_key = original_label_key, save_path= save_path_subfolder)
    # metric_type = df.loc["Metric Type"]
    # df = df.drop("Metric Type")

    results_df.insert(0, "method", results_df.index)
    results_df.insert(0, "n_components", args.components)

    results_df["time"] = results_df["method"].map(times_dict)
    results_df["memory"] = results_df["method"].map(memory_dict)
       
    print(results_df)
    # save in the subfolder for this specific run
    results_df.to_csv(f"{save_path_subfolder}/{save_name}_results.csv", index=False)
                    
    # add results to a common file in the parent
    if os.path.exists(f"{save_path_parent}/{save_name}_results.csv"):
        print("Adding to previously computed results...")
        all_results_df = pd.read_csv(f"{save_path_parent}/{save_name}_results.csv")
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    else:
        all_results_df = results_df       
        print("Saving results as a new result file in the parent folder...")   

    all_results_df.to_csv(f"{save_path_parent}/{save_name}_results.csv", index=False)
                    
                    
                    
def paired_evaluate_and_save_results(adata, save_path_subfolder, save_path_parent, original_label_key, batch_key, args, save_name =""):
    times_dict = adata.uns.get("times_dict", {})
    memory_dict = adata.uns.get("memory_dict", {})
    
    methods_to_benchmark = list(adata.obsm.keys())

    if adata.uns["global_masking_fraction"]>0:
        benchmark_adata = adata[adata.obs['mask_indices'] == 1].copy()
    else:
        benchmark_adata = adata.copy() # if we masked nothing, evaluate on everything

    for method in methods_to_benchmark:
        res, sil_dom, foscttm, alignment_score = run_metrics_from_adata(benchmark_adata, method, batch_key = batch_key, label_key = encoded_label_key, masked_label_key = masked_encoded_label_key)
        rows_list = []
        result_row = {"n_components": args.components, 
                    "model": method, 
                    "FOSCTTM": foscttm, 
                    "Silhouette_domain": sil_dom,
                    "Accuracy_missing": res['acc_missing'], 
                    "Accuracy_visible": res['acc_visible'],
                    "Alignment_score": alignment_score,
                    "time": times_dict.get(method, None),
                    "memory": memory_dict.get(method, None)}

        rows_list.append(result_row)

    results_df = pd.DataFrame(rows_list) 

    results_df.to_csv(f"{save_path_subfolder}/{save_name}_results.csv", index=False)
                    
    # add results to a common file in the parent
    if os.path.exists(f"{save_path_parent}/{save_name}_results.csv"):
        print("Adding to previously computed results...")
        all_results_df = pd.read_csv(f"{save_path_parent}/{save_name}_results.csv")
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
    else:
        all_results_df = results_df       
        print("Saving results as a new result file in the parent folder...")   
    
    all_results_df.to_csv(f"{save_path_parent}/{save_name}_results.csv", index=False)
                            
                    
def save_embeddings(adata, save_path, label_key, batch_key):
    for method in adata.obsm.keys():
        sc.pl.embedding(adata, basis=method, color=[batch_key, label_key], title=method)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{method}.png")

        #%%     



