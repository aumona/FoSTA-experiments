#%%
import numpy as np
import argparse
import scanpy as sc
import pandas as pd
import pickle
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import os
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import visualization, run_our_models, run_models_from_adata, run_metrics

from utils.simulation_utils import add_noise, dropout, split_and_transform_batch, clean_and_encode_labels, preprocess_adata, mask_labels


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, PAIRED_DATA_PATH, RESULTS_PATH
base_path = BASE_PATH
data_path = PAIRED_DATA_PATH
scratch_path = RESULTS_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('-t', default = 2) 
args = parser.parse_args()
t = int(args.t) if args.t != "auto" else "auto"
seed= args.seed



# set save location
save_name = "pbmc_paired"
save_path = f"{scratch_path}/results/{save_name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
# load data 
data_path = f"{mali_path}/batch_correction/data/"
adata_d1 = sc.read(f"{data_path}pbmc_peak.h5ad")
adata_d2 = sc.read(f"{data_path}pbmc_gex.h5ad")

label_key = "CellType"


# take only the paired parts
indices_intersect = adata_d1.obs.index.intersection(adata_d2.obs.index)

# remove one, so the n is not prime
# indices_intersect = indices_intersect[:-3]


# # for TESTING -- keeping only 100 cells
# indices_intersect = indices_intersect[:100]

adata_d1 = adata_d1[indices_intersect]
adata_d2 = adata_d2[indices_intersect]



# from anndata import concat
# adata = concat([adata_d1, adata_d2], join="outer", label="modality", keys=["ATAC", "GEX"], fill_value=0)




# # checking that all the indices are the same (and in the same order)

# idx_atac = adata.obs.loc[adata.obs["modality"] == "ATAC"].index
# idx_gex = adata.obs.loc[adata.obs["modality"] == "GEX"].index

# (idx_atac == idx_gex).all()


adata_d1 = clean_and_encode_labels(adata_d1, label_key=label_key, encoded_label_key="cell_type_cleaned_encoded", min_cells=0)
# we don't do the second, because we dont want a different encoding in each modality. and anyway the labels are the same
labels = adata_d1.obs["cell_type_cleaned_encoded"].values # should be the same for both adatas, because the cells are the same and in the same order
# mask some labels
labels_masked = mask_labels(labels, mask_fraction=0.5, random_state = seed)


# adata = preprocess_adata(adata)

# hack so we can get X from obsm
# adata.obsm["X"] = adata.X.copy()


models = ["FoSTA", "RFMALI", "MALI", "Pamona", "KEMArbf", "KEMAlin"]
# models = ["FoSTA", "MALI", "Pamona", "KEMArbf", "KEMAlin", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI"]


rows_list = []
for method in models:
    print(f"Running method: {method}, seed: {seed}, t: {t}")
    # try: 
    x_source = adata_d2.X
    x_target = adata_d1.X
    y_source = np.array(labels)
    y_target = np.array(labels_masked)
    
    try:
        embedding = run_our_models(method, x_source, x_target, y_source, y_target, embedder="PHATE", seed=seed, t=t)
    except Exception as e:
        x_source = x_source.toarray()
        x_target = x_target.toarray()
        print("(!) Sparse matrix conversion to dense due to error:", e)
        embedding = run_our_models(method, x_source, x_target, y_source, y_target, embedder="PHATE", seed=seed, t=t)
    
    viz_save_path = f"{save_path}/{method}.png"
    visualization(embedding, y_source, y_target, title=f"{method}", save_path=viz_save_path, seed= seed)
    
    # plot with matching lines
    T_true = np.eye(len(y_source))
    visualization(embedding, y_source, y_target, title=f"{method}", T_true = T_true, save_path=viz_save_path.replace(".png", "_matching_lines.png"), seed= seed)
    
    mask_missing_target_full = (y_target == -1) 
    res, sil_dom, foscttm, alignment_score = run_metrics(embedding, y_source, y_target, y_target_true = labels, mask_missing_target_full = mask_missing_target_full)
    
    result_row = {"model": method, 
                    "seed": seed,
                    "t": t,
                    "FOSCTTM": foscttm, 
                    "Silhouette_domain": sil_dom,
                    "Accuracy_missing": res['acc_missing'], 
                    "Accuracy_visible": res['acc_visible'],
                    "Alignment_score": alignment_score}
    
    # except Exception as e:
    #     print(f"Method {method} failed with error: {e}")
    #     result_row = {"model": method, 
    #               "FOSCTTM": None,
    #               "Silhouette_domain": None,
    #               "Accuracy_missing": None, 
    #               "Accuracy_visible": None,
    #               "Alignment_score": None}
    
    rows_list.append(result_row)
            

results_df = pd.DataFrame(rows_list) 

print("Saving results...")
# update results file (add new results to existing file if exists)
results_file = f"{save_path}/real_paired_results.csv"
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    results_df = pd.concat([existing_results, results_df], ignore_index=True)

results_df.to_csv(results_file, index=False)
                  