#%%
import argparse
import pandas as pd
import numpy as np
import os
import time
import tracemalloc

import sys, pathlib
sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents) if (p/"src").is_dir())))
from utils.benchmark_utils import ablation_fosta, visualization, run_models_from_adata, run_our_models, run_metrics
from utils.utils import dataprep

from utils.simulation_utils import add_noise, dropout, random_feature_split, importance_split, alternating_importance_split, add_noise_features_split, random_rotate, mask_labels


# methods = ["MALI", "RFMALI", "RFMALI_WIP", "Scanorama", "LIGER", "Harmony", "scVI", "scANVI", "Pamona", "KEMArbf"]
# methods = ["RFMALI", "FoSTA", "MALI", "Pamona", "KEMArbf", "KEMAlin"]
# methods = ["RFMALI_WIP"]


base_path = "/home/mila/m/myriam.lizotte/RF-MALI"
mali_path = "/home/mila/m/myriam.lizotte/MALI"
scratch_path = "/home/mila/m/myriam.lizotte/scratch/RF-MALI"


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default = "diabetes") 
parser.add_argument('-s', '--split', default = "random") 
parser.add_argument('--seed', default = 3008874, type=int) 
parser.add_argument('-t', default = "auto") 
parser.add_argument('--label_mask_fraction', default = 0.5, type=float) 
parser.add_argument('--savename', default = "ablation_fosta_params", type=str)

args = parser.parse_args()
data_name = args.dataset
seed = args.seed
split = args.split
t = int(args.t) if args.t != "auto" else "auto"
label_mask_fraction = args.label_mask_fraction

# set save location (will create subfolders per dataset and split type)
save_name = args.savename #" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/results/{save_name}"
    
os.makedirs(save_path, exist_ok=True)    
    


# LOAD DATA 
datasets_path = "/home/mila/m/myriam.lizotte/scratch/RF-MALI/data"
# for these datasets, we assume the first column is the target (labels)

df = pd.read_csv(f"{datasets_path}/{data_name}.csv")
df, labels = dataprep(df) # labels are encoded
# # remove and store labels
# labels = df.pop(df.columns[0])


# mask some labels in target domain
labels_masked = mask_labels(labels, label_mask_fraction, random_state=seed)

if split == "random":
    df1, df2 = random_feature_split(df, random_state=seed) # split
elif split == "importance":
    df1, df2 = importance_split(df, labels)
elif split == "alternating_importance":
    df1, df2 = alternating_importance_split(df, labels)
elif split == "add_noise_features":
    df1, df2 = add_noise_features_split(df, signal_to_noise_ratio=0.1, random_state=seed)
elif split == "distort": # add noise
    df1 = df.copy()
    df2 = add_noise(df, sigma=0.1, random_state=seed)    
elif split == "rotate":
    df1 = df.copy()
    df2 = random_rotate(df, random_state=seed)
    
else: 
    raise ValueError(f"Unknown split type: {split}")
    
# # add noise to one half to simulate batch effect
# df2 = add_noise(df2, sigma=0.1, random_state=seed)


save_path_subfolder = f"{save_path}/{split}/{data_name}"
if not os.path.exists(save_path_subfolder):
    os.makedirs(save_path_subfolder)

rows_list = []
tracemalloc.start()


fosta_params_dict = {"fosta": {"prior_correct": True, "euclidean_mode": False, "t": 2},
                    "fosta_no_prior_normalization": {"prior_correct": False, "euclidean_mode": False, "t": 2}, 
                    "fosta_euclidean": {"prior_correct": True, "euclidean_mode": True, "t": 2},
                    "fosta_euclidean_t_auto": {"prior_correct": True, "euclidean_mode": True, "t": "auto"},
                    "RFMALI": {"t": 2}
                    }

# euclidean vs. rfgap
# prior vs no prior normalization

# FoSTA vs. RFMALI




for method, params in fosta_params_dict.items():   
    method_start_time = time.perf_counter()
    current_mem_start, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try: 
        x_source = np.array(df2)
        x_target = np.array(df1)
        y_source = np.array(labels)
        y_target = labels_masked
        
        embedding = ablation_fosta(method_name = method, params_dict = params, x_source = x_source, x_target = x_target, y_source = y_source, y_target = y_target)  
        
        elapsed_seconds = time.perf_counter() - method_start_time
        _, peak_mem_after = tracemalloc.get_traced_memory()
        peak_mem_delta_mb = max(0.0, (peak_mem_after - current_mem_start) / (1024 ** 2))
        
        
        # viz_save_path = f"{save_path_subfolder}/{method}.png"
        # visualization(embedding, y_source, y_target, title=f"{data_name} - {method} - {split} split", save_path=viz_save_path, seed= seed)
        
        # plot with matching lines
        # T_true = np.eye(len(y_source))
        # visualization(embedding, y_source, y_target, title=f"{data_name} - {method} - {split} split", T_true = T_true, save_path=viz_save_path.replace(".png", "_matching_lines.png"), seed= seed)
        
        mask_missing_target_full = (y_target == -1) 
        res, sil_dom, foscttm, alignment_score = run_metrics(embedding, y_source, y_target, y_target_true = labels, mask_missing_target_full = mask_missing_target_full)

        
        result_row = {"model": method, 
                  "FOSCTTM": foscttm, 
                  "Silhouette_domain": sil_dom,
                  "Accuracy_missing": res['acc_missing'], 
                  "Accuracy_visible": res['acc_visible'],
                  "Alignment_score": alignment_score,
                  "Time_seconds": elapsed_seconds,
                  "Peak_memory_MB": peak_mem_delta_mb}
    
    except Exception as e:
        print(f"Method {method} failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        elapsed_seconds = time.perf_counter() - method_start_time
        _, peak_mem_after = tracemalloc.get_traced_memory()
        peak_mem_delta_mb = max(0.0, (peak_mem_after - current_mem_start) / (1024 ** 2))
        result_row = {"model": method, 
                  "FOSCTTM": None,
                  "Silhouette_domain": None,
                  "Accuracy_missing": None, 
                  "Accuracy_visible": None,
                  "Alignment_score": None,
                  "Time_seconds": elapsed_seconds,
                  "Peak_memory_MB": peak_mem_delta_mb}
    
    rows_list.append(result_row)

tracemalloc.stop()
            

results_df = pd.DataFrame(rows_list) 
results_df.insert(0, "dataset", data_name)   
results_df.insert(0, "split", split)
results_df.insert(0, "seed", seed)


print("Saving results...")
results_df.to_csv(f"{save_path_subfolder}/simulated_multimodal_results.csv", index=False)

# update results file (add new results to existing file if exists)
results_file = f"{save_path}/simulated_multimodal_results.csv"
if os.path.exists(results_file):
    existing_results = pd.read_csv(results_file)
    results_df = pd.concat([existing_results, results_df], ignore_index=True)

results_df.to_csv(results_file, index=False)
                  

        #%%     



