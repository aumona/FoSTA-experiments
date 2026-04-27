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
from utils.script_utils import main_argparser, prepare_adata, run_benchmarks, evaluate_and_save_results, save_embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from personal_paths import BASE_PATH, LUNG_BATCHES_DATA_PATH, RESULTS_PATH
base_path = BASE_PATH
scratch_path = RESULTS_PATH
data_path = LUNG_BATCHES_DATA_PATH


parser = main_argparser()
parser.add_argument('--batch1', default = "4")
parser.add_argument('--batch2', default = "5")
args = parser.parse_args()

batches = [args.batch1, args.batch2]

# set save location 
save_name = f"{args.savename}/{batches[0]}_{batches[1]}" #dataset}".format(dataset = dataset_name)
save_path = f"{scratch_path}/{save_name}"

# LOAD DATA 
adata_full = sc.read(data_path)

label_key = "cell_type"
batch_key = "batch"

# subset to the current batches
adata = adata_full[(adata_full.obs["batch"].isin(batches))].copy()
# adata, labels_not_in1, labels_not_in2 = remove_dataset_specific_cells(adata, batch_key, label_key)


methods_params_dict = {"FoSTA t=2": {"kernel_method": "original", 
                                    "ot_solver": "hiref", 
                                    "t": 2},
                       "FoSTA t=auto": {"kernel_method": "original", 
                                        "ot_solver": "hiref", 
                                        "t": "auto"},
                       "Scanorama":{},
                        "LIGER":{},
                        "Harmony":{},
                        "scVI":{},
                        "scANVI":{},
                        "Pamona":{},
                        "KEMArbf": {},
                        "KEMAlin": {}
                    }


adata, original_label_key, label_key = prepare_adata(adata, save_path, label_key, batch_key, args=args)
adata, times_dict, memory_dict = run_methods(adata, save_path, label_key, batch_key, methods_params_dict=methods_params_dict, args=args)
evaluate_and_save_results(adata, save_path, original_label_key, batch_key, times_dict, memory_dict, args)
save_embeddings(adata, save_path, methods_to_benchmark, label_key, batch_key)
